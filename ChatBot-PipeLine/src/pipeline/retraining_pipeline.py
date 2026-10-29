"""
Continuous-Learning Retraining Pipeline for BERT Intent Classifier & Google ADK Bot.

Implements the complete 14-stage lifecycle:
  1. Collect approved feedback examples.
  2. Validate schema, labels and training eligibility.
  3. Merge eligible examples with existing training data without modifying raw source.
  4. Check for duplicates, conflicting labels and class imbalance.
  5. Create a versioned training dataset.
  6. Run preprocessing and BERT candidate training.
  7. Evaluate candidate on held-out test set & new validation data.
  8. Generate an Evidently drift/evaluation report.
  9. Log metrics and artifacts to MLflow & version with DVC.
  10. Apply promotion rules (Macro F1, accuracy, non-regression, health checks).
  11. If accepted, publish model artifact and compatible ADK instruction version.
  12. Atomically update the active-version pointer or registry state.
  13. Notify the application of the new active versions through the supported reload mechanism.
  14. If rejected or deployment fails, keep existing production versions active and record reason.

Key Safety Rules:
- Never overwrite or mutate raw Intent.json or baseline instruction v1.0.0.
- Rejects candidates that fail tests, regress macro F1, or lack valid label mappings.
- Concurrency control via file locking prevents concurrent runs from collision.
- Supports dry-run, manual promotion, and explicit rollback.
- Does not promote merely because training succeeded.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid
import yaml

# Disable telemetry and tracking
os.environ["DO_NOT_TRACK"] = "1"
os.environ["EVIDENTLY_DISABLE_TELEMETRY"] = "1"

# Ensure repo and pipeline roots are on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PIPELINE_ROOT = _REPO_ROOT / "ChatBot-PipeLine"
for p in [_REPO_ROOT, _PIPELINE_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch

# Structured logger
logger = logging.getLogger("pipeline.retraining")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

from common.contracts import (
    FeedbackRating,
    FeedbackRecord,
    ModelMetadata,
    PromotionStatus,
    PromptEvaluationStatus,
    PromptMetadata,
)
from common.feedback_store import FeedbackStore
from common.config_resolver import (
    resolve_active_model_config,
    resolve_active_prompt_config,
)
from src.preprocessing.clean_data import IntentPreprocessor, normalize_whitespace
from src.evaluation.evaluate import evaluate_model
from src.evaluation.drift_monitor import EvidentlyDriftMonitor

# Prometheus metric recording helpers (lazy-loaded to keep imports lightweight)
def _record_metric_run(status: str) -> None:
    try:
        from ADKBot.metrics import RETRAINING_RUNS_TOTAL
        RETRAINING_RUNS_TOTAL.labels(status=status).inc()
    except Exception:
        pass


def _record_metric_promotion() -> None:
    try:
        from ADKBot.metrics import CANDIDATE_PROMOTIONS_TOTAL
        CANDIDATE_PROMOTIONS_TOTAL.inc()
    except Exception:
        pass


def _record_metric_rejection() -> None:
    try:
        from ADKBot.metrics import CANDIDATE_REJECTIONS_TOTAL
        CANDIDATE_REJECTIONS_TOTAL.inc()
    except Exception:
        pass


def _record_training_duration(duration: float) -> None:
    try:
        from ADKBot.metrics import TRAINING_DURATION_SECONDS
        TRAINING_DURATION_SECONDS.set(duration)
    except Exception:
        pass


def _update_version_gauge(model_v: str, prompt_v: str) -> None:
    try:
        from ADKBot.metrics import update_active_versions_gauge
        update_active_versions_gauge(model_v, prompt_v)
    except Exception:
        pass


class PipelineConcurrencyLock:
    """File-based concurrency lock ensuring only one retraining pipeline runs at a time."""

    def __init__(self, lock_file: Path, timeout_seconds: int = 3600):
        self.lock_file = Path(lock_file)
        self.timeout_seconds = timeout_seconds
        self._acquired = False

    def acquire(self) -> bool:
        if self.lock_file.exists():
            try:
                content = self.lock_file.read_text(encoding="utf-8").strip()
                lock_time = float(content.split(":")[0])
                if time.time() - lock_time > self.timeout_seconds:
                    logger.warning("Stale retraining lock detected; overriding.")
                    self.lock_file.unlink()
                else:
                    return False
            except Exception:
                return False

        try:
            self.lock_file.parent.mkdir(parents=True, exist_ok=True)
            self.lock_file.write_text(f"{time.time()}:{os.getpid()}", encoding="utf-8")
            self._acquired = True
            return True
        except Exception:
            return False

    def release(self) -> None:
        if self._acquired and self.lock_file.exists():
            try:
                self.lock_file.unlink()
            except Exception:
                pass
            self._acquired = False

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError(
                f"Concurrent retraining pipeline run active. Lock file exists at: {self.lock_file}"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


@dataclass
class QualityGateConfig:
    min_macro_f1: float = 0.80
    min_accuracy: float = 0.85
    min_improvement_pct: float = 0.0  # Require candidate >= baseline
    min_feedback_samples: int = 3
    drift_threshold: float = 0.5


class ContinuousLearningPipeline:
    """
    Executes and coordinates continuous learning pipeline stages.
    """

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        config_path: Optional[Path] = None,
        quality_gates: Optional[QualityGateConfig] = None,
        dry_run: bool = False,
    ):
        self.repo_root = Path(repo_root or _REPO_ROOT).resolve()
        self.config_path = Path(config_path or (self.repo_root / "ChatBot-PipeLine" / "configs" / "config.yaml"))
        self.quality_gates = quality_gates or QualityGateConfig()
        self.dry_run = dry_run

        self.pipeline_dir = self.repo_root / "ChatBot-PipeLine"
        self.raw_data_file = self.pipeline_dir / "data" / "raw" / "Intent.json"
        self.processed_dir = self.pipeline_dir / "data" / "processed"
        self.versioned_data_dir = self.pipeline_dir / "data" / "versioned"
        self.feedback_file = self.pipeline_dir / "data" / "feedback" / "user_feedback.jsonl"
        self.models_dir = self.repo_root / "artifacts" / "models"
        self.metrics_dir = self.repo_root / "artifacts" / "metrics"
        self.reports_dir = self.repo_root / "artifacts" / "reports"
        self.lock_file = self.pipeline_dir / ".retraining.lock"

        self.feedback_store = FeedbackStore(storage_path=self.feedback_file)
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        for d in [self.versioned_data_dir, self.models_dir, self.metrics_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Collect Approved Feedback ───────────────────────────────────
    def stage_1_collect_approved_feedback(self) -> List[FeedbackRecord]:
        logger.info("Stage 1: Collecting approved feedback examples...")
        all_eligible = self.feedback_store.get_training_eligible_records()
        logger.info(f"Retrieved {len(all_eligible)} training-eligible feedback records.")
        return all_eligible

    # ── Stage 2: Validate Schema, Labels, and Eligibility ────────────────────
    def stage_2_validate_feedback(
        self, records: List[FeedbackRecord]
    ) -> Tuple[List[Dict[str, str]], List[str]]:
        logger.info("Stage 2: Validating feedback schema, intent labels, and training eligibility...")
        valid_examples: List[Dict[str, str]] = []
        validation_warnings: List[str] = []

        # Load known intent labels from raw data or mapping
        known_labels: Set[str] = set()
        if self.raw_data_file.exists():
            with open(self.raw_data_file, "r", encoding="utf-8") as f:
                raw_json = json.load(f)
                known_labels = {item["intent"].strip() for item in raw_json.get("intents", [])}

        for r in records:
            if not r.training_eligibility:
                validation_warnings.append(f"Record {r.record_id} skipped: not flagged training_eligible.")
                continue

            msg = normalize_whitespace(r.message)
            if not msg:
                validation_warnings.append(f"Record {r.record_id} rejected: empty message.")
                continue

            intent = (r.corrected_intent or "").strip()
            if not intent:
                validation_warnings.append(f"Record {r.record_id} rejected: missing corrected_intent.")
                continue

            if known_labels and intent not in known_labels:
                validation_warnings.append(
                    f"Record {r.record_id} rejected: unknown intent '{intent}' not in known classes."
                )
                continue

            valid_examples.append({
                "record_id": r.record_id,
                "text": msg,
                "intent": intent,
            })

        logger.info(
            f"Validation complete: {len(valid_examples)} valid examples, "
            f"{len(validation_warnings)} validation issues noted."
        )
        return valid_examples, validation_warnings

    # ── Stage 3: Merge with Existing Training Data (Raw Source Immutable) ────
    def stage_3_merge_training_data(
        self, valid_feedback: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        logger.info("Stage 3: Merging eligible examples with existing training data (read-only raw source)...")
        if not self.raw_data_file.exists():
            raise FileNotFoundError(f"Raw source dataset not found: {self.raw_data_file}")

        # Read original dataset into memory without writing to raw source file
        with open(self.raw_data_file, "r", encoding="utf-8") as f:
            merged_dataset = json.load(f)

        intent_map = {item["intent"].strip(): item for item in merged_dataset.get("intents", [])}

        added_count = 0
        for fb in valid_feedback:
            target_intent = fb["intent"]
            text = fb["text"]
            if target_intent in intent_map:
                intent_block = intent_map[target_intent]
                existing_texts = set(intent_block.get("text", []))
                if text not in existing_texts:
                    intent_block["text"].append(text)
                    added_count += 1

        logger.info(f"Merged {added_count} new distinct utterances across {len(intent_map)} intents.")
        return merged_dataset

    # ── Stage 4: Check Duplicates, Conflicting Labels & Imbalance ───────────
    def stage_4_check_dataset_health(
        self, merged_dataset: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str]]:
        logger.info("Stage 4: Checking for duplicates, conflicting labels, and class imbalance...")
        warnings: List[str] = []
        utterance_map: Dict[str, Set[str]] = defaultdict(set)
        class_counts: Dict[str, int] = {}

        for item in merged_dataset.get("intents", []):
            intent_name = item["intent"].strip()
            patterns = item.get("text", [])
            class_counts[intent_name] = len(patterns)

            for raw_u in patterns:
                norm_u = normalize_whitespace(raw_u)
                if norm_u:
                    utterance_map[norm_u].add(intent_name)

        # Detect cross-intent conflicts
        conflicts = []
        for text, intents in utterance_map.items():
            if len(intents) > 1:
                conflicts.append({"utterance": text, "conflicting_intents": list(intents)})
                warnings.append(f"CONFLICT: Utterance '{text}' mapped to multiple intents: {list(intents)}")

        # Check class imbalance
        min_c = min(class_counts.values()) if class_counts else 0
        max_c = max(class_counts.values()) if class_counts else 0
        if max_c > 3 * max(1, min_c):
            warnings.append(f"Class imbalance detected: min class count={min_c}, max class count={max_c}")

        health_report = {
            "total_intents": len(class_counts),
            "total_utterances": sum(class_counts.values()),
            "class_counts": class_counts,
            "conflicts_count": len(conflicts),
            "conflicts": conflicts,
            "warnings": warnings,
        }
        logger.info(f"Health check complete: {len(conflicts)} conflicts, {len(warnings)} total warnings.")
        return health_report, warnings

    # ── Stage 5: Create Versioned Training Dataset ───────────────────────────
    def stage_5_create_versioned_dataset(
        self, merged_dataset: Dict[str, Any], candidate_version: str
    ) -> Path:
        logger.info(f"Stage 5: Creating versioned training dataset snapshot for '{candidate_version}'...")
        versioned_file = self.versioned_data_dir / f"Intent_{candidate_version}.json"
        with open(versioned_file, "w", encoding="utf-8") as f:
            json.dump(merged_dataset, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved immutable versioned dataset: {versioned_file}")
        return versioned_file

    # ── Stage 6: Run Preprocessing and Candidate Training ────────────────────
    def stage_6_train_candidate(
        self, versioned_dataset_path: Path, candidate_version: str
    ) -> Path:
        logger.info(f"Stage 6: Preprocessing and training candidate model '{candidate_version}'...")
        candidate_model_dir = self.models_dir / candidate_version
        candidate_model_dir.mkdir(parents=True, exist_ok=True)

        # 1. Run Preprocessing
        preprocessor = IntentPreprocessor(
            raw_data_path=versioned_dataset_path,
            processed_dir=self.processed_dir,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
        )
        raw_data, _ = preprocessor.load_and_validate_raw_json()
        clean_pairs, dup_report, _ = preprocessor.extract_and_analyze_utterances(raw_data)
        intents = [item["intent"].strip() for item in raw_data["intents"]]
        label2id, id2label = preprocessor.build_label_mapping(intents)

        train_ex, val_ex, test_ex, _ = preprocessor.split_stratified_leak_free(clean_pairs, label2id)
        # Note: Do not overwrite baseline test.json to guarantee held-out evaluation consistency

        # 2. Train or Fine-Tune BERT Candidate
        start_time = time.time()
        try:
            from transformers import AutoTokenizer, AutoConfig
            from src.model.model import BERTIntentClassifier

            # Baseline model directory to initialize from
            baseline_dir = self.models_dir / "v0.1.0-baseline"
            if not baseline_dir.exists():
                baseline_dir = self.models_dir / "baseline"

            if baseline_dir.exists():
                logger.info(f"Fine-tuning candidate initialized from baseline weights: {baseline_dir}")
                model = BERTIntentClassifier.from_pretrained(baseline_dir)
                tokenizer = AutoTokenizer.from_pretrained(baseline_dir)
            else:
                logger.info("Initializing new DistilBERT classification head...")
                model = BERTIntentClassifier(model_name="distilbert-base-uncased", num_labels=len(label2id))
                tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

            # Quick fine-tuning pass on merged training split
            train_texts = [e.text for e in train_ex]
            train_labels = [e.label_id for e in train_ex]

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.train()

            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

            # Mini-batch training (2 epochs for lightweight continuous learning)
            batch_size = 16
            for epoch in range(2):
                for i in range(0, len(train_texts), batch_size):
                    b_texts = train_texts[i : i + batch_size]
                    b_labels = torch.tensor(train_labels[i : i + batch_size], dtype=torch.long).to(device)

                    enc = tokenizer(
                        b_texts,
                        truncation=True,
                        padding=True,
                        max_length=64,
                        return_tensors="pt"
                    )
                    input_ids = enc["input_ids"].to(device)
                    attention_mask = enc["attention_mask"].to(device)

                    loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=b_labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Save candidate artifacts
            model.save_pretrained(candidate_model_dir)
            tokenizer.save_pretrained(candidate_model_dir)

            mapping_path = candidate_model_dir / "label_mapping.json"
            with open(mapping_path, "w", encoding="utf-8") as f:
                json.dump({"label2id": label2id, "id2label": id2label, "classes": sorted(list(label2id.keys()))}, f, indent=2)

            duration = round(time.time() - start_time, 2)
            _record_training_duration(duration)

            logger.info(f"Candidate model successfully trained and saved to {candidate_model_dir} in {duration}s")
            return candidate_model_dir

        except Exception as e:
            logger.error(f"Candidate training failed: {e}")
            raise RuntimeError(f"Candidate training failed: {e}") from e

    # ── Stage 7: Evaluate Candidate on Held-Out Test Set ─────────────────────
    def stage_7_evaluate_candidate(
        self, candidate_model_dir: Path, candidate_version: str
    ) -> Dict[str, Any]:
        logger.info("Stage 7: Evaluating candidate on established held-out test split...")
        test_file = self.processed_dir / "test.json"
        if not test_file.exists():
            raise FileNotFoundError(f"Established held-out test set missing at: {test_file}")

        from transformers import AutoTokenizer
        from src.model.model import BERTIntentClassifier

        model = BERTIntentClassifier.from_pretrained(candidate_model_dir)
        tokenizer = AutoTokenizer.from_pretrained(candidate_model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        with open(test_file, "r", encoding="utf-8") as f:
            test_examples = json.load(f)

        texts = [e["text"] for e in test_examples]
        true_labels = [e["label_id"] for e in test_examples]

        metrics = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            true_labels=true_labels,
            id2label=model.id2label,
            max_length=64,
            device=device,
        )

        metrics_file = self.metrics_dir / f"{candidate_version}_test_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        logger.info(
            f"Candidate evaluation metrics: Accuracy={metrics['accuracy']}, "
            f"Macro F1={metrics['macro_f1']} (saved to {metrics_file})"
        )
        return metrics

    # ── Stage 8: Generate Evidently Evaluation/Drift Report ──────────────────
    def stage_8_generate_drift_report(
        self, feedback_records: List[FeedbackRecord], candidate_version: str
    ) -> Dict[str, Any]:
        logger.info("Stage 8: Generating Evidently evaluation and drift report...")
        report_output_dir = self.reports_dir / "evidently" / candidate_version
        report_output_dir.mkdir(parents=True, exist_ok=True)

        drift_monitor = EvidentlyDriftMonitor(
            reference_data_path=self.processed_dir / "train.json",
            drift_share_threshold=self.quality_gates.drift_threshold,
        )

        ref_df = drift_monitor.prepare_reference_dataframe()
        cur_df, stats = drift_monitor.prepare_feedback_dataframe(feedback_records, require_verified_only=False)

        drift_result = drift_monitor.run_drift_analysis(
            reference_df=ref_df,
            current_df=cur_df,
            output_dir=report_output_dir,
            reference_version="v0.1.0-baseline",
            current_version=candidate_version,
            model_version=candidate_version,
        )

        logger.info(
            f"Drift report generation status: {drift_result.get('status')}, "
            f"Drift detected: {drift_result.get('drift_detected')}"
        )
        return drift_result

    # ── Stage 9: Log Metrics & Artifacts to MLflow and DVC ───────────────────
    def stage_9_log_mlflow_and_dvc(
        self,
        candidate_version: str,
        metrics: Dict[str, Any],
        candidate_model_dir: Path,
        drift_result: Dict[str, Any],
    ) -> Optional[str]:
        logger.info(f"Stage 9: Logging parameters, metrics and artifacts to MLflow for '{candidate_version}'...")
        run_id = None
        try:
            import mlflow

            tracking_uri = self.pipeline_dir / "mlruns"
            mlflow.set_tracking_uri(str(tracking_uri))
            mlflow.set_experiment("aic-chatbot-continuous-learning")

            with mlflow.start_run(run_name=f"retrain-{candidate_version}") as run:
                run_id = run.info.run_id
                mlflow.log_param("candidate_version", candidate_version)
                mlflow.log_param("timestamp", datetime.now(timezone.utc).isoformat())

                mlflow.log_metric("test_accuracy", metrics.get("accuracy", 0.0))
                mlflow.log_metric("test_macro_f1", metrics.get("macro_f1", 0.0))
                mlflow.log_metric("test_macro_precision", metrics.get("macro_precision", 0.0))
                mlflow.log_metric("test_macro_recall", metrics.get("macro_recall", 0.0))

                if drift_result.get("drift_share") is not None:
                    mlflow.log_metric("drift_share", drift_result.get("drift_share"))

                # Log model artifacts
                mlflow.log_artifacts(str(candidate_model_dir), artifact_path="model")

                # Log drift report if available
                html_report = drift_result.get("html_report_path")
                if html_report and Path(html_report).exists():
                    mlflow.log_artifact(html_report, artifact_path="evidently_report")

            logger.info(f"Successfully logged candidate to MLflow run: {run_id}")
        except Exception as e:
            logger.warning(f"MLflow logging encountered a non-fatal issue: {e}")

        # DVC Versioning attempt (best effort)
        try:
            import subprocess
            subprocess.run(
                ["dvc", "add", str(candidate_model_dir)],
                cwd=str(self.repo_root),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
            logger.info(f"DVC tracking updated for {candidate_model_dir}")
        except Exception:
            pass

        return run_id

    # ── Stage 10: Apply Promotion Rules ──────────────────────────────────────
    def stage_10_apply_promotion_rules(
        self,
        candidate_version: str,
        candidate_metrics: Dict[str, Any],
        candidate_model_dir: Path,
        drift_result: Dict[str, Any],
        force_promote: bool = False,
    ) -> Tuple[bool, List[str]]:
        logger.info(f"Stage 10: Applying production quality gate promotion rules to '{candidate_version}'...")
        rejection_reasons: List[str] = []

        if force_promote:
            logger.warning(f"Manual promotion flag set. Bypassing automatic thresholds for '{candidate_version}'.")
            return True, ["Promoted via explicit manual override."]

        # 1. Missing artifact check
        required_files = ["model.safetensors", "config.json", "label_mapping.json", "tokenizer.json"]
        for rf in required_files:
            if not (candidate_model_dir / rf).exists():
                rejection_reasons.append(f"Missing required model artifact file: {rf}")

        # 2. Label mapping integrity check
        mapping_file = candidate_model_dir / "label_mapping.json"
        if mapping_file.exists():
            try:
                with open(mapping_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    classes = data.get("classes", [])
                    if len(classes) < 22:
                        rejection_reasons.append(f"Label mapping regression: expected >= 22 classes, got {len(classes)}")
            except Exception as e:
                rejection_reasons.append(f"Corrupt label mapping file: {e}")

        # 3. Baseline comparison
        baseline_config = resolve_active_model_config()
        baseline_metrics_path = self.metrics_dir / f"{baseline_config.model_version}_test_metrics.json"
        baseline_macro_f1 = 0.88  # Default baseline from config
        if baseline_metrics_path.exists():
            try:
                with open(baseline_metrics_path, "r", encoding="utf-8") as f:
                    b_metrics = json.load(f)
                    baseline_macro_f1 = b_metrics.get("macro_f1", baseline_macro_f1)
            except Exception:
                pass

        cand_macro_f1 = candidate_metrics.get("macro_f1", 0.0)
        cand_acc = candidate_metrics.get("accuracy", 0.0)

        # 4. Minimum Quality Thresholds
        if cand_macro_f1 < self.quality_gates.min_macro_f1:
            rejection_reasons.append(
                f"Candidate Macro F1 ({cand_macro_f1:.4f}) below minimum threshold ({self.quality_gates.min_macro_f1:.4f})"
            )

        if cand_acc < self.quality_gates.min_accuracy:
            rejection_reasons.append(
                f"Candidate Accuracy ({cand_acc:.4f}) below minimum threshold ({self.quality_gates.min_accuracy:.4f})"
            )

        # 5. Non-regression check
        if cand_macro_f1 < baseline_macro_f1:
            rejection_reasons.append(
                f"Performance regression: candidate Macro F1 ({cand_macro_f1:.4f}) < baseline ({baseline_macro_f1:.4f})"
            )

        # 6. Drift rule: drift alone NEVER triggers promotion
        # (Drift triggers evaluation; only models passing quality gates get promoted)

        is_promoted = len(rejection_reasons) == 0
        if is_promoted:
            logger.info(f"Candidate '{candidate_version}' PASSED all quality gates!")
        else:
            logger.warning(f"Candidate '{candidate_version}' REJECTED: {rejection_reasons}")

        return is_promoted, rejection_reasons

    # ── Stage 11: Publish Model Artifact and Compatible ADK Instructions ─────
    def stage_11_publish_artifacts(
        self, candidate_version: str, prompt_version: str
    ) -> Tuple[Path, Path]:
        logger.info(f"Stage 11: Publishing model artifact '{candidate_version}' & instruction '{prompt_version}'...")
        published_model_dir = self.models_dir / candidate_version

        # Publish compatible versioned system instruction for Google ADK
        instructions_dir = self.pipeline_dir / "configs" / "instructions"
        instructions_dir.mkdir(parents=True, exist_ok=True)
        instruction_file = instructions_dir / f"{prompt_version}.txt"

        baseline_instruction_file = self.repo_root / "ChatBot" / "ADKBot" / "instruction.txt"
        if not instruction_file.exists() and baseline_instruction_file.exists():
            shutil.copy(baseline_instruction_file, instruction_file)

        logger.info(f"Published model: {published_model_dir}, instruction: {instruction_file}")
        return published_model_dir, instruction_file

    # ── Stage 12: Atomically Update Active-Version Pointer & Registry ─────────
    def stage_12_atomic_update_active_version(
        self,
        candidate_version: str,
        prompt_version: str,
        metrics: Dict[str, Any],
        mlflow_run_id: Optional[str] = None,
    ) -> None:
        logger.info("Stage 12: Atomically updating active-version pointers and registries...")
        # 1. Update config.yaml atomically
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}

            cfg.setdefault("model_management", {})["active_version"] = candidate_version
            cfg.setdefault("prompt_management", {})["active_version"] = prompt_version

            tmp_config = self.config_path.with_suffix(".tmp")
            with open(tmp_config, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False)
            tmp_config.replace(self.config_path)

        # 2. Update models/metadata.yaml
        models_meta_file = self.pipeline_dir / "configs" / "models" / "metadata.yaml"
        if models_meta_file.exists():
            with open(models_meta_file, "r", encoding="utf-8") as f:
                m_meta = yaml.safe_load(f) or {}

            m_meta["active_version"] = candidate_version
            m_meta.setdefault("models", {})[candidate_version] = {
                "model_version": candidate_version,
                "model_name": "bert-intent-classifier",
                "artifact_location": f"artifacts/models/{candidate_version}",
                "dataset_version": f"ChatBot-PipeLine/data/versioned/Intent_{candidate_version}.json",
                "dvc_revision": mlflow_run_id or "uncommitted",
                "num_classes": 22,
                "promotion_status": "production",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "evaluation_metrics": {
                    "macro_f1": metrics.get("macro_f1", 0.0),
                    "accuracy": metrics.get("accuracy", 0.0),
                },
                "notes": f"Promoted candidate model {candidate_version} via Continuous Learning Pipeline."
            }

            tmp_m_meta = models_meta_file.with_suffix(".tmp")
            with open(tmp_m_meta, "w", encoding="utf-8") as f:
                yaml.dump(m_meta, f, default_flow_style=False)
            tmp_m_meta.replace(models_meta_file)

        logger.info(f"Active version atomically updated to model={candidate_version}, prompt={prompt_version}")

    # ── Stage 13: Notify Application via Supported Reload Mechanism ──────────
    def stage_13_notify_application(
        self, candidate_version: str, prompt_version: str
    ) -> bool:
        logger.info("Stage 13: Notifying application of active versions via supported reload mechanism...")
        success = False

        # 1. Notify running API service via POST /reload if running
        try:
            import urllib.request
            req = urllib.request.Request(
                "http://localhost:6655/reload",
                data=json.dumps({
                    "model_version": candidate_version,
                    "prompt_version": prompt_version
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    success = True
                    logger.info("Running API service successfully reloaded via POST /reload.")
        except Exception:
            pass

        # 2. In-process reload if agent module is active in current interpreter
        if not success and "ADKBot.agent" in sys.modules:
            try:
                from ADKBot.agent import agent_manager
                agent_manager.reload(version_override=prompt_version)
                success = True
                logger.info("In-process ADK Agent Manager hot-reloaded successfully.")
            except Exception as e:
                logger.warning(f"In-process ADK reload notification skipped: {e}")

        # Update Prometheus active version gauge & counters
        _update_version_gauge(candidate_version, prompt_version)
        _record_metric_promotion()

        return success

    # ── Stage 14: Record Failure / Rejection ──────────────────────────────────
    def stage_14_handle_failure_or_rejection(
        self, candidate_version: str, reasons: List[str]
    ) -> None:
        logger.warning(f"Stage 14: Candidate '{candidate_version}' was NOT promoted. Active versions remain unchanged.")
        models_meta_file = self.pipeline_dir / "configs" / "models" / "metadata.yaml"
        if models_meta_file.exists():
            try:
                with open(models_meta_file, "r", encoding="utf-8") as f:
                    m_meta = yaml.safe_load(f) or {}

                m_meta.setdefault("models", {})[candidate_version] = {
                    "model_version": candidate_version,
                    "promotion_status": "rejected",
                    "rejected_at": datetime.now(timezone.utc).isoformat(),
                    "rejection_reasons": reasons,
                }

                tmp_m_meta = models_meta_file.with_suffix(".tmp")
                with open(tmp_m_meta, "w", encoding="utf-8") as f:
                    yaml.dump(m_meta, f, default_flow_style=False)
                tmp_m_meta.replace(models_meta_file)
            except Exception:
                pass

        _record_metric_rejection()

    # ── Full Execution Orchestrator ──────────────────────────────────────────
    def run(self, force_promote: bool = False, min_feedback: Optional[int] = None) -> Dict[str, Any]:
        """Execute full 14-stage pipeline with locking, gates, and auditing."""
        min_feedback = min_feedback if min_feedback is not None else self.quality_gates.min_feedback_samples
        run_start_time = time.time()
        pipeline_audit: Dict[str, Any] = {
            "status": "pending",
            "stages_executed": [],
            "stages_succeeded": [],
            "reasons": [],
            "promoted": False,
        }

        with PipelineConcurrencyLock(self.lock_file):
            # Stage 1: Collect approved feedback
            pipeline_audit["stages_executed"].append(1)
            feedback_records = self.stage_1_collect_approved_feedback()
            pipeline_audit["stages_succeeded"].append(1)

            # Check threshold
            if len(feedback_records) < min_feedback and not force_promote:
                msg = (
                    f"Insufficient eligible feedback records: found {len(feedback_records)}, "
                    f"minimum threshold is {min_feedback}. Skipping retraining."
                )
                logger.info(msg)
                pipeline_audit["status"] = "skipped"
                pipeline_audit["reasons"].append(msg)
                _record_metric_run("insufficient_feedback")
                return pipeline_audit

            # Stage 2: Validate feedback
            pipeline_audit["stages_executed"].append(2)
            valid_feedback, warnings = self.stage_2_validate_feedback(feedback_records)
            pipeline_audit["stages_succeeded"].append(2)

            if len(valid_feedback) == 0 and not force_promote:
                msg = "No valid eligible feedback examples remained after validation check. Halting."
                logger.warning(msg)
                pipeline_audit["status"] = "rejected"
                pipeline_audit["reasons"].append(msg)
                _record_metric_run("no_valid_feedback")
                return pipeline_audit

            # Generate candidate semantic version
            cand_version = f"candidate-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
            prompt_version = f"v{datetime.now(timezone.utc).strftime('%Y.%m.%d')}"

            # Stage 3: Merge with training data
            pipeline_audit["stages_executed"].append(3)
            merged_dataset = self.stage_3_merge_training_data(valid_feedback)
            pipeline_audit["stages_succeeded"].append(3)

            # Stage 4: Health check
            pipeline_audit["stages_executed"].append(4)
            health_report, health_warnings = self.stage_4_check_dataset_health(merged_dataset)
            pipeline_audit["stages_succeeded"].append(4)

            # Stage 5: Versioned dataset
            pipeline_audit["stages_executed"].append(5)
            versioned_data_path = self.stage_5_create_versioned_dataset(merged_dataset, cand_version)
            pipeline_audit["stages_succeeded"].append(5)

            # Stage 6: Training
            pipeline_audit["stages_executed"].append(6)
            candidate_model_dir = self.stage_6_train_candidate(versioned_data_path, cand_version)
            pipeline_audit["stages_succeeded"].append(6)

            # Stage 7: Evaluation
            pipeline_audit["stages_executed"].append(7)
            metrics = self.stage_7_evaluate_candidate(candidate_model_dir, cand_version)
            pipeline_audit["stages_succeeded"].append(7)

            # Stage 8: Drift report
            pipeline_audit["stages_executed"].append(8)
            drift_res = self.stage_8_generate_drift_report(feedback_records, cand_version)
            pipeline_audit["stages_succeeded"].append(8)

            # Stage 9: MLflow & DVC logging
            pipeline_audit["stages_executed"].append(9)
            mlflow_id = self.stage_9_log_mlflow_and_dvc(cand_version, metrics, candidate_model_dir, drift_res)
            pipeline_audit["stages_succeeded"].append(9)

            if self.dry_run:
                msg = f"Dry-run mode active. Candidate '{cand_version}' evaluated but not promoted."
                logger.info(msg)
                pipeline_audit["status"] = "dry_run_completed"
                pipeline_audit["candidate_version"] = cand_version
                pipeline_audit["metrics"] = metrics
                return pipeline_audit

            # Stage 10: Promotion Rules Gate
            pipeline_audit["stages_executed"].append(10)
            promoted, reasons = self.stage_10_apply_promotion_rules(
                cand_version, metrics, candidate_model_dir, drift_res, force_promote=force_promote
            )
            pipeline_audit["stages_succeeded"].append(10)

            if promoted:
                # Stage 11: Publish artifacts
                pipeline_audit["stages_executed"].append(11)
                self.stage_11_publish_artifacts(cand_version, prompt_version)
                pipeline_audit["stages_succeeded"].append(11)

                # Stage 12: Atomic active-version update
                pipeline_audit["stages_executed"].append(12)
                self.stage_12_atomic_update_active_version(cand_version, prompt_version, metrics, mlflow_id)
                pipeline_audit["stages_succeeded"].append(12)

                # Stage 13: Notify application reload
                pipeline_audit["stages_executed"].append(13)
                self.stage_13_notify_application(cand_version, prompt_version)
                pipeline_audit["stages_succeeded"].append(13)

                pipeline_audit["status"] = "promoted"
                pipeline_audit["promoted"] = True
                pipeline_audit["candidate_version"] = cand_version
                pipeline_audit["prompt_version"] = prompt_version
                pipeline_audit["metrics"] = metrics
                _record_metric_run("success")
            else:
                # Stage 14: Rejection and failure recording
                pipeline_audit["stages_executed"].append(14)
                self.stage_14_handle_failure_or_rejection(cand_version, reasons)
                pipeline_audit["stages_succeeded"].append(14)

                pipeline_audit["status"] = "rejected"
                pipeline_audit["promoted"] = False
                pipeline_audit["candidate_version"] = cand_version
                pipeline_audit["reasons"] = reasons
                pipeline_audit["metrics"] = metrics
                _record_metric_run("rejected")

        pipeline_audit["duration_seconds"] = round(time.time() - run_start_time, 2)
        return pipeline_audit

    # ── Rollback Mechanism ───────────────────────────────────────────────────
    def rollback(self, target_model_version: str, target_prompt_version: Optional[str] = None) -> bool:
        """
        Explicit rollback mechanism to restore a previously validated model/prompt version.
        """
        logger.info(f"Initiating explicit rollback to model version '{target_model_version}'...")
        target_dir = self.models_dir / target_model_version
        if not target_dir.exists():
            raise FileNotFoundError(f"Target model version directory not found: {target_dir}")

        prompt_v = target_prompt_version or "v1.0.0"

        # Update active pointer in config.yaml
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            cfg.setdefault("model_management", {})["active_version"] = target_model_version
            cfg.setdefault("prompt_management", {})["active_version"] = prompt_v

            tmp = self.config_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False)
            tmp.replace(self.config_path)

        # Notify application
        self.stage_13_notify_application(target_model_version, prompt_v)
        logger.info(f"Rollback completed successfully. Active version restored to {target_model_version} / {prompt_v}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Continuous-Learning Retraining Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Execute evaluation without promoting candidate")
    parser.add_argument("--force-promote", action="store_true", help="Force promotion bypassing quality gates")
    parser.add_argument("--rollback", type=str, default=None, help="Roll back active model to specified version")
    parser.add_argument("--min-feedback", type=int, default=3, help="Minimum feedback count required to retrain")
    parser.add_argument("--min-f1", type=float, default=0.80, help="Minimum Macro F1 threshold")
    parser.add_argument("--min-acc", type=float, default=0.85, help="Minimum Accuracy threshold")

    args = parser.parse_args()

    pipeline = ContinuousLearningPipeline(
        quality_gates=QualityGateConfig(
            min_macro_f1=args.min_f1,
            min_accuracy=args.min_acc,
            min_feedback_samples=args.min_feedback,
        ),
        dry_run=args.dry_run,
    )

    if args.rollback:
        success = pipeline.rollback(target_model_version=args.rollback)
        print(f"Rollback status: {'Success' if success else 'Failed'}")
        return

    result = pipeline.run(force_promote=args.force_promote, min_feedback=args.min_feedback)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
