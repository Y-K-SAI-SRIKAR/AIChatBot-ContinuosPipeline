"""
Integration and Unit Tests for Continuous-Learning Retraining Pipeline.

Covers:
- No eligible feedback records
- Insufficient new feedback records (< min_feedback threshold)
- Concurrency lock prevents duplicate simultaneous triggers
- Invalid model artifacts rejection
- Candidate quality gate rejection (e.g. macro F1 regression)
- Successful candidate promotion and registry updates
- Explicit rollback to known baseline
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import shutil
import tempfile
import yaml

from common.contracts import FeedbackRating, FeedbackRecord
from src.pipeline.retraining_pipeline import (
    ContinuousLearningPipeline,
    PipelineConcurrencyLock,
    QualityGateConfig,
)


@pytest.fixture
def mock_pipeline_env(tmp_path):
    """Sets up an isolated workspace for pipeline testing."""
    repo_root = tmp_path / "repo"
    pipeline_dir = repo_root / "ChatBot-PipeLine"
    configs_dir = pipeline_dir / "configs"
    data_dir = pipeline_dir / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    feedback_dir = data_dir / "feedback"
    models_dir = repo_root / "artifacts" / "models"
    metrics_dir = repo_root / "artifacts" / "metrics"
    chatbot_dir = repo_root / "ChatBot" / "ADKBot"

    for d in [raw_dir, processed_dir, feedback_dir, configs_dir / "models", configs_dir / "instructions", models_dir, metrics_dir, chatbot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Mock Raw Intent.json with 22 intents
    intents = []
    classes = [
        "Greeting", "TimeQuery", "CourtesyGreeting", "CurrentHumanQuery", "NameQuery",
        "RealNameQuery", "TimeQuery2", "GoodBye", "UnderstandQuery", "WhoAmI",
        "Clever", "SelfAware", "CourtesyGoodBye", "Jokes", "PodBayDoor",
        "PodBayDoorResponse", "NotTalking2U", "Shutup", "WhatAreYou", "WhatIsMyName",
        "WhereAreYou", "WhoMadeYou"
    ]
    for c in classes:
        intents.append({
            "intent": c,
            "text": [f"Example text 1 for {c}", f"Example text 2 for {c}", f"Example text 3 for {c}"],
            "responses": [f"Response for {c}"]
        })
    raw_file = raw_dir / "Intent.json"
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump({"intents": intents}, f, indent=2)

    # 2. Mock test.json in processed
    test_examples = []
    for idx, c in enumerate(classes):
        test_examples.append({"text": f"Test sample for {c}", "intent": c, "label_id": idx, "split": "test"})
    with open(processed_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_examples, f, indent=2)

    # 3. Mock train.json
    with open(processed_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(test_examples, f, indent=2)

    # 4. Mock instruction.txt and config.yaml
    (chatbot_dir / "instruction.txt").write_text("Baseline bot instruction", encoding="utf-8")
    (configs_dir / "instructions" / "v1.0.0.txt").write_text("Baseline bot instruction", encoding="utf-8")

    cfg_dict = {
        "model_management": {"active_version": "v0.1.0-baseline"},
        "prompt_management": {"active_version": "v1.0.0"},
    }
    with open(configs_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg_dict, f)

    # 5. Mock baseline model directory
    baseline_dir = models_dir / "v0.1.0-baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "model.safetensors").write_text("fake weights", encoding="utf-8")
    (baseline_dir / "config.json").write_text("{}", encoding="utf-8")
    (baseline_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    with open(baseline_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "label2id": {c: i for i, c in enumerate(classes)}}, f)

    # Baseline metrics
    with open(metrics_dir / "v0.1.0-baseline_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"macro_f1": 0.88, "accuracy": 0.90}, f)

    return repo_root


class TestRetrainingPipeline:

    def test_insufficient_new_examples(self, mock_pipeline_env):
        pipeline = ContinuousLearningPipeline(
            repo_root=mock_pipeline_env,
            quality_gates=QualityGateConfig(min_feedback_samples=5),
        )
        # Store is empty
        audit = pipeline.run()
        assert audit["status"] == "skipped"
        assert audit["promoted"] is False
        assert "Insufficient eligible feedback" in audit["reasons"][0]

    def test_no_eligible_feedback(self, mock_pipeline_env):
        # Insert feedback records with training_eligibility=False
        feedback_file = mock_pipeline_env / "ChatBot-PipeLine" / "data" / "feedback" / "user_feedback.jsonl"
        with open(feedback_file, "w", encoding="utf-8") as f:
            rec = FeedbackRecord(
                request_id="r1",
                message="bad response",
                rating=FeedbackRating.NEGATIVE,
                corrected_intent=None,  # Ineligible
                model_version="v0.1.0-baseline",
                prompt_version="v1.0.0",
            )
            f.write(rec.model_dump_json() + "\n")

        pipeline = ContinuousLearningPipeline(
            repo_root=mock_pipeline_env,
            quality_gates=QualityGateConfig(min_feedback_samples=1),
        )
        audit = pipeline.run()
        assert audit["status"] == "skipped"
        assert audit["promoted"] is False

    def test_concurrency_lock_prevents_duplicate_triggers(self, mock_pipeline_env):
        lock_file = mock_pipeline_env / "ChatBot-PipeLine" / ".retraining.lock"
        lock1 = PipelineConcurrencyLock(lock_file)
        assert lock1.acquire() is True

        # Second lock attempt must fail
        lock2 = PipelineConcurrencyLock(lock_file)
        assert lock2.acquire() is False

        # Release first lock
        lock1.release()
        assert lock2.acquire() is True
        lock2.release()

    def test_rejection_on_invalid_artifacts(self, mock_pipeline_env):
        pipeline = ContinuousLearningPipeline(repo_root=mock_pipeline_env)
        candidate_dir = mock_pipeline_env / "artifacts" / "models" / "cand-invalid"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        # Empty candidate dir (missing model.safetensors, config.json, etc.)

        promoted, reasons = pipeline.stage_10_apply_promotion_rules(
            candidate_version="cand-invalid",
            candidate_metrics={"macro_f1": 0.95, "accuracy": 0.95},
            candidate_model_dir=candidate_dir,
            drift_result={},
        )
        assert promoted is False
        assert any("Missing required model artifact" in r for r in reasons)

    def test_rejection_on_severe_metric_regression(self, mock_pipeline_env):
        pipeline = ContinuousLearningPipeline(
            repo_root=mock_pipeline_env,
            quality_gates=QualityGateConfig(min_macro_f1=0.80, min_accuracy=0.85),
        )
        candidate_dir = mock_pipeline_env / "artifacts" / "models" / "v0.1.0-baseline"

        # Candidate with low Macro F1 (e.g. 0.70 < min 0.80 and < baseline 0.88)
        promoted, reasons = pipeline.stage_10_apply_promotion_rules(
            candidate_version="cand-low-f1",
            candidate_metrics={"macro_f1": 0.70, "accuracy": 0.75},
            candidate_model_dir=candidate_dir,
            drift_result={},
        )
        assert promoted is False
        assert any("below minimum threshold" in r or "regression" in r for r in reasons)

    def test_successful_promotion_and_atomic_update(self, mock_pipeline_env):
        pipeline = ContinuousLearningPipeline(
            repo_root=mock_pipeline_env,
            quality_gates=QualityGateConfig(min_macro_f1=0.80, min_accuracy=0.85),
        )
        candidate_dir = mock_pipeline_env / "artifacts" / "models" / "cand-good"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(mock_pipeline_env / "artifacts" / "models" / "v0.1.0-baseline" / "model.safetensors", candidate_dir)
        shutil.copy(mock_pipeline_env / "artifacts" / "models" / "v0.1.0-baseline" / "config.json", candidate_dir)
        shutil.copy(mock_pipeline_env / "artifacts" / "models" / "v0.1.0-baseline" / "tokenizer.json", candidate_dir)
        shutil.copy(mock_pipeline_env / "artifacts" / "models" / "v0.1.0-baseline" / "label_mapping.json", candidate_dir)

        # High metrics exceeding baseline
        cand_metrics = {"macro_f1": 0.92, "accuracy": 0.94}
        promoted, reasons = pipeline.stage_10_apply_promotion_rules(
            candidate_version="cand-good",
            candidate_metrics=cand_metrics,
            candidate_model_dir=candidate_dir,
            drift_result={},
        )
        assert promoted is True

        # Run publication and atomic update
        pipeline.stage_11_publish_artifacts("cand-good", "v2.0.0")
        pipeline.stage_12_atomic_update_active_version("cand-good", "v2.0.0", cand_metrics)

        # Verify config.yaml updated
        with open(mock_pipeline_env / "ChatBot-PipeLine" / "configs" / "config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["model_management"]["active_version"] == "cand-good"
        assert cfg["prompt_management"]["active_version"] == "v2.0.0"

    def test_explicit_rollback_to_baseline(self, mock_pipeline_env):
        pipeline = ContinuousLearningPipeline(repo_root=mock_pipeline_env)
        # Roll back to baseline
        success = pipeline.rollback(target_model_version="v0.1.0-baseline", target_prompt_version="v1.0.0")
        assert success is True

        with open(mock_pipeline_env / "ChatBot-PipeLine" / "configs" / "config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["model_management"]["active_version"] == "v0.1.0-baseline"
        assert cfg["prompt_management"]["active_version"] == "v1.0.0"

