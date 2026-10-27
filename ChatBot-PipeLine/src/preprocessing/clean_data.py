"""
Preprocessing and Dataset Validation Module for Intent Classification.

This module processes raw intent JSON datasets (e.g. ChatBot-PipeLine/data/raw/Intent.json)
into validated, reproducible, stratified train/validation/test splits.

KEY REQUIREMENTS ENFORCED:
1. Never alters the original raw source file.
2. Validates JSON schema, intent names, utterance types, and empty values.
3. Normalizes whitespace while preserving original casing, words, and punctuation.
4. Detects intra-intent duplicates and cross-intent conflicts without silent overwrites.
5. Implements stratified, leak-free train/val/test splitting using a deterministic seed.
6. Reports limitations for tiny classes instead of fabricating artificial records.
7. Saves processed splits (train, val, test, training_data), label mappings, and summary report.
8. Exposes a reproducible CLI entry point suitable for DVC pipeline execution.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Setup structured logger
logger = logging.getLogger("preprocessing.clean_data")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class DatasetValidationError(ValueError):
    """Raised when the raw dataset violates structural or semantic requirements."""
    pass


@dataclass
class ProcessedExample:
    """Standardized representation of an individual training utterance."""
    text: str
    intent: str
    label_id: int
    split: str  # 'train' | 'val' | 'test'


@dataclass
class DuplicateReport:
    """Summary of duplicate utterances detected in the raw dataset."""
    intra_intent_duplicates: List[Dict[str, Any]] = field(default_factory=list)
    cross_intent_conflicts: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def has_conflicts(self) -> bool:
        return len(self.cross_intent_conflicts) > 0


@dataclass
class DatasetSummary:
    """Comprehensive dataset statistics, split sizes, and validation warnings."""
    raw_file_path: str
    timestamp: str
    random_seed: int
    total_raw_intents: int
    total_raw_examples: int
    total_unique_cleaned_examples: int
    intra_intent_duplicates_count: int
    cross_intent_conflicts_count: int
    train_size: int
    val_size: int
    test_size: int
    label_counts: Dict[str, int]
    split_distribution: Dict[str, Dict[str, int]]
    warnings: List[str] = field(default_factory=list)


def normalize_whitespace(text: str) -> str:
    """
    Normalize consecutive whitespace characters to a single space and strip leading/trailing spaces.
    Preserves all words, contractions, casing, and punctuation intact.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


class IntentPreprocessor:
    """
    Deterministic dataset validator, cleaner, and stratified splitter for Intent datasets.
    """

    def __init__(
        self,
        raw_data_path: Path,
        processed_dir: Path,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        seed: Optional[int] = None
    ):
        self.raw_data_path = Path(raw_data_path).resolve()
        self.processed_dir = Path(processed_dir).resolve()
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = seed if seed is not None else random_seed

        # Validate split proportions
        total_ratio = round(self.train_ratio + self.val_ratio + self.test_ratio, 4)
        if total_ratio != 1.0:
            raise ValueError(
                f"Split ratios must sum to 1.0 (got train={train_ratio}, "
                f"val={val_ratio}, test={test_ratio}; sum={total_ratio})"
            )

    def load_and_validate_raw_json(self) -> Tuple[Dict[str, Any], List[str]]:
        """
        Load raw JSON without modifying the file and validate required schema.
        Returns the parsed data and any non-fatal validation warnings.
        """
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw intent dataset not found at: {self.raw_data_path}")

        logger.info(f"Loading raw intent data from {self.raw_data_path} (read-only)...")
        try:
            with open(self.raw_data_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise DatasetValidationError(f"Invalid JSON in {self.raw_data_path}: {e}")

        warnings: List[str] = []

        if not isinstance(raw_data, dict):
            raise DatasetValidationError(
                f"Top-level JSON structure must be a dictionary/object, got {type(raw_data).__name__}"
            )

        if "intents" not in raw_data:
            raise DatasetValidationError(
                "Missing required top-level key 'intents' in dataset."
            )

        intents_list = raw_data["intents"]
        if not isinstance(intents_list, list):
            raise DatasetValidationError(
                f"'intents' key must map to a JSON list, got {type(intents_list).__name__}"
            )

        if len(intents_list) == 0:
            raise DatasetValidationError("'intents' list is empty. Expected at least 1 intent.")

        # Check each intent object
        seen_intent_names: Set[str] = set()
        for idx, item in enumerate(intents_list):
            if not isinstance(item, dict):
                raise DatasetValidationError(f"Intent item at index {idx} is not a dictionary.")

            intent_name = item.get("intent")
            if not intent_name or not isinstance(intent_name, str) or not intent_name.strip():
                raise DatasetValidationError(
                    f"Intent item at index {idx} has invalid or missing 'intent' name: {intent_name}"
                )

            clean_name = intent_name.strip()
            if clean_name in seen_intent_names:
                raise DatasetValidationError(
                    f"Duplicate intent block detected for name: '{clean_name}' at index {idx}."
                )
            seen_intent_names.add(clean_name)

            # Validate text array
            if "text" not in item:
                raise DatasetValidationError(
                    f"Intent '{clean_name}' at index {idx} is missing required 'text' array."
                )
            if not isinstance(item["text"], list):
                raise DatasetValidationError(
                    f"Intent '{clean_name}' 'text' field must be a list, got {type(item['text']).__name__}."
                )

            # Check optional schema fields (e.g. entityType, responses)
            if "entityType" not in item:
                warnings.append(
                    f"Intent '{clean_name}' is missing optional field 'entityType'. Will default to 'NA'."
                )
            if "responses" not in item or not isinstance(item["responses"], list) or len(item["responses"]) == 0:
                warnings.append(
                    f"Intent '{clean_name}' has missing or empty response templates."
                )

        logger.info(
            f"Successfully validated {len(intents_list)} intent blocks. "
            f"Warnings detected: {len(warnings)}"
        )
        return raw_data, warnings

    def extract_and_analyze_utterances(
        self, raw_data: Dict[str, Any]
    ) -> Tuple[List[Tuple[str, str]], DuplicateReport, List[str]]:
        """
        Extract utterances, normalize whitespace, detect duplicates and cross-intent conflicts.
        Returns:
            - List of unique cleaned (utterance_text, intent_name) pairs.
            - DuplicateReport containing intra-intent and cross-intent details.
            - List of extraction warnings.
        """
        warnings: List[str] = []
        duplicate_report = DuplicateReport()

        # Map normalized utterance -> list of (original_intent, original_text)
        utterance_registry: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        raw_count = 0

        for item in raw_data.get("intents", []):
            intent_name = item["intent"].strip()
            patterns = item.get("text", [])

            if len(patterns) == 0:
                warnings.append(f"Intent '{intent_name}' contains zero utterance examples.")
                continue

            for u_idx, raw_utterance in enumerate(patterns):
                raw_count += 1
                if not isinstance(raw_utterance, str):
                    warnings.append(
                        f"Non-string utterance at intent '{intent_name}' index {u_idx}: {type(raw_utterance)}"
                    )
                    continue

                cleaned_text = normalize_whitespace(raw_utterance)
                if not cleaned_text:
                    warnings.append(
                        f"Empty or whitespace-only utterance discarded in intent '{intent_name}' index {u_idx}."
                    )
                    continue

                utterance_registry[cleaned_text].append((intent_name, raw_utterance))

        # Analyze duplicates and conflicts
        unique_clean_pairs: List[Tuple[str, str]] = []

        for cleaned_text, occurrences in utterance_registry.items():
            intents_involved = {intent for intent, _ in occurrences}
            count = len(occurrences)

            if len(intents_involved) > 1:
                # Cross-intent conflict!
                conflict_info = {
                    "utterance": cleaned_text,
                    "intents": list(intents_involved),
                    "occurrences_count": count
                }
                duplicate_report.cross_intent_conflicts.append(conflict_info)
                warnings.append(
                    f"CONFLICT: Utterance '{cleaned_text}' appears in multiple conflicting intents: {list(intents_involved)}."
                )
                # Keep the first occurrence to avoid silent drops, but record warning
                unique_clean_pairs.append((cleaned_text, occurrences[0][0]))

            elif count > 1:
                # Intra-intent duplicate!
                intent_name = occurrences[0][0]
                duplicate_report.intra_intent_duplicates.append({
                    "utterance": cleaned_text,
                    "intent": intent_name,
                    "count": count
                })
                # Deduplicate: only 1 instance enters training to prevent split leakage
                unique_clean_pairs.append((cleaned_text, intent_name))
            else:
                # Exactly 1 occurrence
                unique_clean_pairs.append((cleaned_text, occurrences[0][0]))

        logger.info(
            f"Extracted {raw_count} raw utterances -> {len(unique_clean_pairs)} unique cleaned utterances. "
            f"Intra-intent duplicates: {len(duplicate_report.intra_intent_duplicates)}, "
            f"Cross-intent conflicts: {len(duplicate_report.cross_intent_conflicts)}."
        )
        return unique_clean_pairs, duplicate_report, warnings

    @staticmethod
    def build_label_mapping(intents: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Build a deterministic label2id and id2label mapping sorted alphabetically.
        """
        unique_sorted_intents = sorted(list(set(intents)))
        label2id = {name: idx for idx, name in enumerate(unique_sorted_intents)}
        id2label = {idx: name for idx, name in enumerate(unique_sorted_intents)}
        return label2id, id2label

    def split_stratified_leak_free(
        self,
        unique_pairs: List[Tuple[str, str]],
        label2id: Dict[str, int]
    ) -> Tuple[List[ProcessedExample], List[ProcessedExample], List[ProcessedExample], List[str]]:
        """
        Split examples into train, validation, and test sets.
        
        Stratification & Leak-Free Rules:
        - Deterministically shuffled per-intent using self.random_seed.
        - For classes with N >= 3: allocates >= 1 to train, >= 1 to val, >= 1 to test.
        - For classes with N < 3: reports limitation warnings instead of fabricating data.
        - Exactly 0 duplicate utterance overlap across splits.
        """
        rng = random.Random(self.random_seed)
        split_warnings: List[str] = []

        # Group examples by intent
        by_intent: Dict[str, List[str]] = defaultdict(list)
        for text, intent in unique_pairs:
            by_intent[intent].append(text)

        train_examples: List[ProcessedExample] = []
        val_examples: List[ProcessedExample] = []
        test_examples: List[ProcessedExample] = []

        # Process intents in sorted order for complete determinism
        for intent in sorted(by_intent.keys()):
            texts = by_intent[intent]
            # Deterministic shuffle
            shuffled_texts = list(texts)
            rng.shuffle(shuffled_texts)
            n = len(shuffled_texts)
            lbl_id = label2id[intent]

            if n < 3:
                split_warnings.append(
                    f"Class '{intent}' has only {n} example(s). Cannot populate all 3 splits; "
                    f"allocating available samples to train split."
                )
                for t in shuffled_texts:
                    train_examples.append(ProcessedExample(text=t, intent=intent, label_id=lbl_id, split="train"))
                continue

            # Target counts
            n_val = max(1, round(n * self.val_ratio))
            n_test = max(1, round(n * self.test_ratio))
            n_train = n - n_val - n_test

            # Ensure at least 1 in train if n >= 3
            if n_train < 1:
                n_train = 1
                if n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1

            train_texts = shuffled_texts[:n_train]
            val_texts = shuffled_texts[n_train:n_train + n_val]
            test_texts = shuffled_texts[n_train + n_val:]

            for t in train_texts:
                train_examples.append(ProcessedExample(text=t, intent=intent, label_id=lbl_id, split="train"))
            for t in val_texts:
                val_examples.append(ProcessedExample(text=t, intent=intent, label_id=lbl_id, split="val"))
            for t in test_texts:
                test_examples.append(ProcessedExample(text=t, intent=intent, label_id=lbl_id, split="test"))

        # Verify no leakage between splits
        train_set = {e.text for e in train_examples}
        val_set = {e.text for e in val_examples}
        test_set = {e.text for e in test_examples}

        leakage_tv = train_set.intersection(val_set)
        leakage_tt = train_set.intersection(test_set)
        leakage_vt = val_set.intersection(test_set)

        if leakage_tv or leakage_tt or leakage_vt:
            raise RuntimeError(
                f"Data leakage detected! Train/Val overlap: {len(leakage_tv)}, "
                f"Train/Test overlap: {len(leakage_tt)}, Val/Test overlap: {len(leakage_vt)}"
            )

        logger.info(
            f"Splits generated successfully: Train={len(train_examples)}, "
            f"Val={len(val_examples)}, Test={len(test_examples)}. Total={len(unique_pairs)}"
        )
        return train_examples, val_examples, test_examples, split_warnings

    def save_processed_outputs(
        self,
        train_examples: List[ProcessedExample],
        val_examples: List[ProcessedExample],
        test_examples: List[ProcessedExample],
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        summary: DatasetSummary
    ) -> Dict[str, Path]:
        """
        Save all processed artifacts, JSON splits, label mappings, and markdown summary.
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        saved_files: Dict[str, Path] = {}

        # 1. train.json
        train_path = self.processed_dir / "train.json"
        with open(train_path, "w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in train_examples], f, indent=2, ensure_ascii=False)
        saved_files["train"] = train_path

        # 2. val.json
        val_path = self.processed_dir / "val.json"
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in val_examples], f, indent=2, ensure_ascii=False)
        saved_files["val"] = val_path

        # 3. test.json
        test_path = self.processed_dir / "test.json"
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in test_examples], f, indent=2, ensure_ascii=False)
        saved_files["test"] = test_path

        # 4. training_data.json (combined full dataset with split tags)
        combined_path = self.processed_dir / "training_data.json"
        all_examples = [asdict(e) for e in (train_examples + val_examples + test_examples)]
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(all_examples, f, indent=2, ensure_ascii=False)
        saved_files["combined"] = combined_path

        # 5. label_mapping.json
        mapping_path = self.processed_dir / "label_mapping.json"
        mapping_payload = {
            "num_classes": len(label2id),
            "label2id": label2id,
            "id2label": {str(k): v for k, v in id2label.items()},
            "classes": sorted(list(label2id.keys()))
        }
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping_payload, f, indent=2, ensure_ascii=False)
        saved_files["label_mapping"] = mapping_path

        # 6. dataset_summary.json
        summary_path = self.processed_dir / "dataset_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, indent=2, ensure_ascii=False)
        saved_files["summary_json"] = summary_path

        # 7. dataset_report.md
        report_path = self.processed_dir / "dataset_report.md"
        report_text = self.format_markdown_report(summary, label2id)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        saved_files["summary_md"] = report_path

        logger.info(f"All processed artifacts saved successfully to {self.processed_dir}")
        return saved_files

    @staticmethod
    def format_markdown_report(summary: DatasetSummary, label2id: Dict[str, int]) -> str:
        """Generate human-readable markdown dataset report."""
        lines = [
            "# Dataset Preprocessing & Validation Report",
            f"\n- **Generated At:** {summary.timestamp}",
            f"- **Source File:** `{summary.raw_file_path}`",
            f"- **Random Seed:** `{summary.random_seed}`",
            f"- **Total Raw Intents:** {summary.total_raw_intents}",
            f"- **Total Raw Utterances:** {summary.total_raw_examples}",
            f"- **Total Cleaned Unique Utterances:** {summary.total_unique_cleaned_examples}",
            f"- **Intra-Intent Duplicates Deduplicated:** {summary.intra_intent_duplicates_count}",
            f"- **Cross-Intent Conflicts Detected:** {summary.cross_intent_conflicts_count}",
            "\n## Split Sizes",
            f"- **Train Set:** {summary.train_size} examples ({summary.train_size / summary.total_unique_cleaned_examples * 100:.1f}%)",
            f"- **Validation Set:** {summary.val_size} examples ({summary.val_size / summary.total_unique_cleaned_examples * 100:.1f}%)",
            f"- **Test Set:** {summary.test_size} examples ({summary.test_size / summary.total_unique_cleaned_examples * 100:.1f}%)",
            "\n## Intent Class Distribution across Splits",
            "| Class ID | Intent Name | Total | Train | Val | Test |",
            "| :--- | :--- | :---: | :---: | :---: | :---: |"
        ]

        for intent_name, lbl_id in sorted(label2id.items(), key=lambda x: x[1]):
            tot = summary.label_counts.get(intent_name, 0)
            tr = summary.split_distribution.get("train", {}).get(intent_name, 0)
            va = summary.split_distribution.get("val", {}).get(intent_name, 0)
            te = summary.split_distribution.get("test", {}).get(intent_name, 0)
            lines.append(f"| {lbl_id} | `{intent_name}` | {tot} | {tr} | {va} | {te} |")

        if summary.warnings:
            lines.append("\n## Validation & Split Warnings")
            for w in summary.warnings:
                lines.append(f"- ⚠️ {w}")
        else:
            lines.append("\n## Validation Warnings\n- None. Dataset schema perfectly compliant.")

        return "\n".join(lines) + "\n"

    def run(self) -> Tuple[DatasetSummary, Dict[str, Path]]:
        """
        Execute full preprocessing pipeline.
        """
        raw_data, schema_warnings = self.load_and_validate_raw_json()
        raw_examples_count = sum(len(item.get("text", [])) for item in raw_data.get("intents", []))

        unique_pairs, duplicate_report, extraction_warnings = self.extract_and_analyze_utterances(raw_data)

        distinct_intents = sorted(list({intent for _, intent in unique_pairs}))
        label2id, id2label = self.build_label_mapping(distinct_intents)

        train_examples, val_examples, test_examples, split_warnings = self.split_stratified_leak_free(
            unique_pairs, label2id
        )

        # Compute summary metrics
        all_warnings = schema_warnings + extraction_warnings + split_warnings
        label_counts = dict(Counter(intent for _, intent in unique_pairs))
        split_distribution = {
            "train": dict(Counter(e.intent for e in train_examples)),
            "val": dict(Counter(e.intent for e in val_examples)),
            "test": dict(Counter(e.intent for e in test_examples)),
        }

        summary = DatasetSummary(
            raw_file_path=str(self.raw_data_path),
            timestamp=datetime.now(timezone.utc).isoformat(),
            random_seed=self.random_seed,
            total_raw_intents=len(raw_data.get("intents", [])),
            total_raw_examples=raw_examples_count,
            total_unique_cleaned_examples=len(unique_pairs),
            intra_intent_duplicates_count=len(duplicate_report.intra_intent_duplicates),
            cross_intent_conflicts_count=len(duplicate_report.cross_intent_conflicts),
            train_size=len(train_examples),
            val_size=len(val_examples),
            test_size=len(test_examples),
            label_counts=label_counts,
            split_distribution=split_distribution,
            warnings=all_warnings
        )

        saved_files = self.save_processed_outputs(
            train_examples, val_examples, test_examples, label2id, id2label, summary
        )

        return summary, saved_files


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess and split Intent dataset for continuous learning pipeline."
    )
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    default_raw = repo_root / "ChatBot-PipeLine" / "data" / "raw" / "Intent.json"
    default_out = repo_root / "ChatBot-PipeLine" / "data" / "processed"

    parser.add_argument("--input", "-i", type=Path, default=default_raw, help="Path to raw Intent.json")
    parser.add_argument("--output-dir", "-o", type=Path, default=default_out, help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train set ratio (default: 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation set ratio (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test set ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting")
    return parser.parse_args()


def main():
    args = parse_arguments()
    logger.info("Starting Intent preprocessing stage...")
    preprocessor = IntentPreprocessor(
        raw_data_path=args.input,
        processed_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )
    summary, saved_files = preprocessor.run()
    logger.info("Preprocessing complete.")
    logger.info(f"Train size: {summary.train_size}, Val size: {summary.val_size}, Test size: {summary.test_size}")
    logger.info(f"Saved artifacts: {list(saved_files.keys())}")


if __name__ == "__main__":
    main()
