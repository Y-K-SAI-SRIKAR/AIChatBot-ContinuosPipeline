"""
Unit and integration tests for the Intent dataset preprocessor.
"""

import json
from pathlib import Path
import pytest

from src.preprocessing.clean_data import (
    DatasetSummary,
    DatasetValidationError,
    DuplicateReport,
    IntentPreprocessor,
    normalize_whitespace,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REAL_RAW_DATA_PATH = REPO_ROOT / "ChatBot-PipeLine" / "data" / "raw" / "Intent.json"


class TestWhitespaceNormalization:
    def test_normalizes_multiple_spaces_and_tabs(self):
        text = "  Hello   there, \t how are   you? \n "
        assert normalize_whitespace(text) == "Hello there, how are you?"

    def test_preserves_punctuation_and_casing(self):
        text = "What's your real name?!"
        assert normalize_whitespace(text) == "What's your real name?!"

    def test_handles_non_string_gracefully(self):
        assert normalize_whitespace(None) == ""
        assert normalize_whitespace(123) == ""


class TestRawJsonValidation:
    def test_malformed_json_syntax(self, tmp_path):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{not valid json: 123", encoding="utf-8")
        preprocessor = IntentPreprocessor(bad_json, tmp_path / "out")
        with pytest.raises(DatasetValidationError) as exc:
            preprocessor.load_and_validate_raw_json()
        assert "Invalid JSON" in str(exc.value)

    def test_not_a_dict(self, tmp_path):
        bad_json = tmp_path / "list.json"
        bad_json.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
        preprocessor = IntentPreprocessor(bad_json, tmp_path / "out")
        with pytest.raises(DatasetValidationError) as exc:
            preprocessor.load_and_validate_raw_json()
        assert "must be a dictionary" in str(exc.value)

    def test_missing_intents_key(self, tmp_path):
        bad_json = tmp_path / "no_intents.json"
        bad_json.write_text(json.dumps({"data": []}), encoding="utf-8")
        preprocessor = IntentPreprocessor(bad_json, tmp_path / "out")
        with pytest.raises(DatasetValidationError) as exc:
            preprocessor.load_and_validate_raw_json()
        assert "Missing required top-level key 'intents'" in str(exc.value)

    def test_empty_intents_list(self, tmp_path):
        bad_json = tmp_path / "empty_intents.json"
        bad_json.write_text(json.dumps({"intents": []}), encoding="utf-8")
        preprocessor = IntentPreprocessor(bad_json, tmp_path / "out")
        with pytest.raises(DatasetValidationError) as exc:
            preprocessor.load_and_validate_raw_json()
        assert "Expected at least 1 intent" in str(exc.value)

    def test_missing_or_blank_intent_name(self, tmp_path):
        bad_json = tmp_path / "blank_name.json"
        payload = {"intents": [{"intent": "  ", "text": ["hi"]}]}
        bad_json.write_text(json.dumps(payload), encoding="utf-8")
        preprocessor = IntentPreprocessor(bad_json, tmp_path / "out")
        with pytest.raises(DatasetValidationError) as exc:
            preprocessor.load_and_validate_raw_json()
        assert "invalid or missing 'intent' name" in str(exc.value)

    def test_duplicate_intent_blocks(self, tmp_path):
        bad_json = tmp_path / "dupe_intents.json"
        payload = {
            "intents": [
                {"intent": "Greeting", "text": ["hi"]},
                {"intent": "Greeting", "text": ["hello"]}
            ]
        }
        bad_json.write_text(json.dumps(payload), encoding="utf-8")
        preprocessor = IntentPreprocessor(bad_json, tmp_path / "out")
        with pytest.raises(DatasetValidationError) as exc:
            preprocessor.load_and_validate_raw_json()
        assert "Duplicate intent block detected" in str(exc.value)

    def test_missing_text_array(self, tmp_path):
        bad_json = tmp_path / "no_text.json"
        payload = {"intents": [{"intent": "Greeting", "patterns": ["hi"]}]}
        bad_json.write_text(json.dumps(payload), encoding="utf-8")
        preprocessor = IntentPreprocessor(bad_json, tmp_path / "out")
        with pytest.raises(DatasetValidationError) as exc:
            preprocessor.load_and_validate_raw_json()
        assert "missing required 'text' array" in str(exc.value)


class TestDuplicateAndConflictDetection:
    def test_intra_intent_duplicate_detection(self, tmp_path):
        raw_file = tmp_path / "intra_dupe.json"
        payload = {
            "intents": [
                {
                    "intent": "RealNameQuery",
                    "text": ["What is your name?", "Your real name please?", "Your real name please?"]
                }
            ]
        }
        raw_file.write_text(json.dumps(payload), encoding="utf-8")
        preprocessor = IntentPreprocessor(raw_file, tmp_path / "out")
        raw_data, _ = preprocessor.load_and_validate_raw_json()
        unique_pairs, dupe_report, _ = preprocessor.extract_and_analyze_utterances(raw_data)

        # 3 raw utterances -> 2 unique utterances
        assert len(unique_pairs) == 2
        assert len(dupe_report.intra_intent_duplicates) == 1
        assert dupe_report.intra_intent_duplicates[0]["utterance"] == "Your real name please?"
        assert dupe_report.intra_intent_duplicates[0]["count"] == 2
        assert len(dupe_report.cross_intent_conflicts) == 0

    def test_cross_intent_conflict_detection(self, tmp_path):
        raw_file = tmp_path / "cross_conflict.json"
        payload = {
            "intents": [
                {"intent": "Greeting", "text": ["Hello", "Hi"]},
                {"intent": "GreetingResponse", "text": ["Hello", "Hey back"]}
            ]
        }
        raw_file.write_text(json.dumps(payload), encoding="utf-8")
        preprocessor = IntentPreprocessor(raw_file, tmp_path / "out")
        raw_data, _ = preprocessor.load_and_validate_raw_json()
        unique_pairs, dupe_report, warnings = preprocessor.extract_and_analyze_utterances(raw_data)

        # "Hello" is present in both Greeting and GreetingResponse
        assert len(dupe_report.cross_intent_conflicts) == 1
        conflict = dupe_report.cross_intent_conflicts[0]
        assert conflict["utterance"] == "Hello"
        assert set(conflict["intents"]) == {"Greeting", "GreetingResponse"}
        # Must report a conflict warning
        assert any("CONFLICT" in w for w in warnings)


class TestSplitsAndDeterminism:
    def test_stratification_and_no_split_leakage(self, tmp_path):
        raw_file = tmp_path / "strat_test.json"
        # 3 intents with 6 examples each
        payload = {
            "intents": [
                {"intent": "A", "text": [f"a_{i}" for i in range(6)]},
                {"intent": "B", "text": [f"b_{i}" for i in range(6)]},
                {"intent": "C", "text": [f"c_{i}" for i in range(6)]},
            ]
        }
        raw_file.write_text(json.dumps(payload), encoding="utf-8")
        preprocessor = IntentPreprocessor(raw_file, tmp_path / "out", seed=42)
        summary, _ = preprocessor.run()

        # Each class has 6 examples: train=4, val=1, test=1
        assert summary.train_size == 12
        assert summary.val_size == 3
        assert summary.test_size == 3

        # Check label mapping
        with open(tmp_path / "out" / "label_mapping.json") as f:
            mapping = json.load(f)
        assert mapping["classes"] == ["A", "B", "C"]
        assert mapping["label2id"] == {"A": 0, "B": 1, "C": 2}

        # Check zero leakage across splits
        with open(tmp_path / "out" / "train.json") as f:
            train_texts = {x["text"] for x in json.load(f)}
        with open(tmp_path / "out" / "val.json") as f:
            val_texts = {x["text"] for x in json.load(f)}
        with open(tmp_path / "out" / "test.json") as f:
            test_texts = {x["text"] for x in json.load(f)}

        assert len(train_texts.intersection(val_texts)) == 0
        assert len(train_texts.intersection(test_texts)) == 0
        assert len(val_texts.intersection(test_texts)) == 0

    def test_reproducible_splits_with_fixed_seed(self, tmp_path):
        raw_file = tmp_path / "seed_test.json"
        payload = {
            "intents": [
                {"intent": "X", "text": [f"x_{i}" for i in range(8)]},
                {"intent": "Y", "text": [f"y_{i}" for i in range(8)]},
            ]
        }
        raw_file.write_text(json.dumps(payload), encoding="utf-8")

        p1 = IntentPreprocessor(raw_file, tmp_path / "out1", seed=42)
        _, f1 = p1.run()

        p2 = IntentPreprocessor(raw_file, tmp_path / "out2", seed=42)
        _, f2 = p2.run()

        p3 = IntentPreprocessor(raw_file, tmp_path / "out3", seed=999)
        _, f3 = p3.run()

        # Seed 42 runs must be byte-for-byte identical
        t1 = (tmp_path / "out1" / "train.json").read_text(encoding="utf-8")
        t2 = (tmp_path / "out2" / "train.json").read_text(encoding="utf-8")
        t3 = (tmp_path / "out3" / "train.json").read_text(encoding="utf-8")

        assert t1 == t2
        assert t1 != t3  # different seed produces different split permutation

    def test_tiny_class_limitation_warning(self, tmp_path):
        raw_file = tmp_path / "tiny_class.json"
        payload = {
            "intents": [
                {"intent": "NormalIntent", "text": ["n1", "n2", "n3", "n4", "n5"]},
                {"intent": "TinyIntent", "text": ["only_one_example"]}
            ]
        }
        raw_file.write_text(json.dumps(payload), encoding="utf-8")
        preprocessor = IntentPreprocessor(raw_file, tmp_path / "out", seed=42)
        summary, _ = preprocessor.run()

        assert any("TinyIntent" in w and "only 1 example" in w for w in summary.warnings)


class TestRealDatasetPreprocessing:
    def test_real_intent_json_execution_and_artifacts(self, tmp_path):
        """Run preprocessing on the real repository Intent.json file."""
        assert REAL_RAW_DATA_PATH.exists()
        original_bytes = REAL_RAW_DATA_PATH.read_bytes()

        preprocessor = IntentPreprocessor(
            raw_data_path=REAL_RAW_DATA_PATH,
            processed_dir=tmp_path / "processed",
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        summary, saved_files = preprocessor.run()

        # 1. Source file must NEVER be modified
        assert REAL_RAW_DATA_PATH.read_bytes() == original_bytes

        # 2. Check counts on real dataset
        assert summary.total_raw_intents == 22
        assert summary.total_raw_examples == 143
        assert summary.total_unique_cleaned_examples == 142
        assert summary.intra_intent_duplicates_count == 1
        assert summary.cross_intent_conflicts_count == 0

        # 3. Check split sizes
        assert summary.train_size == 98
        assert summary.val_size == 22
        assert summary.test_size == 22
        assert summary.train_size + summary.val_size + summary.test_size == 142

        # 4. Check all output files exist
        assert (tmp_path / "processed" / "train.json").exists()
        assert (tmp_path / "processed" / "val.json").exists()
        assert (tmp_path / "processed" / "test.json").exists()
        assert (tmp_path / "processed" / "training_data.json").exists()
        assert (tmp_path / "processed" / "label_mapping.json").exists()
        assert (tmp_path / "processed" / "dataset_summary.json").exists()
        assert (tmp_path / "processed" / "dataset_report.md").exists()

        # 5. Check missing entityType warning is logged
        assert any("UnderstandQuery" in w and "entityType" in w for w in summary.warnings)

        # 6. Check duplicate in RealNameQuery is documented
        summary_data = json.loads((tmp_path / "processed" / "dataset_summary.json").read_text(encoding="utf-8"))
        assert summary_data["intra_intent_duplicates_count"] == 1
