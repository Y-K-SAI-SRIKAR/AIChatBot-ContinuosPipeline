"""
Unit tests for BERT intent classifier: model loading, inference, label mapping
and malformed input handling.

These tests run with a tiny randomly-initialised model so they complete
without network access and in seconds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

# ── path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── helpers ───────────────────────────────────────────────────────────────────

LABEL2ID = {
    "Greeting": 0,
    "GoodBye": 1,
    "TimeQuery": 2,
    "Thanks": 3,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


def _write_label_mapping(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "num_classes": NUM_LABELS,
                "model_name": "distilbert-base-uncased",
                "label2id": LABEL2ID,
                "id2label": {str(k): v for k, v in ID2LABEL.items()},
                "classes": sorted(LABEL2ID.keys()),
            }
        ),
        encoding="utf-8",
    )


def _build_and_save_tiny_model(artifact_dir: Path) -> None:
    """
    Construct an untrained DistilBERT with a small config to avoid downloading
    a full pre-trained checkpoint (tests run offline).
    """
    from transformers import AutoTokenizer, DistilBertConfig, DistilBertForSequenceClassification

    cfg = DistilBertConfig(
        vocab_size=100,
        max_position_embeddings=64,
        n_layers=1,
        n_heads=2,
        dim=64,
        hidden_dim=128,
        num_labels=NUM_LABELS,
        id2label={v: k for k, v in LABEL2ID.items()},
        label2id=LABEL2ID,
    )
    backbone = DistilBertForSequenceClassification(cfg)
    backbone.save_pretrained(artifact_dir)
    _write_label_mapping(artifact_dir / "label_mapping.json")

    # Minimal tokenizer backed by a character-level vocab
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
        vocab_size=100,
    )
    tok.train_from_iterator(
        [
            "hello goodbye time thanks what is",
            "the who are you please stop",
        ],
        trainer=trainer,
    )
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    fast_tok.save_pretrained(artifact_dir)


# ─────────────────────────────────────────────────────────────────────────────
class TestLabelMapping:
    def test_label2id_consistent_with_id2label(self):
        for intent, idx in LABEL2ID.items():
            assert ID2LABEL[idx] == intent

    def test_label_mapping_json_roundtrip(self, tmp_path):
        mapping_path = tmp_path / "label_mapping.json"
        _write_label_mapping(mapping_path)
        loaded = json.loads(mapping_path.read_text(encoding="utf-8"))
        assert loaded["num_classes"] == NUM_LABELS
        assert loaded["label2id"] == LABEL2ID
        assert {int(k): v for k, v in loaded["id2label"].items()} == ID2LABEL
        assert len(loaded["classes"]) == NUM_LABELS

    def test_missing_label_mapping_raises(self, tmp_path):
        from src.model.model import BERTIntentClassifier

        with pytest.raises(FileNotFoundError, match="label_mapping.json"):
            BERTIntentClassifier.from_pretrained(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
class TestModelLoading:
    def test_from_pretrained_loads_correctly(self, tmp_path):
        _build_and_save_tiny_model(tmp_path)
        from src.model.model import BERTIntentClassifier

        model = BERTIntentClassifier.from_pretrained(tmp_path)
        assert model.num_labels == NUM_LABELS
        assert model.label2id == LABEL2ID
        assert model.id2label[0] == "Greeting"

    def test_save_and_reload_roundtrip(self, tmp_path):
        _build_and_save_tiny_model(tmp_path)
        from src.model.model import BERTIntentClassifier

        original = BERTIntentClassifier.from_pretrained(tmp_path)
        save_dir = tmp_path / "resaved"
        original.save_pretrained(save_dir)
        reloaded = BERTIntentClassifier.from_pretrained(save_dir)
        assert reloaded.num_labels == original.num_labels
        assert reloaded.label2id == original.label2id


# ─────────────────────────────────────────────────────────────────────────────
class TestInference:
    def test_output_shape(self, tmp_path):
        _build_and_save_tiny_model(tmp_path)
        from src.model.model import BERTIntentClassifier
        from transformers import AutoTokenizer

        model = BERTIntentClassifier.from_pretrained(tmp_path)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(tmp_path)

        enc = tokenizer(
            ["Hello there"],
            truncation=True,
            padding="max_length",
            max_length=16,
            return_tensors="pt",
        )
        with torch.no_grad():
            loss, logits = model(enc["input_ids"], enc["attention_mask"])
        assert logits.shape == (1, NUM_LABELS)
        assert loss is None  # no labels provided → no loss

    def test_logits_for_batch(self, tmp_path):
        _build_and_save_tiny_model(tmp_path)
        from src.model.model import BERTIntentClassifier
        from transformers import AutoTokenizer

        model = BERTIntentClassifier.from_pretrained(tmp_path)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(tmp_path)

        texts = ["Hi", "Bye", "What time"]
        enc = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=16,
            return_tensors="pt",
        )
        with torch.no_grad():
            _, logits = model(enc["input_ids"], enc["attention_mask"])
        assert logits.shape == (3, NUM_LABELS)

    def test_predictor_empty_string(self, tmp_path):
        _build_and_save_tiny_model(tmp_path)
        from src.model.predict import IntentPredictor

        predictor = IntentPredictor(artifact_dir=tmp_path, device="cpu", max_length=16)
        result = predictor.predict("")
        assert result.intent == "UNKNOWN"
        assert result.confidence == 0.0
        assert result.label_id == -1

    def test_predictor_none_input(self, tmp_path):
        _build_and_save_tiny_model(tmp_path)
        from src.model.predict import IntentPredictor

        predictor = IntentPredictor(artifact_dir=tmp_path, device="cpu", max_length=16)
        result = predictor.predict(None)  # type: ignore
        assert result.intent == "UNKNOWN"

    def test_predictor_whitespace_only(self, tmp_path):
        _build_and_save_tiny_model(tmp_path)
        from src.model.predict import IntentPredictor

        predictor = IntentPredictor(artifact_dir=tmp_path, device="cpu", max_length=16)
        result = predictor.predict("    \t\n   ")
        assert result.intent == "UNKNOWN"

    def test_predictor_valid_text_returns_correct_types(self, tmp_path):
        _build_and_save_tiny_model(tmp_path)
        from src.model.predict import IntentPredictor

        predictor = IntentPredictor(artifact_dir=tmp_path, device="cpu", max_length=16, top_k=3)
        result = predictor.predict("Hello how are you")
        assert isinstance(result.intent, str)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.label_id, int)
        assert result.intent in LABEL2ID or result.intent.startswith("LABEL_")
        assert len(result.top_k) <= 3
        assert all(isinstance(p[0], str) and 0.0 <= p[1] <= 1.0 for p in result.top_k)


# ─────────────────────────────────────────────────────────────────────────────
class TestUnknownLabel:
    def test_id2label_fallback_for_out_of_range(self, tmp_path):
        """
        If the model ever predicts a label index not in the mapping,
        the predictor returns a safe LABEL_N string rather than crashing.
        """
        _build_and_save_tiny_model(tmp_path)
        from src.model.model import BERTIntentClassifier

        model = BERTIntentClassifier.from_pretrained(tmp_path)
        # Ask for a label that doesn't exist
        name = model.id2label.get(9999, f"LABEL_{9999}")
        assert name == "LABEL_9999"
