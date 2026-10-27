"""
model.py — BERT-family sequence-classification wrapper.

The model architecture is fully configurable: swap the backbone by changing
`model_name` in params.yaml (or the CLI arg).  The ADK Agent is unaffected.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn

logger = logging.getLogger("model.bert_classifier")


def _require_transformers() -> None:
    try:
        import transformers  # noqa: F401
    except ImportError:  # pragma: no cover
        raise ImportError(
            "The 'transformers' package is not installed.\n"
            "Install it with:  pip install transformers>=4.40.0\n"
            "Then re-run training."
        )


class BERTIntentClassifier(nn.Module):
    """
    Thin wrapper around AutoModelForSequenceClassification.

    Architecture
    ------------
    Pre-trained BERT-family encoder (frozen or fine-tuned)
      └── Linear classification head  →  num_labels logits

    The ADK/Gemini agent is completely independent of this model.
    This module is responsible only for intent detection.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        _require_transformers()
        from transformers import AutoConfig, AutoModelForSequenceClassification

        self.model_name = model_name
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id

        try:
            config = AutoConfig.from_pretrained(
                model_name,
                num_labels=num_labels,
                id2label={int(k): v for k, v in id2label.items()},
                label2id=label2id,
            )
            # Set dropout dynamically depending on model architecture
            if hasattr(config, "dropout"):
                config.dropout = dropout
            if hasattr(config, "hidden_dropout_prob"):
                config.hidden_dropout_prob = dropout
            if hasattr(config, "attention_dropout"):
                config.attention_dropout = dropout
            if hasattr(config, "attention_probs_dropout_prob"):
                config.attention_probs_dropout_prob = dropout

            self.backbone = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config=config,
                ignore_mismatched_sizes=True,
            )
        except OSError as exc:
            raise OSError(
                f"Failed to download or load '{model_name}' from Hugging Face Hub.\n"
                f"Original error: {exc}\n\n"
                "Actionable steps:\n"
                "  1. Check your internet connection.\n"
                "  2. Verify the model name is correct (e.g. 'distilbert-base-uncased').\n"
                "  3. Try:  pip install --upgrade transformers huggingface_hub\n"
                "  4. If behind a proxy, set HF_HUB_DISABLE_IMPLICIT_TOKEN=1 and TRANSFORMERS_OFFLINE=0."
            ) from exc

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss, outputs.logits

    # ------------------------------------------------------------------
    def save_pretrained(self, save_dir: Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(save_dir)
        mapping = {
            "num_classes": self.num_labels,
            "model_name": self.model_name,
            "label2id": self.label2id,
            "id2label": {str(k): v for k, v in self.id2label.items()},
            "classes": sorted(list(self.label2id.keys())),
        }
        with open(save_dir / "label_mapping.json", "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)
        logger.info(f"Model and label mapping saved to {save_dir}")

    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, load_dir: Path) -> "BERTIntentClassifier":
        """Load a persisted checkpoint produced by save_pretrained()."""
        _require_transformers()
        from transformers import AutoModelForSequenceClassification

        load_dir = Path(load_dir)
        mapping_path = load_dir / "label_mapping.json"
        if not mapping_path.exists():
            raise FileNotFoundError(
                f"label_mapping.json not found in {load_dir}. "
                "Ensure the model was saved with BERTIntentClassifier.save_pretrained()."
            )
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        id2label = {int(k): v for k, v in mapping["id2label"].items()}
        label2id = mapping["label2id"]
        num_labels = mapping["num_classes"]
        model_name = mapping.get("model_name", "distilbert-base-uncased")

        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.model_name = model_name
        instance.num_labels = num_labels
        instance.id2label = id2label
        instance.label2id = label2id
        instance.backbone = AutoModelForSequenceClassification.from_pretrained(load_dir)
        return instance

