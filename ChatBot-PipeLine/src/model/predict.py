"""
predict.py — Thread-safe, lazy-loading inference engine for the trained BERT classifier.

Responsibilities
----------------
- Load the fine-tuned model and tokenizer from a versioned artifact directory.
- Return predicted intent name and confidence score for any text input.
- Never alter the ADK Agent or its instructions.
- Handle empty, malformed and out-of-vocabulary inputs gracefully.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger("model.predict")


@dataclass(frozen=True)
class IntentPrediction:
    """Immutable inference result."""
    intent: str
    confidence: float           # Probability [0.0 – 1.0]
    label_id: int
    top_k: List[Tuple[str, float]]  # Top-k (intent, prob) pairs


class IntentPredictor:
    """
    Lazy-loading, thread-safe BERT intent predictor.

    Usage
    -----
    predictor = IntentPredictor(artifact_dir=Path("artifacts/models/v0.1.0-baseline"))
    result = predictor.predict("What time is it?")
    print(result.intent, result.confidence)
    """

    def __init__(
        self,
        artifact_dir: Path,
        device: Optional[str] = None,
        max_length: int = 64,
        top_k: int = 3,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.max_length = max_length
        self.top_k = top_k
        self._model = None
        self._tokenizer = None
        self._lock = Lock()

        if device is not None:
            self._device = torch.device(device)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return  # double-checked
            try:
                from transformers import AutoTokenizer
            except ImportError:
                raise ImportError(
                    "The 'transformers' package is required for inference.\n"
                    "Install with: pip install transformers>=4.40.0"
                )
            from src.model.model import BERTIntentClassifier

            if not self.artifact_dir.exists():
                raise FileNotFoundError(
                    f"Artifact directory not found: {self.artifact_dir}\n"
                    "Run the training pipeline first to generate a model checkpoint."
                )

            logger.info(f"Loading model from {self.artifact_dir} on {self._device}")
            self._model = BERTIntentClassifier.from_pretrained(self.artifact_dir)
            self._model.to(self._device)
            self._model.eval()
            self._tokenizer = AutoTokenizer.from_pretrained(self.artifact_dir)
            logger.info("Model and tokenizer loaded successfully.")

    # ------------------------------------------------------------------
    def predict(self, text: str) -> IntentPrediction:
        """
        Predict the intent for a single utterance.

        Parameters
        ----------
        text : str
            User utterance (will be cleaned; empty strings are handled gracefully).

        Returns
        -------
        IntentPrediction
            Frozen dataclass containing intent name, confidence, label_id, and top-k.
        """
        if not isinstance(text, str) or not text.strip():
            logger.warning("Received empty or non-string input; returning UNKNOWN intent.")
            return IntentPrediction(
                intent="UNKNOWN",
                confidence=0.0,
                label_id=-1,
                top_k=[],
            )

        self._ensure_loaded()
        clean_text = " ".join(text.strip().split())

        encoding = self._tokenizer(
            clean_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self._device)
        attention_mask = encoding["attention_mask"].to(self._device)

        with torch.no_grad():
            _, logits = self._model(input_ids=input_ids, attention_mask=attention_mask)

        probs = torch.softmax(logits, dim=-1).squeeze(0)
        top_indices = torch.topk(probs, k=min(self.top_k, len(probs))).indices.tolist()

        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])
        intent_name = self._model.id2label.get(pred_idx, f"LABEL_{pred_idx}")

        top_k_list = [
            (self._model.id2label.get(i, f"LABEL_{i}"), float(probs[i]))
            for i in top_indices
        ]

        return IntentPrediction(
            intent=intent_name,
            confidence=confidence,
            label_id=pred_idx,
            top_k=top_k_list,
        )

    def predict_batch(self, texts: List[str]) -> List[IntentPrediction]:
        """Predict intents for a list of utterances."""
        return [self.predict(t) for t in texts]

