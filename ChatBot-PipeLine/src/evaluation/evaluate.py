"""
evaluate.py — Evaluation utilities used by both training and retraining pipelines.

Computes accuracy, macro P/R/F1, per-class metrics, and confusion matrix.
Results are written to JSON for MLflow logging and DVC metrics comparison.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger("model.evaluate")


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError:  # pragma: no cover
        raise ImportError("scikit-learn is required: pip install scikit-learn>=1.4.0")


def evaluate_model(
    model: "BERTIntentClassifier",  # noqa: F821 – forward ref
    tokenizer,
    texts: List[str],
    true_labels: List[int],
    id2label: Dict[int, str],
    max_length: int = 64,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Run the model over texts and return a comprehensive metrics dict.

    WARNING — Dataset size limitation
    ----------------------------------
    This project trains on 142 unique utterances (22 classes × ~6 each).
    With only 1 sample per class in the test split, per-class precision/recall
    is either 0 or 1 (no interpolation possible).  Macro-averaged scores
    should be interpreted as orientation estimates, not production benchmarks.
    """
    _require_sklearn()
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    all_preds: List[int] = []

    # Mini-batch inference
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_labels = true_labels[start : start + batch_size]

        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            _, logits = model(input_ids=input_ids, attention_mask=attention_mask)

        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)

    label_names = [id2label[i] for i in sorted(id2label.keys())]

    accuracy = accuracy_score(true_labels, all_preds)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        true_labels, all_preds, average="macro", zero_division=0
    )
    weighted_f1 = precision_recall_fscore_support(
        true_labels, all_preds, average="weighted", zero_division=0
    )[2]

    per_class_report = classification_report(
        true_labels,
        all_preds,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(true_labels, all_preds, labels=sorted(id2label.keys())).tolist()

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class_report,
        "confusion_matrix": cm,
        "class_names": label_names,
        "n_samples": len(true_labels),
        "dataset_size_warning": (
            "CAUTION: Test set contains only 1 sample per class (22 total). "
            "Metrics are directional estimates only and must not be cited as "
            "production-grade benchmarks."
        ),
    }

    logger.info(
        f"Evaluation complete — accuracy={accuracy:.4f}, macro_f1={macro_f1:.4f}, "
        f"n_samples={len(true_labels)}"
    )
    return metrics


def save_metrics(metrics: Dict, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {output_path}")

