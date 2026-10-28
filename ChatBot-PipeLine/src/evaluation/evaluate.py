"""
evaluate.py — Evaluation utilities and CLI entry point for intent classification.

Computes accuracy, macro P/R/F1, per-class metrics, and confusion matrix.
Results are written to JSON for MLflow logging and DVC metrics comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_PIPELINE_ROOT = _REPO_ROOT / "ChatBot-PipeLine"
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("model.evaluate")


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError:  # pragma: no cover
        raise ImportError("scikit-learn is required: pip install scikit-learn>=1.4.0")


def evaluate_model(
    model: "BERTIntentClassifier",  # noqa: F821
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
    This project trains on 142 unique utterances (22 classes x ~6 each).
    With only 1 sample per class in the test split, per-class precision/recall
    is either 0 or 1 (no interpolation possible). Macro-averaged scores
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained BERT intent classifier on test split")
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/models/v0.1.0-baseline"), help="Path to model checkpoint")
    parser.add_argument("--test-data", type=Path, default=Path("ChatBot-PipeLine/data/processed/test.json"), help="Path to test.json")
    parser.add_argument("--output-metrics", type=Path, default=Path("artifacts/metrics/v0.1.0-baseline_test_metrics.json"), help="Output metrics path")
    parser.add_argument("--max-length", type=int, default=64, help="Max sequence length")

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    test_path = Path(args.test_data)
    output_path = Path(args.output_metrics)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found at: {model_dir}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at: {test_path}")

    from transformers import AutoTokenizer
    from src.model.model import BERTIntentClassifier

    logger.info(f"Loading model from {model_dir}...")
    model = BERTIntentClassifier.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    with open(test_path, "r", encoding="utf-8") as f:
        test_examples = json.load(f)

    texts = [e["text"] for e in test_examples]
    true_labels = [e["label_id"] for e in test_examples]

    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        true_labels=true_labels,
        id2label=model.id2label,
        max_length=args.max_length,
        device=device,
    )

    save_metrics(metrics, output_path)
    logger.info(f"Held-out test evaluation completed. Metrics saved to {output_path}")


if __name__ == "__main__":
    main()
