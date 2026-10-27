"""
train.py — BERT intent classification training pipeline.

This script:
  1. Reads processed train/val splits and the label mapping.
  2. Fine-tunes a configurable BERT-family encoder.
  3. Evaluates on val set each epoch (with early stopping).
  4. Runs final evaluation on the held-out test set.
  5. Saves model + tokenizer + label mapping to a versioned artifact dir.
  6. Logs all params, metrics and artifacts to MLflow (local or DagsHub).

The Google ADK Agent is NEVER modified.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure repo root and ChatBot-PipeLine are on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_PIPELINE_ROOT = _REPO_ROOT / "ChatBot-PipeLine"
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, TensorDataset

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("training.train")


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainingConfig:
    """All training hyperparameters in one place — sourced from params.yaml or CLI."""
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 22
    batch_size: int = 16
    learning_rate: float = 3e-5
    epochs: int = 5
    max_length: int = 64
    weight_decay: float = 0.01
    dropout: float = 0.1
    early_stopping_patience: int = 3
    random_seed: int = 42
    warmup_ratio: float = 0.1
    processed_dir: Path = Path("ChatBot-PipeLine/data/processed")
    artifact_dir: Path = Path("artifacts/models")
    model_version: str = "v0.1.0-baseline"
    mlflow_experiment: str = "aic-chatbot-intent-classifier"
    mlflow_tracking_uri: str = "ChatBot-PipeLine/mlruns"


# ─────────────────────────────────────────────────────────────────────────────
def set_seeds(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"All random seeds set to {seed}")


def load_split(split_path: Path) -> Tuple[List[str], List[int]]:
    with open(split_path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    texts = [e["text"] for e in examples]
    labels = [e["label_id"] for e in examples]
    return texts, labels


def load_label_mapping(mapping_path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(mapping_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    label2id: Dict[str, int] = m["label2id"]
    id2label: Dict[int, str] = {int(k): v for k, v in m["id2label"].items()}
    return label2id, id2label


def tokenize_texts(tokenizer, texts: List[str], max_length: int) -> Dict:
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )


def make_dataloader(
    tokenizer,
    texts: List[str],
    labels: List[int],
    max_length: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    enc = tokenize_texts(tokenizer, texts, max_length)
    dataset = TensorDataset(
        enc["input_ids"],
        enc["attention_mask"],
        torch.tensor(labels, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


# ─────────────────────────────────────────────────────────────────────────────
def compute_class_weights(labels: List[int], num_labels: int, device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency weights for the training set to mitigate class imbalance.
    Classes with fewer examples receive higher weight.
    """
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(num_labels):
        count = counts.get(i, 1)
        weights.append(total / (num_labels * count))
    tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    logger.info(f"Class weights (first 5): {tensor[:5].tolist()}")
    return tensor


# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scheduler,
    loss_fn: nn.CrossEntropyLoss,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


# ─────────────────────────────────────────────────────────────────────────────
def setup_mlflow(cfg: TrainingConfig):
    """
    Configure MLflow. Uses DagsHub remote if DAGSHUB_REPO_OWNER is set;
    otherwise falls back to local tracking URI.
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not installed — skipping experiment tracking.")
        return None, None

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    dagshub_owner = os.getenv("DAGSHUB_REPO_OWNER")
    dagshub_repo = os.getenv("DAGSHUB_REPO_NAME", "AIChatBot-ContinuosPipeline")

    if dagshub_owner:
        try:
            import dagshub
            dagshub.init(repo_owner=dagshub_owner, repo_name=dagshub_repo, mlflow=True)
            logger.info(f"DagsHub MLflow tracking enabled for {dagshub_owner}/{dagshub_repo}")
        except Exception as e:
            logger.warning(f"DagsHub init failed ({e}); falling back to local tracking.")
            mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    else:
        local_uri = Path(_REPO_ROOT) / cfg.mlflow_tracking_uri
        local_uri.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(str(local_uri))
        logger.info(f"MLflow tracking locally at: {local_uri}")

    mlflow.set_experiment(cfg.mlflow_experiment)
    return mlflow, mlflow.start_run(run_name=cfg.model_version)


# ─────────────────────────────────────────────────────────────────────────────
def run_training(cfg: TrainingConfig) -> Dict:
    """
    Full training loop. Returns a metrics dict of results.
    """
    set_seeds(cfg.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load data ────────────────────────────────────────────────────────────
    processed_dir = _REPO_ROOT / cfg.processed_dir
    train_texts, train_labels = load_split(processed_dir / "train.json")
    val_texts, val_labels = load_split(processed_dir / "val.json")
    test_texts, test_labels = load_split(processed_dir / "test.json")
    label2id, id2label = load_label_mapping(processed_dir / "label_mapping.json")
    logger.info(
        f"Dataset — Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}"
    )

    # ── Load tokenizer ───────────────────────────────────────────────────────
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers not installed. Run: pip install transformers>=4.40.0"
        )

    logger.info(f"Loading tokenizer: {cfg.model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    except OSError as e:
        raise OSError(
            f"Cannot load tokenizer for '{cfg.model_name}'. Check your internet connection.\n"
            f"Original error: {e}"
        ) from e

    # ── Build model ──────────────────────────────────────────────────────────
    from src.model.model import BERTIntentClassifier

    logger.info(f"Loading model: {cfg.model_name}")
    model = BERTIntentClassifier(
        model_name=cfg.model_name,
        num_labels=cfg.num_labels,
        id2label=id2label,
        label2id=label2id,
        dropout=cfg.dropout,
    )
    model.to(device)

    # ── Dataloaders ──────────────────────────────────────────────────────────
    train_loader = make_dataloader(
        tokenizer, train_texts, train_labels, cfg.max_length, cfg.batch_size, shuffle=True
    )
    val_loader = make_dataloader(
        tokenizer, val_texts, val_labels, cfg.max_length, cfg.batch_size, shuffle=False
    )

    # ── Weighted loss for class imbalance ────────────────────────────────────
    class_weights = compute_class_weights(train_labels, cfg.num_labels, device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer & scheduler ────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow, run = setup_mlflow(cfg)
    if mlflow and run:
        mlflow.log_params(
            {
                "model_name": cfg.model_name,
                "num_labels": cfg.num_labels,
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.learning_rate,
                "epochs": cfg.epochs,
                "max_length": cfg.max_length,
                "weight_decay": cfg.weight_decay,
                "dropout": cfg.dropout,
                "random_seed": cfg.random_seed,
                "train_samples": len(train_texts),
                "val_samples": len(val_texts),
                "test_samples": len(test_texts),
            }
        )

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    history = []
    best_model_state = None
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        ep_t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, loss_fn, device)
        ep_dur = time.time() - ep_t0

        logger.info(
            f"Epoch {epoch}/{cfg.epochs} — "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  ({ep_dur:.1f}s)"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 4),
                "duration_s": round(ep_dur, 2),
            }
        )

        if mlflow and run:
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                step=epoch,
            )

        # Early stopping on val_loss; keep best weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

    total_dur = round(time.time() - t0, 2)
    logger.info(f"Training finished in {total_dur}s. Best val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}")

    # Restore best weights before final evaluation
    if best_model_state:
        model.load_state_dict(best_model_state)

    # ── Final test evaluation ────────────────────────────────────────────────
    from src.evaluation.evaluate import evaluate_model, save_metrics

    test_metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        texts=test_texts,
        true_labels=test_labels,
        id2label=id2label,
        max_length=cfg.max_length,
        device=device,
    )

    logger.info("=" * 60)
    logger.info("HELD-OUT TEST SET RESULTS")
    logger.info(f"  Accuracy       : {test_metrics['accuracy']:.4f}")
    logger.info(f"  Macro F1       : {test_metrics['macro_f1']:.4f}")
    logger.info(f"  Macro Precision: {test_metrics['macro_precision']:.4f}")
    logger.info(f"  Macro Recall   : {test_metrics['macro_recall']:.4f}")
    logger.info(f"  Weighted F1    : {test_metrics['weighted_f1']:.4f}")
    logger.info(f"  {test_metrics['dataset_size_warning']}")
    logger.info("=" * 60)

    if mlflow and run:
        mlflow.log_metrics(
            {
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_macro_precision": test_metrics["macro_precision"],
                "test_macro_recall": test_metrics["macro_recall"],
                "test_weighted_f1": test_metrics["weighted_f1"],
                "best_val_accuracy": best_val_acc,
                "best_val_loss": best_val_loss,
                "training_duration_s": total_dur,
            }
        )

    # ── Save artifacts ───────────────────────────────────────────────────────
    artifact_dir = _REPO_ROOT / cfg.artifact_dir / cfg.model_version
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(artifact_dir)
    tokenizer.save_pretrained(artifact_dir)

    # Training history
    history_path = artifact_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Test metrics
    metrics_dir = _REPO_ROOT / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(test_metrics, metrics_dir / f"{cfg.model_version}_test_metrics.json")

    # Model card
    model_card = {
        "model_version": cfg.model_version,
        "base_model": cfg.model_name,
        "num_classes": cfg.num_labels,
        "artifact_dir": str(artifact_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_config": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in asdict(cfg).items()
        },
        "training_duration_s": total_dur,
        "best_val_accuracy": round(best_val_acc, 4),
        "best_val_loss": round(best_val_loss, 6),
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "dataset_size_warning": test_metrics["dataset_size_warning"],
    }
    with open(artifact_dir / "model_card.json", "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)

    if mlflow and run:
        mlflow.log_artifacts(str(artifact_dir), artifact_path="model")
        mlflow.end_run()
        logger.info(f"MLflow run complete. Run ID: {run.info.run_id}")

    logger.info(f"All artifacts saved to: {artifact_dir}")
    return {
        "artifact_dir": str(artifact_dir),
        "model_version": cfg.model_version,
        "test_metrics": test_metrics,
        "training_history": history,
        "model_card": model_card,
    }


# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BERT intent classifier")
    p.add_argument("--model-name", default="distilbert-base-uncased")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--processed-dir", default="ChatBot-PipeLine/data/processed")
    p.add_argument("--artifact-dir", default="artifacts/models")
    p.add_argument("--model-version", default="v0.1.0-baseline")
    p.add_argument("--mlflow-experiment", default="aic-chatbot-intent-classifier")
    p.add_argument("--mlflow-uri", default="ChatBot-PipeLine/mlruns")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainingConfig(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        early_stopping_patience=args.patience,
        random_seed=args.seed,
        processed_dir=Path(args.processed_dir),
        artifact_dir=Path(args.artifact_dir),
        model_version=args.model_version,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_tracking_uri=args.mlflow_uri,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()

