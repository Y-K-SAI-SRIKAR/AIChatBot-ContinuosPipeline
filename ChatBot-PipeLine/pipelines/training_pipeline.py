"""
training_pipeline.py — End-to-end training pipeline orchestrator for BERT intent classification.

Orchestrates:
1. Verification of preprocessed dataset splits (train/val/test/label_mapping).
   If absent, runs clean_data.py to generate them from raw Intent.json.
2. Configuration loading from params.yaml / config.yaml with CLI override support.
3. Model training with PyTorch and Hugging Face Transformers.
4. Validation during training with early stopping.
5. Final held-out test evaluation with macro-F1, accuracy, per-class metrics and confusion matrix.
6. Artifact persistence (model weights, tokenizer, label_mapping.json, model_card.json).
7. Syncing baseline checkpoint to artifacts/models/baseline for API inference compatibility.
8. MLflow logging (local or DagsHub).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure repository root and ChatBot-PipeLine are on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_PIPELINE_ROOT = _REPO_ROOT / "ChatBot-PipeLine"
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

import yaml
from src.model.train import TrainingConfig, run_training

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pipeline.training")


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}


def ensure_preprocessed_data(repo_root: Path, params: Dict[str, Any]) -> Path:
    """Ensure preprocessed train/val/test splits exist; generate them if missing."""
    prep_params = params.get("preprocessing", {})
    processed_dir = repo_root / prep_params.get("processed_dir", "ChatBot-PipeLine/data/processed")
    required_files = ["train.json", "val.json", "test.json", "label_mapping.json"]

    missing = [f for f in required_files if not (processed_dir / f).exists()]
    if missing:
        logger.info(f"Missing preprocessed files: {missing}. Triggering data preprocessing...")
        from src.preprocessing.clean_data import main as clean_data_main

        raw_path = repo_root / prep_params.get("raw_data_path", "ChatBot-PipeLine/data/raw/Intent.json")
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw intent data not found at: {raw_path}")

        # Run preprocessing via clean_data
        from src.preprocessing.clean_data import IntentPreprocessor
        preprocessor = IntentPreprocessor(
            raw_file_path=raw_path,
            output_dir=processed_dir,
            train_ratio=prep_params.get("train_ratio", 0.70),
            val_ratio=prep_params.get("val_ratio", 0.15),
            test_ratio=prep_params.get("test_ratio", 0.15),
            random_seed=prep_params.get("random_seed", 42),
        )
        preprocessor.run()
        logger.info(f"Data preprocessing complete. Artifacts written to {processed_dir}")
    else:
        logger.info(f"Preprocessed splits validated in: {processed_dir}")

    return processed_dir


def run_pipeline(
    model_name: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    max_length: Optional[int] = None,
    model_version: Optional[str] = None,
    seed: Optional[int] = None,
    params_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute the complete training pipeline."""
    t_start = time.time()
    params_file = params_path or (_REPO_ROOT / "ChatBot-PipeLine" / "params.yaml")
    params = load_yaml(params_file)
    train_params = params.get("training", {})

    processed_dir = ensure_preprocessed_data(_REPO_ROOT, params)

    cfg = TrainingConfig(
        model_name=model_name or train_params.get("model_name", "distilbert-base-uncased"),
        num_labels=train_params.get("num_labels", 22),
        batch_size=batch_size or train_params.get("batch_size", 16),
        learning_rate=learning_rate or float(train_params.get("learning_rate", 3e-5)),
        epochs=epochs or int(train_params.get("epochs", 4)),
        max_length=max_length or int(train_params.get("max_length", 64)),
        weight_decay=float(train_params.get("weight_decay", 0.01)),
        random_seed=seed if seed is not None else int(params.get("preprocessing", {}).get("random_seed", 42)),
        processed_dir=Path("ChatBot-PipeLine/data/processed"),
        artifact_dir=Path("artifacts/models"),
        model_version=model_version or "v0.1.0-baseline",
        mlflow_experiment="aic-chatbot-intent-classifier",
        mlflow_tracking_uri="ChatBot-PipeLine/mlruns",
    )

    logger.info(f"Starting BERT intent training pipeline with config:")
    logger.info(f"  model_name     : {cfg.model_name}")
    logger.info(f"  epochs         : {cfg.epochs}")
    logger.info(f"  batch_size     : {cfg.batch_size}")
    logger.info(f"  learning_rate  : {cfg.learning_rate}")
    logger.info(f"  max_length     : {cfg.max_length}")
    logger.info(f"  model_version  : {cfg.model_version}")

    # Execute training
    results = run_training(cfg)

    # Sync baseline model to artifacts/models/baseline for backward compatibility
    trained_artifact_dir = Path(results["artifact_dir"])
    baseline_dir = _REPO_ROOT / "artifacts" / "models" / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    for item in trained_artifact_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, baseline_dir / item.name)
    logger.info(f"Synchronized checkpoint to baseline directory: {baseline_dir}")

    total_pipeline_time = round(time.time() - t_start, 2)
    logger.info(f"Pipeline finished successfully in {total_pipeline_time}s.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BERT Intent Classification Training Pipeline")
    parser.add_argument("--model-name", default=None, help="Backbone Hugging Face model name")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=None, help="Max sequence length")
    parser.add_argument("--model-version", default="v0.1.0-baseline", help="Model version tag")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--params", default=None, help="Path to params.yaml")

    args = parser.parse_args()
    params_path = Path(args.params) if args.params else None

    run_pipeline(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        model_version=args.model_version,
        seed=args.seed,
        params_path=params_path,
    )


if __name__ == "__main__":
    main()
