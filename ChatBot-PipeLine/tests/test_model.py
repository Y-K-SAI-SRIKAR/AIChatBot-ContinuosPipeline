"""
Unit tests for model configuration and model metadata.
"""

from pathlib import Path
import pytest

from common.contracts import ModelMetadata, PromotionStatus
from common.config_resolver import resolve_active_model_config


class TestModelConfiguration:
    def test_baseline_model_metadata_integrity(self):
        resolved = resolve_active_model_config()
        assert resolved.model_version == "v0.1.0-baseline"
        assert resolved.metadata.model_name == "bert-intent-classifier"
        assert resolved.metadata.num_classes == 22
        assert resolved.metadata.promotion_status in {
            PromotionStatus.PRODUCTION,
            PromotionStatus.CANDIDATE,
        }

    def test_candidate_model_promotion_lifecycle(self):
        candidate = ModelMetadata(
            model_version="v0.2.0-candidate",
            artifact_location="artifacts/models/v0.2.0-candidate",
            dvc_revision="dvc-hash-xyz",
            dataset_version="ChatBot-PipeLine/data/processed/training_data.json",
            num_classes=22,
            evaluation_metrics={"macro_f1": 0.89, "accuracy": 0.93},
            promotion_status=PromotionStatus.CANDIDATE
        )
        assert candidate.promotion_status == PromotionStatus.CANDIDATE

        # Simulate promotion after evaluation gate passes
        candidate.promotion_status = PromotionStatus.PRODUCTION
        assert candidate.promotion_status == PromotionStatus.PRODUCTION

