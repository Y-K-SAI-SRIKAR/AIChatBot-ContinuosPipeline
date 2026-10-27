"""
Unit tests for prompt configuration and prompt metadata.
"""

from pathlib import Path
import pytest

from common.contracts import PromptEvaluationStatus, PromptMetadata
from common.config_resolver import resolve_active_prompt_config


class TestPromptConfiguration:
    def test_baseline_prompt_metadata_integrity(self):
        resolved = resolve_active_prompt_config()
        assert resolved.prompt_version == "v1.0.0"
        assert resolved.metadata.prompt_version == "v1.0.0"
        assert resolved.metadata.associated_adk_model == "gemini-3-flash-preview"
        assert resolved.metadata.evaluation_status in {
            PromptEvaluationStatus.BASELINE,
            PromptEvaluationStatus.PRODUCTION,
        }

    def test_candidate_prompt_metadata_creation(self):
        meta = PromptMetadata(
            prompt_version="v1.1.0-candidate",
            instruction_path="ChatBot-PipeLine/configs/instructions/v1.1.0-candidate.txt",
            associated_adk_model="gemini-3-flash-preview",
            evaluation_status=PromptEvaluationStatus.CANDIDATE,
            description="Enhanced prompt with few-shot guidance",
            evaluation_metrics={"instruction_following_score": 0.94}
        )
        assert meta.prompt_version == "v1.1.0-candidate"
        assert meta.evaluation_status == PromptEvaluationStatus.CANDIDATE
        assert meta.evaluation_metrics["instruction_following_score"] == 0.94

