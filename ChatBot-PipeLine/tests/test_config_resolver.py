"""
Unit tests for explicit prompt and model configuration resolution.
"""

import os
from pathlib import Path
import pytest

from common.config_resolver import (
    PromptConfigResolver,
    ModelConfigResolver,
    resolve_active_prompt_config,
    resolve_active_model_config,
)
from common.contracts import PromotionStatus, PromptEvaluationStatus


class TestPromptConfigResolver:
    def test_default_resolution_loads_baseline_v1(self):
        resolved = resolve_active_prompt_config()
        assert resolved.prompt_version == "v1.0.0"
        assert len(resolved.instruction_text) > 0
        assert "AI Chatbot Agent" in resolved.instruction_text
        assert resolved.instruction_path.exists()

    def test_environment_variable_override(self, monkeypatch):
        monkeypatch.setenv("ACTIVE_PROMPT_VERSION", "v1.0.0")
        resolved = resolve_active_prompt_config()
        assert resolved.prompt_version == "v1.0.0"

    def test_never_overwrites_known_good_baseline(self):
        """CRITICAL: Ensure the baseline file exists and is intact."""
        baseline_file = Path("ChatBot/ADKBot/instruction.txt").resolve()
        assert baseline_file.exists()
        original_content = baseline_file.read_text(encoding="utf-8")

        # Resolving any configuration must never alter the baseline file
        resolved = resolve_active_prompt_config()
        after_content = baseline_file.read_text(encoding="utf-8")
        assert original_content == after_content
        assert resolved.instruction_text.strip() == original_content.strip()

    def test_fallback_when_version_file_not_found(self):
        """When an unknown version is specified without its file, fall back gracefully to baseline."""
        resolver = PromptConfigResolver()
        resolved = resolver.resolve(version_override="v99.9.9-nonexistent")
        assert resolved.prompt_version == "v99.9.9-nonexistent"
        # Must resolve to fallback baseline instructions safely
        assert "AI Chatbot Agent" in resolved.instruction_text


class TestModelConfigResolver:
    def test_default_model_resolution(self):
        resolved = resolve_active_model_config()
        assert resolved.model_version == "v0.1.0-baseline"
        assert resolved.metadata.model_name == "bert-intent-classifier"
        assert resolved.metadata.num_classes == 22

    def test_environment_override_model(self, monkeypatch):
        monkeypatch.setenv("ACTIVE_MODEL_VERSION", "candidate-20261027")
        resolved = resolve_active_model_config()
        assert resolved.model_version == "candidate-20261027"
        assert resolved.metadata.promotion_status == PromotionStatus.CANDIDATE
