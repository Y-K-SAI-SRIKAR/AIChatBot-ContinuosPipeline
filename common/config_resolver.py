"""
Explicit Configuration Resolution for Prompts and Models.

This module enforces a single, deterministic mechanism for resolving:
1. Active versioned Google ADK Agent system instructions.
2. Active BERT intent classification model checkpoint.

CRITICAL SAFETY RULES:
- Never overwrite the known-good baseline prompt ('ChatBot/ADKBot/instruction.txt').
- Deterministic precedence: Environment Variable -> Config File -> Known-good baseline fallback.
- Explicit immutable resolution results (ResolvedPromptConfig, ResolvedModelConfig).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional
import yaml

from common.contracts import (
    ModelMetadata,
    PromotionStatus,
    PromptEvaluationStatus,
    PromptMetadata,
)


class ResolvedPromptConfig(NamedTuple):
    """Immutable resolved prompt configuration container."""
    prompt_version: str
    instruction_text: str
    instruction_path: Path
    metadata: PromptMetadata


class ResolvedModelConfig(NamedTuple):
    """Immutable resolved model configuration container."""
    model_version: str
    artifact_location: Path
    metadata: ModelMetadata


def find_repo_root(start_dir: Optional[Path] = None) -> Path:
    """Deterministically locate the root directory of the repository."""
    current = (start_dir or Path(__file__)).resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or ((parent / "ChatBot").exists() and (parent / "ChatBot-PipeLine").exists()):
            return parent
    # Default to current directory if markers not found
    return Path.cwd().resolve()


class PromptConfigResolver:
    """
    Deterministic resolver for versioned ADK system prompts.
    
    Resolution order:
    1. Runtime override parameter (explicitly passed to resolve)
    2. Environment variable 'ACTIVE_PROMPT_VERSION'
    3. 'config.yaml' -> prompt_management.active_version
    4. Fallback default: 'v1.0.0'
    """

    DEFAULT_PROMPT_VERSION = "v1.0.0"

    def __init__(self, repo_root: Optional[Path] = None):
        self.repo_root = repo_root or find_repo_root()
        self.configs_dir = self.repo_root / "ChatBot-PipeLine" / "configs"
        self.instructions_dir = self.configs_dir / "instructions"
        self.fallback_file = self.repo_root / "ChatBot" / "ADKBot" / "instruction.txt"

    def _read_config_active_version(self) -> Optional[str]:
        cfg_file = self.configs_dir / "config.yaml"
        if not cfg_file.exists():
            return None
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    return data.get("prompt_management", {}).get("active_version")
        except Exception:
            return None
        return None

    def resolve(self, version_override: Optional[str] = None) -> ResolvedPromptConfig:
        """Resolve the active prompt configuration safely."""
        target_version = (
            version_override
            or os.getenv("ACTIVE_PROMPT_VERSION")
            or self._read_config_active_version()
            or self.DEFAULT_PROMPT_VERSION
        ).strip()

        # Check versioned instruction file
        version_file = self.instructions_dir / f"{target_version}.txt"
        resolved_path: Path
        resolved_text: str

        if version_file.exists():
            resolved_path = version_file
            with open(version_file, "r", encoding="utf-8") as f:
                resolved_text = f.read()
        elif self.fallback_file.exists():
            # Safe fallback to known-good baseline prompt
            resolved_path = self.fallback_file
            with open(self.fallback_file, "r", encoding="utf-8") as f:
                resolved_text = f.read()
        else:
            raise FileNotFoundError(
                f"Neither requested prompt version '{target_version}' at {version_file} "
                f"nor baseline fallback at {self.fallback_file} could be located."
            )

        metadata = PromptMetadata(
            prompt_version=target_version,
            instruction_path=str(resolved_path),
            associated_adk_model="gemini-3-flash-preview",
            evaluation_status=(
                PromptEvaluationStatus.BASELINE
                if target_version == self.DEFAULT_PROMPT_VERSION
                else PromptEvaluationStatus.PRODUCTION
            ),
            description=f"Active instruction version '{target_version}' resolved via PromptConfigResolver."
        )

        return ResolvedPromptConfig(
            prompt_version=target_version,
            instruction_text=resolved_text,
            instruction_path=resolved_path,
            metadata=metadata
        )


class ModelConfigResolver:
    """
    Deterministic resolver for BERT intent classification models.
    
    Resolution order:
    1. Runtime override parameter
    2. Environment variable 'ACTIVE_MODEL_VERSION'
    3. 'config.yaml' -> model_management.active_version
    4. Fallback default: 'v0.1.0-baseline'
    """

    DEFAULT_MODEL_VERSION = "v0.1.0-baseline"

    def __init__(self, repo_root: Optional[Path] = None):
        self.repo_root = repo_root or find_repo_root()
        self.configs_dir = self.repo_root / "ChatBot-PipeLine" / "configs"
        # Serving deployments may mount model artifacts on durable storage.  A
        # local development run continues to use the repository artifacts.
        configured_artifacts = os.getenv("MODEL_ARTIFACTS_DIR", "").strip()
        self.artifacts_dir = (
            Path(configured_artifacts).expanduser().resolve()
            if configured_artifacts
            else self.repo_root / "artifacts" / "models"
        )
        # The shipped baseline checkpoint is versioned; the old unversioned
        # ``artifacts/models/baseline`` placeholder intentionally contains no
        # model files and must not be used for a readiness fallback.
        self.fallback_dir = self.artifacts_dir / self.DEFAULT_MODEL_VERSION

    def _read_config_active_version(self) -> Optional[str]:
        cfg_file = self.configs_dir / "config.yaml"
        if not cfg_file.exists():
            return None
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    return data.get("model_management", {}).get("active_version")
        except Exception:
            return None
        return None

    def resolve(self, version_override: Optional[str] = None) -> ResolvedModelConfig:
        """Resolve the active model configuration safely."""
        target_version = (
            version_override
            or os.getenv("ACTIVE_MODEL_VERSION")
            or self._read_config_active_version()
            or self.DEFAULT_MODEL_VERSION
        ).strip()

        version_dir = self.artifacts_dir / target_version
        resolved_path = version_dir if version_dir.exists() else self.fallback_dir

        metadata = ModelMetadata(
            model_version=target_version,
            model_name="bert-intent-classifier",
            artifact_location=str(resolved_path),
            dataset_version="ChatBot-PipeLine/data/raw/Intent.json",
            num_classes=22,
            promotion_status=(
                PromotionStatus.PRODUCTION
                if target_version == self.DEFAULT_MODEL_VERSION
                else PromotionStatus.CANDIDATE
            ),
            notes=f"Model resolved via ModelConfigResolver with target version '{target_version}'."
        )

        return ResolvedModelConfig(
            model_version=target_version,
            artifact_location=resolved_path,
            metadata=metadata
        )


# Global singleton accessors
_prompt_resolver = PromptConfigResolver()
_model_resolver = ModelConfigResolver()


def resolve_active_prompt_config(version_override: Optional[str] = None) -> ResolvedPromptConfig:
    """Convenience function to resolve the active prompt configuration."""
    return _prompt_resolver.resolve(version_override=version_override)


def resolve_active_model_config(version_override: Optional[str] = None) -> ResolvedModelConfig:
    """Convenience function to resolve the active model configuration."""
    return _model_resolver.resolve(version_override=version_override)

