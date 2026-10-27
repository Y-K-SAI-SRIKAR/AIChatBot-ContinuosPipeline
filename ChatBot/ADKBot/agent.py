"""
agent.py — Google ADK Agent and Runner management with versioned instruction support.

Guarantees:
- Never replaces or rebuilds the Google ADK foundation model (gemini-3-flash-preview).
- Atomic version resolution: Each chat turn executes against an immutable (Agent, Runner)
  pair corresponding to the active prompt_version.
- In-place mutation prevention: Existing agent instruction text is never modified in-place
  during an in-flight request.
- Session continuity: Shared InMemorySessionService preserves multi-turn conversational context
  even when instructions or configurations reload.
- Known-good fallback: If a candidate prompt fails or is unavailable, resolution deterministically
  falls back to baseline v1.0.0 (ChatBot/ADKBot/instruction.txt).
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from threading import Lock
from typing import Dict, Optional, Tuple

# Ensure repository root is on sys.path for common imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pyrefly: ignore[missing-import]
from google.adk import Agent
# pyrefly: ignore[missing-import]
from google.adk.runners import Runner
# pyrefly: ignore[missing-import]
from google.adk.sessions.in_memory_session_service import InMemorySessionService

from common.config_resolver import (
    PromptConfigResolver,
    ResolvedPromptConfig,
    resolve_active_prompt_config,
)

logger = logging.getLogger("ADKBot.agent")

# Shared session service preserving conversation memory across turns and version switches
shared_session_service = InMemorySessionService()


class ADKAgentManager:
    """
    Thread-safe, atomic manager for versioned Google ADK Agents and Runners.
    """

    DEFAULT_MODEL_NAME = "gemini-3-flash-preview"

    def __init__(self, session_service: Optional[InMemorySessionService] = None):
        self.session_service = session_service or shared_session_service
        self.prompt_resolver = PromptConfigResolver()
        self._lock = Lock()
        self._agents: Dict[str, Agent] = {}
        self._runners: Dict[str, Runner] = {}
        self._active_version: Optional[str] = None

        # Eagerly initialize baseline
        self.get_active_runner()

    def get_agent(self, prompt_version: str, instruction_text: str) -> Agent:
        """Retrieve or build an immutable Agent instance for a specific prompt version."""
        with self._lock:
            if prompt_version not in self._agents:
                logger.info(f"Instantiating new ADK Agent for prompt version: {prompt_version}")
                agent = Agent(
                    name=f"ChatBot-{prompt_version}",
                    description="AI ChatBot Agent with versioned instructions",
                    instruction=instruction_text,
                    model=self.DEFAULT_MODEL_NAME,
                    tools=[],
                )
                self._agents[prompt_version] = agent
            return self._agents[prompt_version]

    def get_runner(self, prompt_version: str, instruction_text: str) -> Runner:
        """Retrieve or build a Runner instance bound to the versioned Agent and shared session."""
        with self._lock:
            if prompt_version not in self._runners:
                agent = self.get_agent(prompt_version, instruction_text)
                runner = Runner(
                    app_name="WPChatBot",
                    agent=agent,
                    session_service=self.session_service,
                    auto_create_session=True,
                )
                self._runners[prompt_version] = runner
            return self._runners[prompt_version]

    def get_active_runner(self, version_override: Optional[str] = None) -> Tuple[Runner, ResolvedPromptConfig]:
        """
        Atomically resolve active prompt configuration and return the associated Runner.
        Falls back to known-good baseline v1.0.0 if version_override or target fails.
        """
        resolved_prompt = self.prompt_resolver.resolve(version_override=version_override)
        runner = self.get_runner(resolved_prompt.prompt_version, resolved_prompt.instruction_text)
        with self._lock:
            self._active_version = resolved_prompt.prompt_version
        return runner, resolved_prompt

    def reload(self, version_override: Optional[str] = None) -> ResolvedPromptConfig:
        """
        Controlled reload mechanism: forces fresh resolution of active prompt configuration.
        """
        with self._lock:
            self.prompt_resolver = PromptConfigResolver()
        runner, resolved = self.get_active_runner(version_override=version_override)
        logger.info(f"ADK Agent configuration reloaded. Active prompt version: {resolved.prompt_version}")
        return resolved

    @property
    def active_prompt_version(self) -> str:
        with self._lock:
            return self._active_version or PromptConfigResolver.DEFAULT_PROMPT_VERSION


# Global agent manager instance
agent_manager = ADKAgentManager(session_service=shared_session_service)

# Backward-compatible exports
_baseline_runner, _baseline_prompt = agent_manager.get_active_runner()
active_prompt_version = _baseline_prompt.prompt_version
instruction_text = _baseline_prompt.instruction_text
root_agent = _baseline_runner.agent
runner = _baseline_runner
session_service = shared_session_service
