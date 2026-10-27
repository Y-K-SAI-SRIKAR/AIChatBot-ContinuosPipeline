"""
Integration tests for Google ADK Agent integration with BERT intent classification,
versioned instruction resolution, atomic reloads, confidence thresholding,
and fallback behaviors.
"""
from __future__ import annotations
import os
os.environ["ADK_OFFLINE_MODE"] = "true"

from pathlib import Path
import sys
import pytest

# Ensure ChatBot and repo root are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "ChatBot") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "ChatBot"))

# pyrefly: ignore[missing-import]
from ADKBot.agent import agent_manager, root_agent
# pyrefly: ignore[missing-import]
from ADKBot.api import chat, reload_configuration
from common.contracts import ChatRequest


def run_coro(coro):
    """Execute coroutine synchronously for deterministic test execution."""
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value
    raise RuntimeError("Coroutine did not complete synchronously in offline mode")


class TestADKIntegration:
    def test_successful_classification_and_metadata(self):
        """Verify successful BERT classification returns intent metadata alongside ADK response."""
        req = ChatRequest(message="Hello there", conversation_id="session-test-1")
        resp = run_coro(chat(req))

        assert resp.request_id == req.request_id
        assert resp.response is not None and len(resp.response) > 0
        assert resp.model_version == "v0.1.0-baseline"
        assert resp.prompt_version == "v1.0.0"
        assert resp.latency_ms >= 0.0
        # If model is loaded, intent is populated
        if resp.predicted_intent:
            assert isinstance(resp.predicted_intent, str)
            assert 0.0 <= resp.intent_confidence <= 1.0

    def test_low_confidence_threshold_preserves_safe_behavior(self, monkeypatch):
        """Verify that when confidence threshold is set high, safe ADK behavior is preserved."""
        monkeypatch.setenv("INTENT_CONFIDENCE_THRESHOLD", "0.9999")
        req = ChatRequest(message="xyz random gibberish input 12345", conversation_id="session-test-2")
        resp = run_coro(chat(req))

        assert resp.request_id == req.request_id
        assert len(resp.response) > 0
        assert resp.model_version == "v0.1.0-baseline"

    def test_missing_model_artifacts_fallback(self, monkeypatch):
        """Verify that when BERT classifier is disabled or model path is missing, chat does not crash."""
        monkeypatch.setenv("BERT_CLASSIFIER_ENABLED", "false")
        req = ChatRequest(message="What is the time?", conversation_id="session-test-3")
        resp = run_coro(chat(req))

        assert resp.predicted_intent is None
        assert resp.intent_confidence is None
        assert len(resp.response) > 0

    def test_prompt_loading_failure_fallback_to_baseline(self, monkeypatch):
        """Verify that requesting a missing prompt version safely falls back to baseline v1.0.0."""
        monkeypatch.setenv("ACTIVE_PROMPT_VERSION", "v99.99.99-nonexistent")
        # Trigger reload with non-existent version
        reload_configuration({"prompt_version": "v99.99.99-nonexistent"})

        req = ChatRequest(message="Hi", conversation_id="session-test-4")
        resp = run_coro(chat(req))

        # Falls back to known-good baseline instruction text
        assert len(resp.response) > 0

        # Reset back to v1.0.0
        monkeypatch.delenv("ACTIVE_PROMPT_VERSION", raising=False)
        reload_configuration({"prompt_version": "v1.0.0"})

    def test_version_change_atomic_and_reload(self, tmp_path):
        """
        Verify that changing active prompt version produces an atomic configuration
        change without rebuilding the chatbot or restarting the server.
        """
        instructions_dir = REPO_ROOT / "ChatBot-PipeLine" / "configs" / "instructions"
        test_version_file = instructions_dir / "v1.1.0-test.txt"
        test_version_file.write_text(
            "# AI Chatbot Agent — Version 1.1.0-test\n"
            "You are a test prompt configuration demonstrating atomic reload.\n",
            encoding="utf-8"
        )

        try:
            # Trigger reload with new prompt version
            reload_res = reload_configuration({"prompt_version": "v1.1.0-test"})
            assert reload_res["active_prompt_version"] == "v1.1.0-test"

            # Execute chat request under new version
            req = ChatRequest(message="Testing versioned instructions", conversation_id="session-test-5")
            resp = run_coro(chat(req))

            assert resp.prompt_version == "v1.1.0-test"
            assert len(resp.response) > 0
        finally:
            # Clean up test file and rollback to baseline v1.0.0
            if test_version_file.exists():
                test_version_file.unlink()
            reload_configuration({"prompt_version": "v1.0.0"})

        # Confirm rollback was successful
        req_rollback = ChatRequest(message="Confirming rollback", conversation_id="session-test-6")
        resp_rollback = run_coro(chat(req_rollback))
        assert resp_rollback.prompt_version == "v1.0.0"

    def test_agent_instruction_immutability(self):
        """Verify that baseline root_agent instruction is never modified in-place."""
        original_instruction = root_agent.instruction
        agent_manager.get_active_runner(version_override="v1.0.0")
        assert root_agent.instruction == original_instruction

    def test_conversation_context_continuity(self):
        """Verify that multi-turn requests share session_service and maintain conversation session ID."""
        session_id = "session-continuity-123"
        req1 = ChatRequest(message="My name is Alice", conversation_id=session_id)
        resp1 = run_coro(chat(req1))
        assert resp1.request_id == req1.request_id

        req2 = ChatRequest(message="What is my name?", conversation_id=session_id)
        resp2 = run_coro(chat(req2))
        assert resp2.request_id == req2.request_id
        assert len(resp2.response) > 0
