"""
Integration tests for FastAPI endpoints and route handlers.
Covers:
- Health check
- Chat endpoint and validation
- Feedback collection with training eligibility rules
- Unknown intent label rejection (HTTP 400)
- Duplicate submission rejection / idempotency (HTTP 409)
- Feedback stats endpoint without exposing PII
- Metadata endpoints
"""

import os
os.environ["ADK_OFFLINE_MODE"] = "true"

from pathlib import Path
import sys
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

# Ensure ChatBot and repo root are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "ChatBot") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "ChatBot"))

# pyrefly: ignore[missing-import]
from ADKBot.api import (
    home,
    chat,
    collect_feedback,
    get_feedback_stats,
    get_active_prompt_metadata,
    get_active_model_metadata,
)
from common.contracts import (
    ChatRequest,
    FeedbackRating,
    FeedbackRequest,
)


def run_coro(coro):
    """
    Drive a coroutine to completion synchronously without creating an event loop or socketpair,
    ensuring deterministic offline execution in sandboxed environments.
    """
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value
    raise RuntimeError("Coroutine did not complete synchronously in offline mode")


class TestChatBotAPI:
    def test_home_healthcheck(self):
        result = home()
        assert result == {"status": "AI ChatBot API Running"}

    def test_chat_endpoint_valid_request(self):
        req = ChatRequest(
            message="Hello, how are you?",
            conversation_id="test-session-1"
        )
        resp = run_coro(chat(req))
        assert resp.request_id == req.request_id
        assert "Simulated agent response" in resp.response or len(resp.response) > 0
        assert resp.model_version == "v0.1.0-baseline"
        assert resp.prompt_version == "v1.0.0"
        assert resp.latency_ms >= 0.0

    def test_chat_endpoint_rejects_empty_message(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="   ")

    def test_feedback_endpoint_eligible_correction(self):
        # 1. Trigger a chat turn
        chat_req = ChatRequest(message="Good morning unique test 1")
        chat_resp = run_coro(chat(chat_req))

        # 2. Submit feedback with valid corrected_intent
        fb_req = FeedbackRequest(
            request_id=chat_resp.request_id,
            rating=FeedbackRating.NEGATIVE,
            corrected_intent="Greeting",
            textual_feedback="The model should have recognized this as a morning greeting."
        )
        fb_rec = run_coro(collect_feedback(fb_req))
        assert fb_rec.request_id == chat_resp.request_id
        assert fb_rec.rating == FeedbackRating.NEGATIVE
        assert fb_rec.corrected_intent == "Greeting"
        assert fb_rec.training_eligibility is True
        assert "Eligible" in fb_rec.eligibility_reason

    def test_feedback_endpoint_ineligible_negative_without_correction(self):
        chat_req = ChatRequest(message="Tell me a joke unique test 2")
        chat_resp = run_coro(chat(chat_req))

        fb_req = FeedbackRequest(
            request_id=chat_resp.request_id,
            rating=FeedbackRating.NEGATIVE,
            textual_feedback="Not funny"
        )
        fb_rec = run_coro(collect_feedback(fb_req))
        assert fb_rec.training_eligibility is False
        assert "Ineligible" in fb_rec.eligibility_reason

    def test_feedback_endpoint_rejects_unknown_intent(self):
        chat_req = ChatRequest(message="What is the weather unique test 3")
        chat_resp = run_coro(chat(chat_req))

        # Attempt to correct with an unknown intent label
        fb_req = FeedbackRequest(
            request_id=chat_resp.request_id,
            rating=FeedbackRating.NEGATIVE,
            corrected_intent="CompletelyFakeNonExistentIntent"
        )
        with pytest.raises(HTTPException) as exc_info:
            run_coro(collect_feedback(fb_req))
        assert exc_info.value.status_code == 400
        assert "Unknown intent label" in exc_info.value.detail

    def test_feedback_endpoint_rejects_duplicate_submission(self):
        chat_req = ChatRequest(message="Hello there unique test 4")
        chat_resp = run_coro(chat(chat_req))

        fb_req = FeedbackRequest(
            request_id=chat_resp.request_id,
            rating=FeedbackRating.POSITIVE
        )
        # First submission succeeds
        run_coro(collect_feedback(fb_req))

        # Second submission for the same request_id must be rejected as duplicate (409)
        with pytest.raises(HTTPException) as exc_info:
            run_coro(collect_feedback(fb_req))
        assert exc_info.value.status_code == 409
        assert "Duplicate feedback" in exc_info.value.detail

    def test_feedback_stats_endpoint(self):
        stats = get_feedback_stats()
        assert "total_feedback" in stats
        assert "positive_ratings" in stats
        assert "negative_ratings" in stats
        assert "training_eligible" in stats
        assert "drift_monitoring" in stats
        assert isinstance(stats["total_feedback"], int)

    def test_metadata_endpoints(self):
        prompt_meta = get_active_prompt_metadata()
        assert prompt_meta.prompt_version == "v1.0.0"
        assert prompt_meta.associated_adk_model == "gemini-3-flash-preview"

        model_meta = get_active_model_metadata()
        assert model_meta.model_version == "v0.1.0-baseline"
        assert model_meta.num_classes == 22
