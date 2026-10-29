"""
Unit and Integration tests for Prometheus monitoring and API readiness.

Uses direct deterministic handler invocation (like test_api.py) to prevent
Windows socketpair / event-loop deadlocks.

Covers:
- /metrics endpoint exposure and Prometheus text format
- Token protection when METRICS_AUTH_TOKEN is configured
- /health and /ready endpoints distinguishing liveness from readiness
- Counter and histogram updates upon chat and feedback calls
- Active version info gauge reflection
"""

from __future__ import annotations

import os
os.environ["ADK_OFFLINE_MODE"] = "true"

from pathlib import Path
import sys
import pytest
from fastapi import HTTPException

# Ensure ChatBot and repo root are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "ChatBot") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "ChatBot"))

from ADKBot.api import (
    home,
    readiness_check,
    metrics,
    chat,
    collect_feedback,
    feedback_store,
)
from ADKBot.metrics import (
    HTTP_REQUESTS_TOTAL,
    CHATBOT_REQUESTS_TOTAL,
    FEEDBACK_SUBMISSIONS_TOTAL,
    ACTIVE_VERSION_INFO,
    update_active_versions_gauge,
    format_prometheus_metrics,
)
from common.contracts import ChatRequest, FeedbackRating, FeedbackRequest


def run_coro(coro):
    """Drive coroutine to completion synchronously without creating an event loop."""
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value
    raise RuntimeError("Coroutine did not complete synchronously in offline mode.")


@pytest.fixture(autouse=True)
def clean_feedback_store():
    feedback_store.clear()
    yield
    feedback_store.clear()


class TestPrometheusMonitoring:

    def test_health_endpoint_liveness(self):
        data = home()
        assert data["status"] == "AI ChatBot API Running"

    def test_ready_endpoint_readiness(self):
        data = readiness_check()
        assert data["ready"] is True
        assert "model_version" in data
        assert "prompt_version" in data
        assert data["status"] == "Ready"

    def test_metrics_endpoint_open_by_default(self, monkeypatch):
        monkeypatch.delenv("METRICS_AUTH_TOKEN", raising=False)
        resp = metrics(authorization=None)
        assert resp.status_code == 200
        content = resp.body.decode("utf-8")
        assert "http_requests_total" in content
        assert "chatbot_requests_total" in content
        assert "chatbot_active_version_info" in content

    def test_metrics_endpoint_token_protection(self, monkeypatch):
        monkeypatch.setenv("METRICS_AUTH_TOKEN", "super-secret-token")

        # Missing token raises 401
        with pytest.raises(HTTPException) as exc_info:
            metrics(authorization=None)
        assert exc_info.value.status_code == 401

        # Invalid token raises 401
        with pytest.raises(HTTPException) as exc_info:
            metrics(authorization="Bearer wrong-token")
        assert exc_info.value.status_code == 401

        # Valid token succeeds
        resp = metrics(authorization="Bearer super-secret-token")
        assert resp.status_code == 200
        assert "http_requests_total" in resp.body.decode("utf-8")

    def test_chat_updates_prometheus_counters(self):
        req = ChatRequest(message="hello testing metrics")
        resp = run_coro(chat(req))
        assert resp.response is not None

        content = format_prometheus_metrics().decode("utf-8")
        assert "chatbot_requests_total" in content

    def test_feedback_updates_feedback_counters(self):
        # 1. Chat turn
        c_req = ChatRequest(message="feedback metric test query")
        c_resp = run_coro(chat(c_req))
        req_id = c_resp.request_id

        # 2. Submit verified feedback
        fb_req = FeedbackRequest(
            request_id=req_id,
            rating=FeedbackRating.POSITIVE,
            corrected_intent="Greeting"
        )
        fb_resp = run_coro(collect_feedback(fb_req))
        assert fb_resp.training_eligibility is True

        content = format_prometheus_metrics().decode("utf-8")
        assert 'chatbot_feedback_submissions_total{eligible="true",rating="positive"}' in content

    def test_active_versions_gauge(self):
        update_active_versions_gauge("test-model-v99", "test-prompt-v99")
        samples = list(ACTIVE_VERSION_INFO.collect())
        assert len(samples) > 0
        found = False
        for s in samples[0].samples:
            if s.labels.get("model_version") == "test-model-v99" and s.labels.get("prompt_version") == "test-prompt-v99":
                assert s.value == 1.0
                found = True
        assert found is True
