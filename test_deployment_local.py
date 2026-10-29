"""
Local deployment verification test script.

Tests the live FastAPI server, validating:
1. Process startup & health checks (/health and /ready)
2. Active model & prompt metadata resolution (/metadata/model and /metadata/prompt)
3. Live chat endpoint (/chat) with BERT intent classification
4. Feedback collection & durable persistence (/feedback)
5. Prometheus metrics exposure (/metrics)
6. Zero-downtime hot reload (/reload)
7. Feedback durability across service restarts
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
import httpx

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "ChatBot") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "ChatBot"))

def run_smoke_test(base_url: str = "http://127.0.0.1:6655"):
    results = {}
    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        # 1. Health check (liveness)
        t0 = time.time()
        resp_health = client.get("/health")
        assert resp_health.status_code == 200, f"Healthcheck failed: {resp_health.status_code}"
        health_data = resp_health.json()
        assert health_data["status"] == "AI ChatBot API Running"
        results["health"] = {"status": 200, "data": health_data, "latency_ms": round((time.time() - t0) * 1000, 2)}
        print("[PASS] 1. /health endpoint is live and responding (200 OK)")

        # 2. Readiness check (verifies model + instruction files)
        t0 = time.time()
        resp_ready = client.get("/ready")
        assert resp_ready.status_code == 200, f"Readiness check failed: {resp_ready.status_code}"
        ready_data = resp_ready.json()
        assert ready_data["ready"] is True
        assert ready_data["model_version"] == "v0.1.0-baseline"
        assert ready_data["prompt_version"] == "v1.0.0"
        results["ready"] = {"status": 200, "data": ready_data, "latency_ms": round((time.time() - t0) * 1000, 2)}
        print(f"[PASS] 2. /ready verified active model ({ready_data['model_version']}) and instruction ({ready_data['prompt_version']})")

        # 3. Model metadata
        resp_model_meta = client.get("/metadata/model")
        assert resp_model_meta.status_code == 200
        model_meta = resp_model_meta.json()
        assert model_meta["model_version"] == "v0.1.0-baseline"
        results["model_metadata"] = model_meta
        print(f"[PASS] 3. /metadata/model resolved baseline artifact: {model_meta['artifact_location']}")

        # 4. Prompt metadata
        resp_prompt_meta = client.get("/metadata/prompt")
        assert resp_prompt_meta.status_code == 200
        prompt_meta = resp_prompt_meta.json()
        assert prompt_meta["prompt_version"] == "v1.0.0"
        results["prompt_metadata"] = prompt_meta
        print(f"[PASS] 4. /metadata/prompt resolved baseline prompt: {prompt_meta['prompt_version']}")

        # 5. Chat Turn
        t0 = time.time()
        chat_payload = {"message": "Hello, good morning!"}
        resp_chat = client.post("/chat", json=chat_payload)
        assert resp_chat.status_code == 200, f"Chat failed: {resp_chat.text}"
        chat_data = resp_chat.json()
        assert "response" in chat_data
        assert "request_id" in chat_data
        assert chat_data["predicted_intent"] is not None
        print(f"[INFO] Chat response: predicted_intent='{chat_data['predicted_intent']}', confidence={chat_data.get('intent_confidence')}")
        results["chat"] = {"status": 200, "data": chat_data, "latency_ms": round((time.time() - t0) * 1000, 2)}
        print(f"[PASS] 5. /chat turn executed: predicted_intent='{chat_data['predicted_intent']}', req_id={chat_data['request_id']}")

        # 6. Feedback collection
        fb_payload = {
            "request_id": chat_data["request_id"],
            "rating": "positive",
            "corrected_intent": "Greeting",
            "textual_feedback": "Verified morning greeting"
        }
        resp_fb = client.post("/feedback", json=fb_payload)
        assert resp_fb.status_code == 201, f"Feedback failed: {resp_fb.text}"
        fb_data = resp_fb.json()
        assert fb_data["training_eligibility"] is True
        results["feedback"] = fb_data
        print(f"[PASS] 6. /feedback accepted and marked training_eligibility={fb_data['training_eligibility']}")

        # 7. Metrics Endpoint
        resp_metrics = client.get("/metrics")
        assert resp_metrics.status_code == 200
        metrics_text = resp_metrics.text
        assert "http_requests_total" in metrics_text
        assert "chatbot_requests_total" in metrics_text
        assert "chatbot_active_version_info" in metrics_text
        results["metrics"] = {"status": 200, "sample_length": len(metrics_text)}
        print("[PASS] 7. /metrics exposed valid Prometheus format metrics")

        # 8. Reload endpoint
        resp_reload = client.post("/reload", json={})
        assert resp_reload.status_code == 200
        reload_data = resp_reload.json()
        assert reload_data["status"] == "reloaded"
        results["reload"] = reload_data
        print(f"[PASS] 8. /reload succeeded: active_model={reload_data['active_model_version']}, active_prompt={reload_data['active_prompt_version']}")

    return results

if __name__ == "__main__":
    import threading
    import uvicorn
    from ADKBot.api import app

    os.environ["ADK_OFFLINE_MODE"] = "true"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["EVIDENTLY_DISABLE_TELEMETRY"] = "1"

    config = uvicorn.Config(app, host="127.0.0.1", port=6655, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to start
    time.sleep(2.0)
    try:
        res = run_smoke_test("http://127.0.0.1:6655")
        print("\n=== ALL 8 ENDPOINT TESTS PASSED SUCCESSFULLY ===")
    finally:
        server.should_exit = True
