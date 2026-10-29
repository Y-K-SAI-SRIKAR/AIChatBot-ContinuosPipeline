"""
Prometheus Metrics Instrumentation for AI ChatBot and MLOps Pipeline.

Defines Prometheus metrics counters, histograms, and gauges following
production observability standards:
- Bounded label dimensions (no raw messages, user IDs, or unbounded strings).
- Zero exposure of credentials, feedback text, or sensitive user data.
- Lightweight in-memory tracking via official prometheus-client.
"""

from __future__ import annotations

import os
from typing import Optional
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CollectorRegistry,
    REGISTRY,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# Shared CollectorRegistry (uses default global REGISTRY)
registry: CollectorRegistry = REGISTRY

# ── HTTP Metrics ─────────────────────────────────────────────────────────────
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total count of HTTP requests handled by the API",
    ["method", "endpoint", "status_code"],
    registry=registry,
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "Histogram of HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=registry,
)

# ── ChatBot & ADK Agent Metrics ──────────────────────────────────────────────
CHATBOT_REQUESTS_TOTAL = Counter(
    "chatbot_requests_total",
    "Total user chat queries processed",
    registry=registry,
)

CHATBOT_ADK_FAILURES_TOTAL = Counter(
    "chatbot_adk_failures_total",
    "Total failures encountered when invoking Google ADK / Gemini Agent",
    registry=registry,
)

# ── Intent Classifier Metrics ────────────────────────────────────────────────
INTENT_PREDICTIONS_TOTAL = Counter(
    "chatbot_intent_predictions_total",
    "Total intent predictions performed by BERT classifier",
    ["intent"],
    registry=registry,
)

INTENT_PREDICTION_CONFIDENCE = Histogram(
    "chatbot_intent_prediction_confidence",
    "Distribution of confidence scores for intent predictions [0.0 - 1.0]",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
    registry=registry,
)

# ── Feedback Metrics ─────────────────────────────────────────────────────────
FEEDBACK_SUBMISSIONS_TOTAL = Counter(
    "chatbot_feedback_submissions_total",
    "Total feedback submissions received",
    ["rating", "eligible"],
    registry=registry,
)

# ── Active Version Info Gauge ────────────────────────────────────────────────
ACTIVE_VERSION_INFO = Gauge(
    "chatbot_active_version_info",
    "Information on current active model and prompt versions",
    ["model_version", "prompt_version"],
    registry=registry,
)

# ── MLOps & Continuous Retraining Metrics ────────────────────────────────────
RETRAINING_RUNS_TOTAL = Counter(
    "mloops_retraining_runs_total",
    "Total retraining pipeline execution triggers by terminal status",
    ["status"],  # 'success', 'rejected', 'insufficient_feedback', 'failed'
    registry=registry,
)

CANDIDATE_PROMOTIONS_TOTAL = Counter(
    "mloops_candidate_promotions_total",
    "Total candidate model checkpoints promoted to production",
    registry=registry,
)

CANDIDATE_REJECTIONS_TOTAL = Counter(
    "mloops_candidate_rejections_total",
    "Total candidate model checkpoints rejected by quality gates",
    registry=registry,
)

TRAINING_DURATION_SECONDS = Gauge(
    "mloops_training_duration_seconds",
    "Duration of the most recent model training execution in seconds",
    registry=registry,
)

MODEL_EVALUATION_MACRO_F1 = Gauge(
    "mloops_model_macro_f1",
    "Held-out test set Macro F1 score of the active production model",
    registry=registry,
)

MODEL_EVALUATION_ACCURACY = Gauge(
    "mloops_model_accuracy",
    "Held-out test set accuracy score of the active production model",
    registry=registry,
)


def update_active_versions_gauge(model_version: str, prompt_version: str) -> None:
    """Update active version gauge while clearing stale label combinations."""
    ACTIVE_VERSION_INFO.clear()
    ACTIVE_VERSION_INFO.labels(
        model_version=str(model_version).strip(),
        prompt_version=str(prompt_version).strip()
    ).set(1.0)


def format_prometheus_metrics() -> bytes:
    """Format all registered metrics for Prometheus scraping."""
    return generate_latest(registry)

