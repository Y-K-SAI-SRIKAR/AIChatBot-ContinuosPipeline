from datetime import datetime, timezone
import json
from pathlib import Path
import os
import sys
import time
from typing import Any, Dict, Optional, Set

# Ensure repository root and module subpaths are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for p in [str(REPO_ROOT), str(REPO_ROOT / "ChatBot"), str(REPO_ROOT / "ChatBot-PipeLine")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
loaded = load_dotenv(dotenv_path=env_path)

from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import PlainTextResponse
# pyrefly: ignore[missing-import]
from google.genai import types
# pyrefly: ignore[missing-import]
from google.adk.runners import Runner
# pyrefly: ignore[missing-import]
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from ADKBot.agent import root_agent, active_prompt_version, agent_manager
from ADKBot.metrics import (
    HTTP_REQUESTS_TOTAL,
    HTTP_REQUEST_DURATION_SECONDS,
    CHATBOT_REQUESTS_TOTAL,
    CHATBOT_ADK_FAILURES_TOTAL,
    INTENT_PREDICTIONS_TOTAL,
    INTENT_PREDICTION_CONFIDENCE,
    FEEDBACK_SUBMISSIONS_TOTAL,
    update_active_versions_gauge,
    format_prometheus_metrics,
    CONTENT_TYPE_LATEST,
)
from common.contracts import (
    ChatRequest,
    ChatResponse,
    FeedbackRating,
    FeedbackRequest,
    FeedbackRecord,
    ModelMetadata,
    PromptMetadata,
)
from common.config_resolver import (
    resolve_active_prompt_config,
    resolve_active_model_config,
)
from common.feedback_store import (
    FeedbackStore,
    DuplicateFeedbackError,
    PersistenceError,
)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI ChatBot API", version="1.0.0")

# Allowed Origins configuration for web client integrations
raw_origins = os.getenv("ALLOWED_ORIGINS", "").strip()
allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
if not allowed_origins and os.getenv("ENVIRONMENT", "development").lower() == "development":
    # Keep local Vite/React development convenient without exposing a deployed
    # API to arbitrary browser origins.  Production must set ALLOWED_ORIGINS.
    allowed_origins = ["http://localhost:3000", "http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=bool(allowed_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)

session_service = InMemorySessionService()

runner = Runner(
    app_name="WPChatBot",
    agent=root_agent,
    session_service=session_service,
    auto_create_session=True
)

feedback_storage_override = os.getenv("FEEDBACK_STORAGE_PATH", "").strip()
feedback_store = FeedbackStore(storage_path=Path(feedback_storage_override) if feedback_storage_override else None)

# In-memory recent turn cache for correlating feedback to original user queries
_recent_turns: Dict[str, Dict[str, Any]] = {}

# Cached intent predictor instance
_active_model_version: Optional[str] = None
_predictor_instance = None
_predictor_model_path = None

# Initialize Prometheus active version gauge
try:
    _init_model = resolve_active_model_config()
    _init_prompt = resolve_active_prompt_config()
    update_active_versions_gauge(_init_model.model_version, _init_prompt.prompt_version)
except Exception:
    pass


@app.middleware("http")
async def prometheus_metrics_middleware(request: Request, call_next):
    """
    Lightweight middleware capturing HTTP request counts and durations.
    Uses normalized endpoint path templates to avoid metric label explosion.
    """
    start_time = time.perf_counter()
    path = request.url.path

    # Mask IDs to prevent unbounded Prometheus label cardinality
    endpoint = path
    if path.startswith("/chat"):
        endpoint = "/chat"
    elif path.startswith("/feedback"):
        endpoint = "/feedback" if path == "/feedback" else "/feedback/stats"
    elif path.startswith("/api/v1/metadata"):
        endpoint = "/api/v1/metadata"

    response = await call_next(request)
    duration = time.perf_counter() - start_time

    try:
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=str(response.status_code)
        ).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=request.method,
            endpoint=endpoint
        ).observe(duration)
    except Exception:
        pass

    return response


def _get_intent_predictor(artifact_path: Path):
    global _predictor_instance, _predictor_model_path
    if _predictor_instance is not None and _predictor_model_path == artifact_path:
        return _predictor_instance
    try:
        # pyrefly: ignore[missing-import]
        from src.model.predict import IntentPredictor
        if artifact_path.exists():
            _predictor_instance = IntentPredictor(artifact_dir=artifact_path)
            _predictor_model_path = artifact_path
            return _predictor_instance
    except Exception as e:
        import logging
        logging.getLogger("ADKBot.api").warning(f"Could not initialize intent predictor from {artifact_path}: {e}")
    return None


@app.get("/")
@app.get("/health")
def home():
    """Health check endpoint — verifies process liveness."""
    return {"status": "AI ChatBot API Running"}


@app.get("/ready")
def readiness_check():
    """
    Readiness endpoint — verifies model artifact and system instruction availability.
    Distinguishes process liveness from operational model readiness.
    """
    try:
        active_model = resolve_active_model_config()
        active_prompt = resolve_active_prompt_config()
        model_ready = Path(active_model.artifact_location).exists()
        prompt_ready = Path(active_prompt.instruction_path).exists()
        if not model_ready or not prompt_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "ready": False,
                    "model_ready": model_ready,
                    "prompt_ready": prompt_ready,
                    "model_version": active_model.model_version,
                    "prompt_version": active_prompt.prompt_version,
                }
            )
        return {
            "ready": True,
            "status": "Ready",
            "model_version": active_model.model_version,
            "prompt_version": active_prompt.prompt_version,
            "artifact_location": str(active_model.artifact_location)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Readiness check failed: {e}"
        )


@app.get("/metrics")
def metrics(authorization: Optional[str] = Header(None)):
    """
    Prometheus scrape endpoint.
    If METRICS_AUTH_TOKEN is configured in environment, enforces token-based protection.
    """
    required_token = os.getenv("METRICS_AUTH_TOKEN", "").strip()
    if required_token:
        # Require Bearer token authentication
        auth_header = authorization or ""
        expected = f"Bearer {required_token}"
        if auth_header != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: invalid or missing Prometheus metrics authentication token.",
                headers={"WWW-Authenticate": "Bearer"}
            )
    
    return Response(content=format_prometheus_metrics(), media_type=CONTENT_TYPE_LATEST)


@app.post("/reload")
def reload_configuration(payload: Optional[Dict[str, str]] = None):
    """
    Controlled reload mechanism for the active model and prompt configurations.
    Enables zero-downtime hot reloading of models and instructions upon pipeline promotion.
    """
    global _predictor_instance, _predictor_model_path, runner, _active_model_version
    payload = payload or {}
    model_override = payload.get("model_version")
    prompt_override = payload.get("prompt_version")

    # Reload agent manager and prompt configuration
    resolved_prompt = agent_manager.reload(version_override=prompt_override)
    runner, _ = agent_manager.get_active_runner(version_override=resolved_prompt.prompt_version)

    # Force reset of intent predictor so it picks up the newly resolved active model
    _predictor_instance = None
    _predictor_model_path = None
    resolved_model = resolve_active_model_config(version_override=model_override)
    _active_model_version = resolved_model.model_version
    _get_intent_predictor(resolved_model.artifact_location)

    # Update Prometheus active version gauge
    update_active_versions_gauge(resolved_model.model_version, resolved_prompt.prompt_version)

    return {
        "status": "reloaded",
        "active_model_version": resolved_model.model_version,
        "active_prompt_version": resolved_prompt.prompt_version,
        "artifact_location": str(resolved_model.artifact_location),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Handle user chat message, classify intent with BERT, invoke Google ADK Agent,
    and return standardized response contract.
    """
    start_time = time.perf_counter()
    session_id = request.conversation_id or "default-session"
    response_text = ""

    CHATBOT_REQUESTS_TOTAL.inc()

    # Resolve active model & prompt configurations
    active_prompt = resolve_active_prompt_config(version_override=agent_manager.active_prompt_version)
    active_model = resolve_active_model_config(version_override=_active_model_version)

    # Update Prometheus gauge
    update_active_versions_gauge(active_model.model_version, active_prompt.prompt_version)

    # Predict intent via fine-tuned BERT classifier
    predicted_intent = None
    intent_confidence = None
    bert_enabled = os.getenv("BERT_CLASSIFIER_ENABLED", "true").strip().lower() not in {"false", "0", "no"}
    if bert_enabled:
        predictor = _get_intent_predictor(active_model.artifact_location)
        if predictor:
            try:
                pred_res = predictor.predict(request.message)
                predicted_intent = pred_res.intent
                intent_confidence = round(pred_res.confidence, 4)
                INTENT_PREDICTIONS_TOTAL.labels(intent=predicted_intent or "unknown").inc()
                if intent_confidence is not None:
                    INTENT_PREDICTION_CONFIDENCE.observe(intent_confidence)
            except Exception:
                predicted_intent = None
                intent_confidence = None

    # Check for offline test mode (prevents sandbox network hanging during automated test runs)
    if os.getenv("ADK_OFFLINE_MODE", "false").lower() in {"true", "1", "yes"}:
        response_text = f"Simulated agent response for: '{request.message}'"
    else:
        try:
            active_runner, _ = agent_manager.get_active_runner(active_prompt.prompt_version)
            async for event in active_runner.run_async(
                user_id="web-user",
                session_id=session_id,
                new_message=types.UserContent(
                    parts=[
                        types.Part(text=request.message)
                    ]
                )
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if getattr(part, "text", None):
                            response_text += part.text
        except Exception as e:
            CHATBOT_ADK_FAILURES_TOTAL.inc()
            if not response_text:
                response_text = f"Agent response generated for: '{request.message}'"

    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    # Cache recent turn for feedback correlation
    _recent_turns[request.request_id] = {
        "message": request.message,
        "response": response_text,
        "predicted_intent": predicted_intent,
        "model_version": active_model.model_version,
        "prompt_version": active_prompt.prompt_version,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    return ChatResponse(
        request_id=request.request_id,
        response=response_text,
        predicted_intent=predicted_intent,
        intent_confidence=intent_confidence,
        model_version=active_model.model_version,
        prompt_version=active_prompt.prompt_version,
        latency_ms=latency_ms
    )


def get_known_intent_labels() -> Optional[Set[str]]:
    """Retrieve known valid intent classes from active model or label mapping."""
    try:
        active_model = resolve_active_model_config()
        mapping_file = active_model.artifact_location / "label_mapping.json"
        if not mapping_file.exists():
            mapping_file = REPO_ROOT / "ChatBot-PipeLine" / "data" / "processed" / "label_mapping.json"
        if mapping_file.exists():
            with open(mapping_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "classes" in data:
                    return set(data["classes"])
                if "label2id" in data:
                    return set(data["label2id"].keys())
    except Exception:
        pass
    return None


@app.post("/feedback", response_model=FeedbackRecord, status_code=status.HTTP_201_CREATED)
async def collect_feedback(request: FeedbackRequest):
    """
    Record user feedback on a chat turn.
    
    CRITICAL RULES:
    1. Negative feedback alone is NOT eligible for supervised retraining.
    2. Only feedback with a verified, valid corrected_intent is marked training_eligibility=True.
    3. Unknown corrected_intent values are rejected when label mapping is available.
    4. Duplicate feedback submissions are detected and rejected via idempotency check.
    5. No retraining occurs during this request (collection & retraining are separate).
    """
    # 1. Idempotency / Duplicate Check
    if feedback_store.has_feedback_for_request(request.request_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Duplicate feedback: feedback for request_id '{request.request_id}' has already been submitted."
        )

    # 2. Validate corrected_intent if provided
    if request.corrected_intent and request.corrected_intent.strip():
        clean_intent = request.corrected_intent.strip()
        known_labels = get_known_intent_labels()
        if known_labels and clean_intent not in known_labels:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown intent label '{clean_intent}'. Valid labels: {sorted(list(known_labels))}"
            )

    cached_turn = _recent_turns.get(request.request_id, {})
    message_text = cached_turn.get("message", f"Turn [{request.request_id}]")
    predicted_intent = cached_turn.get("predicted_intent")

    active_prompt = resolve_active_prompt_config()
    active_model = resolve_active_model_config()

    record = FeedbackRecord(
        request_id=request.request_id,
        message=message_text,
        predicted_intent=predicted_intent,
        rating=request.rating,
        corrected_intent=request.corrected_intent.strip() if request.corrected_intent else None,
        corrected_response=request.corrected_response.strip() if request.corrected_response else None,
        textual_feedback=request.textual_feedback.strip() if request.textual_feedback else None,
        model_version=cached_turn.get("model_version", active_model.model_version),
        prompt_version=cached_turn.get("prompt_version", active_prompt.prompt_version),
    )

    try:
        persisted_record = feedback_store.save_feedback(record)
    except DuplicateFeedbackError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except PersistenceError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to persist feedback: {e}")

    try:
        FEEDBACK_SUBMISSIONS_TOTAL.labels(
            rating=persisted_record.rating.value,
            eligible=str(persisted_record.training_eligibility).lower()
        ).inc()
    except Exception:
        pass

    return persisted_record


@app.get("/feedback/stats")
@app.get("/api/v1/feedback/stats")
def get_feedback_stats():
    """
    Expose aggregate feedback health and count metrics without exposing
    raw feedback records, user messages, or personal data publicly.
    """
    return feedback_store.get_stats()


@app.get("/metadata/prompt", response_model=PromptMetadata)
@app.get("/api/v1/metadata/prompt", response_model=PromptMetadata)
def get_active_prompt_metadata():
    """Retrieve metadata for currently active system instructions."""
    resolved = resolve_active_prompt_config(version_override=agent_manager.active_prompt_version)
    return resolved.metadata


@app.get("/metadata/model", response_model=ModelMetadata)
@app.get("/api/v1/metadata/model", response_model=ModelMetadata)
def get_active_model_metadata():
    """Retrieve metadata for currently active intent classification model."""
    resolved = resolve_active_model_config(version_override=_active_model_version)
    return resolved.metadata
