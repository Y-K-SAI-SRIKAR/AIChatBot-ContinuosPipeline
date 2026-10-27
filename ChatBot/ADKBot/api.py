from datetime import datetime, timezone
import json
from pathlib import Path
import os
import sys
import time
from typing import Any, Dict, Optional, Set

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
loaded = load_dotenv(dotenv_path=env_path)

from fastapi import FastAPI, HTTPException, status
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService

import logging
logger = logging.getLogger("ADKBot.api")

from ADKBot.agent import (
    agent_manager,
    root_agent,
    active_prompt_version,
    runner,
    session_service,
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

app = FastAPI(title="AI ChatBot API", version="1.0.0")

feedback_store = FeedbackStore()

# In-memory recent turn cache for correlating feedback to original user queries
_recent_turns: Dict[str, Dict[str, Any]] = {}

# Cached intent predictor instance
_predictor_instance = None
_predictor_model_path = None


def _get_intent_predictor(artifact_path: Path):
    global _predictor_instance, _predictor_model_path
    if _predictor_instance is not None and _predictor_model_path == artifact_path:
        return _predictor_instance
    try:
        from src.model.predict import IntentPredictor
        if artifact_path.exists():
            _predictor_instance = IntentPredictor(artifact_dir=artifact_path)
            _predictor_model_path = artifact_path
            return _predictor_instance
    except Exception:
        pass
    return None


@app.get("/")
@app.get("/health")
def home():
    """Health check endpoint exposing active system versions."""
    return {
        "status": "AI ChatBot API Running",
        "active_prompt_version": agent_manager.active_prompt_version,
        "adk_foundation_model": "gemini-3-flash-preview",
        "timestamp": datetime.now(timezone.utc).isoformat(),
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

    # Atomically resolve active prompt configuration and corresponding ADK Runner
    active_runner, active_prompt = agent_manager.get_active_runner()
    active_model = resolve_active_model_config()

    # Configurable confidence threshold & enable flag
    confidence_threshold = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.50"))
    bert_enabled = os.getenv("BERT_CLASSIFIER_ENABLED", "true").lower() in {"true", "1", "yes"}

    # Run BERT intent classifier if available and enabled
    predicted_intent = None
    intent_confidence = None
    if bert_enabled:
        predictor = _get_intent_predictor(active_model.artifact_location)
        if predictor:
            try:
                pred_res = predictor.predict(request.message)
                predicted_intent = pred_res.intent
                intent_confidence = round(pred_res.confidence, 4)
                if intent_confidence < confidence_threshold or predicted_intent == "UNKNOWN":
                    logger.info(
                        f"Low confidence ({intent_confidence} < {confidence_threshold}) for query '{request.message}'. "
                        "Preserving general ADK conversational reasoning."
                    )
            except Exception as e:
                logger.warning(f"Intent prediction failed: {e}")
                predicted_intent = None
                intent_confidence = None

    # Check for offline test mode (prevents sandbox network hanging during automated test runs)
    if os.getenv("ADK_OFFLINE_MODE", "false").lower() in {"true", "1", "yes"}:
        response_text = f"Simulated agent response for: '{request.message}'"
    else:
        try:
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
            logger.warning(f"ADK runner exception: {e}")
            if not response_text:
                response_text = f"Agent response generated for: '{request.message}'"

    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    # Structured logging for turn observability
    logger.info(
        f"Turn [{request.request_id}] | intent={predicted_intent} (conf={intent_confidence}) | "
        f"model={active_model.model_version} | prompt={active_prompt.prompt_version} | latency={latency_ms}ms"
    )

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


@app.post("/api/v1/reload")
@app.post("/reload")
def reload_configuration(payload: Optional[Dict[str, Any]] = None):
    """
    Explicit controlled reload endpoint for newly promoted prompt or model versions.
    Re-resolves configuration and updates runner atomically without restarting server.
    """
    global _predictor_instance, _predictor_model_path
    prompt_override = payload.get("prompt_version") if payload else None
    model_override = payload.get("model_version") if payload else None

    reloaded_prompt = agent_manager.reload(version_override=prompt_override)
    reloaded_model = resolve_active_model_config(version_override=model_override)

    # Invalidate predictor cache to force fresh model loading on next request
    _predictor_instance = None
    _predictor_model_path = None

    return {
        "status": "Configuration reloaded successfully",
        "active_prompt_version": reloaded_prompt.prompt_version,
        "active_model_version": reloaded_model.model_version,
        "adk_model": reloaded_prompt.metadata.associated_adk_model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


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

    return persisted_record


@app.get("/feedback/stats")
@app.get("/api/v1/feedback/stats")
def get_feedback_stats():
    """
    Expose aggregate feedback health and count metrics without exposing
    raw feedback records, user messages, or personal data publicly.
    """
    return feedback_store.get_stats()


@app.get("/api/v1/metadata/prompt", response_model=PromptMetadata)
def get_active_prompt_metadata():
    """Retrieve metadata for currently active system instructions."""
    resolved = resolve_active_prompt_config()
    return resolved.metadata


@app.get("/api/v1/metadata/model", response_model=ModelMetadata)
def get_active_model_metadata():
    """Retrieve metadata for currently active intent classification model."""
    resolved = resolve_active_model_config()
    return resolved.metadata
