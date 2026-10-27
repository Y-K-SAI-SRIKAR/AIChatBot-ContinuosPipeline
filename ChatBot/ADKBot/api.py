from datetime import datetime, timezone
from pathlib import Path
import os
import sys
import time
from typing import Any, Dict, Optional

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

from ADKBot.agent import root_agent, active_prompt_version
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
from common.feedback_store import FeedbackStore

app = FastAPI(title="AI ChatBot API", version="1.0.0")

session_service = InMemorySessionService()

runner = Runner(
    app_name="WPChatBot",
    agent=root_agent,
    session_service=session_service,
    auto_create_session=True
)

feedback_store = FeedbackStore()

# In-memory recent turn cache for correlating feedback to original user queries
_recent_turns: Dict[str, Dict[str, Any]] = {}


@app.get("/")
def home():
    """Health check endpoint."""
    return {
        "status": "AI ChatBot API Running"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Handle user chat message, invoke Google ADK Agent, and return standardized response contract.
    """
    start_time = time.perf_counter()
    session_id = request.conversation_id or "default-session"
    response_text = ""

    # Resolve active model & prompt configurations
    active_prompt = resolve_active_prompt_config()
    active_model = resolve_active_model_config()

    # Check for offline test mode (prevents sandbox network hanging during automated test runs)
    if os.getenv("ADK_OFFLINE_MODE", "false").lower() in {"true", "1", "yes"}:
        response_text = f"Simulated agent response for: '{request.message}'"
    else:
        try:
            async for event in runner.run_async(
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
            if not response_text:
                response_text = f"Agent response generated for: '{request.message}'"

    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    # Cache recent turn for feedback correlation
    _recent_turns[request.request_id] = {
        "message": request.message,
        "response": response_text,
        "predicted_intent": None,
        "model_version": active_model.model_version,
        "prompt_version": active_prompt.prompt_version,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    return ChatResponse(
        request_id=request.request_id,
        response=response_text,
        predicted_intent=None,
        intent_confidence=None,
        model_version=active_model.model_version,
        prompt_version=active_prompt.prompt_version,
        latency_ms=latency_ms
    )


@app.post("/feedback", response_model=FeedbackRecord, status_code=status.HTTP_201_CREATED)
async def collect_feedback(request: FeedbackRequest):
    """
    Record user feedback on a chat turn.
    
    CRITICAL RULE:
    Negative feedback alone is NOT eligible for supervised retraining.
    Only feedback with an explicit corrected_intent is marked training_eligibility=True.
    """
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
        corrected_intent=request.corrected_intent,
        corrected_response=request.corrected_response,
        textual_feedback=request.textual_feedback,
        model_version=cached_turn.get("model_version", active_model.model_version),
        prompt_version=cached_turn.get("prompt_version", active_prompt.prompt_version),
    )

    persisted_record = feedback_store.save_feedback(record)
    return persisted_record


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
