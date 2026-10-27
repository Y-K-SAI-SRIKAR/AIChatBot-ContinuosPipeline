"""
Unit tests for shared Pydantic data contracts and validation rules.
"""

import pytest
from pydantic import ValidationError

from common.contracts import (
    ChatRequest,
    ChatResponse,
    FeedbackRating,
    FeedbackRecord,
    FeedbackRequest,
    ModelMetadata,
    PromotionStatus,
    PromptEvaluationStatus,
    PromptMetadata,
)


class TestChatRequestContract:
    def test_valid_chat_request_with_defaults(self):
        req = ChatRequest(message="Hello there!")
        assert req.message == "Hello there!"
        assert req.conversation_id is None
        assert req.request_id is not None
        assert len(req.request_id) > 0

    def test_valid_chat_request_preserves_custom_ids(self):
        req = ChatRequest(
            message="What is the weather?",
            conversation_id="conv-123",
            request_id="custom-req-id-999"
        )
        assert req.message == "What is the weather?"
        assert req.conversation_id == "conv-123"
        assert req.request_id == "custom-req-id-999"

    def test_rejects_empty_message(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="")

    def test_rejects_whitespace_only_message(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="   \t\n  ")


class TestChatResponseContract:
    def test_valid_response_minimal(self):
        resp = ChatResponse(
            request_id="req-123",
            response="Hello human, I am your assistant.",
            model_version="v0.1.0-baseline",
            prompt_version="v1.0.0",
            latency_ms=15.4
        )
        assert resp.request_id == "req-123"
        assert resp.response == "Hello human, I am your assistant."
        assert resp.reply == "Hello human, I am your assistant."
        assert resp.predicted_intent is None
        assert resp.intent_confidence is None
        assert resp.latency_ms == 15.4

    def test_valid_response_with_intent(self):
        resp = ChatResponse(
            request_id="req-456",
            response="Good morning!",
            predicted_intent="Greeting",
            intent_confidence=0.985,
            model_version="v0.1.0-baseline",
            prompt_version="v1.0.0",
            latency_ms=42.1
        )
        assert resp.predicted_intent == "Greeting"
        assert resp.intent_confidence == 0.985

    def test_confidence_boundary_validation(self):
        with pytest.raises(ValidationError):
            ChatResponse(
                request_id="req-1",
                response="test",
                intent_confidence=1.5,  # must be <= 1.0
                model_version="v1",
                prompt_version="v1",
                latency_ms=10.0
            )


class TestFeedbackContracts:
    def test_feedback_request_positive_normalization(self):
        req = FeedbackRequest(request_id="req-1", rating="positive")
        assert req.rating == FeedbackRating.POSITIVE

        req2 = FeedbackRequest(request_id="req-1", rating="+1")
        assert req2.rating == FeedbackRating.POSITIVE

        req3 = FeedbackRequest(request_id="req-1", rating="thumbs_up")
        assert req3.rating == FeedbackRating.POSITIVE

    def test_feedback_request_negative_normalization(self):
        req = FeedbackRequest(request_id="req-1", rating="negative")
        assert req.rating == FeedbackRating.NEGATIVE

        req2 = FeedbackRequest(request_id="req-1", rating="-1")
        assert req2.rating == FeedbackRating.NEGATIVE

    def test_feedback_request_invalid_rating(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(request_id="req-1", rating="neutral_invalid")

    def test_training_eligibility_negative_without_correction_is_false(self):
        """CRITICAL RULE: Negative rating alone is NOT an eligible supervised training label."""
        record = FeedbackRecord(
            request_id="req-1",
            message="What time is it?",
            predicted_intent="Greeting",
            rating=FeedbackRating.NEGATIVE,
            corrected_intent=None,
            model_version="v0.1.0-baseline",
            prompt_version="v1.0.0"
        )
        assert record.training_eligibility is False
        assert "Ineligible" in record.eligibility_reason
        assert "drift" in record.eligibility_reason.lower()

    def test_training_eligibility_with_explicit_correction_is_true(self):
        """CRITICAL RULE: Verified ground-truth correction makes the sample eligible."""
        record = FeedbackRecord(
            request_id="req-2",
            message="What time is it?",
            predicted_intent="Greeting",
            rating=FeedbackRating.NEGATIVE,
            corrected_intent="TimeQuery",
            model_version="v0.1.0-baseline",
            prompt_version="v1.0.0"
        )
        assert record.training_eligibility is True
        assert "Eligible" in record.eligibility_reason
        assert "TimeQuery" in record.eligibility_reason


class TestMetadataContracts:
    def test_model_metadata_defaults_and_serialization(self):
        meta = ModelMetadata(
            model_version="v0.1.0-baseline",
            artifact_location="artifacts/models/baseline",
            dvc_revision="abc1234",
            evaluation_metrics={"macro_f1": 0.88, "accuracy": 0.92},
            promotion_status=PromotionStatus.PRODUCTION
        )
        json_data = meta.model_dump()
        assert json_data["model_version"] == "v0.1.0-baseline"
        assert json_data["num_classes"] == 22
        assert json_data["promotion_status"] == "production"

    def test_prompt_metadata_defaults_and_serialization(self):
        meta = PromptMetadata(
            prompt_version="v1.0.0",
            instruction_path="ChatBot/ADKBot/instruction.txt",
            evaluation_status=PromptEvaluationStatus.BASELINE
        )
        assert meta.associated_adk_model == "gemini-3-flash-preview"
        assert meta.evaluation_status == "baseline"

