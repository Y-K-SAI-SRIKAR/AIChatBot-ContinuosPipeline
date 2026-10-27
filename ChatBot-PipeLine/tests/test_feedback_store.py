"""
Unit tests for the FeedbackStore local JSONL persistence.
"""

from pathlib import Path
import pytest

from common.contracts import FeedbackRating, FeedbackRecord
from common.feedback_store import FeedbackStore


class TestFeedbackStore:
    def test_save_and_load_records(self, tmp_path):
        test_file = tmp_path / "test_feedback.jsonl"
        store = FeedbackStore(storage_path=test_file)

        # Record 1: Negative rating without correction (Ineligible)
        r1 = FeedbackRecord(
            request_id="req-1",
            message="Show me the menu",
            predicted_intent="Greeting",
            rating=FeedbackRating.NEGATIVE,
            corrected_intent=None,
            model_version="v0.1.0-baseline",
            prompt_version="v1.0.0"
        )
        # Record 2: Negative rating with explicit correction (Eligible)
        r2 = FeedbackRecord(
            request_id="req-2",
            message="Bye see you later",
            predicted_intent="Swearing",
            rating=FeedbackRating.NEGATIVE,
            corrected_intent="GoodBye",
            model_version="v0.1.0-baseline",
            prompt_version="v1.0.0"
        )
        # Record 3: Positive rating (Ineligible for intent relabeling)
        r3 = FeedbackRecord(
            request_id="req-3",
            message="Thanks a lot",
            predicted_intent="Thanks",
            rating=FeedbackRating.POSITIVE,
            model_version="v0.1.0-baseline",
            prompt_version="v1.0.0"
        )

        store.save_feedback(r1)
        store.save_feedback(r2)
        store.save_feedback(r3)

        all_records = store.load_all_records()
        assert len(all_records) == 3

        # Verify filtering rule
        eligible = store.get_training_eligible_records()
        assert len(eligible) == 1
        assert eligible[0].request_id == "req-2"
        assert eligible[0].corrected_intent == "GoodBye"
        assert eligible[0].training_eligibility is True

        drift_records = store.get_drift_monitoring_records()
        assert len(drift_records) == 2
        assert {r.request_id for r in drift_records} == {"req-1", "req-3"}

