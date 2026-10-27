"""
Unit tests for the FeedbackStore local JSONL persistence.
Covers:
- Valid submissions & record loading
- Training eligibility separation (thumbs-up/down vs. verified corrections)
- Duplicate submission prevention (idempotency check)
- Corrupted / partial line resilience
- Persistence failure handling
- Aggregate stats reporting
"""

import json
from pathlib import Path
import pytest

from common.contracts import FeedbackRating, FeedbackRecord
from common.feedback_store import (
    FeedbackStore,
    DuplicateFeedbackError,
    PersistenceError,
)


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

    def test_duplicate_submission_raises_error(self, tmp_path):
        test_file = tmp_path / "test_duplicates.jsonl"
        store = FeedbackStore(storage_path=test_file)

        r1 = FeedbackRecord(
            request_id="req-dup-1",
            message="Hello",
            predicted_intent="Greeting",
            rating=FeedbackRating.POSITIVE,
            model_version="v0.1.0-baseline",
            prompt_version="v1.0.0",
        )
        store.save_feedback(r1)
        assert store.has_feedback_for_request("req-dup-1") is True

        # Second submission with same request_id should be rejected
        with pytest.raises(DuplicateFeedbackError, match="already been recorded"):
            store.save_feedback(r1)

    def test_skips_corrupted_lines_gracefully(self, tmp_path):
        test_file = tmp_path / "corrupted_feedback.jsonl"
        valid_rec = FeedbackRecord(
            request_id="req-good",
            message="Valid query",
            rating=FeedbackRating.POSITIVE,
            model_version="v0.1.0-baseline",
            prompt_version="v1.0.0",
        )
        # Write 1 valid record, 1 corrupt line, 1 valid record
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(valid_rec.model_dump_json() + "\n")
            f.write("CORRUPTED_NON_JSON_DATA_HERE\n")
            f.write('{"incomplete": \n')
            valid_rec2 = valid_rec.model_copy(update={"request_id": "req-good-2"})
            f.write(valid_rec2.model_dump_json() + "\n")

        store = FeedbackStore(storage_path=test_file)
        loaded = store.load_all_records()
        assert len(loaded) == 2
        assert {r.request_id for r in loaded} == {"req-good", "req-good-2"}

    def test_persistence_failure_handling(self, tmp_path):
        # Create a directory where the file path is expected to be a file
        read_only_dir = tmp_path / "protected_target"
        read_only_dir.mkdir(parents=True, exist_ok=True)
        # Point storage_path to a directory instead of a file so append writing fails with IsADirectoryError / PermissionError
        store = FeedbackStore(storage_path=tmp_path / "test.jsonl")
        # Artificially point store.file_path to the directory to simulate an OS write failure
        store.file_path = read_only_dir

        record = FeedbackRecord(
            request_id="req-fail-test",
            message="Fail test",
            rating=FeedbackRating.POSITIVE,
            model_version="v0.1.0-baseline",
            prompt_version="v1.0.0",
        )
        with pytest.raises(PersistenceError, match="Failed to append feedback record"):
            store.save_feedback(record)

    def test_get_stats_aggregates_properly(self, tmp_path):
        test_file = tmp_path / "stats_test.jsonl"
        store = FeedbackStore(storage_path=test_file)

        store.save_feedback(
            FeedbackRecord(
                request_id="s1",
                message="Hi",
                rating=FeedbackRating.POSITIVE,
                model_version="v0.1.0-baseline",
                prompt_version="v1.0.0",
            )
        )
        store.save_feedback(
            FeedbackRecord(
                request_id="s2",
                message="Bad",
                rating=FeedbackRating.NEGATIVE,
                model_version="v0.1.0-baseline",
                prompt_version="v1.0.0",
            )
        )
        store.save_feedback(
            FeedbackRecord(
                request_id="s3",
                message="Time",
                rating=FeedbackRating.NEGATIVE,
                corrected_intent="TimeQuery",
                model_version="v0.1.0-baseline",
                prompt_version="v1.0.0",
            )
        )

        stats = store.get_stats()
        assert stats["total_feedback"] == 3
        assert stats["positive_ratings"] == 1
        assert stats["negative_ratings"] == 2
        assert stats["training_eligible"] == 1
        assert stats["drift_monitoring"] == 2
