"""
Local Persistence Store for Feedback Records.

Uses JSON Lines (JSONL) append-only storage for maximum reliability,
crash resistance, and zero external database dependencies.

Key Features & Guarantees:
- Thread-safe append writes using threading.Lock.
- Idempotency / duplicate submission detection based on request_id.
- Corrupted / partial line resilience (skips malformed lines during read).
- Strict separation of training-eligible samples vs. drift-monitoring records.
- Aggregate stats reporting without exposing raw records publicly.

Documented Limitations:
- File locking is process-local via threading.Lock (not distributed across multi-node clusters).
- Storage capacity is bounded by local disk space.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Set

from common.contracts import FeedbackRating, FeedbackRecord

logger = logging.getLogger("common.feedback_store")


class DuplicateFeedbackError(ValueError):
    """Raised when feedback for a given request_id has already been recorded."""
    pass


class PersistenceError(IOError):
    """Raised when writing feedback to disk fails."""
    pass


class FeedbackStore:
    """
    Append-only thread-safe local feedback storage.
    
    Persists feedback records to 'ChatBot-PipeLine/data/feedback/user_feedback.jsonl'.
    Allows filtering by training eligibility (verified corrections only)
    versus quality/drift monitoring samples.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path:
            self.file_path = Path(storage_path)
        else:
            repo_root = Path(__file__).resolve().parent.parent
            self.file_path = repo_root / "ChatBot-PipeLine" / "data" / "feedback" / "user_feedback.jsonl"
        
        self._lock = Lock()
        self._known_request_ids: Dict[str, FeedbackRecord] = {}
        self._ensure_storage()
        self._index_existing_records()

    def _ensure_storage(self) -> None:
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.file_path.exists():
                self.file_path.touch()
        except OSError as e:
            raise PersistenceError(f"Cannot initialize feedback storage directory {self.file_path.parent}: {e}") from e

    def _index_existing_records(self) -> None:
        """Index existing request IDs from disk to support idempotency checks."""
        self._known_request_ids.clear()
        if not self.file_path.exists():
            return

        with self._lock:
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line_str = line.strip()
                        if not line_str:
                            continue
                        try:
                            data = json.loads(line_str)
                            rec = FeedbackRecord.model_validate(data)
                            self._known_request_ids[rec.request_id] = rec
                        except Exception:
                            continue
            except OSError:
                pass

    def save_feedback(self, record: FeedbackRecord, allow_duplicate: bool = False) -> FeedbackRecord:
        """
        Append a feedback record safely to JSONL storage with duplicate check.
        
        Raises DuplicateFeedbackError if a record with the same request_id exists
        and allow_duplicate is False.
        """
        with self._lock:
            if not allow_duplicate and record.request_id in self._known_request_ids:
                raise DuplicateFeedbackError(
                    f"Feedback for request_id '{record.request_id}' has already been recorded."
                )

            try:
                line_data = record.model_dump_json() + "\n"
                with open(self.file_path, "a", encoding="utf-8") as f:
                    f.write(line_data)
                    f.flush()
                self._known_request_ids[record.request_id] = record
            except OSError as e:
                raise PersistenceError(f"Failed to append feedback record to {self.file_path}: {e}") from e

        return record

    def has_feedback_for_request(self, request_id: str) -> bool:
        """Check if feedback for request_id has already been recorded."""
        with self._lock:
            return request_id in self._known_request_ids

    def load_all_records(self) -> List[FeedbackRecord]:
        """Load all persisted feedback records, skipping corrupted lines."""
        records: List[FeedbackRecord] = []
        if not self.file_path.exists():
            return records

        with self._lock:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        data = json.loads(line_str)
                        records.append(FeedbackRecord.model_validate(data))
                    except Exception as e:
                        # Skip corrupted or partial lines without failing the entire batch
                        logger.warning(f"Skipping corrupt feedback line: {e}")
                        continue
        return records

    def get_training_eligible_records(self) -> List[FeedbackRecord]:
        """
        Filter only verified feedback eligible for supervised intent retraining.
        
        Enforces rule: Thumbs-up/down alone is NOT eligible.
        Only records with explicit, verified corrected_intent are returned.
        """
        return [r for r in self.load_all_records() if r.training_eligibility]

    def get_drift_monitoring_records(self) -> List[FeedbackRecord]:
        """Retrieve feedback records relevant for drift and failure mode inspection."""
        return [r for r in self.load_all_records() if not r.training_eligibility]

    def get_stats(self) -> Dict[str, int]:
        """
        Return aggregate counts without exposing raw user records or PII publicly.
        """
        records = self.load_all_records()
        total = len(records)
        pos = sum(1 for r in records if r.rating == FeedbackRating.POSITIVE)
        neg = sum(1 for r in records if r.rating == FeedbackRating.NEGATIVE)
        eligible = sum(1 for r in records if r.training_eligibility)
        ineligible = total - eligible

        return {
            "total_feedback": total,
            "positive_ratings": pos,
            "negative_ratings": neg,
            "training_eligible": eligible,
            "drift_monitoring": ineligible,
        }

    def clear(self) -> None:
        """Clear storage and index (used predominantly in test teardowns)."""
        with self._lock:
            if self.file_path.exists():
                self.file_path.write_text("", encoding="utf-8")
            self._known_request_ids.clear()
