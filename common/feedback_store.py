"""
Local Persistence Store for Feedback Records.

Uses JSON Lines (JSONL) append-only storage for maximum reliability,
crash resistance, and zero external database dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import List, Optional

from common.contracts import FeedbackRating, FeedbackRecord


class FeedbackStore:
    """
    Append-only thread-safe local feedback storage.
    
    Persists feedback records to 'data/feedback/user_feedback.jsonl'.
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
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.touch()

    def save_feedback(self, record: FeedbackRecord) -> FeedbackRecord:
        """Append a feedback record safely to JSONL storage."""
        with self._lock:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
        return record

    def load_all_records(self) -> List[FeedbackRecord]:
        """Load all persisted feedback records."""
        records: List[FeedbackRecord] = []
        if not self.file_path.exists():
            return records

        with self._lock:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        data = json.loads(line_str)
                        records.append(FeedbackRecord.model_validate(data))
                    except Exception as e:
                        # Log and skip corrupted line rather than failing the whole file
                        continue
        return records

    def get_training_eligible_records(self) -> List[FeedbackRecord]:
        """
        Filter only verified feedback eligible for supervised intent retraining.
        
        Enforces rule: Negative feedback alone is NOT eligible.
        Only records with explicit corrected_intent are returned.
        """
        return [r for r in self.load_all_records() if r.training_eligibility]

    def get_drift_monitoring_records(self) -> List[FeedbackRecord]:
        """Retrieve feedback records relevant for drift and failure mode inspection."""
        return [r for r in self.load_all_records() if not r.training_eligibility]

    def clear(self) -> None:
        """Clear storage (used predominantly in unit tests)."""
        with self._lock:
            if self.file_path.exists():
                self.file_path.write_text("", encoding="utf-8")

