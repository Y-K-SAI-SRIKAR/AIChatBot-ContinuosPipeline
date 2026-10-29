"""
Unit tests for Evidently evaluation and drift monitoring module.

Covers:
- Empty current dataset handling (no crash, status=skipped)
- Insufficient data (<2 samples, status=insufficient_data)
- Missing labels and verified-only separation
- Successful HTML and JSON metadata report generation
- Retraining recommendation logic
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import pandas as pd

from src.evaluation.drift_monitor import EvidentlyDriftMonitor
from common.contracts import FeedbackRating, FeedbackRecord


@pytest.fixture
def sample_reference_df():
    df = pd.DataFrame([
        {"text": "hello there", "intent": "Greeting"},
        {"text": "good morning", "intent": "Greeting"},
        {"text": "what time is it", "intent": "TimeQuery"},
        {"text": "tell me current time", "intent": "TimeQuery"},
        {"text": "thanks a lot", "intent": "CourtesyGreeting"},
        {"text": "thank you very much", "intent": "CourtesyGreeting"},
    ])
    return EvidentlyDriftMonitor.extract_features(df)


@pytest.fixture
def mock_feedback_records():
    return [
        FeedbackRecord(
            request_id="req-1",
            message="hey buddy",
            predicted_intent="Greeting",
            rating=FeedbackRating.POSITIVE,
            corrected_intent="Greeting",
            model_version="v0.1.0-baseline",
            prompt_version="v1.0.0",
        ),
        FeedbackRecord(
            request_id="req-2",
            message="what is the time now",
            predicted_intent="Greeting",
            rating=FeedbackRating.NEGATIVE,
            corrected_intent="TimeQuery",
            model_version="v0.1.0-baseline",
            prompt_version="v1.0.0",
        ),
        FeedbackRecord(
            request_id="req-3",
            message="i hate this answer",
            predicted_intent="Greeting",
            rating=FeedbackRating.NEGATIVE,
            corrected_intent=None,  # Ineligible for supervised training
            model_version="v0.1.0-baseline",
            prompt_version="v1.0.0",
        ),
    ]


class TestEvidentlyDriftMonitor:

    def test_feature_extraction(self):
        df = pd.DataFrame({"text": ["Hello world!", "Hi"]})
        feat_df = EvidentlyDriftMonitor.extract_features(df)
        assert "char_count" in feat_df.columns
        assert "word_count" in feat_df.columns
        assert "avg_word_length" in feat_df.columns
        assert feat_df.iloc[0]["word_count"] == 2
        assert feat_df.iloc[1]["word_count"] == 1

    def test_empty_feedback_window(self, tmp_path, sample_reference_df):
        monitor = EvidentlyDriftMonitor()
        empty_cur_df = pd.DataFrame()

        res = monitor.run_drift_analysis(
            reference_df=sample_reference_df,
            current_df=empty_cur_df,
            output_dir=tmp_path,
        )

        assert res["status"] == "skipped"
        assert res["drift_detected"] is False
        assert res["retraining_recommended"] is False
        assert (tmp_path / "drift_report_metadata.json").exists()

    def test_insufficient_feedback_sample(self, tmp_path, sample_reference_df):
        monitor = EvidentlyDriftMonitor()
        single_df = pd.DataFrame([{"text": "just one sample", "intent": "Greeting"}])
        single_df = EvidentlyDriftMonitor.extract_features(single_df)

        res = monitor.run_drift_analysis(
            reference_df=sample_reference_df,
            current_df=single_df,
            output_dir=tmp_path,
        )

        assert res["status"] == "insufficient_data"
        assert res["sample_count"] == 1
        assert res["drift_detected"] is False
        assert res["retraining_recommended"] is False

    def test_verified_only_label_separation(self, mock_feedback_records):
        monitor = EvidentlyDriftMonitor()
        df, stats = monitor.prepare_feedback_dataframe(mock_feedback_records, require_verified_only=True)

        # Only 2 out of 3 records have corrected_intent
        assert stats["eligible_records"] == 2
        assert stats["ineligible_records"] == 1
        assert len(df) == 2
        assert all(df["is_verified"] == True)
        assert "Greeting" in df["intent"].values
        assert "TimeQuery" in df["intent"].values

    def test_report_generation_with_valid_samples(self, tmp_path, sample_reference_df, mock_feedback_records):
        monitor = EvidentlyDriftMonitor()
        cur_df, stats = monitor.prepare_feedback_dataframe(mock_feedback_records, require_verified_only=False)

        res = monitor.run_drift_analysis(
            reference_df=sample_reference_df,
            current_df=cur_df,
            output_dir=tmp_path,
            reference_version="v0.1.0-baseline",
            current_version="test-cur-v1",
        )

        assert res["status"] == "success"
        assert (tmp_path / "drift_report.html").exists()
        assert (tmp_path / "drift_report.json").exists()
        assert (tmp_path / "drift_report_metadata.json").exists()
        assert "drift_share" in res
        assert "retraining_recommended" in res

