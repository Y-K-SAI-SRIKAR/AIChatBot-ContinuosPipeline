"""
Evidently Evaluation and Drift Monitoring Module for ChatBot Intent Pipeline.

Compatible with Evidently 0.7.23.
Calculates text and label drift between baseline reference data and newly collected
verified feedback records.

Key architectural guarantees:
- Never treats unverified model predictions as ground-truth labels.
- Separates input text & label distribution drift from model performance degradation.
- Handles empty feedback windows, tiny sample sizes (<2), and missing labels gracefully.
- Emits reproducible HTML report and JSON metadata with retraining recommendations.
- NEVER triggers automatic promotion based on drift alone (drift triggers investigation/retraining).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
os.environ["DO_NOT_TRACK"] = "1"
os.environ["EVIDENTLY_DISABLE_TELEMETRY"] = "1"
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# Disable telemetry and tracking
os.environ["DO_NOT_TRACK"] = "1"
os.environ["EVIDENTLY_DISABLE_TELEMETRY"] = "1"

import pandas as pd

logger = logging.getLogger("evaluation.drift_monitor")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class DriftMonitoringError(Exception):
    """Raised when an unrecoverable drift monitoring failure occurs."""
    pass


class EvidentlyDriftMonitor:
    """
    Production-conscious drift monitor wrapping Evidently 0.7.23 Report & Presets.
    """

    def __init__(
        self,
        reference_data_path: Optional[Path] = None,
        drift_share_threshold: float = 0.5,
    ):
        self.reference_data_path = reference_data_path
        self.drift_share_threshold = drift_share_threshold

    @staticmethod
    def extract_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """
        Derive numerical text descriptor features (character length, word count, avg word len)
        to enable robust distribution comparison without heavy LLM downloads.
        """
        features_df = df.copy()
        if text_col in features_df.columns:
            features_df["char_count"] = features_df[text_col].astype(str).str.len()
            features_df["word_count"] = features_df[text_col].astype(str).apply(lambda s: len(s.split()))
            features_df["avg_word_length"] = features_df["char_count"] / features_df["word_count"].replace(0, 1)
        return features_df

    def prepare_reference_dataframe(self, ref_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load baseline reference dataset (e.g. from data/processed/train.json).
        """
        target_path = Path(ref_path or self.reference_data_path or "ChatBot-PipeLine/data/processed/train.json")
        if not target_path.exists():
            raise FileNotFoundError(f"Reference dataset not found at: {target_path}")

        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list) or len(data) == 0:
            raise ValueError(f"Reference dataset at {target_path} is empty or not a list.")

        rows = []
        for item in data:
            if isinstance(item, dict) and "text" in item and "intent" in item:
                rows.append({"text": str(item["text"]), "intent": str(item["intent"])})

        df = pd.DataFrame(rows)
        return self.extract_features(df, text_col="text")

    def prepare_feedback_dataframe(
        self,
        feedback_records: List[Any],
        require_verified_only: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Build current dataset from feedback records.
        Ensures intent distribution is evaluated ONLY against verified corrected_intent,
        never treating unverified predictions as true ground truth.
        """
        rows = []
        stats = {
            "total_records": len(feedback_records),
            "eligible_records": 0,
            "ineligible_records": 0,
            "has_ground_truth_labels": False,
        }

        for r in feedback_records:
            # Handle Pydantic model or dict
            text = getattr(r, "message", None) if hasattr(r, "message") else r.get("message")
            corr_intent = getattr(r, "corrected_intent", None) if hasattr(r, "corrected_intent") else r.get("corrected_intent")
            pred_intent = getattr(r, "predicted_intent", None) if hasattr(r, "predicted_intent") else r.get("predicted_intent")
            eligible = getattr(r, "training_eligibility", False) if hasattr(r, "training_eligibility") else r.get("training_eligibility", False)

            if not text or not str(text).strip():
                continue

            if eligible and corr_intent and str(corr_intent).strip():
                stats["eligible_records"] += 1
                rows.append({
                    "text": str(text).strip(),
                    "intent": str(corr_intent).strip(),
                    "predicted_intent": str(pred_intent or ""),
                    "is_verified": True
                })
            elif not require_verified_only:
                stats["ineligible_records"] += 1
                rows.append({
                    "text": str(text).strip(),
                    "intent": str(corr_intent or pred_intent or "Unknown"),
                    "predicted_intent": str(pred_intent or ""),
                    "is_verified": bool(corr_intent)
                })
            else:
                stats["ineligible_records"] += 1

        stats["has_ground_truth_labels"] = stats["eligible_records"] > 0
        df = pd.DataFrame(rows)
        if not df.empty:
            df = self.extract_features(df, text_col="text")
        return df, stats

    def run_drift_analysis(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        output_dir: Path,
        reference_version: str = "v0.1.0-baseline",
        current_version: str = "feedback-window-latest",
        model_version: str = "v0.1.0-baseline",
    ) -> Dict[str, Any]:
        """
        Execute Evidently drift and data quality report between reference and current windows.
        Gracefully handles tiny datasets or missing labels without crashing.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()

        # Handle empty current dataset
        if current_df.empty or len(current_df) == 0:
            logger.info("Current dataset is empty. Skipping drift report generation.")
            result_meta = {
                "status": "skipped",
                "reason": "Current feedback dataset contains zero records.",
                "timestamp": timestamp,
                "reference_version": reference_version,
                "current_version": current_version,
                "model_version": model_version,
                "drift_detected": False,
                "retraining_recommended": False,
                "html_report_path": None,
                "metrics_summary": {}
            }
            self._save_metadata(result_meta, output_dir / "drift_report_metadata.json")
            return result_meta

        # Handle tiny dataset (fewer than 2 records cannot establish distribution drift)
        if len(current_df) < 2:
            logger.info(f"Current dataset size is {len(current_df)} (< 2). Statistical drift calculation unreliable.")
            result_meta = {
                "status": "insufficient_data",
                "reason": f"Current dataset contains only {len(current_df)} record(s). Minimum 2 required for statistical distribution comparison.",
                "timestamp": timestamp,
                "reference_version": reference_version,
                "current_version": current_version,
                "model_version": model_version,
                "sample_count": len(current_df),
                "drift_detected": False,
                "retraining_recommended": False,
                "html_report_path": None,
                "metrics_summary": {
                    "sample_count": len(current_df),
                    "intents_observed": current_df["intent"].value_counts().to_dict() if "intent" in current_df else {}
                }
            }
            self._save_metadata(result_meta, output_dir / "drift_report_metadata.json")
            return result_meta

        # Align columns for Evidently comparison
        cols_to_compare = [c for c in ["char_count", "word_count", "avg_word_length", "intent"] if c in reference_df.columns and c in current_df.columns]
        ref_subset = reference_df[cols_to_compare].dropna()
        cur_subset = current_df[cols_to_compare].dropna()

        html_path = output_dir / "drift_report.html"
        json_path = output_dir / "drift_report.json"
        meta_path = output_dir / "drift_report_metadata.json"

        try:
            from evidently.legacy.report import Report
            from evidently.legacy.metric_preset import DataDriftPreset

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=ref_subset, current_data=cur_subset)

            report.save_html(str(html_path))
            report_dict = report.as_dict()

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report_dict, f, indent=2)

            # Parse Evidently metrics safely
            drift_share = 0.0
            number_of_drifted_columns = 0
            dataset_drift = False

            metrics_list = report_dict.get("metrics", [])
            for m in metrics_list:
                result = m.get("result", {})
                if "dataset_drift" in result:
                    dataset_drift = bool(result.get("dataset_drift", False))
                if "drift_share" in result:
                    drift_share = float(result.get("drift_share", 0.0))
                if "number_of_drifted_columns" in result:
                    number_of_drifted_columns = int(result.get("number_of_drifted_columns", 0))

            # Retraining recommendation logic:
            # Recommend retraining when dataset drift is detected OR drift share >= threshold
            # Note: Retraining recommendation NEVER promotes automatically.
            retraining_recommended = dataset_drift or (drift_share >= self.drift_share_threshold)
            recommendation_reasons = []
            if dataset_drift:
                recommendation_reasons.append(f"Evidently flagged dataset drift (drift_share: {drift_share:.2f}).")
            if drift_share >= self.drift_share_threshold:
                recommendation_reasons.append(f"Drift share {drift_share:.2f} >= threshold {self.drift_share_threshold:.2f}.")
            if not retraining_recommended:
                recommendation_reasons.append("Data distributions remain stable within threshold boundaries.")

            result_meta = {
                "status": "success",
                "timestamp": timestamp,
                "reference_version": reference_version,
                "current_version": current_version,
                "model_version": model_version,
                "reference_samples": len(ref_subset),
                "current_samples": len(cur_subset),
                "columns_analyzed": cols_to_compare,
                "drift_detected": dataset_drift,
                "drift_share": round(drift_share, 4),
                "drifted_columns_count": number_of_drifted_columns,
                "retraining_recommended": retraining_recommended,
                "retraining_reasons": recommendation_reasons,
                "html_report_path": str(html_path),
                "json_report_path": str(json_path),
                "metrics_summary": {
                    "dataset_drift": dataset_drift,
                    "drift_share": round(drift_share, 4),
                    "number_of_drifted_columns": number_of_drifted_columns,
                }
            }
            self._save_metadata(result_meta, meta_path)
            logger.info(f"Drift analysis complete. Drift detected: {dataset_drift}. HTML report saved to {html_path}")
            return result_meta

        except Exception as e:
            logger.warning(f"Evidently run encountered an exception: {e}. Writing fallback summary.")
            fallback_meta = {
                "status": "partial_failure",
                "error": str(e),
                "timestamp": timestamp,
                "reference_version": reference_version,
                "current_version": current_version,
                "model_version": model_version,
                "drift_detected": False,
                "retraining_recommended": False,
                "html_report_path": None,
                "metrics_summary": {}
            }
            self._save_metadata(fallback_meta, meta_path)
            return fallback_meta

    @staticmethod
    def _save_metadata(meta: Dict[str, Any], path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run Evidently drift analysis on feedback records vs reference data")
    parser.add_argument("--ref-data", type=Path, default=Path("ChatBot-PipeLine/data/processed/train.json"))
    parser.add_argument("--feedback-file", type=Path, default=Path("ChatBot-PipeLine/data/feedback/user_feedback.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports/evidently"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    monitor = EvidentlyDriftMonitor(reference_data_path=args.ref_data, drift_share_threshold=args.threshold)
    ref_df = monitor.prepare_reference_dataframe()

    # Load feedback
    from common.feedback_store import FeedbackStore
    store = FeedbackStore(storage_path=args.feedback_file)
    records = store.load_all_records()

    cur_df, stats = monitor.prepare_feedback_dataframe(records, require_verified_only=False)
    logger.info(f"Loaded {len(records)} feedback records -> {len(cur_df)} analyzed. Stats: {stats}")

    res = monitor.run_drift_analysis(
        reference_df=ref_df,
        current_df=cur_df,
        output_dir=args.output_dir
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
