"""Supplementary calibrated IF telemetry, using the schema-2 artifact contract."""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np

from symptom_ai.models.evaluation import SYMPTOM_FEATURES

CALIBRATION_REPORT = Path("reports/evaluation_v2/isolation_forest_benign_only/isolation_forest_benign_only.json")
CALIBRATED_MODEL_PATH = Path("models/unknown_behavior_detector_benign_only.joblib")
DEFAULT_TARGET_CALIBRATION_FPR = 0.05
NORMALIZATION = "fixed_nonnegative_saturation_v1"


def load_calibrated_shadow_artifacts():
    report = json.loads(CALIBRATION_REPORT.read_text(encoding="utf-8"))
    if report.get("feature_schema_version") != 2 or report.get("normalization") != NORMALIZATION:
        raise ValueError("Legacy shadow calibration requires re-ingestion and retraining with schema 2")
    digest = hashlib.sha256(CALIBRATED_MODEL_PATH.read_bytes()).hexdigest()
    if report.get("artifact_sha256") != digest:
        raise ValueError("Shadow artifact does not match its calibration report")
    bundle = joblib.load(CALIBRATED_MODEL_PATH)
    if not isinstance(bundle, dict):
        raise ValueError("Shadow artifact must be a schema-2 bundle")
    for key in ("feature_schema_version", "normalization", "feature_columns"):
        if bundle.get(key) != report.get(key):
            raise ValueError(f"Shadow artifact/report mismatch: {key}")
    features = report.get("feature_columns")
    if not features or len(features) != len(set(features)) or not set(features) <= set(SYMPTOM_FEATURES):
        raise ValueError("Invalid shadow observation features")
    model = bundle["model"]
    if list(model.feature_names_in_) != features:
        raise ValueError("Shadow model feature names mismatch")
    return model, report


def select_threshold(report, target_fpr):
    for row in report.get("threshold_results", []):
        if abs(float(row.get("target_calibration_fpr", -1)) - target_fpr) < 1e-12:
            return row
    raise ValueError(f"No calibrated threshold found for target FPR={target_fpr}")


def evaluate_unknown_detector_shadow(X, active_model, target_fpr=DEFAULT_TARGET_CALIBRATION_FPR):
    """The active IF is already benign-calibrated; shadow never changes policy."""
    result = {
        "mode": "benign_calibrated_active_with_calibrated_shadow",
        "active": {"decision_function_score": round(float(active_model.decision_function(X)[0]), 6),
                   "prediction": int(active_model.predict(X)[0])},
        "calibrated_shadow": {"status": "unavailable", "mode": "benign_only_calibrated_shadow",
                              "changes_unknown_risk": False},
    }
    try:
        model, report = load_calibrated_shadow_artifacts()
        features = report["feature_columns"]
        if set(X.columns) != set(features):
            raise ValueError("Shadow/active observation feature names differ; retrain matching artifacts")
        values = X.loc[:, features]
        if not np.isfinite(values.to_numpy()).all() or ((values < 0) | (values > 1)).any().any():
            raise ValueError("Shadow observations must be finite and in [0, 1]")
        row = select_threshold(report, target_fpr)
        threshold = float(row["threshold_anomaly_score"])
        score = float(-model.score_samples(values)[0])
        if not np.isfinite([threshold, score]).all():
            raise ValueError("Nonfinite shadow threshold/score")
        result["calibrated_shadow"].update({
            "status": "available", "target_calibration_fpr": target_fpr,
            "threshold_anomaly_score": round(threshold, 6), "anomaly_score": round(score, 6),
            "is_anomalous": score > threshold,
            "threshold_comparison": "anomaly_score > threshold_anomaly_score",
            "independent_benign_test_fpr": row.get("independent_benign_test_fpr"),
            "macro_family_unknown_detection_recall": row.get("macro_family_unknown_detection_recall"),
            "scope_note": "Supplementary shadow evidence; cannot independently trigger containment.",
        })
    except (OSError, ValueError, TypeError, KeyError, AttributeError, EOFError) as exc:
        result["calibrated_shadow"]["reason"] = str(exc)
    return result
