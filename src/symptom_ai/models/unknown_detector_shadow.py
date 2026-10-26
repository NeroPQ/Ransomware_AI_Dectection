from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import json
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd


MODEL_DIR = Path("models")
CALIBRATION_REPORT = Path(
    "reports/evaluation_v2/isolation_forest_benign_only/"
    "isolation_forest_benign_only.json"
)

CALIBRATED_MODEL_PATH = (
    MODEL_DIR / "unknown_behavior_detector_benign_only.joblib"
)

DEFAULT_TARGET_CALIBRATION_FPR = 0.05


@lru_cache(maxsize=1)
def load_calibrated_shadow_artifacts():
    if not CALIBRATED_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing calibrated IF artifact: {CALIBRATED_MODEL_PATH}"
        )

    if not CALIBRATION_REPORT.exists():
        raise FileNotFoundError(
            f"Missing calibration report: {CALIBRATION_REPORT}"
        )

    model = joblib.load(CALIBRATED_MODEL_PATH)

    report = json.loads(
        CALIBRATION_REPORT.read_text(encoding="utf-8")
    )

    return model, report


def select_threshold(report: Dict[str, Any], target_fpr: float):
    for row in report.get("threshold_results", []):
        value = float(row.get("target_calibration_fpr", -1))

        if abs(value - target_fpr) < 1e-12:
            return row

    raise ValueError(
        f"No calibrated threshold found for target FPR={target_fpr}"
    )


def evaluate_unknown_detector_shadow(
    X: pd.DataFrame,
    legacy_model,
    target_fpr: float = DEFAULT_TARGET_CALIBRATION_FPR,
) -> Dict[str, Any]:
    """
    Compare legacy Isolation Forest with benign-only calibrated IF.

    Important:
    - legacy result remains the live decision source
    - calibrated result is shadow evidence only
    - this helper never changes unknown_risk itself
    """
    legacy_raw_score = float(legacy_model.decision_function(X)[0])
    legacy_prediction = int(legacy_model.predict(X)[0])

    result = {
        "mode": "legacy_with_calibrated_shadow",
        "legacy": {
            "decision_function_score": round(legacy_raw_score, 6),
            "prediction": legacy_prediction,
            "is_anomalous": legacy_prediction == -1,
        },
        "calibrated_shadow": {
            "status": "unavailable",
            "mode": "benign_only_calibrated_shadow",
            "changes_unknown_risk": False,
        },
    }

    try:
        calibrated_model, report = load_calibrated_shadow_artifacts()

        expected_features = getattr(
            calibrated_model,
            "n_features_in_",
            None,
        )

        calibrated_X = X.to_numpy(dtype=np.float32)

        if (
            expected_features is not None
            and calibrated_X.shape[1] != int(expected_features)
        ):
            raise ValueError(
                "Calibrated model feature mismatch: "
                f"expected {expected_features}, got {calibrated_X.shape[1]}"
            )

        threshold_row = select_threshold(report, target_fpr)

        anomaly_score = float(
            -calibrated_model.score_samples(calibrated_X)[0]
        )

        threshold = float(
            threshold_row["threshold_anomaly_score"]
        )

        result["calibrated_shadow"] = {
            "status": "available",
            "mode": "benign_only_calibrated_shadow",
            "target_calibration_fpr": target_fpr,
            "threshold_anomaly_score": round(threshold, 6),
            "anomaly_score": round(anomaly_score, 6),
            "is_anomalous": anomaly_score >= threshold,
            "independent_benign_test_fpr": threshold_row.get(
                "independent_benign_test_fpr"
            ),
            "macro_family_unknown_detection_recall": (
                threshold_row.get(
                    "macro_family_unknown_detection_recall"
                )
            ),
            "changes_unknown_risk": False,
            "scope_note": (
                "Supplementary shadow evidence only. "
                "It does not independently trigger containment."
            ),
        }

    except Exception as exc:
        result["calibrated_shadow"] = {
            "status": "unavailable",
            "mode": "benign_only_calibrated_shadow",
            "changes_unknown_risk": False,
            "reason": str(exc),
        }

    return result
