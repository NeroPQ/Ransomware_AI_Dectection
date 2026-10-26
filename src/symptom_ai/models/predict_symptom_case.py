from pathlib import Path
import json
import joblib
import pandas as pd

from symptom_ai.models.risk_scorer import calculate_symptom_risk
from symptom_ai.models.unknown_detector_shadow import (
    evaluate_unknown_detector_shadow,
)
from symptom_ai.response_engine.response_policy import recommend_response


MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/symptom_ai")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_models():
    clf = joblib.load(MODEL_DIR / "ransomware_symptom_classifier.joblib")
    iso = joblib.load(MODEL_DIR / "unknown_behavior_detector.joblib")

    with open(MODEL_DIR / "model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return clf, iso, metadata


def build_known_ransomware_like_case():
    return {
        "file_write_burst": 0.95,
        "file_rename_burst": 0.90,
        "high_entropy_write": 0.92,
        "mass_file_modification": 0.87,
        "suspicious_extension_change": 0.83,
        "suspicious_process_spawn": 0.75,
        "process_tree_anomaly": 0.65,
        "registry_run_key_modified": 0.35,
        "persistence_attempt": 0.25,
        "c2_beaconing": 0.20,
        "data_exfiltration_pattern": 0.10
    }


def build_unknown_high_risk_case():
    return {
        "file_write_burst": 0.82,
        "file_rename_burst": 0.25,
        "high_entropy_write": 0.91,
        "mass_file_modification": 0.60,
        "suspicious_extension_change": 0.35,
        "backup_disable_attempt": 0.88,
        "shadow_copy_delete_attempt": 0.82,
        "security_tool_tamper": 0.80,
        "c2_beaconing": 0.76,
        "data_exfiltration_pattern": 0.74
    }


def predict_case(case_name: str, case: dict):
    clf, iso, metadata = load_models()
    feature_cols = metadata["feature_columns"]

    row = {col: float(case.get(col, 0.0)) for col in feature_cols}
    X = pd.DataFrame([row], columns=feature_cols)

    pred = clf.predict(X)[0]

    probabilities = {}
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        for cls, prob in zip(clf.classes_, proba):
            probabilities[str(cls)] = round(float(prob), 4)

    iso_raw = float(iso.decision_function(X)[0])
    iso_pred = int(iso.predict(X)[0])

    detector_shadow = evaluate_unknown_detector_shadow(
        X,
        iso,
        target_fpr=0.05,
    )

    unknown_risk = "high" if iso_pred == -1 else "low"

    if (
        case.get("backup_disable_attempt", 0) >= 0.7
        or case.get("shadow_copy_delete_attempt", 0) >= 0.7
        or case.get("security_tool_tamper", 0) >= 0.7
    ):
        unknown_risk = "high"

    risk_score = calculate_symptom_risk(case)

    response = recommend_response(
        symptoms=case,
        unknown_risk=unknown_risk
    )

    return {
        "case_name": case_name,
        "input_symptoms": case,
        "predicted_label": pred,
        "prediction_probabilities": probabilities,
        "risk_score": risk_score,
        "isolation_forest_raw_score": round(iso_raw, 4),
        "isolation_forest_prediction": iso_pred,
        "unknown_detector_mode": detector_shadow["mode"],
        "unknown_detector_shadow": detector_shadow["calibrated_shadow"],
        "unknown_risk": unknown_risk,
        "response": response
    }


def main():
    results = [
        predict_case("known_ransomware_like_demo", build_known_ransomware_like_case()),
        predict_case("unknown_high_risk_demo", build_unknown_high_risk_case()),
    ]

    out = REPORT_DIR / "prediction_demo_result.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
