from pathlib import Path
import json
from typing import Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from symptom_ai.models.risk_scorer import calculate_symptom_risk
from symptom_ai.models.unknown_detector_shadow import (
    evaluate_unknown_detector_shadow,
)
from symptom_ai.response_engine.response_policy import recommend_response
from symptom_ai.response_engine.protective_lockdown import (
    run_protective_lockdown,
    unlock_protected_files_demo,
    ensure_demo_protected_docs,
)


MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/symptom_ai")
UNKNOWN_CASE_DIR = Path("data/unknown_cases")
BACKUP_DIR = Path("data/emergency_backup")
PROTECTED_DIR = Path("data/protected_docs")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
UNKNOWN_CASE_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)
PROTECTED_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="AI Ransomware Symptom Learning and Response API",
    version="0.3.0"
)


class SymptomRequest(BaseModel):
    symptoms: Dict[str, float]


def load_artifacts():
    with open(MODEL_DIR / "model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    primary = metadata.get("primary_classifier", "ransomware_symptom_classifier.joblib")

    clf = joblib.load(MODEL_DIR / primary)
    iso = joblib.load(MODEL_DIR / "unknown_behavior_detector.joblib")

    return clf, iso, metadata


def normalize_prediction(pred, metadata):
    labels = metadata.get("labels", [])
    id_to_label = metadata.get("id_to_label", {})

    value = pred

    try:
        if hasattr(value, "item"):
            value = value.item()
    except Exception:
        pass

    s = str(value)

    if s in labels:
        return s

    if s in id_to_label:
        return id_to_label[s]

    try:
        i = int(value)
        if str(i) in id_to_label:
            return id_to_label[str(i)]
    except Exception:
        pass

    return s


def normalize_probabilities(clf, proba, metadata):
    labels = metadata.get("labels", [])
    id_to_label = metadata.get("id_to_label", {})

    result = {}

    if hasattr(clf, "classes_"):
        classes = list(clf.classes_)
    else:
        classes = list(range(len(proba)))

    for cls, prob in zip(classes, proba):
        label = normalize_prediction(cls, metadata)
        result[str(label)] = round(float(prob), 4)

    # If XGBoost only returns one probability column in some binary cases, guard fallback.
    if not result and len(labels) == len(proba):
        for label, prob in zip(labels, proba):
            result[label] = round(float(prob), 4)

    return result


def infer(req: SymptomRequest) -> Dict[str, Any]:
    clf, iso, metadata = load_artifacts()
    feature_cols = metadata["feature_columns"]

    row = {col: float(req.symptoms.get(col, 0.0)) for col in feature_cols}
    X = pd.DataFrame([row], columns=feature_cols)

    raw_pred = clf.predict(X)[0]
    pred = normalize_prediction(raw_pred, metadata)

    probabilities = {}
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        probabilities = normalize_probabilities(clf, proba, metadata)

    iso_raw = float(iso.decision_function(X)[0])
    iso_pred = int(iso.predict(X)[0])

    detector_shadow = evaluate_unknown_detector_shadow(
        X,
        iso,
        target_fpr=0.05,
    )

    unknown_risk = "high" if iso_pred == -1 else "low"

    if (
        req.symptoms.get("backup_disable_attempt", 0) >= 0.7
        or req.symptoms.get("shadow_copy_delete_attempt", 0) >= 0.7
        or req.symptoms.get("security_tool_tamper", 0) >= 0.7
        or req.symptoms.get("event_log_clear_attempt", 0) >= 0.7
    ):
        unknown_risk = "high"

    risk_score = calculate_symptom_risk(req.symptoms)

    response = recommend_response(
        symptoms=req.symptoms,
        unknown_risk=unknown_risk
    )

    return {
        "predicted_label": pred,
        "prediction_probabilities": probabilities,
        "risk_score": risk_score,
        "isolation_forest_raw_score": round(iso_raw, 4),
        "isolation_forest_prediction": iso_pred,
        "unknown_detector_mode": detector_shadow["mode"],
        "unknown_detector_shadow": detector_shadow["calibrated_shadow"],
        "unknown_risk": unknown_risk,
        "response": response,
        "model_used": metadata.get("primary_classifier"),
        "available_models": metadata.get("models")
    }


@app.get("/")
def root():
    return {
        "system": "AI Ransomware Symptom Learning and Response API",
        "version": "0.3.0",
        "status": "running"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/info")
def model_info():
    with open(MODEL_DIR / "model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "models": metadata.get("models"),
        "primary_classifier": metadata.get("primary_classifier"),
        "labels": metadata.get("labels"),
        "feature_count": metadata.get("feature_count"),
        "training_rows": metadata.get("training_rows"),
        "lstm_status": metadata.get("lstm_status")
    }


@app.post("/predict")
def predict(req: SymptomRequest) -> Dict[str, Any]:
    return infer(req)


@app.post("/respond")
def respond(req: SymptomRequest) -> Dict[str, Any]:
    result = infer(req)

    lockdown_result = None

    if result["response"]["policy"] == "protective_lockdown":
        case = {
            "input_symptoms": req.symptoms,
            "inference_result": result
        }
        lockdown_result = run_protective_lockdown(case)

    result["lockdown_result"] = lockdown_result

    out = REPORT_DIR / "last_response_result.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


@app.get("/cases/unknown")
def list_unknown_cases():
    files = sorted(UNKNOWN_CASE_DIR.glob("*.json"), reverse=True)
    return {
        "count": len(files),
        "cases": [{"file": str(p), "size_bytes": p.stat().st_size} for p in files[:50]]
    }


@app.get("/backups")
def list_backups():
    dirs = sorted([p for p in BACKUP_DIR.glob("*") if p.is_dir()], reverse=True)
    return {
        "count": len(dirs),
        "backups": [{"path": str(p), "file_count": len([x for x in p.glob("*") if x.is_file()])} for p in dirs[:50]]
    }


@app.post("/prepare-demo-docs")
def prepare_demo_docs():
    ensure_demo_protected_docs()
    return {
        "status": "ok",
        "protected_dir": str(PROTECTED_DIR),
        "files": [str(p) for p in PROTECTED_DIR.glob("*") if p.is_file()]
    }


@app.post("/unlock-demo")
def unlock_demo():
    unlocked = unlock_protected_files_demo()
    return {"status": "ok", "unlocked_files": unlocked}


try:
    from symptom_ai.explainability.evidence_matcher import build_decision_explanation

    @app.post("/explain")
    def explain(req: SymptomRequest) -> Dict[str, Any]:
        result = infer(req)

        explanation = build_decision_explanation(
            symptoms=req.symptoms,
            predicted_label=result["predicted_label"],
            probabilities=result["prediction_probabilities"],
            risk_score=result["risk_score"],
            unknown_risk=result["unknown_risk"],
            response=result["response"],
            block_threshold_0_10=7.0
        )

        return {
            "inference": result,
            "decision_explanation": explanation
        }

except Exception as e:
    @app.post("/explain")
    def explain_unavailable(req: SymptomRequest) -> Dict[str, Any]:
        return {
            "status": "explainability_not_ready",
            "detail": str(e),
            "inference": infer(req)
        }
