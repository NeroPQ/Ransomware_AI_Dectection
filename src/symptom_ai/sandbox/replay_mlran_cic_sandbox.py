import os
from pathlib import Path
from datetime import datetime
import json
import urllib.request

import pandas as pd


MLRAN = Path("data/symptom_labels/mlran_symptom_dataset.csv")
CIC = Path("data/symptom_labels/cic_malmem2022_symptom_dataset.csv")

OUT_DIR = Path("reports/sandbox")
OUT_DIR.mkdir(parents=True, exist_ok=True)

API_BASE_URL = "http://localhost:8000"

META_COLS = {
    "sample_id",
    "dataset_source",
    "family",
    "behavior_type",
    "collection_type",
    "platform",
    "is_simulated",
    "is_real_malware_executed",
    "label",
    "response_policy",
}

# Các symptom mình muốn replay để giống custom sandbox logs
REPLAY_SYMPTOMS = [
    # MLRan / dynamic sandbox behavior
    "file_api_usage",
    "process_api_usage",
    "registry_api_usage",
    "network_api_usage",
    "suspicious_process_spawn",
    "process_tree_anomaly",
    "anti_analysis",
    "c2_beaconing",
    "data_exfiltration_pattern",

    # CIC-MalMem / memory forensic behavior
    "memory_access_spike",
    "memory_entropy_region_high",
    "process_injection_suspected",
    "anti_vm",
    "anti_debugging",
    "packed_binary",
    "high_section_entropy",
    "service_api_usage",
    "persistence_attempt",

    # General ransomware impact symptoms
    "file_write_burst",
    "file_read_burst",
    "file_rename_burst",
    "high_entropy_write",
    "mass_file_modification",
    "suspicious_extension_change",
    "ransom_note_created",
    "backup_disable_attempt",
    "shadow_copy_delete_attempt",
    "security_tool_tamper",
]


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def json_safe(obj):
    """
    Convert pandas/numpy values to plain Python values for JSON export.
    """
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [json_safe(v) for v in obj]

    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass

    return obj


def call_api(symptoms: dict, execute_response: bool = False):
    endpoint = "/respond" if execute_response else "/predict"
    url = API_BASE_URL + endpoint

    clean_symptoms = {}
    for k, v in symptoms.items():
        try:
            clean_symptoms[str(k)] = float(v)
        except Exception:
            clean_symptoms[str(k)] = 0.0

    payload = json.dumps({"symptoms": clean_symptoms}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "X-API-Key": os.getenv("SYMPTOM_AI_API_KEY", "")},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            result["_api_endpoint"] = endpoint
            return result
    except Exception as e:
        return {
            "api_error": str(e),
            "api_endpoint": endpoint,
            "note": "Make sure FastAPI is running at http://localhost:8000"
        }


def load_dataset(path: Path, source_name: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"[!] Missing {source_name}: {path}")

    df = pd.read_csv(path)

    if "label" not in df.columns:
        raise SystemExit(f"[!] Missing label column in {path}")

    print(f"[+] Loaded {source_name}: {df.shape}")
    return df


def pick_sample(df: pd.DataFrame, label_preference="known_ransomware_like") -> pd.Series:
    preferred = df[df["label"].astype(str) == label_preference]

    if preferred.empty:
        preferred = df

    # Chọn dòng có tổng symptom cao để replay dễ thấy
    feature_cols = [
        c for c in REPLAY_SYMPTOMS
        if c in preferred.columns
    ]

    if feature_cols:
        scores = preferred[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        idx = scores.sort_values(ascending=False).index[0]
        return preferred.loc[idx]

    return preferred.sample(n=1, random_state=42).iloc[0]


def row_to_symptoms(row: pd.Series, scale: float = 1.0) -> dict:
    symptoms = {}

    for c in REPLAY_SYMPTOMS:
        if c not in row.index:
            continue

        try:
            v = float(row[c])
        except Exception:
            v = 0.0

        symptoms[c] = max(0.0, min(v * scale, 1.0))

    return symptoms


def merge_symptoms(*items: dict) -> dict:
    merged = {}

    for d in items:
        for k, v in d.items():
            merged[k] = max(float(merged.get(k, 0.0)), float(v))

    return merged


def add_stage_escalation(symptoms: dict, stage: int, mode: str) -> dict:
    s = dict(symptoms)

    if stage == 1:
        # Very early stage: only weak sandbox indicators.
        # Do not replay full CIC/MLRan anomaly yet, otherwise unknown detector may lockdown too early.
        s = {
            "file_api_usage": 0.20,
            "process_api_usage": 0.20,
            "suspicious_process_spawn": 0.15,
            "memory_access_spike": 0.15,
            "network_api_usage": 0.10,
        }

    elif stage == 2:
        # More suspicious sandbox behavior, but still pre-impact stage.
        s["suspicious_process_spawn"] = max(s.get("suspicious_process_spawn", 0), 0.50)
        s["process_api_usage"] = max(s.get("process_api_usage", 0), 0.50)
        s["file_api_usage"] = max(s.get("file_api_usage", 0), 0.45)
        s["memory_access_spike"] = max(s.get("memory_access_spike", 0), 0.45)
        s["anti_analysis"] = max(s.get("anti_analysis", 0), 0.40)
        s["c2_beaconing"] = max(s.get("c2_beaconing", 0), 0.45)
        s["data_exfiltration_pattern"] = max(s.get("data_exfiltration_pattern", 0), 0.35)
        s["file_write_burst"] = max(s.get("file_write_burst", 0), 0.30)

    elif stage == 3:
        # Ransomware-like file impact starts
        s["file_write_burst"] = max(s.get("file_write_burst", 0), 0.78)
        s["file_rename_burst"] = max(s.get("file_rename_burst", 0), 0.72)
        s["high_entropy_write"] = max(s.get("high_entropy_write", 0), 0.80)
        s["mass_file_modification"] = max(s.get("mass_file_modification", 0), 0.75)
        s["suspicious_extension_change"] = max(s.get("suspicious_extension_change", 0), 0.70)

    elif stage == 4:
        # Critical stage
        s["file_write_burst"] = max(s.get("file_write_burst", 0), 0.90)
        s["file_rename_burst"] = max(s.get("file_rename_burst", 0), 0.88)
        s["high_entropy_write"] = max(s.get("high_entropy_write", 0), 0.92)
        s["mass_file_modification"] = max(s.get("mass_file_modification", 0), 0.90)
        s["ransom_note_created"] = max(s.get("ransom_note_created", 0), 0.80)

        if mode == "unknown":
            s["backup_disable_attempt"] = max(s.get("backup_disable_attempt", 0), 0.82)
            s["shadow_copy_delete_attempt"] = max(s.get("shadow_copy_delete_attempt", 0), 0.80)
            s["security_tool_tamper"] = max(s.get("security_tool_tamper", 0), 0.78)
            s["memory_access_spike"] = max(s.get("memory_access_spike", 0), 0.88)
            s["process_injection_suspected"] = max(s.get("process_injection_suspected", 0), 0.78)

    return s


def summarize_stage(stage: int, symptoms: dict, ai_result: dict) -> dict:
    active = [
        k for k, v in symptoms.items()
        if isinstance(v, (int, float)) and float(v) >= 0.5
    ]

    response = ai_result.get("response", {})

    return {
        "time": now(),
        "stage": stage,
        "active_symptom_count": len(active),
        "active_symptoms": ",".join(active),
        "predicted_label": ai_result.get("predicted_label"),
        "known_ransomware_like_prob": ai_result.get("prediction_probabilities", {}).get("known_ransomware_like"),
        "benign_prob": ai_result.get("prediction_probabilities", {}).get("benign"),
        "risk_score": ai_result.get("risk_score"),
        "unknown_risk": ai_result.get("unknown_risk"),
        "severity": response.get("severity"),
        "policy": response.get("policy"),
        "recommended_actions": " | ".join(response.get("recommended_actions", [])),
        "model_used": ai_result.get("model_used"),
    }


def run_replay(mode="known"):
    incident_id = f"mlran_cic_replay_{mode}_{stamp()}"

    mlran = load_dataset(MLRAN, "MLRan")
    cic = load_dataset(CIC, "CIC_MalMem2022")

    mlran_sample = pick_sample(mlran, "known_ransomware_like")
    cic_sample = pick_sample(cic, "known_ransomware_like")

    mlran_symptoms = row_to_symptoms(mlran_sample, scale=1.0)
    cic_symptoms = row_to_symptoms(cic_sample, scale=1.0)

    base_symptoms = merge_symptoms(mlran_symptoms, cic_symptoms)

    timeline = []
    replay_rows = []

    print("=" * 80)
    print(f"[+] Custom Sandbox Replay from MLRan + CIC-MalMem2022")
    print(f"[+] Incident ID: {incident_id}")
    print(f"[+] Mode: {mode}")
    print(f"[+] MLRan sample: {mlran_sample.get('sample_id')}")
    print(f"[+] CIC sample: {cic_sample.get('sample_id')}")
    print("=" * 80)

    stopped = False
    stop_stage = None
    stop_reason = None

    for stage in [1, 2, 3, 4]:
        symptoms = add_stage_escalation(base_symptoms, stage, mode)
        ai_result = call_api(symptoms, execute_response=(stage >= 3))

        row = summarize_stage(stage, symptoms, ai_result)
        replay_rows.append(row)

        event = {
            "stage": stage,
            "time": row["time"],
            "source_samples": {
                "mlran_sample_id": str(mlran_sample.get("sample_id")),
                "mlran_family": str(mlran_sample.get("family")),
                "cic_sample_id": str(cic_sample.get("sample_id")),
                "cic_family": str(cic_sample.get("family")),
            },
            "symptoms": symptoms,
            "ai_result": ai_result,
            "summary": row,
        }
        timeline.append(event)

        print()
        print(f"Stage {stage}")
        print(f"Predicted: {row['predicted_label']}")
        print(f"Prob ransomware-like: {row['known_ransomware_like_prob']}")
        print(f"Risk: {row['risk_score']}")
        print(f"Unknown risk: {row['unknown_risk']}")
        print(f"Severity: {row['severity']}")
        print(f"Policy: {row['policy']}")
        print(f"Active symptoms: {row['active_symptoms']}")
        if "api_error" in ai_result:
            print(f"[API ERROR] {ai_result.get('api_error')}")

        if "api_error" not in ai_result and row["policy"] in {"isolate_and_backup", "protective_lockdown"} and stage >= 3:
            stopped = True
            stop_stage = stage
            stop_reason = f"AI response policy triggered: {row['policy']}"
            print(f"[!] Replay stopped at stage {stage}: {stop_reason}")
            break

    log_path = OUT_DIR / f"sandbox_replay_log_{incident_id}.csv"
    report_path = OUT_DIR / f"sandbox_replay_report_{incident_id}.json"

    pd.DataFrame(replay_rows).to_csv(log_path, index=False)

    report = {
        "incident_id": incident_id,
        "mode": mode,
        "created_at": now(),
        "stopped": stopped,
        "stop_stage": stop_stage,
        "stop_reason": stop_reason,
        "source": {
            "mlran_dataset": str(MLRAN),
            "cic_dataset": str(CIC),
            "mlran_sample_id": str(mlran_sample.get("sample_id")),
            "mlran_family": str(mlran_sample.get("family")),
            "cic_sample_id": str(cic_sample.get("sample_id")),
            "cic_family": str(cic_sample.get("family")),
        },
        "timeline": timeline,
        "safety_note": "Replay uses normalized symptoms from MLRan and CIC-MalMem2022. No real malware is executed."
    }

    report_path.write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")

    print()
    print("=" * 80)
    print(f"[+] Sandbox replay log saved: {log_path}")
    print(f"[+] Sandbox replay report saved: {report_path}")

    return log_path, report_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["known", "unknown"], default="known")

    args = parser.parse_args()
    run_replay(mode=args.mode)
