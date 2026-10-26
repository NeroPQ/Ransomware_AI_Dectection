from pathlib import Path
import json

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/symptom_ai")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_models():
    from symptom_ai.api.main import load_artifacts
    return load_artifacts()


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
    from symptom_ai.api.main import infer, SymptomRequest
    return {"case_name": case_name, "input_symptoms": case,
            **infer(SymptomRequest(symptoms=case))}


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
