"""Recommendations only. An anomaly by itself is insufficient for containment."""
from symptom_ai.models.evaluation import OUTCOME_FEATURES


def recommend_response(symptoms: dict, unknown_risk: str = "low") -> dict:
    active = {k for k, value in symptoms.items() if k not in OUTCOME_FEATURES and float(value) >= 0.7}
    file_evidence = bool(active & {"file_write_burst", "file_rename_burst", "mass_file_modification"})
    corroborated = file_evidence and bool(active & {"high_entropy_write", "suspicious_extension_change", "ransom_note_created"})
    destructive = bool(active & {"shadow_copy_delete_attempt", "backup_disable_attempt"})
    network = {"c2_beaconing", "data_exfiltration_pattern"} <= active
    policy, severity = "monitor_only", "Low"
    actions = ["Continue monitoring", "Collect more telemetry"]
    if active or unknown_risk == "high":
        policy, severity = "early_warning", "Medium"
        actions = ["Collect evidence and request analyst review"]
    if corroborated:
        policy, severity = "isolate_and_backup", "High"
        actions = ["Recommend containment and backup verification", "Preserve file activity evidence"]
    if destructive or network or (corroborated and unknown_risk == "high"):
        policy, severity = "protective_lockdown", "Critical"
        actions = ["Recommend containment", "Preserve evidence", "Request analyst review"]
    return {"severity": severity, "policy": policy, "active_symptoms": sorted(active),
            "recommended_actions": actions, "containment_recommended": policy in {"isolate_and_backup", "protective_lockdown"},
            "containment_executed": False, "response_mode": "simulation",
            "suspicious": policy != "monitor_only"}
