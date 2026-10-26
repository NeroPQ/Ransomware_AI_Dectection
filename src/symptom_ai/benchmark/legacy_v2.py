"""Frozen Task 02 synthetic replay semantics. Never used by live detection.

Preserves historical comparability, including documented semantic defects.
Use telemetry.behavior and response_engine.evidence_policy for current policy.
"""
import math
from collections import Counter
from symptom_ai.models.evaluation import OUTCOME_FEATURES

HIGH_IMPACT_EVENTS = {
    "file_rename",
    "extension_change",
    "high_entropy_write",
    "ransom_note_created",

    # Safe zero-day simulation impact events
    "stealth_file_lock",
    "stealth_content_scramble",
    "stealth_extension_mutation",
    "stealth_note_marker",
}

EARLY_DETECTION_ALERT_TYPES = {
    "early_warning_known_ransomware",
    "early_warning_unknown_behavior",
    "early_containment_recommended",
    "file_impact_containment_recommended",
    "learned_unknown_ransomware_detected",
}

def clamp01(x):
    value = float(x)
    return max(0.0, min(value, 1.0)) if math.isfinite(value) else 0.0

def aggregate_symptoms(events):
    count = Counter(e.get("event_type") for e in events)
    hosts = {e.get("host") for e in events if e.get("host")}
    paths = {e.get("path") for e in events if e.get("path")}

    entropy_scores = []
    memory_scores = []

    for e in events:
        extra = e.get("extra") or {}
        if not isinstance(extra, dict):
            extra = {}
        try:
            entropy_scores.append(float(extra.get("entropy_score", 0.0)))
        except Exception:
            pass
        try:
            memory_scores.append(float(extra.get("memory_score", 0.0)))
        except Exception:
            pass

    symptoms = {
        "file_write_burst": clamp01((count["file_write"] + count["storage_write_spike"]) / 10),
        "file_read_burst": clamp01(count["file_read"] / 10),
        "file_rename_burst": clamp01(count["file_rename"] / 6),
        "mass_file_modification": clamp01((count["file_write"] + count["file_rename"] + count["extension_change"]) / 12),
        "suspicious_extension_change": clamp01(count["extension_change"] / 4),
        "high_entropy_write": clamp01(max([count["high_entropy_write"] / 4] + entropy_scores)),
        "ransom_note_created": clamp01(count["ransom_note_created"] / 2),
        "multi_directory_impact": clamp01(len(paths) / 15),
        "user_document_impact_high": clamp01(len(paths) / 20),

        "suspicious_process_spawn": clamp01((count["process_spawn"] + count["rare_process"]) / 5),
        "rare_process_name": clamp01(count["rare_process"] / 4),
        "process_tree_anomaly": clamp01(count["rare_process"] / 4),
        "process_injection_suspected": clamp01(count["process_injection_suspected"] / 3),

        "memory_access_spike": clamp01(max([count["memory_anomaly"] / 4] + memory_scores)),
        "memory_entropy_region_high": clamp01(count["memory_anomaly"] / 4),
        "anti_vm": clamp01(count["anti_vm"] / 3),
        "anti_analysis": clamp01((count["anti_vm"] + count["api_behavior"]) / 6),

        "network_api_usage": clamp01((count["network_beacon"] + count["api_behavior"]) / 8),
        "c2_beaconing": clamp01(count["network_beacon"] / 3),
        "data_exfiltration_pattern": clamp01(count["network_beacon"] / 5),

        "storage_write_spike": clamp01(count["storage_write_spike"] / 4),
        "novel_symptom_combination": clamp01((count["memory_anomaly"] + count["rare_process"] + count["storage_write_spike"]) / 8),
        "unknown_high_risk": 0.0,
        "analyst_review_required": 0.0,
        "retraining_candidate": 0.0,
    }

    if symptoms["novel_symptom_combination"] >= 0.5 and (
        symptoms["mass_file_modification"] >= 0.4 or symptoms["high_entropy_write"] >= 0.7
    ):
        symptoms["unknown_high_risk"] = 0.9
        symptoms["analyst_review_required"] = 0.9
        symptoms["retraining_candidate"] = 0.9

    damage = {
        "event_count": len(events),
        "host_count": len(hosts),
        "hosts": sorted(hosts),
        "unique_path_count": len(paths),
        "high_impact_event_count": sum(count[e] for e in HIGH_IMPACT_EVENTS),
        "event_type_counts": dict(count),
        "has_file_impact": any(count[e] > 0 for e in HIGH_IMPACT_EVENTS),
    }

    return symptoms, damage

def decide_alert_type(ai_result, damage, policy, unknown_risk):
    if "api_error" in ai_result:
        return "api_error_needs_review"
    recommended = policy in {"isolate_and_backup", "protective_lockdown"}
    impact = bool(damage.get("has_file_impact"))
    if impact and recommended:
        return "file_impact_containment_recommended"
    if impact:
        return "file_impact_needs_review"
    if recommended:
        return "early_containment_recommended"
    if ai_result.get("predicted_label") == "known_ransomware_like":
        return "early_warning_known_ransomware"
    if unknown_risk == "high" or policy == "early_warning":
        return "early_warning_unknown_behavior"
    return "monitor"

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

RULE_IMPACT_EVENTS = {
    "file_rename",
    "extension_change",
    "high_entropy_write",
    "ransom_note_created",
    "storage_write_spike",
}

def is_rule_detected(damage):
    counts = damage.get("event_type_counts", {}) or {}

    impact_count = sum(
        int(counts.get(event_name, 0) or 0)
        for event_name in RULE_IMPACT_EVENTS
    )

    return bool(
        damage.get("has_file_impact")
        and impact_count >= 3
    )
