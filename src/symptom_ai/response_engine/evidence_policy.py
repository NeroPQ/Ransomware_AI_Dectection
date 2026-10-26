"""Single pure decision engine. Detector evidence is immutable policy input.

Stages are recommendations, never endpoint actions. Model consensus is not
independent corroboration of damage; classifiers and IF share feature inputs.
"""
from copy import deepcopy
import math

POLICY_VERSION = 1
ALTERNATIVES = ("corroboration_only", "evidence_balanced", "analyst_sensitive")
STAGES = ("monitor", "suspicious_activity", "analyst_alert", "containment_recommendation")
GENERIC_ACTIVITY = {"file_write_burst", "file_read_burst", "file_rename_burst", "mass_file_modification",
                    "multi_directory_impact", "storage_write_spike", "network_api_usage",
                    "suspicious_process_spawn", "memory_access_spike"}
SPECIFIC = {"high_entropy_write", "suspicious_extension_change", "ransom_note_created",
            "process_injection_suspected", "shadow_copy_delete_attempt", "backup_disable_attempt",
            "c2_beaconing", "data_exfiltration_pattern"}


def detector_evidence(symptoms, damage=None, ai_result=None, learned_match=None):
    damage, ai = damage or {}, ai_result or {}
    categories = damage.get("evidence_categories", {})
    active = {key for key, value in symptoms.items() if isinstance(value, (float, int)) and math.isfinite(value) and value >= .7}
    specific = sorted(active & SPECIFIC)
    rule_hit = bool(categories.get("possible_impact") and (len(specific) >= 2 or categories.get("corroborated_high_confidence_detection")))
    # No early return on impact: retain every individual detector result.
    compatible = ai.get("model_feature_semantics_version") == 3 and ai.get("input_feature_semantics_version") == 3
    classifier_present = "predicted_label" in ai and "api_error" not in ai
    anomaly_present = "isolation_forest_prediction" in ai and "api_error" not in ai
    return {"schema_version": 1,
            "rule": {"available": True, "hit": rule_hit, "specific_features": specific, "semantics_version": 3},
            "classifier": {"available": classifier_present,
                           "hit": ai.get("predicted_label") == "known_ransomware_like" if classifier_present else None,
                           "label": ai.get("predicted_label"), "probabilities": deepcopy(ai.get("prediction_probabilities", {})),
                           "usable_for_policy": classifier_present and compatible,
                           "compatibility": "v3" if compatible else "unverified_legacy_semantics_shadow_only"},
            "anomaly": {"available": anomaly_present,
                        "hit": ai.get("isolation_forest_prediction") == -1 if anomaly_present else None,
                        "decision": ai.get("isolation_forest_raw_score"),
                        "usable_for_policy": anomaly_present and compatible,
                        "shadow": deepcopy(ai.get("unknown_detector_shadow")),
                        "compatibility": "v3" if compatible else "unverified_legacy_semantics_shadow_only"},
            "reviewed_signature": {"available": learned_match is not None, "hit": bool(learned_match and learned_match.get("matched")),
                                   "match": deepcopy(learned_match), "note": "Reviewed similarity remains a heuristic"}}


def decide_policy(symptoms, damage=None, detectors=None, alternative="evidence_balanced"):
    if alternative not in ALTERNATIVES:
        raise ValueError("Unknown policy alternative")
    damage = damage or {}
    evidence = deepcopy(detectors if detectors is not None else detector_evidence(symptoms, damage))
    categories = deepcopy(damage.get("evidence_categories", {"observed_activity": [], "suspicious_behavior": [],
                              "possible_impact": [], "corroborated_high_confidence_detection": []}))
    active = sorted(key for key, value in symptoms.items() if isinstance(value, (int, float)) and math.isfinite(value) and value >= .7)
    specific = set(active) & SPECIFIC
    rule = bool(evidence["rule"]["hit"])
    model_hits = sum(bool(evidence[key].get("usable_for_policy") and evidence[key].get("hit")) for key in ("classifier", "anomaly"))
    signature = bool(evidence["reviewed_signature"]["hit"])
    possible = bool(categories.get("possible_impact"))
    corroborated = bool(categories.get("corroborated_high_confidence_detection"))
    stage, reasons = "monitor", ["Activity alone does not establish harmful intent"]
    if specific or categories.get("suspicious_behavior") or possible or model_hits or signature:
        stage, reasons = "suspicious_activity", ["Collect and preserve additional evidence"]
    context = bool(specific or categories.get("suspicious_behavior") or possible)
    analyst = ((possible and (rule or model_hits > 0 or signature)) or (model_hits == 2 and context))
    if alternative == "analyst_sensitive":
        analyst = analyst or possible or rule or (model_hits > 0 and context) or signature
    elif alternative == "corroboration_only":
        analyst = possible and (rule or signature)
    if analyst:
        stage, reasons = "analyst_alert", ["Analyst review required; detector signals retained separately"]
    if corroborated:
        stage, reasons = "containment_recommendation", ["Attributed destructive-content and note/recovery evidence corroborate possible impact"]
    actions = {"monitor": ["Continue monitoring"],
               "suspicious_activity": ["Collect evidence and process context"],
               "analyst_alert": ["Request analyst review", "Preserve timeline and affected-file observations"],
               "containment_recommendation": ["Recommend analyst-authorized containment", "Verify backups", "Preserve evidence"]}[stage]
    # Compatibility aliases for old dashboards. Never use aliases to execute actions.
    alias = {"monitor": "monitor_only", "suspicious_activity": "early_warning", "analyst_alert": "analyst_review",
             "containment_recommendation": "protective_lockdown"}[stage]
    return {"policy_version": POLICY_VERSION, "policy_alternative": alternative, "stage": stage, "policy": alias,
            "severity": dict(zip(STAGES, ("Low", "Low", "Medium", "High")))[stage], "reasons": reasons,
            "evidence_categories": categories, "detector_evidence": evidence, "active_symptoms": active,
            "recommended_actions": actions, "suspicious": stage != "monitor",
            "analyst_review_required": stage in STAGES[2:], "containment_recommended": stage == STAGES[3],
            "containment_executed": False, "response_mode": "simulation"}
