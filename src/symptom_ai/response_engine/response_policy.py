"""Compatibility entry point for symptom-only callers of the pure policy engine."""
from symptom_ai.response_engine.evidence_policy import decide_policy, detector_evidence


def recommend_response(symptoms: dict, unknown_risk: str = "low", *, damage=None, detectors=None) -> dict:
    # unknown_risk is a historical derived summary, not independent evidence of
    # anomaly or damage. Preserve it as an unverified observation for old callers.
    evidence = detectors if detectors is not None else detector_evidence(symptoms, damage)
    result = decide_policy(symptoms, damage, evidence)
    result["legacy_unknown_risk"] = unknown_risk
    return result
