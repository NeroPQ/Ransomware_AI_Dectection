import math

HIGH_WEIGHT_SYMPTOMS = {
    "file_write_burst": 0.18,
    "file_rename_burst": 0.18,
    "high_entropy_write": 0.22,
    "mass_file_modification": 0.16,
    "suspicious_extension_change": 0.14,
    "ransom_note_created": 0.18,
    "shadow_copy_delete_attempt": 0.25,
    "backup_disable_attempt": 0.25,
    "c2_beaconing": 0.16,
    "data_exfiltration_pattern": 0.18
}


def calculate_symptom_risk(symptoms: dict) -> float:
    score = 0.0
    used_weight = 0.0

    for symptom, weight in HIGH_WEIGHT_SYMPTOMS.items():
        value = float(symptoms.get(symptom, 0.0))

        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Symptoms must be finite values in [0, 1]")

        if value > 0:
            score += value * weight
            used_weight += weight

    if used_weight == 0:
        return 0.0

    normalized = score / sum(HIGH_WEIGHT_SYMPTOMS.values())

    if (
        symptoms.get("file_write_burst", 0) >= 0.7
        and symptoms.get("file_rename_burst", 0) >= 0.7
        and symptoms.get("high_entropy_write", 0) >= 0.7
    ):
        normalized = max(normalized, 0.85)

    if (
        symptoms.get("shadow_copy_delete_attempt", 0) >= 0.7
        or symptoms.get("backup_disable_attempt", 0) >= 0.7
    ):
        normalized = max(normalized, 0.95)

    if (
        symptoms.get("c2_beaconing", 0) >= 0.7
        and symptoms.get("data_exfiltration_pattern", 0) >= 0.7
    ):
        normalized = max(normalized, 0.90)

    return round(min(max(normalized, 0.0), 1.0), 4)
