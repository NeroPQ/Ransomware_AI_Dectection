from pathlib import Path
import json
import joblib
import pandas as pd


REF_DIR = Path("models/explainability")


def load_reference():
    nn = joblib.load(REF_DIR / "nearest_training_matcher.joblib")
    meta = pd.read_csv(REF_DIR / "training_reference_meta.csv")

    with open(REF_DIR / "reference_metadata.json", "r", encoding="utf-8") as f:
        ref_info = json.load(f)

    feature_file = ref_info.get("feature_file", "training_reference_features.csv.gz")
    features = pd.read_csv(REF_DIR / feature_file)

    return nn, meta, features, ref_info


def get_active_symptoms(symptoms: dict, threshold: float = 0.5):
    active = []

    for k, v in symptoms.items():
        try:
            value = float(v)
        except Exception:
            continue

        if value >= threshold:
            active.append({
                "symptom": k,
                "value": round(value, 4)
            })

    return sorted(active, key=lambda x: x["value"], reverse=True)


def find_nearest_training_rows(symptoms: dict, top_k: int = 5):
    nn, meta, features, ref_info = load_reference()
    feature_cols = ref_info["feature_columns"]

    row = {col: float(symptoms.get(col, 0.0)) for col in feature_cols}
    X = pd.DataFrame([row], columns=feature_cols)

    distances, indices = nn.kneighbors(X, n_neighbors=min(top_k, len(meta)))

    results = []

    for rank, idx in enumerate(indices[0], start=1):
        distance = float(distances[0][rank - 1])
        similarity = 1.0 - distance

        m = meta.iloc[int(idx)].to_dict()
        f = features.iloc[int(idx)]

        shared = []
        for symptom, input_value in symptoms.items():
            if symptom not in features.columns:
                continue

            try:
                iv = float(input_value)
                tv = float(f[symptom])
            except Exception:
                continue

            if iv >= 0.5 and tv >= 0.5:
                shared.append({
                    "symptom": symptom,
                    "input_value": round(iv, 4),
                    "matched_row_value": round(tv, 4)
                })

        shared = sorted(
            shared,
            key=lambda x: x["input_value"] + x["matched_row_value"],
            reverse=True
        )[:10]

        results.append({
            "rank": rank,
            "similarity_score": round(similarity, 4),
            "distance": round(distance, 4),
            "matched_sample_id": m.get("sample_id"),
            "matched_dataset_source": m.get("dataset_source"),
            "matched_family": m.get("family"),
            "matched_behavior_type": m.get("behavior_type"),
            "matched_label": m.get("label"),
            "matched_response_policy": m.get("response_policy"),
            "shared_active_symptoms": shared
        })

    return results


def build_decision_explanation(
    symptoms: dict,
    predicted_label: str,
    probabilities: dict,
    risk_score: float,
    unknown_risk: str,
    response: dict,
    block_threshold_0_10: float = 7.0
):
    risk_0_10 = round(float(risk_score) * 10, 2)

    if risk_0_10 >= block_threshold_0_10:
        decision = "block"
    elif risk_0_10 >= 4:
        decision = "warn"
    else:
        decision = "allow"

    active_symptoms = get_active_symptoms(symptoms, threshold=0.5)
    nearest_rows = find_nearest_training_rows(symptoms, top_k=5)

    matched_label_counts = {}
    for r in nearest_rows:
        label = r["matched_label"]
        matched_label_counts[label] = matched_label_counts.get(label, 0) + 1

    if nearest_rows:
        top_match = nearest_rows[0]
        top_reason = (
            f"Input case is closest to sample {top_match['matched_sample_id']} "
            f"from {top_match['matched_dataset_source']} with label "
            f"{top_match['matched_label']} and similarity "
            f"{top_match['similarity_score']}."
        )
    else:
        top_reason = "No nearest training rows found."

    rule_reason = (
        f"Risk score is {risk_0_10}/10. "
        f"Block threshold is {block_threshold_0_10}/10. "
    )

    if decision == "block":
        rule_reason += "The case is blocked because risk score reaches or exceeds the block threshold."
    elif decision == "warn":
        rule_reason += "The case is not blocked yet, but it requires analyst review."
    else:
        rule_reason += "The case is allowed because risk score is below warning threshold."

    return {
        "decision": decision,
        "block_threshold_0_10": block_threshold_0_10,
        "risk_score_0_1": round(float(risk_score), 4),
        "risk_score_0_10": risk_0_10,
        "predicted_label": predicted_label,
        "prediction_probabilities": probabilities,
        "unknown_risk": unknown_risk,
        "response_policy": response.get("policy"),
        "severity": response.get("severity"),
        "active_symptoms": active_symptoms,
        "matched_label_counts_top5": matched_label_counts,
        "nearest_training_rows": nearest_rows,
        "explanation": {
            "rule_reason": rule_reason,
            "top_match_reason": top_reason
        }
    }
