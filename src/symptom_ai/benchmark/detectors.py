"""Offline scoring through the existing observation/rule/policy contracts."""
from collections import Counter

import numpy as np
import pandas as pd

from symptom_ai.models.evaluation import feature_matrix, fit_benign_detector, metrics
from symptom_ai.models.train_symptom_models import classifier_candidates
from symptom_ai.benchmark.legacy_v2 import decide_alert_type, EARLY_DETECTION_ALERT_TYPES, recommend_response
from symptom_ai.benchmark.sessions import frame_from_windows
from symptom_ai.benchmark.quality_metrics import detection_metrics

METHODS = ("rule_only", "classifier_only", "anomaly_only", "hybrid")
LABELS = ["benign", "known_ransomware_like"]


def decoded_predictions(model, X):
    return np.array([LABELS[int(value)] if isinstance(value, (int, np.integer)) else str(value)
                     for value in model.predict(X)])


def fit_development(train, validation):
    train_frame, validation_frame = frame_from_windows(train), frame_from_windows(validation)
    X, V = feature_matrix(train_frame), feature_matrix(validation_frame)
    if list(X.columns) != list(V.columns):
        raise ValueError("Training/validation feature schemas differ")
    rf, xgb = classifier_candidates()
    rf.fit(X, train_frame["label"])
    xgb.fit(X, train_frame["label"].map({label: index for index, label in enumerate(LABELS)}))
    models = {"random_forest": rf, "xgboost": xgb}
    validation_metrics = {name: metrics(validation_frame["label"], decoded_predictions(model, V), LABELS)
                          for name, model in models.items()}
    selected = max(models, key=lambda name: validation_metrics[name]["f1"])
    iso, calibration = fit_benign_detector(X, train_frame["label"], V, validation_frame["label"])
    bundle = {"classifier": models[selected], "anomaly": iso, "feature_columns": list(X.columns),
              "selected_classifier": selected, "schema_version": 1}
    info = {"classifier_candidates": validation_metrics, "selected_classifier": selected,
            "selection_rule": "validation macro-F1; RF wins exact tie; fixed model hyperparameters",
            "classifier_decision": "model.predict argmax; no test-derived threshold",
            "anomaly_calibration": calibration, "hybrid_policy": "unchanged foundation watcher and response policy",
            "reviewed_signatures": "disabled in this baseline; no approval database read"}
    return bundle, info, X, V


def score_windows(bundle, rows, X=None):
    X = feature_matrix(frame_from_windows(rows)) if X is None else X
    if list(X.columns) != bundle["feature_columns"]:
        raise ValueError("Scoring feature names/order differ from frozen model")
    classifier = decoded_predictions(bundle["classifier"], X)
    classes = [LABELS[int(value)] if isinstance(value, (int, np.integer)) else str(value)
               for value in bundle["classifier"].classes_]
    probabilities = bundle["classifier"].predict_proba(X)[:, classes.index("known_ransomware_like")]
    decision = bundle["anomaly"].decision_function(X)
    outputs = []
    for position, row in enumerate(rows):
        symptoms = X.iloc[position].to_dict()
        unknown_risk = "high" if decision[position] < 0 else "low"
        if any(symptoms.get(key, 0) >= .7 for key in ("backup_disable_attempt", "shadow_copy_delete_attempt", "security_tool_tamper", "event_log_clear_attempt")):
            unknown_risk = "high"
        response = recommend_response(symptoms, unknown_risk)
        alert = decide_alert_type({"predicted_label": classifier[position]}, row["damage"], response["policy"], unknown_risk)
        outputs.append({"rule_only": bool(row["rule_hit"]), "classifier_only": classifier[position] == LABELS[1],
                        "anomaly_only": bool(decision[position] < 0), "hybrid": alert in EARLY_DETECTION_ALERT_TYPES,
                        "classifier_probability": float(probabilities[position]), "anomaly_decision": float(decision[position]),
                        "hybrid_alert": alert, "policy": response["policy"], "active_symptoms": response["active_symptoms"],
                        "containment_recommended": response["containment_recommended"], "containment_executed": False})
    return outputs


def summarize(rows, outputs):
    return {method: detection_metrics(rows, [row[method] for row in outputs]) for method in METHODS}


def validation_diagnostics(bundle, train, validation, X_train, X_validation):
    baseline = score_windows(bundle, validation, X_validation)
    benign_train = X_train.loc[[row["label"] == "benign" for row in train]]
    benign_indices = [i for i, row in enumerate(validation) if row["label"] == "benign"]
    per_feature = []
    # All counterfactuals use validation; reference medians/ranges use training.
    # Correlated features mean these are sensitivity probes, not causal effects.
    for feature in X_train.columns:
        edited = X_validation.copy()
        edited[feature] = benign_train[feature].median()
        probe = score_windows(bundle, validation, edited)
        counts = {method: {"removed_false_alarms": sum(baseline[i][method] and not probe[i][method] for i in benign_indices),
                           "introduced_false_alarms": sum(not baseline[i][method] and probe[i][method] for i in benign_indices)}
                  for method in ("classifier_only", "anomaly_only", "hybrid")}
        if any(item["removed_false_alarms"] or item["introduced_false_alarms"] for item in counts.values()):
            per_feature.append({"feature": feature, "training_benign_median": float(benign_train[feature].median()),
                                "validation_median_replacement": counts})
    alarms = [baseline[i] for i in benign_indices if baseline[i]["hybrid"]]
    rule_alarms = [row for row, out in zip(validation, baseline) if row["label"] == "benign" and out["rule_only"]]
    combinations = Counter(f"rule={out['rule_only']},classifier={out['classifier_only']},anomaly={out['anomaly_only']},hybrid={out['hybrid']}"
                           for row, out in zip(validation, baseline) if row["label"] == "benign")
    return {"partition": "validation_only",
            "benign_signal_combinations": dict(combinations),
            "rule_false_alarm_event_counts": dict(sum((Counter(row["damage"]["event_type_counts"]) for row in rule_alarms), Counter())),
            "impact_review_suppressed_classifier_or_anomaly": sum(out["hybrid_alert"] == "file_impact_needs_review" and (out["classifier_only"] or out["anomaly_only"]) for out in baseline), "benign_windows": len(benign_indices),
            "method_results": summarize(validation, baseline),
            "hybrid_false_alarm_alert_types": dict(Counter(row["hybrid_alert"] for row in alarms)),
            "hybrid_false_alarm_policies": dict(Counter(row["policy"] for row in alarms)),
            "active_symptoms_in_hybrid_false_alarms": dict(Counter(key for row in alarms for key in row["active_symptoms"])),
            "feature_sensitivity": per_feature,
            "sensitivity_note": "Median replacement with training-benign medians, holding raw impact flags fixed; correlated/off-manifold probes are not causal attribution or approved fixes",
            "feature_statistics": {feature: {"benign_train_min": float(benign_train[feature].min()),
                                              "benign_train_max": float(benign_train[feature].max()),
                                              "benign_validation_median": float(X_validation.iloc[benign_indices][feature].median()),
                                              "validation_benign_outside_training_range": int(((X_validation.iloc[benign_indices][feature] < benign_train[feature].min()) | (X_validation.iloc[benign_indices][feature] > benign_train[feature].max())).sum())}
                                   for feature in X_train.columns}}
