"""Safe behavioral-model demonstration; metrics require independent sessions."""
from pathlib import Path
import argparse
import json

import joblib
import pandas as pd

from symptom_ai.models.train_symptom_models import train_models
from symptom_ai.models.evaluation import feature_matrix, BEHAVIOR_FEATURES

INPUT = Path("data/processed/behavior/behavior_features.csv")
METRICS_OUT = Path("reports/behavior_model_metrics.csv")
SCORED_OUT = Path("reports/behavior_scored_windows.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--allow-row-split", action="store_true")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.exit(1, f"Missing dataset: {args.input}. Run generate_safe_behavior_logs.py then build_behavior_features.py.\n")
    df = pd.read_csv(args.input)
    if "label" in df:
        df["label"] = df["label"].replace({"ransomware": "known_ransomware_like"})
    try:
        models, metadata, report = train_models(df, behavior=True, allow_row_split=args.allow_row_split)
    except ValueError as exc:
        parser.exit(1, f"Training validation failed: {exc}\n")
    Path("models").mkdir(exist_ok=True)
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"model": name, **{k: report[name][k] for k in ("precision", "recall", "f1", "false_positive_rate")}}
            for name in ("random_forest", "xgboost")]
    rows.append({"model": "isolation_forest", **{k: report["unknown_detector"]["test_metrics"][k] for k in ("precision", "recall", "f1", "false_positive_rate")}})
    pd.DataFrame(rows).to_csv(METRICS_OUT, index=False)
    Path("reports/behavior_training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    X = feature_matrix(df, BEHAVIOR_FEATURES, normalized=False)
    scored = df.copy()
    for name, prefix in (("random_forest", "rf"), ("xgboost", "xgb"), ("isolation_forest", "if")):
        model = models[name]
        joblib.dump({"model": model, "feature_cols": list(X.columns), "metadata": metadata}, Path("models") / f"{name}_behavior.joblib")
        if prefix == "if":
            scored["if_pred"] = (model.predict(X) == -1).astype(int)
            scored["if_score"] = -model.decision_function(X)
            scored["iso_pred"] = scored["if_pred"]  # legacy aliases
            scored["iso_score"] = scored["if_score"]
        else:
            prediction = model.predict(X)
            scored[f"{prefix}_pred"] = (prediction == "known_ransomware_like").astype(int) if prefix == "rf" else prediction
            scored[f"{prefix}_score"] = model.predict_proba(X)[:, 1]
    # All-window scores are explicitly exploratory, not held-out lead-time evidence.
    scored["score_scope"] = "exploratory_includes_training_rows"
    scored["rf_detection_lead_events"] = float("nan")
    scored.to_csv(SCORED_OUT, index=False)
    print(f"Saved {METRICS_OUT}, {SCORED_OUT}; synthetic demo scores do not establish model accuracy")


if __name__ == "__main__":
    main()
