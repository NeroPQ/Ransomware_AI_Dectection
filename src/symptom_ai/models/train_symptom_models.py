"""Train existing models using independent train, validation and test partitions."""
from pathlib import Path
import argparse
import hashlib
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from symptom_ai.dataset_ingestion.validation import validate_labels
from symptom_ai.models.evaluation import feature_matrix, split_dataset, metrics, fit_benign_detector

DATASET = Path("data/symptom_labels/unified_symptom_dataset.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/symptom_ai")


def train_models(df, *, behavior=False, allow_row_split=False):
    validate_labels(df)
    from symptom_ai.models.evaluation import BEHAVIOR_FEATURES
    X = feature_matrix(df, BEHAVIOR_FEATURES, normalized=False) if behavior else feature_matrix(df)
    train, validation, test, split_info = split_dataset(df, X, allow_row_split=allow_row_split)
    y = df["label"]
    labels = sorted(y.unique())
    label_to_id = {label: i for i, label in enumerate(labels)}
    rf = RandomForestClassifier(n_estimators=150, max_depth=22, random_state=42,
                                n_jobs=-1, class_weight="balanced")
    xgb = XGBClassifier(n_estimators=250, max_depth=7, learning_rate=0.08,
                        subsample=0.85, colsample_bytree=0.85,
                        objective="binary:logistic", eval_metric="logloss",
                        tree_method="hist", random_state=42, n_jobs=-1)
    rf.fit(X.iloc[train], y.iloc[train])
    xgb.fit(X.iloc[train], y.iloc[train].map(label_to_id))

    def predictions(model, indices):
        prediction = model.predict(X.iloc[indices])
        return [labels[int(p)] for p in prediction] if model is xgb else prediction

    validation_results = {name: metrics(y.iloc[validation], predictions(model, validation), labels)
                          for name, model in (("random_forest", rf), ("xgboost", xgb))}
    # Select only on validation macro F1. The test set is used once for reporting.
    primary = max(validation_results, key=lambda name: validation_results[name]["f1"])
    iso, iso_info = fit_benign_detector(X.iloc[train], y.iloc[train], X.iloc[validation], y.iloc[validation])
    test_results = {name: metrics(y.iloc[test], predictions(model, test), labels)
                    for name, model in (("random_forest", rf), ("xgboost", xgb))}
    iso_predictions = ["known_ransomware_like" if p == -1 else "benign" for p in iso.predict(X.iloc[test])]
    iso_info["test_metrics"] = metrics(y.iloc[test], iso_predictions, labels)
    metadata = {"training_rows": len(train), "feature_count": X.shape[1],
                "feature_columns": list(X.columns), "labels": labels,
                "label_to_id": label_to_id, "id_to_label": {str(i): label for label, i in label_to_id.items()},
                "primary_classifier": "xgboost_classifier.joblib" if primary == "xgboost" else "random_forest_classifier.joblib",
                "models": {"random_forest": "RandomForestClassifier", "xgboost": "XGBClassifier", "unknown_detector": "IsolationForest"},
                "evaluation_split": split_info, "unknown_detector": iso_info,
                "feature_schema_version": 2, "normalization": "raw_behavior_counts_and_byte_entropy" if behavior else "fixed_nonnegative_saturation_v1",
                "requires_retraining_legacy_artifacts": True}
    report = {"validation": validation_results, **test_results, "unknown_detector": iso_info,
              "primary_classifier": metadata["primary_classifier"], "split": split_info,
              "selection_metric": "validation_macro_f1", "test_rows": len(test),
              "limitation": "Proxy observations include impact indicators; results do not establish pre-impact detection or enterprise/zero-day efficacy"}
    return {"random_forest": rf, "xgboost": xgb, "isolation_forest": iso}, metadata, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--allow-row-split", action="store_true", help="Explicit correlated demonstration only; no generalization claims")
    args = parser.parse_args()
    if not args.dataset.is_file():
        parser.exit(1, f"Missing dataset: {args.dataset}. Ingest reviewed source data first; raw datasets are not included.\n")
    try:
        models, metadata, report = train_models(pd.read_csv(args.dataset), allow_row_split=args.allow_row_split)
    except ValueError as exc:
        parser.exit(1, f"Training validation failed: {exc}\n")
    provenance = {"dataset": str(args.dataset), "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest()}
    metadata.update(provenance)
    report.update(provenance)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models["random_forest"], args.model_dir / "random_forest_classifier.joblib")
    joblib.dump(models["xgboost"], args.model_dir / "xgboost_classifier.joblib")
    joblib.dump(models["isolation_forest"], args.model_dir / "unknown_behavior_detector.joblib")
    primary = models["xgboost"] if metadata["primary_classifier"].startswith("xgboost") else models["random_forest"]
    joblib.dump(primary, args.model_dir / "ransomware_symptom_classifier.joblib")
    (args.model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.report_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved candidate models: {args.model_dir}; report: {args.report_dir / 'training_report.json'}")


if __name__ == "__main__":
    main()
