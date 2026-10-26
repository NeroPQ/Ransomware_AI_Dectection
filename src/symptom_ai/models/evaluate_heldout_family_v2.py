from pathlib import Path
import argparse
import json

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from xgboost import XGBClassifier


OUTPUT_ROOT = Path("reports/evaluation_v2")

POSITIVE_LABEL = "known_ransomware_like"
NEGATIVE_LABEL = "benign"

NON_FEATURE_COLS = {
    "sample_id",
    "dataset_source",
    "family",
    "behavior_type",
    "collection_type",
    "platform",
    "is_simulated",
    "is_real_malware_executed",
    "label",
    "response_policy",
}

GENERIC_FAMILIES = {
    "", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "good", "goodware", "benign", "ransom", "ransomware",
    "android_ransomware", "static_pe", "malware_api_sequence",
    "unknown_ransomset_family",
}


def canonical_family(value):
    family = str(value or "").strip()

    if family.lower() in GENERIC_FAMILIES:
        return None

    if "-" in family:
        prefix = family.split("-", 1)[0].strip()
        if prefix:
            return prefix

    return family or None


def load_csv(path):
    if not path.exists():
        raise SystemExit(f"Missing required split file: {path}")
    return pd.read_csv(path)


def metric_summary(y_true, y_pred):
    labels = [NEGATIVE_LABEL, POSITIVE_LABEL]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    tn, fp, fn, tp = cm.ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[POSITIVE_LABEL],
        average=None,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_ransomware": float(precision[0]),
        "recall_ransomware": float(recall[0]),
        "f1_ransomware": float(f1[0]),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else None,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else None,
        "confusion_matrix_labels": labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def validate_split(train_df, test_df, family, source):
    required = {
        "sample_id",
        "dataset_source",
        "family",
        "label",
    }

    missing = required - set(train_df.columns) - set(test_df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    train_sources = sorted(train_df["dataset_source"].astype(str).unique())
    test_sources = sorted(test_df["dataset_source"].astype(str).unique())

    if train_sources != [source] or test_sources != [source]:
        raise SystemExit(
            "Source validation failed. "
            f"train={train_sources}, test={test_sources}, expected={source}"
        )

    train_ids = set(train_df["sample_id"].astype(str))
    test_ids = set(test_df["sample_id"].astype(str))
    overlap_ids = sorted(train_ids & test_ids)

    if overlap_ids:
        raise SystemExit(
            f"Sample-ID leakage detected: {len(overlap_ids)} overlapping IDs."
        )

    train_positive = train_df[
        train_df["label"].astype(str) == POSITIVE_LABEL
    ].copy()

    test_positive = test_df[
        test_df["label"].astype(str) == POSITIVE_LABEL
    ].copy()

    train_target_count = int(
        train_positive["family"]
        .map(canonical_family)
        .eq(family)
        .sum()
    )

    test_non_target_count = int(
        (~test_positive["family"].map(canonical_family).eq(family)).sum()
    )

    if train_target_count != 0:
        raise SystemExit(
            f"Held-out family leakage: {train_target_count} {family} "
            "rows found in training positives."
        )

    if test_non_target_count != 0:
        raise SystemExit(
            f"Test-family validation failed: {test_non_target_count} "
            "non-target ransomware rows found in test positives."
        )

    return {
        "train_test_sample_id_overlap": 0,
        "target_family_rows_in_training_positive": train_target_count,
        "non_target_rows_in_test_positive": test_non_target_count,
        "train_sources": train_sources,
        "test_sources": test_sources,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument(
        "--source",
        default="CSU_Ransomware_Data",
    )
    args = parser.parse_args()

    family = args.family.strip()
    source = args.source.strip()

    run_dir = OUTPUT_ROOT / f"heldout_{source}_{family}"
    data_dir = run_dir / "data"

    train_positive = load_csv(data_dir / "train_positive.csv")
    train_benign = load_csv(data_dir / "train_benign.csv")
    test_positive = load_csv(data_dir / "test_positive.csv")
    test_benign = load_csv(data_dir / "test_benign.csv")

    train_df = pd.concat(
        [train_positive, train_benign],
        ignore_index=True,
    )

    test_df = pd.concat(
        [test_positive, test_benign],
        ignore_index=True,
    )

    print("[+] Validating held-out split...")
    integrity = validate_split(train_df, test_df, family, source)

    feature_cols = [
        col
        for col in train_df.columns
        if col not in NON_FEATURE_COLS
        and pd.api.types.is_numeric_dtype(train_df[col])
    ]

    if not feature_cols:
        raise SystemExit("No numeric feature columns found.")

    X_train = train_df[feature_cols].apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0)

    X_test = test_df.reindex(
        columns=feature_cols,
        fill_value=0,
    ).apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0)

    y_train_text = train_df["label"].astype(str)
    y_test_text = test_df["label"].astype(str)

    expected_train_labels = {POSITIVE_LABEL, NEGATIVE_LABEL}
    if set(y_train_text.unique()) != expected_train_labels:
        raise SystemExit(
            f"Unexpected train labels: {sorted(y_train_text.unique())}"
        )

    if set(y_test_text.unique()) != expected_train_labels:
        raise SystemExit(
            f"Unexpected test labels: {sorted(y_test_text.unique())}"
        )

    y_train_id = (
        y_train_text == POSITIVE_LABEL
    ).astype(int)

    print("\n=== HELD-OUT FAMILY PROTOCOL ===")
    print(f"Source: {source}")
    print(f"Held-out family: {family}")
    print(f"Training rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Feature columns: {len(feature_cols):,}")

    print("\n[+] Training RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=22,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train_text)

    rf_pred = rf.predict(X_test)
    rf_metrics = metric_summary(y_test_text, rf_pred)

    print(
        "[+] RF ransomware recall: "
        f"{rf_metrics['recall_ransomware']:.4f}"
    )
    print(
        "[+] RF false-positive rate: "
        f"{rf_metrics['false_positive_rate']:.4f}"
    )

    print("\n[+] Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=7,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train_id)

    xgb_pred_id = xgb.predict(X_test)
    xgb_pred = pd.Series(xgb_pred_id).map({
        0: NEGATIVE_LABEL,
        1: POSITIVE_LABEL,
    }).values

    xgb_metrics = metric_summary(y_test_text, xgb_pred)

    print(
        "[+] XGB ransomware recall: "
        f"{xgb_metrics['recall_ransomware']:.4f}"
    )
    print(
        "[+] XGB false-positive rate: "
        f"{xgb_metrics['false_positive_rate']:.4f}"
    )

    report = {
        "protocol": (
            "Leave-one-family-out evaluation. The target ransomware "
            "family is excluded from training positives and used only "
            "in the held-out ransomware test set."
        ),
        "dataset_source": source,
        "held_out_family": family,
        "training_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "training_label_counts": {
            str(label): int(count)
            for label, count in y_train_text.value_counts().items()
        },
        "test_label_counts": {
            str(label): int(count)
            for label, count in y_test_text.value_counts().items()
        },
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "split_integrity": integrity,
        "random_forest": rf_metrics,
        "xgboost": xgb_metrics,
        "important_note": (
            "These are held-out family results for the selected CSU "
            "dataset source and feature representation. They do not "
            "by themselves prove real-world ransomware generalization."
        ),
    }

    output_path = run_dir / "heldout_evaluation.json"
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== HELD-OUT EVALUATION COMPLETE ===")
    print(f"Report saved at: {output_path}")


if __name__ == "__main__":
    main()
