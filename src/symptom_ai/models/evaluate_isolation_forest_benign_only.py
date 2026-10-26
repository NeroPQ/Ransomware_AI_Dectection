from pathlib import Path
import csv
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest


SOURCE = "CSU_Ransomware_Data"
ROOT = Path("reports/evaluation_v2")
OUTPUT_DIR = ROOT / "isolation_forest_benign_only"
MODEL_ARTIFACT = Path("models/unknown_behavior_detector_benign_only.joblib")

POSITIVE_LABEL = "known_ransomware_like"
NEGATIVE_LABEL = "benign"

FIT_BENIGN_ROWS = 80000
TARGET_FPRS = [0.01, 0.03, 0.05, 0.10]

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


def stable_score(namespace, sample_id):
    raw = f"{namespace}|{sample_id}".encode(
        "utf-8",
        errors="ignore",
    )
    return int.from_bytes(
        hashlib.blake2b(raw, digest_size=8).digest(),
        "big",
    )


def load_csv(path):
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")
    return pd.read_csv(path)


def validate_benign(df, name):
    labels = sorted(df["label"].astype(str).unique())

    if labels != [NEGATIVE_LABEL]:
        raise SystemExit(
            f"{name} must contain only benign rows. Found: {labels}"
        )

    sources = sorted(df["dataset_source"].astype(str).unique())

    if sources != [SOURCE]:
        raise SystemExit(
            f"{name} source mismatch. Found: {sources}"
        )

    ids = df["sample_id"].astype(str)

    if ids.duplicated().any():
        raise SystemExit(f"{name} contains duplicate sample IDs.")


def validate_positive(df, family):
    labels = sorted(df["label"].astype(str).unique())

    if labels != [POSITIVE_LABEL]:
        raise SystemExit(
            f"{family} test set must contain only ransomware rows."
        )

    sources = sorted(df["dataset_source"].astype(str).unique())

    if sources != [SOURCE]:
        raise SystemExit(
            f"{family} source mismatch. Found: {sources}"
        )

    ids = df["sample_id"].astype(str)

    if ids.duplicated().any():
        raise SystemExit(f"{family} contains duplicate sample IDs.")


def to_matrix(df, feature_cols):
    return (
        df.reindex(columns=feature_cols, fill_value=0)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )


def anomaly_scores(model, X):
    # Higher value means more anomalous.
    return -model.score_samples(X)


def detection_rate(scores, threshold):
    return float(np.mean(scores >= threshold))


def main():
    akira_dir = ROOT / "heldout_CSU_Ransomware_Data_Akira" / "data"
    lockbit_dir = ROOT / "heldout_CSU_Ransomware_Data_LockBit" / "data"

    benign_train = load_csv(akira_dir / "train_benign.csv")
    benign_test = load_csv(akira_dir / "test_benign.csv")

    lockbit_benign_train = load_csv(lockbit_dir / "train_benign.csv")
    lockbit_benign_test = load_csv(lockbit_dir / "test_benign.csv")

    akira_test = load_csv(akira_dir / "test_positive.csv")
    lockbit_test = load_csv(lockbit_dir / "test_positive.csv")

    validate_benign(benign_train, "Akira train benign")
    validate_benign(benign_test, "Akira test benign")
    validate_benign(lockbit_benign_train, "LockBit train benign")
    validate_benign(lockbit_benign_test, "LockBit test benign")

    validate_positive(akira_test, "Akira")
    validate_positive(lockbit_test, "LockBit")

    akira_train_ids = set(benign_train["sample_id"].astype(str))
    lockbit_train_ids = set(
        lockbit_benign_train["sample_id"].astype(str)
    )

    akira_test_ids = set(benign_test["sample_id"].astype(str))
    lockbit_test_ids = set(
        lockbit_benign_test["sample_id"].astype(str)
    )

    if akira_train_ids != lockbit_train_ids:
        raise SystemExit("Akira and LockBit benign train sets differ.")

    if akira_test_ids != lockbit_test_ids:
        raise SystemExit("Akira and LockBit benign test sets differ.")

    if akira_train_ids & akira_test_ids:
        raise SystemExit("Benign train/test sample-ID overlap detected.")

    if set(akira_test["sample_id"].astype(str)) & akira_test_ids:
        raise SystemExit("Akira positive/benign overlap detected.")

    if set(lockbit_test["sample_id"].astype(str)) & akira_test_ids:
        raise SystemExit("LockBit positive/benign overlap detected.")

    feature_cols = [
        col
        for col in benign_train.columns
        if col not in NON_FEATURE_COLS
        and pd.api.types.is_numeric_dtype(benign_train[col])
    ]

    if not feature_cols:
        raise SystemExit("No numeric feature columns found.")

    ranked = benign_train.copy()
    ranked["_stable_score"] = ranked["sample_id"].astype(str).map(
        lambda value: stable_score(
            "if_benign_fit_calibration",
            value,
        )
    )

    ranked = ranked.sort_values(
        "_stable_score",
        kind="stable",
    ).reset_index(drop=True)

    if len(ranked) <= FIT_BENIGN_ROWS:
        raise SystemExit(
            "Not enough benign rows for fit/calibration partition."
        )

    benign_fit = ranked.iloc[:FIT_BENIGN_ROWS].copy()
    benign_calibration = ranked.iloc[FIT_BENIGN_ROWS:].copy()

    fit_ids = set(benign_fit["sample_id"].astype(str))
    calibration_ids = set(
        benign_calibration["sample_id"].astype(str)
    )

    if fit_ids & calibration_ids:
        raise SystemExit("Fit/calibration overlap detected.")

    X_fit = to_matrix(benign_fit, feature_cols)
    X_calibration = to_matrix(benign_calibration, feature_cols)
    X_benign_test = to_matrix(benign_test, feature_cols)
    X_akira = to_matrix(akira_test, feature_cols)
    X_lockbit = to_matrix(lockbit_test, feature_cols)

    print("=== BENIGN-ONLY ISOLATION FOREST PROTOCOL ===")
    print(f"Feature columns: {len(feature_cols):,}")
    print(f"Benign fit rows: {len(benign_fit):,}")
    print(f"Benign calibration rows: {len(benign_calibration):,}")
    print(f"Independent benign test rows: {len(benign_test):,}")
    print(f"Held-out Akira rows: {len(akira_test):,}")
    print(f"Held-out LockBit rows: {len(lockbit_test):,}")

    print("\n[+] Training Isolation Forest on benign-fit data only...")

    model = IsolationForest(
        n_estimators=200,
        max_samples=256,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_fit)

    MODEL_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_ARTIFACT)

    calibration_scores = anomaly_scores(model, X_calibration)
    benign_test_scores = anomaly_scores(model, X_benign_test)
    akira_scores = anomaly_scores(model, X_akira)
    lockbit_scores = anomaly_scores(model, X_lockbit)

    rows = []

    for target_fpr in TARGET_FPRS:
        threshold = float(
            np.quantile(
                calibration_scores,
                1 - target_fpr,
                method="higher",
            )
        )

        calibration_fpr = detection_rate(
            calibration_scores,
            threshold,
        )

        benign_test_fpr = detection_rate(
            benign_test_scores,
            threshold,
        )

        akira_recall = detection_rate(
            akira_scores,
            threshold,
        )

        lockbit_recall = detection_rate(
            lockbit_scores,
            threshold,
        )

        macro_family_recall = float(
            np.mean([akira_recall, lockbit_recall])
        )

        row = {
            "target_calibration_fpr": target_fpr,
            "threshold_anomaly_score": threshold,
            "calibration_benign_fpr": calibration_fpr,
            "independent_benign_test_fpr": benign_test_fpr,
            "akira_unknown_detection_recall": akira_recall,
            "lockbit_unknown_detection_recall": lockbit_recall,
            "macro_family_unknown_detection_recall": (
                macro_family_recall
            ),
        }

        rows.append(row)

        print(
            f"Target FPR={target_fpr * 100:5.1f}% | "
            f"Test FPR={benign_test_fpr * 100:6.3f}% | "
            f"Akira={akira_recall * 100:6.3f}% | "
            f"LockBit={lockbit_recall * 100:6.3f}% | "
            f"Macro Recall={macro_family_recall * 100:6.3f}%"
        )

    integrity = {
        "akira_lockbit_benign_train_sets_identical": True,
        "akira_lockbit_benign_test_sets_identical": True,
        "benign_fit_calibration_overlap": 0,
        "benign_train_test_overlap": 0,
        "akira_positive_benign_test_overlap": 0,
        "lockbit_positive_benign_test_overlap": 0,
    }

    report = {
        "protocol": (
            "Isolation Forest was trained only on benign behavior. "
            "The original 100,000 benign training rows were "
            "deterministically split into 80,000 model-fit rows and "
            "20,000 calibration rows. Thresholds were selected from "
            "calibration-benign anomaly scores and evaluated on an "
            "independent 20,000 benign test set plus held-out Akira "
            "and LockBit ransomware-family test sets."
        ),
        "model": {
            "name": "IsolationForest",
            "n_estimators": 200,
            "max_samples": 256,
            "contamination": "auto",
            "random_state": 42,
            "artifact_path": str(MODEL_ARTIFACT),
        },
        "source": SOURCE,
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "dataset_sizes": {
            "benign_fit": len(benign_fit),
            "benign_calibration": len(benign_calibration),
            "benign_independent_test": len(benign_test),
            "akira_heldout_positive_test": len(akira_test),
            "lockbit_heldout_positive_test": len(lockbit_test),
        },
        "integrity_checks": integrity,
        "threshold_results": rows,
        "important_caveat": (
            "Akira and LockBit use the same deterministic benign test "
            "subset. Their false-positive results are therefore not "
            "independent samples and must not be aggregated as two "
            "separate benign test populations."
        ),
        "scope_note": (
            "This is an offline calibration experiment only. It does "
            "not replace the production unknown detector or modify the "
            "live monitoring API."
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "isolation_forest_benign_only.json"
    csv_path = OUTPUT_DIR / "isolation_forest_benign_only.csv"

    json_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    fields = [
        "target_calibration_fpr",
        "threshold_anomaly_score",
        "calibration_benign_fpr",
        "independent_benign_test_fpr",
        "akira_unknown_detection_recall",
        "lockbit_unknown_detection_recall",
        "macro_family_unknown_detection_recall",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== CALIBRATION COMPLETE ===")
    print(f"JSON report: {json_path}")
    print(f"CSV report:  {csv_path}")
    print(f"Model artifact: {MODEL_ARTIFACT}")


if __name__ == "__main__":
    main()
