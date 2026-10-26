from pathlib import Path
import argparse
import csv
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest


from symptom_ai.models.evaluation import feature_matrix, related_groups, calibrate_benign_threshold
from symptom_ai.models.heldout_validation import validate_partitions, load_family_split
from symptom_ai.models.prepare_heldout_family_v2 import canonical_family, stable_score

SOURCE = "CSU_Ransomware_Data"
ROOT = Path("reports/evaluation_v2")
OUTPUT_DIR = ROOT / "isolation_forest_benign_only"
MODEL_ARTIFACT = Path("models/unknown_behavior_detector_benign_only.joblib")

POSITIVE_LABEL = "known_ransomware_like"
NEGATIVE_LABEL = "benign"

FIT_BENIGN_ROWS = 80000
TARGET_FPRS = [0.01, 0.03, 0.05, 0.10]



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

    if df["sample_id"].isna().any() or df["sample_id"].astype(str).str.strip().eq("").any():
        raise SystemExit("Sample IDs must be nonmissing")
    ids = df["sample_id"].astype(str)

    if ids.duplicated().any():
        raise SystemExit(f"{name} contains duplicate sample IDs.")


def validate_positive(df, family):
    labels = sorted(df["label"].astype(str).unique())

    if labels != [POSITIVE_LABEL]:
        raise SystemExit(
            f"{family} test set must contain only ransomware rows."
        )

    if not df["family"].map(canonical_family).eq(family).all():
        raise SystemExit(f"{family} family labels mismatch")
    sources = sorted(df["dataset_source"].astype(str).unique())
    if sources != [SOURCE]:
        raise SystemExit(f"{family} source mismatch. Found: {sources}")

    if df["sample_id"].isna().any() or df["sample_id"].astype(str).str.strip().eq("").any():
        raise SystemExit("Sample IDs must be nonmissing")
    ids = df["sample_id"].astype(str)

    if ids.duplicated().any():
        raise SystemExit(f"{family} contains duplicate sample IDs.")


def to_matrix(df, feature_cols):
    return feature_matrix(df).loc[:, feature_cols]


def anomaly_scores(model, X):
    # Higher value means more anomalous.
    return -model.score_samples(X)


def detection_rate(scores, threshold):
    return float(np.mean(scores > threshold))


def _main():
    global ROOT, OUTPUT_DIR, MODEL_ARTIFACT, FIT_BENIGN_ROWS
    parser = argparse.ArgumentParser(description="Offline benign-only IF calibration; writes a shadow candidate only")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-artifact", type=Path, default=MODEL_ARTIFACT)
    parser.add_argument("--fit-benign-rows", type=int, default=FIT_BENIGN_ROWS)
    args = parser.parse_args()
    ROOT, OUTPUT_DIR, MODEL_ARTIFACT, FIT_BENIGN_ROWS = args.root, args.output_dir, args.model_artifact, args.fit_benign_rows
    if FIT_BENIGN_ROWS < 2:
        parser.error("fit-benign-rows must be at least 2")
    akira_dir = ROOT / "heldout_CSU_Ransomware_Data_Akira" / "data"
    lockbit_dir = ROOT / "heldout_CSU_Ransomware_Data_LockBit" / "data"

    akira = load_family_split(akira_dir.parent)
    lockbit = load_family_split(lockbit_dir.parent)
    benign_train, benign_test = akira["train_benign"], akira["test_benign"]
    lockbit_benign_train, lockbit_benign_test = lockbit["train_benign"], lockbit["test_benign"]
    akira_test, lockbit_test = akira["test_positive"], lockbit["test_positive"]

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

    if not benign_train.sort_values("sample_id").reset_index(drop=True).equals(lockbit_benign_train.sort_values("sample_id").reset_index(drop=True)):
        raise SystemExit("Benign training observations differ across family splits")
    if not benign_test.sort_values("sample_id").reset_index(drop=True).equals(lockbit_benign_test.sort_values("sample_id").reset_index(drop=True)):
        raise SystemExit("Benign test observations differ across family splits")
    feature_cols = list(feature_matrix(benign_train).columns)
    # Rank whole groups, never split sessions/files/duplicate observations.
    groups, _ = related_groups(benign_train.drop(columns="dataset_source"), feature_matrix(benign_train))
    keys = benign_train.assign(_group=groups).groupby("_group")["sample_id"].min()
    ranked_groups = sorted(keys.index, key=lambda group: stable_score("if_benign_fit_calibration", keys[group]))
    fit_groups = set()
    fit_count = 0
    for group in ranked_groups[:-1]:
        if fit_count >= FIT_BENIGN_ROWS:
            break
        fit_groups.add(group)
        fit_count += int((groups == group).sum())
    if fit_count < FIT_BENIGN_ROWS:
        raise SystemExit("Not enough independent benign groups for requested fit/calibration partition")
    benign_fit = benign_train.loc[np.isin(groups, list(fit_groups))].copy()
    benign_calibration = benign_train.loc[~np.isin(groups, list(fit_groups))].copy()
    validate_partitions({"fit": benign_fit, "calibration": benign_calibration,
                         "test": pd.concat([benign_test, akira_test, lockbit_test], ignore_index=True)})

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


    calibration_scores = anomaly_scores(model, X_calibration)
    benign_test_scores = anomaly_scores(model, X_benign_test)
    akira_scores = anomaly_scores(model, X_akira)
    lockbit_scores = anomaly_scores(model, X_lockbit)

    rows = []

    for target_fpr in TARGET_FPRS:
        calibration_info = calibrate_benign_threshold(model, X_calibration, target_fpr=target_fpr)
        threshold = -calibration_info["threshold"]

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

    # Store the default 5% calibration as well as the multi-threshold report.
    calibrate_benign_threshold(model, X_calibration, target_fpr=0.05)
    joblib.dump({"model": model, "feature_schema_version": 2,
                 "normalization": "fixed_nonnegative_saturation_v1",
                 "feature_columns": feature_cols}, MODEL_ARTIFACT)

    integrity = {
        "akira_lockbit_benign_train_sets_identical": True,
        "akira_lockbit_benign_test_sets_identical": True,
        "benign_fit_calibration_overlap": 0,
        "benign_train_test_overlap": 0,
        "akira_positive_benign_test_overlap": 0,
        "lockbit_positive_benign_test_overlap": 0,
    }

    report = {
        "split_manifests_sha256": {family: hashlib.sha256((ROOT / f"heldout_{SOURCE}_{family}" / "split_manifest.json").read_bytes()).hexdigest() for family in ("Akira", "LockBit")},
        "feature_schema_version": 2,
        "normalization": "fixed_nonnegative_saturation_v1",
        "artifact_sha256": hashlib.sha256(MODEL_ARTIFACT.read_bytes()).hexdigest(),
        "threshold_comparison": "anomaly_score > threshold_anomaly_score",
        "partition_scope": "within_source_heldout_family",
        "protocol": (
            "Isolation Forest fits benign observations only. Whole related groups form fit/calibration/test partitions. "
            "Thresholds use the shared lower-tail benign calibration; equality is an inlier. "
            "This selected, within-source experiment is not a cross-source deployment estimate."
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


def main():
    try:
        _main()
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"Benign calibration failed: {exc}") from exc


if __name__ == "__main__":
    main()
