"""Reproducible holdouts and metrics shared by both existing training pipelines."""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

from symptom_ai.dataset_ingestion.ingest_all_available_datasets import SYMPTOMS

# Decisions, review state and model outputs are not pre-detection observations.
OUTCOME_FEATURES = {
    "known_family_similarity_low", "known_family_similarity_medium",
    "known_family_similarity_high", "anomaly_score_high", "novel_symptom_combination",
    "partial_ransomware_match", "unknown_high_risk", "unknown_low_risk",
    "analyst_review_required", "retraining_candidate",
}
SYMPTOM_FEATURES = sorted(set(SYMPTOMS) - OUTCOME_FEATURES)
BEHAVIOR_FEATURES = [
    "event_count", "file_read_count", "file_write_count", "file_rename_count",
    "file_delete_count", "registry_set_count", "process_create_count",
    "service_stop_count", "shadow_copy_delete_count", "dns_query_count",
    "network_connect_count", "unique_object_count", "unique_process_count",
    "total_bytes_written", "avg_entropy_before", "avg_entropy_after", "entropy_delta",
    "affected_file_events", "rename_burst_score", "write_burst_score", "high_entropy_flag",
]


def feature_matrix(frame, allowed=SYMPTOM_FEATURES, *, normalized=True):
    columns = [c for c in allowed if c in frame]
    if not columns:
        raise ValueError("No compatible observation features found")
    X = frame[columns].apply(pd.to_numeric, errors="raise").fillna(0.0).astype(float)
    if not np.isfinite(X.to_numpy()).all() or (X < 0).any().any():
        raise ValueError("Features must be finite and nonnegative")
    if normalized and (X > 1).any().any():
        raise ValueError("Symptom features must be normalized to [0, 1]")
    return X


def related_groups(frame, X, *, allow_row_split=False):
    """Connected groups enforce every available relationship, plus exact duplicates.

    dataset_source is intentionally conservative: a single source/session cannot
    establish cross-source generalization. Never silently fall back to row splits.
    """
    parent = list(range(len(frame)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join_values(values):
        seen = {}
        for i, value in enumerate(values):
            if pd.isna(value):
                continue
            value = str(value).strip().lower()
            if not value or value in {"unknown", "nan", "none", "benign", "normal", "goodware"} or value.startswith("unknown_"):
                continue
            if value in seen:
                parent[root(i)] = root(seen[value])
            else:
                seen[value] = i

    columns = [c for c in ("dataset_source", "session_id", "source_file", "family", "sample_id") if c in frame and frame[c].notna().any()]
    meaningful = []
    for column in columns:
        values = frame[column].dropna().astype(str).str.strip().str.lower()
        values = values[~values.isin(["", "unknown", "nan", "none", "benign", "normal", "goodware"]) & ~values.str.startswith("unknown_")]
        if not values.empty:
            meaningful.append(column)
    if not (set(meaningful) - {"sample_id"}) and not allow_row_split:
        raise ValueError("Grouping metadata unavailable; supply source/session/family IDs or explicitly use --allow-row-split for a limited demonstration")
    if allow_row_split:
        columns = []  # Explicitly override correlation metadata only for demonstration.
    for column in columns:
        join_values(frame[column])
    join_values(pd.util.hash_pandas_object(X, index=False))
    if allow_row_split:
        warnings.warn("Explicit demonstration split: missing relationships cannot be excluded; metrics do not establish generalization", stacklevel=2)
    groups = np.array([root(i) for i in range(len(frame))])
    return groups, {"group_columns": columns, "exact_duplicates_grouped": True,
                    "allow_row_split": allow_row_split, "group_count": int(len(set(groups)))}


def split_dataset(frame, X, *, allow_row_split=False):
    groups, info = related_groups(frame, X, allow_row_split=allow_row_split)
    labels = frame["label"].to_numpy()
    classes = set(labels)

    def split(indices, seed, size):
        splitter = GroupShuffleSplit(n_splits=64, test_size=size, random_state=seed)
        try:
            for left, right in splitter.split(indices, labels[indices], groups[indices]):
                train, holdout = indices[left], indices[right]
                if set(labels[train]) == classes and set(labels[holdout]) == classes:
                    return train, holdout
        except ValueError:
            pass
        raise ValueError("Insufficient independent groups with both classes for train/validation/test. Collect independent sessions/sources; do not split correlated windows by row.")

    development, test = split(np.arange(len(frame)), 42, 0.2)
    train, validation = split(development, 43, 0.25)
    info.update({"seed": 42, "validation_seed": 43,
                 "limitations": "Only recorded relationships and identical feature rows can be separated; missing provenance, near duplicates, and coarse feature proxies limit generalization claims",
                 "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
                 "group_ids": {name: sorted(set(groups[idx].tolist())) for name, idx in
                               (("train", train), ("validation", validation), ("test", test))}})
    return train, validation, test, info


def metrics(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_fpr = {}
    for i, label in enumerate(labels):
        fp = int(cm[:, i].sum() - cm[i, i])
        negatives = int(cm.sum() - cm[i, :].sum())
        per_class_fpr[str(label)] = fp / negatives if negatives else None
    benign_index = labels.index("benign") if "benign" in labels else None
    benign_count = int(cm[benign_index].sum()) if benign_index is not None else 0
    fpr = (float((benign_count - cm[benign_index, benign_index]) / benign_count)
           if benign_count else None)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": report["macro avg"]["precision"], "recall": report["macro avg"]["recall"],
            "f1": report["macro avg"]["f1-score"], "false_positive_rate": fpr,
            "per_class_false_positive_rate": per_class_fpr, "classification_report": report,
            "confusion_matrix_labels": labels, "confusion_matrix": cm.tolist()}


def calibrate_benign_threshold(model, calibration, *, target_fpr=0.05):
    """Shared lower-tail calibration; ties at the threshold are inliers."""
    if not 0 < target_fpr < 1 or len(calibration) < 2:
        raise ValueError("Calibration requires at least two benign rows and FPR in (0, 1)")
    threshold = float(np.quantile(model.score_samples(calibration), target_fpr))
    model.offset_ = threshold
    return {"threshold": threshold, "target_validation_fpr": target_fpr,
            "benign_calibration_rows": len(calibration),
            "observed_validation_fpr": float((model.predict(calibration) == -1).mean()),
            "threshold_comparison": "score_samples < threshold"}


def fit_benign_detector(X_train, y_train, X_validation, y_validation, *, target_fpr=0.05):
    benign = X_train.loc[np.asarray(y_train) == "benign"]
    calibration = X_validation.loc[np.asarray(y_validation) == "benign"]
    if len(benign) < 2 or len(calibration) < 2:
        raise ValueError("Benign-only Isolation Forest requires at least two training and two validation benign samples")
    iso = IsolationForest(n_estimators=150, contamination="auto", random_state=42, n_jobs=-1)
    iso.fit(benign.sample(n=min(len(benign), 100000), random_state=42))
    calibration_info = calibrate_benign_threshold(iso, calibration, target_fpr=target_fpr)
    return iso, {"training": "benign_only", "training_rows": min(len(benign), 100000),
                 **calibration_info,
                 "limitation": "Anomalies are not confirmed malware and do not prove zero-day detection"}
