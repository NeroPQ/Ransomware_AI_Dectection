from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors


DATASET = Path("data/symptom_labels/unified_symptom_dataset.csv")
MODEL_DIR = Path("models")
OUT_DIR = Path("models/explainability")

OUT_DIR.mkdir(parents=True, exist_ok=True)

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

MAX_REFERENCE_ROWS = 120000


def main():
    if not DATASET.exists():
        raise SystemExit(f"[!] Missing dataset: {DATASET}")

    print(f"[+] Loading {DATASET}")
    df = pd.read_csv(DATASET)

    print("[+] Original shape:", df.shape)

    if len(df) > MAX_REFERENCE_ROWS:
        parts = []
        labels = df["label"].unique()
        per_label = MAX_REFERENCE_ROWS // len(labels)

        for label in labels:
            sub = df[df["label"] == label]
            n = min(len(sub), per_label)
            parts.append(sub.sample(n=n, random_state=42))

        df = pd.concat(parts, ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("[+] Reference shape:", df.shape)

    from symptom_ai.models.evaluation import feature_matrix
    X = feature_matrix(df)
    feature_cols = list(X.columns)

    print("[+] Fitting NearestNeighbors...")
    nn = NearestNeighbors(
        n_neighbors=5,
        metric="cosine",
        algorithm="brute"
    )
    nn.fit(X)

    meta_cols = [
        "sample_id",
        "dataset_source",
        "family",
        "behavior_type",
        "collection_type",
        "platform",
        "label",
        "response_policy"
    ]

    ref_meta = df[meta_cols].copy()
    ref_features = X.copy()

    joblib.dump(nn, OUT_DIR / "nearest_training_matcher.joblib")
    ref_meta.to_csv(OUT_DIR / "training_reference_meta.csv", index=False)

    # Use gzip CSV instead of parquet to avoid pyarrow/fastparquet dependency.
    ref_features.to_csv(
        OUT_DIR / "training_reference_features.csv.gz",
        index=False,
        compression="gzip"
    )

    with open(OUT_DIR / "reference_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "reference_rows": int(len(df)),
            "feature_count": int(len(feature_cols)),
            "feature_columns": feature_cols,
            "nearest_neighbor_metric": "cosine",
            "feature_file": "training_reference_features.csv.gz"
        }, f, indent=2)

    print("[+] Saved explainability reference:")
    print("   ", OUT_DIR / "nearest_training_matcher.joblib")
    print("   ", OUT_DIR / "training_reference_meta.csv")
    print("   ", OUT_DIR / "training_reference_features.csv.gz")
    print("   ", OUT_DIR / "reference_metadata.json")


if __name__ == "__main__":
    main()
