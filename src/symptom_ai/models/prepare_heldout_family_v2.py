from pathlib import Path
import argparse
import hashlib
import json

DATASET = Path("data/symptom_labels/unified_symptom_dataset.csv")
OUTPUT_ROOT = Path("reports/evaluation_v2")

POSITIVE_LABEL = "known_ransomware_like"
NEGATIVE_LABEL = "benign"

GENERIC_FAMILIES = {
    "", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "good", "goodware", "benign", "ransom", "ransomware",
    "android_ransomware", "static_pe", "malware_api_sequence",
    "unknown_ransomset_family", "unknown", "nan", "none",
}


def canonical_family(value):
    family = str(value or "").strip()

    if family.lower() in GENERIC_FAMILIES or family.lower().startswith("unknown_"):
        return None

    # Akira-<hash> -> Akira
    if "-" in family:
        prefix = family.split("-", 1)[0].strip()
        if prefix:
            return prefix

    return family or None


def stable_score(namespace, sample_id):
    raw = f"{namespace}|{sample_id}".encode("utf-8", errors="ignore")
    return int.from_bytes(
        hashlib.blake2b(raw, digest_size=8).digest(),
        "big",
    )


def prepare_split(dataset, output_root, family, source, train_per_class, test_per_class):
    import pandas as pd
    from symptom_ai.models.evaluation import feature_matrix, related_groups
    from symptom_ai.models.heldout_validation import validate_partitions
    from symptom_ai.dataset_ingestion.validation import validate_labels

    if min(train_per_class, test_per_class) <= 0:
        raise ValueError("Split counts must be positive")
    df = pd.read_csv(dataset)
    df = df.loc[df["dataset_source"].eq(source)].reset_index(drop=True)
    validate_labels(df)
    ids = df["sample_id"]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError("Unique nonmissing sample IDs required; resolve duplicates before splitting")
    families = df["family"].map(canonical_family)
    positive = df["label"].eq(POSITIVE_LABEL)
    if families[positive].isna().any():
        raise ValueError("Positive rows require specific reviewed ransomware family names")
    groups, info = related_groups(df.drop(columns="dataset_source"), feature_matrix(df))
    target_groups = set(groups[positive & families.eq(family)])
    other_groups = set(groups[positive & ~families.eq(family)]) - target_groups
    # Group-level deterministic benign split; all known relationships travel together.
    group_keys = df.assign(_group=groups).groupby("_group")["sample_id"].min().to_dict()
    benign_test_groups = {g for g, key in group_keys.items()
                          if stable_score("benign_split", key) % 10 < 2 and g not in other_groups}
    benign_test_groups |= target_groups
    masks = {
        "train_positive": positive & ~families.eq(family) & pd.Series(groups).isin(other_groups),
        "test_positive": positive & families.eq(family),
        "train_benign": ~positive & ~pd.Series(groups).isin(benign_test_groups),
        "test_benign": ~positive & pd.Series(groups).isin(benign_test_groups),
    }
    parts = {}
    for name, mask in masks.items():
        count = train_per_class if name.startswith("train") else test_per_class
        candidates = df.loc[mask].copy()
        if len(candidates) < count:
            raise ValueError(f"Not enough independent records for {name}: need {count}, found {len(candidates)}")
        order = candidates["sample_id"].map(lambda key: stable_score(name, key)).sort_values().index
        parts[name] = candidates.loc[order[:count]].copy()
    train = pd.concat([parts["train_positive"], parts["train_benign"]], ignore_index=True)
    test = pd.concat([parts["test_positive"], parts["test_benign"]], ignore_index=True)
    integrity = validate_partitions({"train": train, "test": test})
    run_dir = Path(output_root) / f"heldout_{source}_{family}"
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    files_sha256 = {}
    for name, part in parts.items():
        path = data_dir / f"{name}.csv"
        part.to_csv(path, index=False)
        files[name] = str(path)
        files_sha256[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"held_out_family": family, "dataset_source": source,
                "feature_schema_version": 2, "normalization": "fixed_nonnegative_saturation_v1",
                "dataset_sha256": hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
                "selected_counts": {name: len(part) for name, part in parts.items()},
                "files": files, "files_sha256": files_sha256, "split_integrity": integrity, "relationships": info,
                "note": "Explicit within-source family protocol with selected class counts; not a cross-source deployment estimate"}
    (run_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Prepare a within-source family holdout with relationship checks")
    parser.add_argument("--family", required=True)
    parser.add_argument("--source", default="CSU_Ransomware_Data")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--train-per-class", type=int, default=100000)
    parser.add_argument("--test-per-class", type=int, default=20000)
    args = parser.parse_args()
    try:
        manifest = prepare_split(args.dataset, args.output_root, args.family.strip(), args.source.strip(),
                                 args.train_per_class, args.test_per_class)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Holdout preparation failed: {exc}\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
