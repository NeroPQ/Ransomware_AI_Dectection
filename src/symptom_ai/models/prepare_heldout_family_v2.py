from pathlib import Path
import argparse
import csv
import hashlib
import heapq
import json
import sys
from collections import Counter

DATASET = Path("data/symptom_labels/unified_symptom_dataset.csv")
OUTPUT_ROOT = Path("reports/evaluation_v2")

POSITIVE_LABEL = "known_ransomware_like"
NEGATIVE_LABEL = "benign"

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


def keep_smallest(heap, capacity, score, record_no):
    """
    Keep only records with the smallest deterministic hash scores.
    This gives stable sampling without loading the entire CSV into RAM.
    """
    item = (-score, record_no)

    if len(heap) < capacity:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def iter_valid_rows(path):
    csv.field_size_limit(sys.maxsize)

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as f:
        reader = csv.reader(f)

        try:
            header = [x.strip() for x in next(reader)]
        except StopIteration:
            raise SystemExit("Dataset is empty.")

        expected_columns = len(header)
        record_no = 0

        for values in reader:
            record_no += 1

            # Skip malformed rows safely.
            if len(values) != expected_columns:
                continue

            yield record_no, header, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument(
        "--source",
        default="CSU_Ransomware_Data",
    )
    parser.add_argument(
        "--train-per-class",
        type=int,
        default=100000,
    )
    parser.add_argument(
        "--test-per-class",
        type=int,
        default=20000,
    )
    args = parser.parse_args()

    if not DATASET.exists():
        raise SystemExit(f"Missing dataset: {DATASET}")

    target_family = args.family.strip()
    source = args.source.strip()

    capacities = {
        "train_positive": args.train_per_class,
        "train_benign": args.train_per_class,
        "test_positive": args.test_per_class,
        "test_benign": args.test_per_class,
    }

    heaps = {name: [] for name in capacities}
    candidate_counts = Counter()
    seen_sample_ids = set()
    header_ref = None

    print("[1/2] Scanning dataset and selecting samples...")

    for record_no, header, values in iter_valid_rows(DATASET):
        header_ref = header
        row = dict(zip(header, values))

        if (row.get("dataset_source") or "").strip() != source:
            continue

        sample_id = (row.get("sample_id") or "").strip()
        if not sample_id:
            sample_id = f"row_{record_no}"

        # Prevent duplicate sample_id leakage.
        if sample_id in seen_sample_ids:
            continue
        seen_sample_ids.add(sample_id)

        label = (row.get("label") or "").strip()
        family = canonical_family(row.get("family"))

        group = None

        # Test positive = Akira only.
        if label == POSITIVE_LABEL and family == target_family:
            group = "test_positive"

        # Train positive = all ransomware except Akira.
        elif label == POSITIVE_LABEL and family != target_family:
            group = "train_positive"

        # Benign data is split deterministically so train/test never overlap.
        elif label == NEGATIVE_LABEL:
            bucket = stable_score("benign_split", sample_id) % 10
            group = "test_benign" if bucket < 2 else "train_benign"

        if group is None:
            continue

        candidate_counts[group] += 1

        keep_smallest(
            heaps[group],
            capacities[group],
            stable_score(group, sample_id),
            record_no,
        )

    if header_ref is None:
        raise SystemExit("No valid records found.")

    selected = {
        group: {record_no for _, record_no in heap}
        for group, heap in heaps.items()
    }

    for group, capacity in capacities.items():
        actual = len(selected[group])

        if actual < capacity:
            raise SystemExit(
                f"Not enough records for {group}: "
                f"need {capacity}, found {actual}"
            )

    print("\n=== Candidate counts ===")
    for group in capacities:
        print(f"{group}: {candidate_counts[group]:,}")

    print("\n=== Selected counts ===")
    for group in capacities:
        print(f"{group}: {len(selected[group]):,}")

    run_dir = OUTPUT_ROOT / f"heldout_{source}_{target_family}"
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    files = {
        group: data_dir / f"{group}.csv"
        for group in capacities
    }

    record_to_group = {}
    for group, records in selected.items():
        for record_no in records:
            record_to_group[record_no] = group

    print("\n[2/2] Writing selected rows to evaluation CSV files...")

    handles = {}
    writers = {}
    written = Counter()

    try:
        for group, path in files.items():
            handle = open(path, "w", encoding="utf-8", newline="")
            writer = csv.writer(handle)
            writer.writerow(header_ref)

            handles[group] = handle
            writers[group] = writer

        for record_no, _header, values in iter_valid_rows(DATASET):
            group = record_to_group.get(record_no)

            if group:
                writers[group].writerow(values)
                written[group] += 1

    finally:
        for handle in handles.values():
            handle.close()

    manifest = {
        "held_out_family": target_family,
        "dataset_source": source,
        "candidate_counts": dict(candidate_counts),
        "selected_counts": {
            group: len(selected[group])
            for group in capacities
        },
        "written_counts": dict(written),
        "files": {
            group: str(path)
            for group, path in files.items()
        },
        "note": (
            "The held-out ransomware family is excluded from the training-positive set. "
            "Benign train and test samples are deterministically separated."
        ),
    }

    manifest_path = run_dir / "split_manifest.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== HELD-OUT SPLIT CREATED SUCCESSFULLY ===")
    print(f"Target family excluded from training: {target_family}")
    print(f"Dataset source: {source}")

    for group in capacities:
        print(f"{group}: {written[group]:,}")

    print(f"\nManifest saved at: {manifest_path}")


if __name__ == "__main__":
    main()
