from pathlib import Path
import pandas as pd
from symptom_ai.dataset_ingestion.validation import combine_frames

from symptom_ai.dataset_ingestion.ingest_all_available_datasets import (
    OUT_DIR,
    UNIFIED_OUT,
    ingest_mlran,
    ingest_windows_pe_api_calls,
    ingest_riss,
    ingest_silrad,
    ingest_kaggle_ransomware_pe,
    ingest_api_call_sequences,
    ingest_ugransome2024,
    ingest_android_ransomware_detection,
    ingest_cic_malmem2022,
    ingest_bodmas_npz,
    ingest_storage_trace_dataset,
)


def build_unified(parts):
    base_files = [
        OUT_DIR / "csu_symptom_dataset.csv",
        OUT_DIR / "ransomset_symptom_dataset.csv",
    ]

    all_files = []
    for p in base_files + [x for x in parts if x is not None]:
        if p and Path(p).exists():
            all_files.append(Path(p))

    frames = []
    for p in all_files:
        print(f"[+] Loading into unified: {p}")
        frames.append(pd.read_csv(p))

    final = combine_frames(frames)
    final.to_csv(UNIFIED_OUT, index=False)

    print("\n[+] Unified saved:", UNIFIED_OUT)
    print("[+] Rows:", len(final))
    print("[+] Columns:", len(final.columns))
    print("\nDataset counts:")
    print(final["dataset_source"].value_counts().to_string())
    print("\nLabel counts:")
    print(final["label"].value_counts().to_string())


def main():
    parts = []

    parts.append(ingest_mlran())
    parts.append(ingest_windows_pe_api_calls())
    parts.append(ingest_riss())

    parts.append(ingest_silrad())
    parts.append(ingest_kaggle_ransomware_pe())
    parts.append(ingest_api_call_sequences())
    parts.append(ingest_ugransome2024())

    # Android is cross-platform and large, so sampled to 80k rows.
    parts.append(ingest_android_ransomware_detection(max_rows=80000))
    parts.append(ingest_cic_malmem2022(max_rows=120000))
    parts.append(ingest_bodmas_npz(max_rows=120000))

    # Storage traces if actual selected run folders are present.
    parts.append(ingest_storage_trace_dataset("RanSAP", "data/raw/ransap"))
    parts.append(ingest_storage_trace_dataset("RanSMAP", "data/raw/ransmap"))

    try:
        build_unified(parts)
    except ValueError as exc:
        raise SystemExit(f"Dataset validation failed: {exc}") from exc


if __name__ == "__main__":
    main()
