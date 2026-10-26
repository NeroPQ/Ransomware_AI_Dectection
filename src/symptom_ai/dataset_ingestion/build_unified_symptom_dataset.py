from pathlib import Path
import pandas as pd
from symptom_ai.dataset_ingestion.validation import combine_frames

OUT = Path("data/symptom_labels/unified_symptom_dataset.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

INPUTS = [
    Path("data/symptom_labels/csu_symptom_dataset.csv"),
    Path("data/symptom_labels/ransomset_symptom_dataset.csv"),
]

def main():
    frames = []

    for p in INPUTS:
        if p.exists():
            print(f"[+] Loading {p}")
            frames.append(pd.read_csv(p))
        else:
            print(f"[!] Missing {p}")

    if not frames:
        raise SystemExit("[!] No symptom datasets found.")

    final = combine_frames(frames)
    final.to_csv(OUT, index=False)

    print(f"\n[+] Unified symptom dataset saved to {OUT}")
    print(f"[+] Rows: {len(final)}")
    print(f"[+] Columns: {len(final.columns)}")
    print("\nDataset source counts:")
    print(final["dataset_source"].value_counts().to_string())
    print("\nLabel counts:")
    print(final["label"].value_counts().to_string())

if __name__ == "__main__":
    main()
