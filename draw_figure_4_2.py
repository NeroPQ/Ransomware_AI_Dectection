from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


SUMMARY_PATH = Path(
    "reports/evaluation_v2/heldout_family_summary.json"
)

OUTPUT_DIR = Path("reports/evaluation_v2/figures")
OUTPUT_PATH = OUTPUT_DIR / "figure_4_2_heldout_family_recall.png"

FAMILIES = ["Akira", "LockBit"]
MODELS = ["Random Forest", "XGBoost"]


def main():
    data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    lookup = {
        (row["family"], row["model"]): row
        for row in data["per_family_results"]
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(FAMILIES))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9, 6))

    all_bars = []

    for index, model in enumerate(MODELS):
        recalls = [
            lookup[(family, model)]["recall_ransomware"] * 100
            for family in FAMILIES
        ]

        offset = (-width / 2) if index == 0 else (width / 2)

        bars = ax.bar(
            x + offset,
            recalls,
            width,
            label=model,
        )

        all_bars.append((model, bars))

    for model, bars in all_bars:
        for family, bar in zip(FAMILIES, bars):
            row = lookup[(family, model)]

            recall = row["recall_ransomware"] * 100
            fpr = row["false_positive_rate"] * 100

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                recall + 0.12,
                f"{recall:.2f}%\nFPR {fpr:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(FAMILIES)

    ax.set_ylim(90, 100)
    ax.set_ylabel("Held-Out Ransomware Recall (%)")
    ax.set_xlabel("Ransomware Family Excluded from Training")
    ax.set_title(
        "Held-Out Ransomware-Family Detection Recall\n"
        "(False-Positive Rate shown above each bar)"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.text(
        0.5,
        0.01,
        "Protocol: each target family was excluded from training positives. "
        "Akira and LockBit runs share the same deterministic benign test subset.",
        ha="center",
        fontsize=8,
    )

    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
