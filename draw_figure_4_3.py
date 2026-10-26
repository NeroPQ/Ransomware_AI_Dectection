from pathlib import Path
import json
import matplotlib.pyplot as plt


REPORT_PATH = Path(
    "reports/evaluation_v2/isolation_forest_benign_only/"
    "isolation_forest_benign_only.json"
)

OUTPUT_DIR = Path("reports/evaluation_v2/figures")
OUTPUT_PATH = OUTPUT_DIR / "figure_4_3_if_calibration_tradeoff.png"


def main():
    data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    rows = sorted(
        data["threshold_results"],
        key=lambda row: row["target_calibration_fpr"],
    )

    x_values = [
        row["independent_benign_test_fpr"] * 100
        for row in rows
    ]

    y_values = [
        row["macro_family_unknown_detection_recall"] * 100
        for row in rows
    ]

    target_labels = [
        f"Target FPR {row['target_calibration_fpr'] * 100:.0f}%"
        for row in rows
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        x_values,
        y_values,
        marker="o",
        linewidth=2,
    )

    for x_value, y_value, label in zip(
        x_values,
        y_values,
        target_labels,
    ):
        ax.annotate(
            label,
            (x_value, y_value),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
        )

    ax.set_xlabel("Independent Benign Test False-Positive Rate (%)")
    ax.set_ylabel(
        "Macro Held-Out Family Unknown Detection Recall (%)"
    )

    ax.set_title(
        "Benign-Only Isolation Forest Calibration Trade-Off"
    )

    ax.grid(alpha=0.3)

    fig.text(
        0.5,
        0.01,
        "Isolation Forest is trained only on benign samples. "
        "Akira and LockBit use the same benign test subset.",
        ha="center",
        fontsize=8,
    )

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
