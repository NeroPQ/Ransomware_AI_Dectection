from pathlib import Path
import csv
import json
from statistics import mean

SOURCE = "CSU_Ransomware_Data"
FAMILIES = ["Akira", "LockBit"]
ROOT = Path("reports/evaluation_v2")


def percent(value):
    return round(float(value) * 100, 4)


def main():
    rows = []
    macro = {
        "random_forest": [],
        "xgboost": [],
    }

    for family in FAMILIES:
        path = (
            ROOT
            / f"heldout_{SOURCE}_{family}"
            / "heldout_evaluation.json"
        )

        if not path.exists():
            raise SystemExit(f"Missing report: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        integrity = data["split_integrity"]

        if integrity["train_test_sample_id_overlap"] != 0:
            raise SystemExit(f"Sample-ID overlap found for {family}")

        if integrity["target_family_rows_in_training_positive"] != 0:
            raise SystemExit(f"Held-out family leakage found for {family}")

        if integrity["non_target_rows_in_test_positive"] != 0:
            raise SystemExit(f"Unexpected test family rows found for {family}")

        for model_key, display_name in [
            ("random_forest", "Random Forest"),
            ("xgboost", "XGBoost"),
        ]:
            metrics = data[model_key]
            cm = metrics["confusion_matrix"]

            row = {
                "family": family,
                "model": display_name,
                "accuracy": metrics["accuracy"],
                "precision_ransomware": metrics["precision_ransomware"],
                "recall_ransomware": metrics["recall_ransomware"],
                "f1_ransomware": metrics["f1_ransomware"],
                "false_positive_rate": metrics["false_positive_rate"],
                "false_negative_rate": metrics["false_negative_rate"],
                "tn": cm[0][0],
                "fp": cm[0][1],
                "fn": cm[1][0],
                "tp": cm[1][1],
                "train_rows": data["training_rows"],
                "test_rows": data["test_rows"],
            }

            rows.append(row)
            macro[model_key].append(metrics)

    macro_summary = {}

    for model_key, metric_list in macro.items():
        macro_summary[model_key] = {
            "accuracy": mean(x["accuracy"] for x in metric_list),
            "precision_ransomware": mean(
                x["precision_ransomware"] for x in metric_list
            ),
            "recall_ransomware": mean(
                x["recall_ransomware"] for x in metric_list
            ),
            "f1_ransomware": mean(
                x["f1_ransomware"] for x in metric_list
            ),
            "false_positive_rate": mean(
                x["false_positive_rate"] for x in metric_list
            ),
            "false_negative_rate": mean(
                x["false_negative_rate"] for x in metric_list
            ),
        }

    summary = {
        "protocol": (
            "Leave-one-family-out evaluation using CSU_Ransomware_Data. "
            "The target ransomware family is excluded from training "
            "positives and evaluated only in the held-out test set."
        ),
        "source": SOURCE,
        "families": FAMILIES,
        "per_family_results": rows,
        "macro_average_across_family_runs": macro_summary,
        "integrity_checks": {
            "all_runs_have_zero_train_test_sample_id_overlap": True,
            "all_runs_have_zero_target_family_rows_in_training_positive": True,
            "all_runs_have_zero_non_target_rows_in_test_positive": True,
        },
        "important_caveat": (
            "The Akira and LockBit runs use the same deterministic benign "
            "test subset. Therefore, the two runs must not be interpreted "
            "as 40,000 independent benign samples."
        ),
        "interpretation": (
            "XGBoost achieved higher held-out ransomware recall across the "
            "tested families, while Random Forest had slightly lower false "
            "positive rates."
        ),
    }

    json_path = ROOT / "heldout_family_summary.json"
    csv_path = ROOT / "heldout_family_summary.csv"

    json_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    fields = [
        "family",
        "model",
        "accuracy",
        "precision_ransomware",
        "recall_ransomware",
        "f1_ransomware",
        "false_positive_rate",
        "false_negative_rate",
        "tn",
        "fp",
        "fn",
        "tp",
        "train_rows",
        "test_rows",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("=== HELD-OUT FAMILY SUMMARY ===")

    for row in rows:
        print(
            f"{row['family']:8} | "
            f"{row['model']:13} | "
            f"Recall={percent(row['recall_ransomware']):7.3f}% | "
            f"FPR={percent(row['false_positive_rate']):6.3f}% | "
            f"F1={percent(row['f1_ransomware']):7.3f}%"
        )

    print("\n=== MACRO AVERAGE ACROSS FAMILY RUNS ===")

    for model_key, metrics in macro_summary.items():
        print(
            f"{model_key:13} | "
            f"Recall={percent(metrics['recall_ransomware']):7.3f}% | "
            f"FPR={percent(metrics['false_positive_rate']):6.3f}% | "
            f"F1={percent(metrics['f1_ransomware']):7.3f}%"
        )

    print(f"\nJSON summary: {json_path}")
    print(f"CSV summary:  {csv_path}")


if __name__ == "__main__":
    main()
