from pathlib import Path
from datetime import datetime
import csv
import json


RUN_DIR = Path("reports/evaluation/ablation_runs")
OUT_DIR = Path("reports/evaluation")

CASES = [
    {
        "case_id": "software_update_like",
        "truth": "benign",
    },
    {
        "case_id": "known_progressive",
        "truth": "ransomware",
    },
    {
        "case_id": "novel_zero_day",
        "truth": "ransomware",
    },
]

METHOD_FIELDS = {
    "rule_only": "rule_only_detected",
    "xgboost_only": "xgboost_only_detected",
    "xgboost_plus_isolation_forest": (
        "xgboost_plus_isolation_forest_detected"
    ),
    "full_hybrid": "full_hybrid_detected",
}


def safe_divide(numerator, denominator):
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def load_case(case_id):
    path = RUN_DIR / f"{case_id}.json"

    if not path.exists():
        raise SystemExit(f"ERROR: Missing ablation result: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    loaded_cases = []

    for case in CASES:
        result = load_case(case["case_id"])
        loaded_cases.append({
            "case_id": case["case_id"],
            "truth": case["truth"],
            "result": result,
        })

    method_rows = []

    for method_name, field_name in METHOD_FIELDS.items():
        tp = fp = tn = fn = 0

        ransomware_scenarios = 0
        ransomware_scenarios_detected = 0

        benign_scenarios = 0
        benign_scenarios_alerted = 0

        lead_times = []
        per_case = []

        for case in loaded_cases:
            case_id = case["case_id"]
            truth = case["truth"]
            result = case["result"]

            windows = result.get("windows", [])
            detections = [
                bool(window.get(field_name))
                for window in windows
            ]

            detected_windows = sum(detections)
            total_windows = len(detections)

            method_summary = (
                result.get("method_summaries", {})
                .get(method_name, {})
            )

            lead_time = method_summary.get(
                "detection_lead_events"
            )

            if truth == "ransomware":
                ransomware_scenarios += 1

                if any(detections):
                    ransomware_scenarios_detected += 1

                tp += detected_windows
                fn += total_windows - detected_windows

                if lead_time is not None and lead_time >= 0:
                    lead_times.append(lead_time)

            else:
                benign_scenarios += 1

                if any(detections):
                    benign_scenarios_alerted += 1

                fp += detected_windows
                tn += total_windows - detected_windows

            per_case.append({
                "case_id": case_id,
                "truth": truth,
                "total_windows": total_windows,
                "detected_windows": detected_windows,
                "first_detection_event_index": (
                    method_summary.get(
                        "first_detection_event_index"
                    )
                ),
                "first_high_impact_event_index": (
                    method_summary.get(
                        "first_high_impact_event_index"
                    )
                ),
                "detection_lead_events": lead_time,
            })

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)

        f1 = None
        if precision is not None and recall is not None:
            if precision + recall > 0:
                f1 = round(
                    2 * precision * recall / (precision + recall),
                    4,
                )
            else:
                f1 = 0.0

        method_rows.append({
            "method": method_name,
            "window_level": {
                "true_positive_windows": tp,
                "false_positive_windows": fp,
                "true_negative_windows": tn,
                "false_negative_windows": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positive_rate": safe_divide(fp, fp + tn),
            },
            "scenario_level": {
                "ransomware_scenarios": ransomware_scenarios,
                "ransomware_scenarios_detected": (
                    ransomware_scenarios_detected
                ),
                "scenario_detection_rate": safe_divide(
                    ransomware_scenarios_detected,
                    ransomware_scenarios,
                ),
                "benign_scenarios": benign_scenarios,
                "benign_scenarios_alerted": benign_scenarios_alerted,
                "scenario_false_positive_rate": safe_divide(
                    benign_scenarios_alerted,
                    benign_scenarios,
                ),
            },
            "lead_time": {
                "positive_lead_times": lead_times,
                "mean_positive_lead_events": (
                    round(sum(lead_times) / len(lead_times), 4)
                    if lead_times else None
                ),
            },
            "per_case": per_case,
        })

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation_scope": (
            "Safe simulated scenario ablation only. "
            "These are not held-out family test-set metrics."
        ),
        "window_label_definition": (
            "All windows from ransomware-labelled scenarios are counted "
            "as ransomware windows for this experimental operational "
            "comparison."
        ),
        "benign_case": "software_update_like",
        "ransomware_cases": [
            "known_progressive",
            "novel_zero_day",
        ],
        "methods": method_rows,
    }

    json_out = OUT_DIR / "ablation_summary.json"
    json_out.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    csv_out = OUT_DIR / "ablation_summary.csv"

    with csv_out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "tp_windows",
                "fp_windows",
                "tn_windows",
                "fn_windows",
                "precision",
                "recall",
                "f1",
                "fpr",
                "scenario_detection_rate",
                "scenario_false_positive_rate",
                "mean_positive_lead_events",
            ],
        )

        writer.writeheader()

        for row in method_rows:
            window = row["window_level"]
            scenario = row["scenario_level"]
            lead = row["lead_time"]

            writer.writerow({
                "method": row["method"],
                "tp_windows": window["true_positive_windows"],
                "fp_windows": window["false_positive_windows"],
                "tn_windows": window["true_negative_windows"],
                "fn_windows": window["false_negative_windows"],
                "precision": window["precision"],
                "recall": window["recall"],
                "f1": window["f1"],
                "fpr": window["false_positive_rate"],
                "scenario_detection_rate": (
                    scenario["scenario_detection_rate"]
                ),
                "scenario_false_positive_rate": (
                    scenario["scenario_false_positive_rate"]
                ),
                "mean_positive_lead_events": (
                    lead["mean_positive_lead_events"]
                ),
            })

    print(f"Saved: {json_out}")
    print(f"Saved: {csv_out}")
    print("\n=== OFFICIAL ABLATION SUMMARY ===")

    for row in method_rows:
        window = row["window_level"]
        scenario = row["scenario_level"]
        lead = row["lead_time"]

        print(
            f"{row['method']}: "
            f"Recall={window['recall']} | "
            f"Precision={window['precision']} | "
            f"F1={window['f1']} | "
            f"FPR={window['false_positive_rate']} | "
            f"ScenarioDetection={scenario['scenario_detection_rate']} | "
            f"MeanLead={lead['mean_positive_lead_events']}"
        )


if __name__ == "__main__":
    main()
