from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import re

from symptom_ai.live_monitoring.live_log_ai_watcher import (
    HIGH_IMPACT_EVENTS,
    aggregate_symptoms,
    call_ai,
    decide_alert_type,
)

RULE_IMPACT_EVENTS = {
    "file_rename",
    "extension_change",
    "high_entropy_write",
    "ransom_note_created",
    "storage_write_spike",
}


def load_events(path):
    events = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line:
            continue

        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return events


def event_index(event, fallback):
    try:
        return int(event.get("scenario_event_index", fallback) or fallback)
    except (TypeError, ValueError):
        return fallback


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def is_xgb_ransomware(ai_result):
    return str(
        ai_result.get("predicted_label", "")
    ).strip().lower() == "known_ransomware_like"


def is_isolation_forest_anomaly(ai_result):
    prediction = ai_result.get(
        "isolation_forest_prediction",
        ai_result.get("anomaly_prediction"),
    )

    prediction_text = str(prediction).strip().lower()

    return (
        prediction_text in {"-1", "anomaly", "outlier"}
        or ai_result.get("unknown_risk") == "high"
    )


def is_rule_detected(damage):
    counts = damage.get("event_type_counts", {}) or {}

    impact_count = sum(
        int(counts.get(event_name, 0) or 0)
        for event_name in RULE_IMPACT_EVENTS
    )

    return bool(
        damage.get("has_file_impact")
        and impact_count >= 3
    )


def first_high_impact_event(events):
    for fallback, event in enumerate(events, start=1):
        if event.get("event_type") in HIGH_IMPACT_EVENTS:
            return event_index(event, fallback)

    return None


def clean_case_id(value):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


def build_summary(name, detections, rows, truth, first_impact):
    first_detection_event = None
    first_detection_window = None

    for row, detected in zip(rows, detections):
        if detected:
            first_detection_event = row["window_last_event_index"]
            first_detection_window = row["window_number"]
            break

    summary = {
        "method": name,
        "total_windows": len(rows),
        "detected_windows": sum(detections),
        "first_detection_window": first_detection_window,
        "first_detection_event_index": first_detection_event,
        "first_high_impact_event_index": first_impact,
        "detection_lead_events": None,
    }

    if (
        first_detection_event is not None
        and first_impact is not None
    ):
        summary["detection_lead_events"] = (
            first_impact - first_detection_event
        )

    if truth == "benign":
        summary["false_positive_windows"] = sum(detections)
        summary["false_positive_window_rate"] = round(
            sum(detections) / len(rows),
            4,
        ) if rows else 0.0
    else:
        summary["scenario_detected"] = any(detections)

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--events",
        default="data/custom_sandbox/events/events.jsonl",
    )

    parser.add_argument(
        "--truth",
        choices=["benign", "ransomware"],
        required=True,
    )

    parser.add_argument(
        "--case-id",
        required=True,
    )

    parser.add_argument(
        "--window-events",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--output-dir",
        default="reports/evaluation/ablation_runs",
    )

    args = parser.parse_args()

    event_path = Path(args.events)

    if not event_path.exists():
        raise SystemExit(f"ERROR: Event log not found: {event_path}")

    events = load_events(event_path)

    if len(events) < args.window_events:
        raise SystemExit(
            f"ERROR: Need at least {args.window_events} events, "
            f"but found {len(events)}."
        )

    usable_count = (
        len(events) // args.window_events
    ) * args.window_events

    usable_events = events[:usable_count]
    first_impact = first_high_impact_event(usable_events)

    rows = []
    rule_results = []
    xgb_results = []
    xgb_if_results = []
    hybrid_results = []

    for window_number, window_events in enumerate(
        chunks(usable_events, args.window_events),
        start=1,
    ):
        symptoms, damage = aggregate_symptoms(window_events)
        ai_result = call_ai(symptoms)

        if "api_error" in ai_result:
            raise SystemExit(
                "ERROR: Cannot reach FastAPI /predict endpoint.\n"
                f"Details: {ai_result.get('api_error')}"
            )

        response = (
            ai_result.get("response")
            or ai_result.get("recommended_response")
            or {}
        )

        policy = response.get("policy")
        unknown_risk = ai_result.get("unknown_risk")

        rule_hit = is_rule_detected(damage)
        xgb_hit = is_xgb_ransomware(ai_result)
        iso_hit = is_isolation_forest_anomaly(ai_result)
        xgb_if_hit = xgb_hit or iso_hit

        hybrid_alert = decide_alert_type(
            ai_result,
            damage,
            policy,
            unknown_risk,
        )

        # A missed_detection_infected result means impact happened
        # without a successful actionable detection.
        hybrid_hit = hybrid_alert not in {
            "monitor",
            "api_error_needs_review",
            "missed_detection_infected",
        }

        last_event_index = event_index(
            window_events[-1],
            window_number * args.window_events,
        )

        rows.append({
            "window_number": window_number,
            "window_last_event_index": last_event_index,
            "predicted_label": ai_result.get("predicted_label"),
            "risk_score": ai_result.get("risk_score"),
            "unknown_risk": unknown_risk,
            "isolation_forest_prediction": ai_result.get(
                "isolation_forest_prediction",
                ai_result.get("anomaly_prediction"),
            ),
            "policy": policy,
            "has_file_impact": damage.get("has_file_impact"),
            "high_impact_event_count": damage.get(
                "high_impact_event_count"
            ),
            "rule_only_detected": rule_hit,
            "xgboost_only_detected": xgb_hit,
            "xgboost_plus_isolation_forest_detected": xgb_if_hit,
            "full_hybrid_alert_type": hybrid_alert,
            "full_hybrid_detected": hybrid_hit,
        })

        rule_results.append(rule_hit)
        xgb_results.append(xgb_hit)
        xgb_if_results.append(xgb_if_hit)
        hybrid_results.append(hybrid_hit)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_id = clean_case_id(args.case_id)

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "safety_note": (
            "This evaluator reads safe JSONL simulation logs only. "
            "It does not create STOP_SIGNAL or trigger containment."
        ),
        "case_id": case_id,
        "truth": args.truth,
        "source_event_log": str(event_path),
        "source_event_log_sha256": hashlib.sha256(
            event_path.read_bytes()
        ).hexdigest(),
        "window_events": args.window_events,
        "total_events_in_log": len(events),
        "events_evaluated": len(usable_events),
        "ignored_tail_events": len(events) - len(usable_events),
        "first_high_impact_event_index": first_impact,
        "method_summaries": {
            "rule_only": build_summary(
                "rule_only",
                rule_results,
                rows,
                args.truth,
                first_impact,
            ),
            "xgboost_only": build_summary(
                "xgboost_only",
                xgb_results,
                rows,
                args.truth,
                first_impact,
            ),
            "xgboost_plus_isolation_forest": build_summary(
                "xgboost_plus_isolation_forest",
                xgb_if_results,
                rows,
                args.truth,
                first_impact,
            ),
            "full_hybrid": build_summary(
                "full_hybrid",
                hybrid_results,
                rows,
                args.truth,
                first_impact,
            ),
        },
        "windows": rows,
    }

    output_path = output_dir / f"{case_id}.json"

    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(f"Saved: {output_path}")
    print("\n=== ABLATION SUMMARY ===")

    for method, summary in result["method_summaries"].items():
        print(
            f"{method}: "
            f"detected_windows={summary['detected_windows']}/"
            f"{summary['total_windows']} | "
            f"first_event={summary['first_detection_event_index']} | "
            f"lead_events={summary['detection_lead_events']}"
        )

        if args.truth == "benign":
            print(
                "  false_positive_windows="
                f"{summary['false_positive_windows']}"
            )
        else:
            print(
                "  scenario_detected="
                f"{summary['scenario_detected']}"
            )


if __name__ == "__main__":
    main()
