from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import re

from symptom_ai.live_monitoring.live_log_ai_watcher import (
    aggregate_symptoms,
    call_ai,
    EARLY_DETECTION_ALERT_TYPES,
    policy_alert_type,
)

from symptom_ai.response_engine.evidence_policy import detector_evidence, decide_policy
from symptom_ai.telemetry.behavior import event_has_possible_impact


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


def is_classifier_ransomware(ai_result):
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
    )


def is_rule_detected(damage, symptoms=None):
    return detector_evidence(symptoms or {}, damage)["rule"]["hit"]


def first_high_impact_event(events):
    for fallback, event in enumerate(events, start=1):
        if event_has_possible_impact(event):
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

    if args.window_events <= 0:
        parser.error("window-events must be positive")

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
    classifier_results = []
    classifier_if_results = []
    hybrid_results = []
    anomaly_results = []

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

        classifier_hit = is_classifier_ransomware(ai_result)
        iso_hit = is_isolation_forest_anomaly(ai_result)
        classifier_if_hit = classifier_hit or iso_hit

        detectors = detector_evidence(symptoms, damage, ai_result)
        response = decide_policy(symptoms, damage, detectors)
        policy = response["policy"]
        rule_hit = detectors["rule"]["hit"]
        hybrid_alert = policy_alert_type(response, ai_result)

        # An impact-only review is not a successful detector decision.
        hybrid_hit = hybrid_alert in EARLY_DETECTION_ALERT_TYPES

        last_event_index = event_index(
            window_events[-1],
            window_number * args.window_events,
        )

        rows.append({
            "window_number": window_number,
            "window_last_event_index": last_event_index,
            "predicted_label": ai_result.get("predicted_label"),
            "model_used": ai_result.get("model_used"),
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
            "policy_stage": response["stage"],
            "detector_evidence": detectors,
            "anomaly_only_detected": iso_hit,
            "rule_only_detected": rule_hit,
            "classifier_only_detected": classifier_hit,
            "classifier_plus_isolation_forest_detected": classifier_if_hit,
            "full_hybrid_alert_type": hybrid_alert,
            "full_hybrid_detected": hybrid_hit,
        })

        anomaly_results.append(iso_hit)
        rule_results.append(rule_hit)
        classifier_results.append(classifier_hit)
        classifier_if_results.append(classifier_if_hit)
        hybrid_results.append(hybrid_hit)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_id = clean_case_id(args.case_id)

    result = {
        "alert_schema_version": 3,
        "feature_semantics_version": 3,
        "classifier_note": "Raw classifier/anomaly evidence uses the recorded active artifact; incompatible v2 semantics are shadow-only in hybrid policy",
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
            "anomaly_only": build_summary("anomaly_only", anomaly_results, rows, args.truth, first_impact),
            "rule_only": build_summary(
                "rule_only",
                rule_results,
                rows,
                args.truth,
                first_impact,
            ),
            "classifier_only": build_summary(
                "classifier_only",
                classifier_results,
                rows,
                args.truth,
                first_impact,
            ),
            "classifier_plus_isolation_forest": build_summary(
                "classifier_plus_isolation_forest",
                classifier_if_results,
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
