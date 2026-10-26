from pathlib import Path
from datetime import datetime
import argparse
import json
import math
import os
import logging
from uuid import uuid4
from symptom_ai.auto_learning.review import is_approved
from symptom_ai.models.evaluation import OUTCOME_FEATURES
from symptom_ai.response_engine.evidence_policy import detector_evidence, decide_policy
from symptom_ai.telemetry.behavior import event_has_possible_impact, normalize_event
import time
import urllib.request
from collections import Counter


EVENT_LOG = Path("data/custom_sandbox/events/events.jsonl")
REPORT_DIR = Path("reports/live_monitoring")
QUEUE_DIR = Path("data/learning_queue")
LEARNED_SIGNATURES = QUEUE_DIR / "learned_behavior_signatures.json"
ERROR_DIR = Path("reports/live_monitoring/error_cases")
CONTROL_DIR = Path("data/custom_sandbox/control")
STOP_SIGNAL = CONTROL_DIR / "STOP_SIGNAL.json"

API_BASE_URL = os.getenv("SYMPTOM_AI_API_BASE_URL", "http://localhost:8000").rstrip("/")
API_URL = API_BASE_URL + "/predict"
API_EXPLAIN_URL = API_BASE_URL + "/explain"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
QUEUE_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)
CONTROL_DIR.mkdir(parents=True, exist_ok=True)

HIGH_IMPACT_EVENTS = {
    "high_entropy_write",
    "ransom_note_created",

    # Safe zero-day simulation impact events
    "stealth_file_lock",
    "stealth_content_scramble",
    "stealth_note_marker",
}



EARLY_DETECTION_ALERT_TYPES = {"analyst_alert", "containment_recommended"}

# Keeps cumulative evidence for each safe simulation scenario.
SCENARIO_EARLY_DETECTION_STATE = {}

def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def read_new_lines(path: Path, offset: int):
    if not path.exists():
        return [], offset

    events = []
    with path.open("r", encoding="utf-8") as stream:
        stream.seek(offset)
        while True:
            position = stream.tell()
            line = stream.readline()
            if not line:
                return events, stream.tell()
            if not line.endswith("\n"):
                return events, position
            try:
                event = json.loads(line)
                if not isinstance(event, dict) or not (isinstance(event.get("event_type"), str) or "schema_version" in event):
                    raise ValueError("Event requires event_type")
                normalize_event(event)
                events.append(event)
            except (ValueError, TypeError):
                logging.getLogger(__name__).warning("Invalid event record skipped at offset %s", position)


def clamp01(x):
    value = float(x)
    return max(0.0, min(value, 1.0)) if math.isfinite(value) else 0.0


def aggregate_symptoms(events):
    from symptom_ai.telemetry.behavior import analyze_events
    return analyze_events(events)


def call_ai(symptoms):
    payload = json.dumps({"symptoms": symptoms, "feature_semantics_version": 3}).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json", "X-API-Key": os.getenv("SYMPTOM_AI_API_KEY", "")},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {
            "api_error": str(e),
            "note": "Make sure FastAPI is running at http://localhost:8000",
        }


def call_explain(symptoms):
    payload = json.dumps({"symptoms": symptoms, "feature_semantics_version": 3}).encode("utf-8")
    req = urllib.request.Request(
        API_EXPLAIN_URL,
        data=payload,
        headers={"Content-Type": "application/json", "X-API-Key": os.getenv("SYMPTOM_AI_API_KEY", "")},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {
            "explain_error": str(e),
            "note": "Explain endpoint is not ready or API is unavailable."
        }


def build_spread_report(events, damage):
    host_counts = Counter(e.get("host") for e in events if e.get("host"))
    path_counts = Counter(e.get("path") for e in events if e.get("path"))

    impacted_dirs = set(damage.get("impacted_directories", []))
    high_impact_paths = []
    for e in events:
        if event_has_possible_impact(e):
            high_impact_paths.append({"event_type": e.get("operation", e.get("event_type")),
                                      "host": e.get("host"), "path": e.get("path"), "timestamp": e.get("timestamp")})

    return {
        "spread_scope": {
            "host_count": damage.get("host_count", 0),
            "hosts": damage.get("hosts", []),
            "unique_path_count": damage.get("unique_path_count", 0),
            "impacted_directory_count": len(impacted_dirs),
            "high_impact_event_count": damage.get("high_impact_event_count", 0),
        },
        "host_event_counts": dict(host_counts),
        "top_touched_paths": [
            {"path": path, "count": count}
            for path, count in path_counts.most_common(10)
        ],
        "impacted_directories": sorted(impacted_dirs)[:20],
        "high_impact_paths": high_impact_paths[:20],
    }


def extract_dataset_evidence(explain_result):
    if not explain_result or "explain_error" in explain_result:
        return {
            "status": "explain_unavailable",
            "detail": (
                explain_result.get("explain_error")
                if isinstance(explain_result, dict)
                else None
            ),
            "top_matches": []
        }

    decision = explain_result.get("decision_explanation", {})

    rows = (
        decision.get("evidence_training_rows")
        or decision.get("nearest_training_rows")
        or []
    )

    top_matches = []

    for record in rows[:5]:
        top_matches.append({
            "rank": record.get("rank"),
            "dataset_source": record.get("matched_dataset_source"),
            "sample_id": record.get("matched_sample_id"),
            "family": record.get("matched_family"),
            "behavior_type": record.get("matched_behavior_type"),
            "label": record.get("matched_label"),
            "response_policy": record.get("matched_response_policy"),
            "evidence_score": record.get(
                "evidence_score",
                record.get("similarity_score"),
            ),
            "shared_active_symptoms": record.get(
                "shared_active_symptoms",
                [],
            ),
        })

    return {
        "status": "ok",
        "decision": decision.get("decision"),
        "rule_reason": decision.get("explanation", {}).get("rule_reason"),
        "top_match_reason": decision.get("explanation", {}).get(
            "top_match_reason"
        ),
        "matched_label_counts_top5": decision.get(
            "matched_label_counts_top5",
            {},
        ),
        "top_matches": top_matches,
    }

def build_remediation(response, alert_type, damage):
    actions = list(response.get("recommended_actions", []) or [])

    if damage.get("has_file_impact"):
        actions.append("Preserve possible impact evidence for analyst review")
    if alert_type == "file_impact_needs_review":
        actions.append("Queue case for explicit analyst labeling before any learning")

    # deduplicate while preserving order
    seen = set()
    clean = []
    for a in actions:
        if a not in seen:
            clean.append(a)
            seen.add(a)

    return clean[:20]


def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def save_learning_case(window_id, events, symptoms, damage, ai_result, reason):
    """
    Save a suspicious safe-sandbox case for analyst review.

    This function does not create a learned signature and does not
    trigger automatic retraining.
    """
    out = QUEUE_DIR / f"learning_case_{window_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    case = {
        "window_id": window_id,
        "saved_at": now_iso(),
        "reason": reason,
        "events": events,
        "symptoms": symptoms,
        "damage": damage,
        "ai_result": ai_result,
        "label": None,
        "family": "unknown_ransomware_candidate",
        "response_policy": ai_result.get("response", {}).get("policy", "analyst_review"),
        "learning_status": "pending_review",
        "confirmed_label": None,
        "feature_semantics_version": 3,
        "safety_note": "Observation-only evaluation; no endpoint response was executed.",
    }

    out.write_text(json.dumps(case, indent=2), encoding="utf-8")
    return str(out)

def save_error_case(window_id, events, symptoms, damage, ai_result):
    out = ERROR_DIR / f"api_error_case_{window_id}.json"
    case = {
        "window_id": window_id,
        "saved_at": now_iso(),
        "events": events,
        "symptoms": symptoms,
        "damage": damage,
        "ai_result": ai_result,
        "status": "needs_api_debug",
    }
    out.write_text(json.dumps(case, indent=2), encoding="utf-8")
    return str(out)


def load_learned_signatures():
    # Legacy automatically generated signature files are deliberately ignored.
    signatures = []
    for path in sorted(QUEUE_DIR.glob("learning_case_*.json")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            if is_approved(case) and case["review"]["confirmed_label"] == "known_ransomware_like":
                signatures.append({"signature_id": path.stem, "case": case})
        except (ValueError, TypeError, OSError):
            logging.getLogger(__name__).warning("Invalid learning case skipped: %s", path.name)
    return signatures


def match_learned_signature(events):
    symptoms, damage = aggregate_symptoms(events)
    current_counts = Counter(normalize_event(e)["operation"] for e in events)
    signature_events = {"file_modify", "file_scramble", "file_lock", "file_extension_change", "ransom_note"}
    if damage["high_impact_event_count"] < 3 or len(set(current_counts) & signature_events) < 2:
        return None
    for signature in load_learned_signatures():
        reference_events = signature["case"].get("events") or []
        if not reference_events:
            continue
        reference, reference_damage = aggregate_symptoms(reference_events)
        counts = Counter(normalize_event(e)["operation"] for e in reference_events)
        if reference_damage["high_impact_event_count"] < 3 or len(set(counts) & signature_events) < 2:
            continue
        # Match observable rates and symptom magnitudes; scenario_type is never used.
        types = set(counts) | set(current_counts)
        rate_distance = sum(abs(counts[t] / len(reference_events) - current_counts[t] / len(events)) for t in types)
        keys = set(symptoms) - OUTCOME_FEATURES
        feature_distance = max(abs(symptoms[k] - reference[k]) for k in keys)
        if rate_distance <= 0.2 and feature_distance <= 0.15:
            return {"matched": True, "signature_id": signature["signature_id"],
                    "reason": "Observable rates and symptoms match a reviewed case; still requires investigation",
                    "shared_event_types": sorted(set(counts) & set(current_counts))}
    return None

def decide_alert_type(ai_result, damage, policy=None, unknown_risk=None):
    """Old function signature delegates to the same canonical policy engine.

    Historical policy/risk strings cannot authorize escalation. Task 02 replay
    keeps its old behavior in the isolated benchmark snapshot.
    """
    symptoms = ai_result.get("symptoms", {})
    observations = dict(damage)
    if "evidence_categories" not in observations:
        observations["evidence_categories"] = {"observed_activity": [], "suspicious_behavior": [],
            "possible_impact": ["untyped_impact_assertion"] if damage.get("has_file_impact") else [],
            "corroborated_high_confidence_detection": []}
    evidence = detector_evidence(symptoms, observations, ai_result)
    return policy_alert_type(decide_policy(symptoms, observations, evidence), ai_result)


def policy_alert_type(response, ai_result):
    if "api_error" in ai_result and response["stage"] in {"monitor", "suspicious_activity"}:
        return "api_error_needs_review"
    return {"monitor": "monitor", "suspicious_activity": "evidence_collection",
            "analyst_alert": "analyst_alert", "containment_recommendation": "containment_recommended"}[response["stage"]]

def write_stop_signal(window_id, alert_type, policy, damage):
    """
    Write a safe-sandbox containment stop signal.

    This does not stop a real host. It only tells the safe demo runner
    to stop generating additional simulated events.
    """
    if STOP_SIGNAL.exists():
        return str(STOP_SIGNAL)

    signal = {
        "time": now_iso(),
        "window_id": window_id,
        "alert_type": alert_type,
        "policy": policy,
        "reason": "AI watcher detected file impact and containment policy. Safe sandbox should stop generating events.",
        "damage": damage,
        "signal_type": "simulated_containment_stop",
        "safety_scope": "safe_sandbox_only",
        "triggered_after_file_impact": bool(damage.get("has_file_impact")),
        "response_mode": "simulation",
        "containment_executed": False,
        "simulated_stop_requested": True,
    }

    STOP_SIGNAL.parent.mkdir(parents=True, exist_ok=True)
    STOP_SIGNAL.write_text(
        json.dumps(signal, indent=2),
        encoding="utf-8",
    )

    return str(STOP_SIGNAL)

def scenario_event_index(event):
    try:
        return int(event.get("scenario_event_index", 0) or 0)
    except (TypeError, ValueError):
        return 0


def update_early_detection_metrics(events, alert_type):
    """
    Track alert timing and file-impact timing for each safe scenario.
    """
    scenario_ids = sorted({
        str(event.get("scenario"))
        for event in events
        if event.get("scenario")
    })

    if len(scenario_ids) != 1:
        return {
            "status": "unavailable_mixed_or_missing_scenario",
            "first_alert_event_index": None,
            "first_high_impact_event_index": None,
            "detection_lead_events": None,
            "high_impact_events_before_first_alert": None,
            "affected_paths_before_first_alert": None,
        }

    scenario_id = scenario_ids[0]

    state = SCENARIO_EARLY_DETECTION_STATE.setdefault(
        scenario_id,
        {
            "last_event_index": 0,
            "high_impact_events_seen": 0,
            "affected_paths_seen": set(),
            "first_high_impact_event_index": None,
            "first_alert_event_index": None,
            "high_impact_events_before_first_alert": None,
            "affected_paths_before_first_alert": None,
        },
    )

    indexed_events = sorted(
        events,
        key=lambda event: scenario_event_index(event),
    )

    for event in indexed_events:
        try:
            event_index = int(event.get("scenario_event_index", 0) or 0)
        except (TypeError, ValueError):
            event_index = state["last_event_index"] + 1

        state["last_event_index"] = max(
            state["last_event_index"],
            event_index,
        )

        if event_has_possible_impact(event):
            if state["first_high_impact_event_index"] is None:
                state["first_high_impact_event_index"] = event_index

            state["high_impact_events_seen"] += 1

        path_value = event.get("path")
        if path_value and event_has_possible_impact(event):
            state["affected_paths_seen"].add(path_value)

    is_detection_alert = alert_type in EARLY_DETECTION_ALERT_TYPES

    if is_detection_alert and state["first_alert_event_index"] is None:
        state["first_alert_event_index"] = state["last_event_index"]
        state["high_impact_events_before_first_alert"] = (
            state["high_impact_events_seen"]
        )
        state["affected_paths_before_first_alert"] = len(
            state["affected_paths_seen"]
        )

    first_alert_index = state["first_alert_event_index"]
    first_impact_index = state["first_high_impact_event_index"]

    detection_lead_events = None

    if first_alert_index is not None and first_impact_index is not None:
        detection_lead_events = first_impact_index - first_alert_index

    return {
        "status": "ok",
        "scenario_id": scenario_id,
        "window_last_event_index": state["last_event_index"],
        "detection_observed_in_this_window": is_detection_alert,
        "first_alert_event_index": first_alert_index,
        "first_high_impact_event_index": first_impact_index,
        "detection_lead_events": detection_lead_events,
        "high_impact_events_before_first_alert": (
            state["high_impact_events_before_first_alert"]
        ),
        "affected_paths_before_first_alert": (
            state["affected_paths_before_first_alert"]
        ),
    }

def evaluate_window(window_events, window_index):
    if not window_events:
        return None

    symptoms, damage = aggregate_symptoms(window_events)
    learned_match = match_learned_signature(window_events)

    ai_result = call_ai(symptoms)

    response = ai_result.get("response", {})
    policy = response.get("policy")
    risk = ai_result.get("risk_score")
    predicted = ai_result.get("predicted_label")
    unknown_risk = ai_result.get("unknown_risk")

    window_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_window_{window_index}_{uuid4().hex[:8]}"

    detectors = detector_evidence(symptoms, damage, ai_result, learned_match)
    response = decide_policy(symptoms, damage, detectors)
    policy = response["policy"]
    ai_result["response"] = response
    ai_result["learned_match"] = learned_match
    alert_type = policy_alert_type(response, ai_result)

    early_detection_metrics = update_early_detection_metrics(
        window_events,
        alert_type,
    )

    learning_case_file = None
    error_case_file = None

    learned_signature_file = None

    if response["analyst_review_required"] and damage.get("has_file_impact"):
        learning_case_file = save_learning_case(
            window_id,
            window_events,
            symptoms,
            damage,
            ai_result,
            reason="Possible impact requires explicit analyst labeling; policy is a recommendation only.",
        )
        # Signature creation is deferred until analyst approval.

    if "api_error" in ai_result:
        error_case_file = save_error_case(
            window_id,
            window_events,
            symptoms,
            damage,
            ai_result,
        )

    stop_signal_file = None
    # Compatibility fields describe the cooperative sandbox request only.
    # No endpoint containment or generator acknowledgement is implied.
    sandbox_safety_stop = bool(
        response["containment_recommended"] and "api_error" not in ai_result
        and all(e.get("source", {}).get("kind") == "synthetic" for e in window_events)
    )
    containment_triggered = False
    triggered_after_file_impact = sandbox_safety_stop
    stop_reason = "simulated_file_impact_with_containment_recommendation" if sandbox_safety_stop else None

    explain_result = None
    dataset_evidence = None

    if alert_type not in {"monitor", "api_error_needs_review"}:
        explain_result = call_explain(symptoms)
        dataset_evidence = extract_dataset_evidence(explain_result)
    else:
        dataset_evidence = {
            "status": "not_requested",
            "top_matches": []
        }

    spread_report = build_spread_report(window_events, damage)
    remediation_actions = build_remediation(response, alert_type, damage)

    alert = {
        "alert_schema_version": 3,
        "feature_semantics_version": 3,
        "policy_stage": response["stage"],
        "policy_version": response["policy_version"],
        "policy_reasons": response["reasons"],
        "model_error": ai_result.get("api_error"),
        "evidence_categories": response["evidence_categories"],
        "detector_evidence": detectors,
        "window_id": window_id,
        "time": now_iso(),
        "alert_type": alert_type,
        "predicted_label": predicted,
        "risk_score": risk,
        "unknown_risk": unknown_risk,
        "unknown_detector_mode": ai_result.get("unknown_detector_mode"),
        "unknown_detector_shadow": ai_result.get("unknown_detector_shadow"),
        "severity": response.get("severity"),
        "policy": policy,
        "containment_triggered": containment_triggered,
        "stop_reason": stop_reason,
        "triggered_after_file_impact": triggered_after_file_impact,
        "sandbox_safety_stop": sandbox_safety_stop,

        "spread_report": spread_report,
        "damage": damage,
        "early_detection_metrics": early_detection_metrics,

        "recommended_actions": response.get("recommended_actions", []),
        "remediation_actions": remediation_actions,

        "dataset_evidence": dataset_evidence,

        "symptoms": symptoms,
        "learning_case_file": learning_case_file,
        "learned_signature_file": learned_signature_file,
        "learned_match": ai_result.get("learned_match"),
        "error_case_file": error_case_file,
        "stop_signal_file": stop_signal_file,
        "api_error": ai_result.get("api_error"),
    }


    if sandbox_safety_stop:
        stop_signal_file = write_stop_signal(window_id, alert_type, policy, damage)
        alert["stop_signal_file"] = stop_signal_file
    alert["suspicious"] = alert_type != "monitor"
    alert["impact_observed"] = bool(damage.get("has_file_impact"))
    alert["containment_recommended"] = policy in {"protective_lockdown", "isolate_and_backup"}
    alert["containment_executed"] = False
    alert["simulated_stop_requested"] = bool(stop_signal_file)
    alert["simulated_containment_executed"] = False  # A stop request is not generator acknowledgement.
    alert["response_mode"] = "simulation"

    append_jsonl(REPORT_DIR / "live_alerts.jsonl", alert)
    (REPORT_DIR / "current_status.json").write_text(json.dumps(alert, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"[AI WATCHER] window={window_index} events={len(window_events)}")
    print(f"alert_type={alert_type}")
    print(f"predicted={predicted}")
    print(f"risk={risk}")
    print(f"unknown_risk={unknown_risk}")
    print(f"policy={policy}")
    print(f"damage={damage}")

    if learning_case_file:
        print(f"[AI WATCHER] learning case saved: {learning_case_file}")
    if learned_signature_file:
        print(f"[AI WATCHER] learned signature saved: {learned_signature_file}")

    if error_case_file:
        print(f"[AI WATCHER] error case saved: {error_case_file}")

    return alert

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-events", type=int, default=20)
    parser.add_argument("--poll", type=float, default=1.0)
    parser.add_argument("--reset-offset", action="store_true")
    args = parser.parse_args()
    if args.window_events <= 0 or args.poll <= 0:
        parser.error("window-events and poll must be positive")

    EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)

    if args.reset_offset:
        offset = 0
    else:
        offset = EVENT_LOG.stat().st_size if EVENT_LOG.exists() else 0

    buffer = []
    window_index = 0

    print("[+] Live Log AI Watcher started")
    print(f"[+] Watching: {EVENT_LOG}")
    print(f"[+] Window size: {args.window_events} events")
    print("[+] Press Ctrl+C to stop")

    while True:
        try:
            if not EVENT_LOG.exists():
                time.sleep(args.poll)
                continue

            size = EVENT_LOG.stat().st_size

            # Nếu file log bị xóa/tạo lại thì reset offset
            if size < offset:
                offset = 0
                buffer.clear()

            new_events, offset = read_new_lines(EVENT_LOG, offset)

            if new_events:
                buffer.extend(new_events)
                print(f"[AI WATCHER] New events: {len(new_events)} | buffer={len(buffer)}")

            while len(buffer) >= args.window_events:
                # Nếu đã có stop signal thì không xử lý window sau nữa.
                # Điều này giúp first STOP_SIGNAL là điểm ngắt thật của demo.
                if STOP_SIGNAL.exists():
                    print("[AI WATCHER] stop signal exists; clearing buffer and pausing window processing.")
                    buffer.clear()
                    break

                window_events = buffer[:args.window_events]
                buffer = buffer[args.window_events:]
                window_index += 1

                evaluate_window(window_events, window_index)

                # Nếu evaluate_window vừa ghi STOP_SIGNAL thì dừng xử lý buffer ngay
                if STOP_SIGNAL.exists():
                    print("[AI WATCHER] containment triggered; stopping further window processing.")
                    buffer.clear()
                    break

            time.sleep(args.poll)

        except KeyboardInterrupt:
            print("\n[+] Live Log AI Watcher stopped")
            break


if __name__ == "__main__":
    main()
