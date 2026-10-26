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

API_URL = "http://localhost:8000/predict"
API_EXPLAIN_URL = "http://localhost:8000/explain"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
QUEUE_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)
CONTROL_DIR.mkdir(parents=True, exist_ok=True)

HIGH_IMPACT_EVENTS = {
    "file_rename",
    "extension_change",
    "high_entropy_write",
    "ransom_note_created",

    # Safe zero-day simulation impact events
    "stealth_file_lock",
    "stealth_content_scramble",
    "stealth_extension_mutation",
    "stealth_note_marker",
}


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
                if not isinstance(event, dict) or not isinstance(event.get("event_type"), str):
                    raise ValueError("Event requires event_type")
                events.append(event)
            except (ValueError, TypeError):
                logging.getLogger(__name__).warning("Invalid event record skipped at offset %s", position)


def clamp01(x):
    value = float(x)
    return max(0.0, min(value, 1.0)) if math.isfinite(value) else 0.0


def aggregate_symptoms(events):
    count = Counter(e.get("event_type") for e in events)
    hosts = {e.get("host") for e in events if e.get("host")}
    paths = {e.get("path") for e in events if e.get("path")}

    entropy_scores = []
    memory_scores = []

    for e in events:
        extra = e.get("extra") or {}
        if not isinstance(extra, dict):
            extra = {}
        try:
            entropy_scores.append(float(extra.get("entropy_score", 0.0)))
        except Exception:
            pass
        try:
            memory_scores.append(float(extra.get("memory_score", 0.0)))
        except Exception:
            pass

    symptoms = {
        "file_write_burst": clamp01((count["file_write"] + count["storage_write_spike"]) / 10),
        "file_read_burst": clamp01(count["file_read"] / 10),
        "file_rename_burst": clamp01(count["file_rename"] / 6),
        "mass_file_modification": clamp01((count["file_write"] + count["file_rename"] + count["extension_change"]) / 12),
        "suspicious_extension_change": clamp01(count["extension_change"] / 4),
        "high_entropy_write": clamp01(max([count["high_entropy_write"] / 4] + entropy_scores)),
        "ransom_note_created": clamp01(count["ransom_note_created"] / 2),
        "multi_directory_impact": clamp01(len(paths) / 15),
        "user_document_impact_high": clamp01(len(paths) / 20),

        "suspicious_process_spawn": clamp01((count["process_spawn"] + count["rare_process"]) / 5),
        "rare_process_name": clamp01(count["rare_process"] / 4),
        "process_tree_anomaly": clamp01(count["rare_process"] / 4),
        "process_injection_suspected": clamp01(count["process_injection_suspected"] / 3),

        "memory_access_spike": clamp01(max([count["memory_anomaly"] / 4] + memory_scores)),
        "memory_entropy_region_high": clamp01(count["memory_anomaly"] / 4),
        "anti_vm": clamp01(count["anti_vm"] / 3),
        "anti_analysis": clamp01((count["anti_vm"] + count["api_behavior"]) / 6),

        "network_api_usage": clamp01((count["network_beacon"] + count["api_behavior"]) / 8),
        "c2_beaconing": clamp01(count["network_beacon"] / 3),
        "data_exfiltration_pattern": clamp01(count["network_beacon"] / 5),

        "storage_write_spike": clamp01(count["storage_write_spike"] / 4),
        "novel_symptom_combination": clamp01((count["memory_anomaly"] + count["rare_process"] + count["storage_write_spike"]) / 8),
        "unknown_high_risk": 0.0,
        "analyst_review_required": 0.0,
        "retraining_candidate": 0.0,
    }

    if symptoms["novel_symptom_combination"] >= 0.5 and (
        symptoms["mass_file_modification"] >= 0.4 or symptoms["high_entropy_write"] >= 0.7
    ):
        symptoms["unknown_high_risk"] = 0.9
        symptoms["analyst_review_required"] = 0.9
        symptoms["retraining_candidate"] = 0.9

    damage = {
        "event_count": len(events),
        "host_count": len(hosts),
        "hosts": sorted(hosts),
        "unique_path_count": len(paths),
        "high_impact_event_count": sum(count[e] for e in HIGH_IMPACT_EVENTS),
        "event_type_counts": dict(count),
        "has_file_impact": any(count[e] > 0 for e in HIGH_IMPACT_EVENTS),
    }

    return symptoms, damage


def call_ai(symptoms):
    payload = json.dumps({"symptoms": symptoms}).encode("utf-8")
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
    payload = json.dumps({"symptoms": symptoms}).encode("utf-8")
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

    impacted_dirs = set()
    high_impact_paths = []

    for e in events:
        event_type = e.get("event_type")
        path = e.get("path")

        if path:
            try:
                impacted_dirs.add(str(Path(path).parent))
            except Exception:
                pass

        if event_type in HIGH_IMPACT_EVENTS and path:
            high_impact_paths.append({
                "event_type": event_type,
                "host": e.get("host"),
                "path": path,
                "timestamp": e.get("timestamp")
            })

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
            "detail": explain_result.get("explain_error") if isinstance(explain_result, dict) else None,
            "top_matches": []
        }

    decision = explain_result.get("decision_explanation", {})
    rows = (
        decision.get("evidence_training_rows")
        or decision.get("nearest_training_rows")
        or []
    )

    top_matches = []
    for r in rows[:5]:
        top_matches.append({
            "rank": r.get("rank"),
            "dataset_source": r.get("matched_dataset_source"),
            "sample_id": r.get("matched_sample_id"),
            "family": r.get("matched_family"),
            "behavior_type": r.get("matched_behavior_type"),
            "label": r.get("matched_label"),
            "response_policy": r.get("matched_response_policy"),
            "evidence_score": r.get("evidence_score", r.get("similarity_score")),
            "shared_active_symptoms": r.get("shared_active_symptoms", []),
        })

    return {
        "status": "ok",
        "decision": decision.get("decision"),
        "rule_reason": decision.get("explanation", {}).get("rule_reason"),
        "top_match_reason": decision.get("explanation", {}).get("top_match_reason"),
        "matched_label_counts_top5": decision.get("matched_label_counts_top5", {}),
        "top_matches": top_matches,
    }


def build_remediation(response, alert_type, damage):
    actions = list(response.get("recommended_actions", []) or [])

    if damage.get("has_file_impact"):
        actions.extend([
            "Immediately isolate affected host(s) from the network",
            "Preserve impacted files and logs for forensic review",
            "Block further write/rename operations on protected folders",
            "Trigger emergency backup verification",
        ])

    if alert_type in {"file_impact_containment_recommended", "file_impact_needs_review"}:
        actions.extend([
            "Send unknown behavior to learning queue",
            "Schedule auto-retraining after validation threshold is reached",
            "Create analyst review ticket for unknown ransomware behavior",
        ])

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
    out = QUEUE_DIR / f"learning_case_{window_id}.json"
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
        "response_policy": "protective_lockdown",
        "learning_status": "pending_review",
        "confirmed_label": None,
        "safety_note": "Safe sandbox logs only. No real ransomware was executed.",
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


def save_learned_signature(window_id, events, damage):
    """Compatibility hook: signatures become available only via explicit case review."""
    return None


def match_learned_signature(events):
    symptoms, damage = aggregate_symptoms(events)
    current_counts = Counter(e.get("event_type") for e in events)
    if damage["high_impact_event_count"] < 3 or len(set(current_counts) & HIGH_IMPACT_EVENTS) < 2:
        return None
    for signature in load_learned_signatures():
        reference_events = signature["case"].get("events") or []
        if not reference_events:
            continue
        reference, reference_damage = aggregate_symptoms(reference_events)
        counts = Counter(e.get("event_type") for e in reference_events)
        if reference_damage["high_impact_event_count"] < 3 or len(set(counts) & HIGH_IMPACT_EVENTS) < 2:
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


def decide_alert_type(ai_result, damage, policy, unknown_risk):
    if "api_error" in ai_result:
        return "api_error_needs_review"
    recommended = policy in {"isolate_and_backup", "protective_lockdown"}
    impact = bool(damage.get("has_file_impact"))
    if impact and recommended:
        return "file_impact_containment_recommended"
    if impact:
        return "file_impact_needs_review"
    if recommended:
        return "early_containment_recommended"
    if unknown_risk == "high" or policy == "early_warning":
        return "early_warning_unknown_behavior"
    return "monitor"


def write_stop_signal(window_id, alert_type, policy, damage):
    # Preserve the FIRST containment point.
    # Later buffered windows must not overwrite the original stop signal.
    if STOP_SIGNAL.exists():
        return str(STOP_SIGNAL)

    signal = {
        "time": now_iso(),
        "window_id": window_id,
        "alert_type": alert_type,
        "policy": policy,
        "reason": "AI watcher detected file impact and containment policy. Safe sandbox should stop generating events.",
        "damage": damage,
        "response_mode": "simulation",
        "containment_executed": False,
        "simulated_stop_requested": True,
    }
    STOP_SIGNAL.write_text(json.dumps(signal, indent=2), encoding="utf-8")
    return str(STOP_SIGNAL)


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

    if learned_match and damage.get("has_file_impact") and "api_error" not in ai_result:
        alert_type = "learned_unknown_ransomware_detected"
        policy = "protective_lockdown"

        # Reviewed similarity is a heuristic recommendation, not a malware probability.
        risk = max(float(risk or 0), 0.7)
        unknown_risk = "high"
        predicted = "learned_unknown_ransomware"

        ai_result["risk_score"] = risk
        ai_result["unknown_risk"] = unknown_risk
        ai_result["predicted_label"] = predicted

        response["policy"] = "protective_lockdown"
        response["severity"] = "Critical"
        response["recommended_actions"] = [
            "Behavior resembles a reviewed simulation case",
            "Immediately isolate affected host(s)",
            "Block further write/rename operations",
            "Preserve evidence and event timeline",
            "Trigger protective lockdown"
        ]
        ai_result["learned_match"] = learned_match
    else:
        alert_type = decide_alert_type(ai_result, damage, policy, unknown_risk)

    learning_case_file = None
    error_case_file = None

    learned_signature_file = None

    if alert_type == "file_impact_needs_review":
        learning_case_file = save_learning_case(
            window_id,
            window_events,
            symptoms,
            damage,
            ai_result,
            reason="File-impact indicators appeared but containment policy was not triggered.",
        )
        learned_signature_file = save_learned_signature(window_id, window_events, damage)

    if alert_type == "api_error_needs_review":
        error_case_file = save_error_case(
            window_id,
            window_events,
            symptoms,
            damage,
            ai_result,
        )

    stop_signal_file = None
    if alert_type == "file_impact_containment_recommended":
        stop_signal_file = write_stop_signal(window_id, alert_type, policy, damage)

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
        "window_id": window_id,
        "time": now_iso(),
        "alert_type": alert_type,
        "predicted_label": predicted,
        "risk_score": risk,
        "unknown_risk": unknown_risk,
        "severity": response.get("severity"),
        "policy": policy,

        "spread_report": spread_report,
        "damage": damage,

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


    if damage.get("has_file_impact") and "api_error" not in ai_result and policy in {"protective_lockdown", "isolate_and_backup"}:
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

    if stop_signal_file:
        print(f"[AI WATCHER] stop signal written: {stop_signal_file}")

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
