#!/usr/bin/env python3
import argparse
import json
import os
import smtplib
import ssl
import time
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path


CURRENT_STATUS = Path("reports/live_monitoring/current_status.json")
LIVE_ALERTS = Path("reports/live_monitoring/live_alerts.jsonl")
STOP_SIGNAL = Path("data/custom_sandbox/control/STOP_SIGNAL.json")


CRITICAL_ALERT_TYPES = {
    "file_impact_containment_recommended",
    "file_impact_needs_review",
    "known_ransomware_contained",
    "unknown_ransomware_contained",
    "ransomware_impact_contained",
    "learned_unknown_ransomware_detected",
    "missed_detection_infected",
}

HIGH_ALERT_TYPES = {
    "early_warning_known_ransomware",
    "early_warning_unknown_behavior",
    "unknown_behavior_detected",
    "suspicious_behavior_detected",
}


def safe_read_json(path: Path):
    try:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_last_jsonl(path: Path):
    try:
        if not path.exists():
            return {}
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except Exception:
                continue
        return {}
    except Exception:
        return {}


def pick_latest_status():
    current = safe_read_json(CURRENT_STATUS)
    last_alert = read_last_jsonl(LIVE_ALERTS)

    if last_alert:
        merged = dict(current)
        merged.update(last_alert)
        return merged

    return current


def get_int(data, keys, default=0):
    for key in keys:
        if key in data and data[key] is not None:
            try:
                return int(data[key])
            except Exception:
                pass

    damage = data.get("damage_summary") or data.get("damage") or {}
    for key in keys:
        if key in damage and damage[key] is not None:
            try:
                return int(damage[key])
            except Exception:
                pass

    return default


def get_bool(data, keys, default=False):
    for key in keys:
        if key in data:
            return bool(data[key])

    damage = data.get("damage_summary") or data.get("damage") or {}
    for key in keys:
        if key in damage:
            return bool(damage[key])

    return default


def classify_severity(status: dict):
    alert_type = str(status.get("alert_type", "monitor")).lower()
    policy = str(status.get("policy", "monitor_only")).lower()
    unknown_risk = str(status.get("unknown_risk", "low")).lower()

    risk = status.get("risk_score", status.get("risk", 0.0))
    try:
        risk = float(risk)
    except Exception:
        risk = 0.0

    file_impact = get_bool(status, ["file_impact", "has_file_impact"], False)
    high_impact = get_int(status, ["high_impact", "high_impact_event_count"], 0)

    stop_signal_exists = STOP_SIGNAL.exists() or bool(status.get("stop_signal"))

    if (
        alert_type in CRITICAL_ALERT_TYPES
        or policy in {"protective_lockdown", "isolate_and_backup"}
        or stop_signal_exists
        or (file_impact and high_impact > 0)
    ):
        return "Critical"

    if (
        alert_type in HIGH_ALERT_TYPES
        or unknown_risk in {"medium", "high"}
        or risk >= 0.60
    ):
        return "High"

    return "Normal"


def is_alert_status(status: dict, severity: str):
    alert_type = str(status.get("alert_type", "monitor")).lower()
    policy = str(status.get("policy", "monitor_only")).lower()

    if severity in {"High", "Critical"}:
        return True

    if alert_type not in {"", "monitor", "none", "normal"}:
        return True

    if policy not in {"", "monitor_only", "none"}:
        return True

    return False


def build_fingerprint(status: dict, severity: str):
    return "|".join([
        str(status.get("window_id", "")),
        str(status.get("time", "")),
        str(status.get("alert_type", "")),
        str(status.get("policy", "")),
        str(status.get("learned_match", "")),
        str(status.get("file_impact", status.get("has_file_impact", ""))),
        str(status.get("high_impact", status.get("high_impact_event_count", ""))),
        severity,
    ])


def summarize_learned_match(value):
    if not value:
        return "None"

    if isinstance(value, dict):
        matched = value.get("matched")
        signature_id = value.get("signature_id")

        if not signature_id and isinstance(value.get("signature"), dict):
            signature_id = value["signature"].get("signature_id")

        shared = value.get("shared_event_types", [])
        if isinstance(shared, list):
            shared = ", ".join(shared[:8])

        return f"matched={matched}, signature_id={signature_id}, shared_event_types=[{shared}]"

    return str(value)


def get_damage_dict(status: dict):
    return status.get("damage_summary") or status.get("damage") or {}


def pick_hosts(status: dict):
    damage = get_damage_dict(status)

    for key in ["hosts", "affected_hosts", "impacted_hosts"]:
        value = status.get(key)
        if value:
            return value

    for key in ["hosts", "affected_hosts", "impacted_hosts"]:
        value = damage.get(key)
        if value:
            return value

    return []


def pick_event_types(status: dict):
    damage = get_damage_dict(status)

    for key in ["event_types", "unique_event_types", "high_impact_event_types"]:
        value = status.get(key)
        if value:
            return value

    for key in ["event_types", "unique_event_types", "high_impact_event_types"]:
        value = damage.get(key)
        if value:
            return value

    learned_match = status.get("learned_match")
    if isinstance(learned_match, dict):
        shared = learned_match.get("shared_event_types")
        if shared:
            return shared

        sig = learned_match.get("signature")
        if isinstance(sig, dict):
            return sig.get("event_types", [])

    return []


def build_email(status: dict, severity: str):
    alert_type = status.get("alert_type", "monitor")
    policy = status.get("policy", "monitor_only")
    predicted = status.get("predicted_label", status.get("predicted", "unknown"))
    risk = status.get("risk_score", status.get("risk", 0.0))
    unknown_risk = status.get("unknown_risk", "low")
    window_id = status.get("window_id", "N/A")
    event_time = status.get("time", datetime.now().isoformat(timespec="seconds"))

    damage = get_damage_dict(status)

    file_impact = get_bool(
        status,
        ["file_impact", "has_file_impact"],
        False
    )

    if not file_impact:
        file_impact = bool(damage.get("has_file_impact", damage.get("file_impact", False)))

    high_impact = get_int(
        status,
        ["high_impact", "high_impact_event_count"],
        0
    )

    if high_impact == 0:
        try:
            high_impact = int(damage.get("high_impact_event_count", damage.get("high_impact", 0)))
        except Exception:
            high_impact = 0

    learned_match = summarize_learned_match(status.get("learned_match", None))
    hosts = pick_hosts(status)
    event_types = pick_event_types(status)

    learning_case = status.get("learning_case") or status.get("learning_case_file")
    learned_signature = status.get("learned_signature") or status.get("learned_signature_file")
    stop_signal = status.get("stop_signal", None)

    infection_proof = "No ransomware impact evidence."
    if str(alert_type).lower() == "missed_detection_infected":
        infection_proof = (
            "Model predicted benign/monitor_only, but watcher observed simulated ransomware impact "
            "and saved a learning signature."
        )
    elif file_impact or high_impact > 0:
        infection_proof = "File-impact/high-impact ransomware-like behavior was observed."

    subject = f"[Ransomware AI][{severity}] {alert_type} | policy={policy}"

    body = f"""
RANSOMWARE AI ALERT NOTIFICATION

Severity        : {severity}
Alert type      : {alert_type}
Policy          : {policy}
Predicted       : {predicted}
Risk score      : {risk}
Unknown risk    : {unknown_risk}

Window ID       : {window_id}
Time            : {event_time}

File impact     : {file_impact}
High impact     : {high_impact}
Affected hosts  : {hosts}
Event types     : {event_types}

Learning case   : {learning_case}
Learned signature: {learned_signature}
Learned match   : {learned_match}
STOP_SIGNAL     : {stop_signal}

Why this proves infection:
{infection_proof}

Recommended response:
- High: investigate suspicious behavior and review affected hosts.
- Critical: isolate host, block write/rename operations, preserve evidence, and verify backup.

This email was generated automatically by the AI Ransomware Symptom Response System.
""".strip()

    return subject, body

def send_email(subject: str, body: str):
    dry_run = os.getenv("EMAIL_DRY_RUN", "0").strip() == "1"

    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    email_from = os.getenv("EMAIL_FROM", smtp_user)
    email_to = os.getenv("EMAIL_TO", "")

    recipients = [x.strip() for x in email_to.split(",") if x.strip()]

    if dry_run:
        print("\n========== EMAIL DRY RUN ==========")
        print("TO:", recipients or ["<not configured>"])
        print("SUBJECT:", subject)
        print(body)
        print("===================================\n")
        return True

    if not smtp_host or not smtp_user or not smtp_password or not recipients:
        print("[EMAIL] Missing SMTP config. Set SMTP_HOST, SMTP_USER, SMTP_PASSWORD, EMAIL_TO.")
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = ", ".join(recipients)
    msg.set_content(body)

    try:
        if smtp_port == 465:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context, timeout=15) as server:
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
                server.ehlo()
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)

        print(f"[EMAIL] Sent: {subject}")
        return True

    except Exception as e:
        print(f"[EMAIL] Send failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll", type=float, default=0.25)
    parser.add_argument("--alert-interval", type=float, default=1.0)
    parser.add_argument(
        "--repeat-alerts",
        action="store_true",
        help="Repeat High/Critical emails every alert interval. Default: send once per new alert."
    )
    parser.add_argument(
        "--send-normal-heartbeat",
        action="store_true",
        help="Send Normal heartbeat emails. Default: disabled."
    )
    args = parser.parse_args()

    print("[EMAIL] AI Email Alert Notifier started")
    print("[EMAIL] Mode                 : alert-only by default")
    print(f"[EMAIL] Repeat alerts        : {args.repeat_alerts}")
    print(f"[EMAIL] Send normal heartbeat: {args.send_normal_heartbeat}")
    print(f"[EMAIL] Alert interval       : {args.alert_interval}s")
    print(f"[EMAIL] Dry run              : {os.getenv('EMAIL_DRY_RUN', '0')}")

    last_sent_at = 0.0
    last_fingerprint = None

    while True:
        status = pick_latest_status()

        if not status:
            time.sleep(args.poll)
            continue

        severity = classify_severity(status)
        fingerprint = build_fingerprint(status, severity)

        alert_status = is_alert_status(status, severity)

        # Default behavior:
        # - Normal / monitor: do not send email.
        # - High / Critical: send once when a new alert/window appears.
        # - If --repeat-alerts is enabled: repeat High/Critical every alert interval.
        if severity == "Normal" and not args.send_normal_heartbeat:
            time.sleep(args.poll)
            continue

        if not alert_status and not args.send_normal_heartbeat:
            time.sleep(args.poll)
            continue

        now = time.time()
        is_new_alert = fingerprint != last_fingerprint

        should_send_now = False

        if is_new_alert:
            should_send_now = True
        elif args.repeat_alerts and severity in {"High", "Critical"}:
            if now - last_sent_at >= args.alert_interval:
                should_send_now = True
        elif args.send_normal_heartbeat and severity == "Normal":
            if now - last_sent_at >= 3.0:
                should_send_now = True

        if should_send_now:
            subject, body = build_email(status, severity)
            ok = send_email(subject, body)
            if ok:
                last_sent_at = now
                last_fingerprint = fingerprint

        time.sleep(args.poll)


if __name__ == "__main__":
    main()
