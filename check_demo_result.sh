#!/usr/bin/env bash
set -e

echo "===== CURRENT STATUS ====="
cat reports/live_monitoring/current_status.json | python3 -m json.tool || true

echo
echo "===== DEMO TIMELINE SUMMARY ====="
python - <<'PY'
import json
from pathlib import Path

p = Path("reports/live_monitoring/live_alerts.jsonl")
if not p.exists():
    print("No live_alerts.jsonl")
    raise SystemExit

alerts = []
for line in p.read_text().splitlines():
    if line.strip():
        alerts.append(json.loads(line))

def brief(a):
    damage = a.get("damage", {}) or {}
    return {
        "window_id": a.get("window_id"),
        "time": a.get("time"),
        "alert_type": a.get("alert_type"),
        "predicted": a.get("predicted_label"),
        "risk": a.get("risk_score"),
        "unknown_risk": a.get("unknown_risk"),
        "policy": a.get("policy"),
        "file_impact": damage.get("has_file_impact"),
        "high_impact": damage.get("high_impact_event_count"),
        "hosts": damage.get("hosts"),
        "learning_case": a.get("learning_case_file"),
        "learned_signature": a.get("learned_signature_file"),
        "learned_match": bool(a.get("learned_match")),
        "stop_signal": a.get("stop_signal_file"),
    }

def first_where(fn):
    for a in alerts:
        if fn(a):
            return a
    return None

first_detection = first_where(
    lambda a: a.get("alert_type") not in {None, "monitor"}
)

first_file_impact = first_where(
    lambda a: (a.get("damage") or {}).get("has_file_impact") is True
)

first_containment = first_where(
    lambda a: a.get("policy") in {"protective_lockdown", "isolate_and_backup"}
)

first_stop = first_where(
    lambda a: bool(a.get("stop_signal_file"))
)

first_learning = first_where(
    lambda a: bool(a.get("learning_case_file") or a.get("learned_signature_file"))
)

first_learned_detection = first_where(
    lambda a: a.get("alert_type") == "learned_unknown_ransomware_detected"
)

items = [
    ("1. First detection / first non-monitor alert", first_detection),
    ("2. First file impact observed", first_file_impact),
    ("3. First containment recommendation", first_containment),
    ("4. First STOP_SIGNAL generated", first_stop),
    ("5. First learning case/signature saved", first_learning),
    ("6. First learned-signature detection", first_learned_detection),
]

for title, obj in items:
    print()
    print(title)
    print("-" * len(title))
    if obj is None:
        print("Not found")
    else:
        b = brief(obj)
        for k, v in b.items():
            print(f"{k}: {v}")

print()
print("===== SHORT WINDOW FLOW =====")
for a in alerts:
    b = brief(a)
    marker = []
    if b["alert_type"] != "monitor":
        marker.append("DETECT")
    if b["file_impact"]:
        marker.append("FILE_IMPACT")
    if b["policy"] in {"protective_lockdown", "isolate_and_backup"}:
        marker.append("CONTAINMENT_RECOMMENDED")
    if b["stop_signal"]:
        marker.append("STOP")
    if b["learning_case"] or b["learned_signature"]:
        marker.append("LEARN")
    if b["learned_match"]:
        marker.append("MATCH_LEARNED")

    marker_text = ",".join(marker) if marker else "normal"

    print(
        f'{b["window_id"]} | {b["time"]} | {marker_text} | '
        f'alert={b["alert_type"]} | policy={b["policy"]} | '
        f'risk={b["risk"]} | impact={b["file_impact"]} | '
        f'high={b["high_impact"]} | hosts={b["hosts"]}'
    )
PY

echo
echo "===== STOP SIGNAL FILE ====="
if [ -f data/custom_sandbox/control/STOP_SIGNAL.json ]; then
  cat data/custom_sandbox/control/STOP_SIGNAL.json | python3 -m json.tool
else
  echo "No STOP_SIGNAL.json"
fi

echo
echo "===== LEARNING QUEUE ====="
ls -lah data/learning_queue || true

echo
echo "===== LAST 10 ALERTS DETAIL ====="
python - <<'PY'
import json
from pathlib import Path

p = Path("reports/live_monitoring/live_alerts.jsonl")
if not p.exists():
    print("No live_alerts.jsonl")
    raise SystemExit

for line in p.read_text().splitlines()[-10:]:
    obj = json.loads(line)
    damage = obj.get("damage", {}) or {}
    print("=" * 90)
    print("window_id:", obj.get("window_id"))
    print("time:", obj.get("time"))
    print("alert_type:", obj.get("alert_type"))
    print("predicted:", obj.get("predicted_label"))
    print("risk:", obj.get("risk_score"))
    print("unknown_risk:", obj.get("unknown_risk"))
    print("policy:", obj.get("policy"))
    print("file_impact:", damage.get("has_file_impact"))
    print("high_impact:", damage.get("high_impact_event_count"))
    print("hosts:", damage.get("hosts"))
    print("learning_case:", obj.get("learning_case_file"))
    print("learned_signature:", obj.get("learned_signature_file"))
    print("learned_match:", obj.get("learned_match"))
    print("stop_signal:", obj.get("stop_signal_file"))
PY
