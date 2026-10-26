import pandas as pd
import os
from pathlib import Path

INPUT = "reports/behavior_scored_windows.csv"
OUTPUT = "reports/mitre_alerts.csv"

if not Path(INPUT).is_file():
    raise SystemExit(f"Missing scored windows: {INPUT}; run train_behavior_models.py first")
df = pd.read_csv(INPUT)
alerts = []

def add(row, technique, tactic, evidence, explanation, severity):
    alerts.append({
        "window_id": row["window_id"],
        "first_event_index": row["first_event_index"],
        "risk_score": max(row.get("rf_score", 0), row.get("xgb_score", 0), row.get("if_score", 0)),
        "technique": technique,
        "tactic": tactic,
        "evidence": evidence,
        "explanation": explanation,
        "severity": severity
    })

for _, row in df.iterrows():
    is_alert = (
        row.get("rf_pred", 0) == 1
        or row.get("xgb_pred", 0) == 1
        or row.get("if_pred", 0) == 1
    )

    if not is_alert:
        continue

    if row["file_write_count"] >= 10 or row["file_rename_count"] >= 8 or row["high_entropy_flag"] == 1:
        add(
            row,
            "T1486",
            "Impact",
            "file write/rename burst and entropy increase",
            "Potential mass file encryption or large-scale file modification behavior.",
            "High"
        )

    if row["shadow_copy_delete_count"] > 0:
        add(
            row,
            "T1490",
            "Impact",
            "shadow copy deletion pattern",
            "Potential attempt to inhibit system recovery by deleting or disabling backup mechanisms.",
            "High"
        )

    if row["service_stop_count"] > 0:
        add(
            row,
            "T1489",
            "Impact",
            "service stop event",
            "Potential attempt to stop backup, security, database, or recovery-related services.",
            "Medium"
        )

    if row["registry_set_count"] > 0:
        add(
            row,
            "T1112",
            "Defense Evasion / Persistence",
            "registry modification",
            "Potential system configuration modification through registry-related activity.",
            "Medium"
        )

    if row["dns_query_count"] >= 5 or row["network_connect_count"] >= 5:
        add(
            row,
            "T1071",
            "Command and Control",
            "network or DNS activity",
            "Network activity requires further investigation for possible command-and-control or external communication behavior.",
            "Medium"
        )

out = pd.DataFrame(alerts, columns=["window_id", "first_event_index", "risk_score", "technique", "tactic", "evidence", "explanation", "severity"])
os.makedirs("reports", exist_ok=True)
out.to_csv(OUTPUT, index=False)

print("[+] Created", OUTPUT)
print(out.head(20))