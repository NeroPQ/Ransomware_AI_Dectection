from pathlib import Path
import json
import time
import argparse

from prometheus_client import start_http_server, Gauge


STATUS_FILE = Path("reports/live_monitoring/current_status.json")
QUEUE_DIR = Path("data/learning_queue")
ERROR_DIR = Path("reports/live_monitoring/error_cases")


risk_score = Gauge("ransomware_risk_score", "Current ransomware risk score")
unknown_risk = Gauge("ransomware_unknown_risk", "Unknown risk flag, 1 high, 0 low")
has_file_impact = Gauge("ransomware_has_file_impact", "Whether file impact has been observed")
high_impact_events = Gauge("ransomware_high_impact_event_count", "High impact event count")
host_count = Gauge("ransomware_impacted_host_count", "Number of impacted hosts")
path_count = Gauge("ransomware_unique_path_count", "Number of touched paths")
directory_count = Gauge("ransomware_impacted_directory_count", "Number of impacted directories")
learning_queue_size = Gauge("ransomware_learning_queue_size", "Number of learning queue cases")
error_case_size = Gauge("ransomware_error_case_size", "Number of API/error cases")

alert_type_gauge = Gauge(
    "ransomware_alert_type",
    "Current alert type encoded as label",
    ["type"]
)

policy_gauge = Gauge(
    "ransomware_policy",
    "Current response policy encoded as label",
    ["policy"]
)

host_event_gauge = Gauge(
    "ransomware_host_event_count",
    "Event count per host in latest window",
    ["host"]
)

dataset_evidence_gauge = Gauge(
    "ransomware_dataset_evidence",
    "Dataset evidence presence from explainability",
    ["dataset_source", "label"]
)


ALERT_TYPES = [
    "monitor",
    "file_impact_containment_recommended",
    "file_impact_needs_review",
    "learned_unknown_ransomware_detected",
    "early_warning_unknown_behavior",
    "early_warning_known_ransomware",
    "early_containment_recommended",
    "unknown_ransomware_contained",
    "ransomware_impact_contained",
    "missed_detection_infected",
    "api_error_needs_review",
]

POLICIES = [
    "monitor_only",
    "early_warning",
    "isolate_and_backup",
    "protective_lockdown",
    "unknown",
]


def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def load_status():
    if not STATUS_FILE.exists():
        return {}
    try:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def update_metrics():
    data = load_status()

    risk_score.set(safe_float(data.get("risk_score"), 0.0))
    unknown_risk.set(1 if data.get("unknown_risk") == "high" else 0)

    damage = data.get("damage", {}) or {}
    spread = data.get("spread_report", {}) or {}
    scope = spread.get("spread_scope", {}) or {}

    has_file_impact.set(1 if damage.get("has_file_impact") else 0)
    high_impact_events.set(safe_float(damage.get("high_impact_event_count"), 0))
    host_count.set(safe_float(damage.get("host_count"), 0))
    path_count.set(safe_float(damage.get("unique_path_count"), 0))
    directory_count.set(safe_float(scope.get("impacted_directory_count"), 0))

    learning_queue_size.set(len(list(QUEUE_DIR.glob("learning_case_*.json"))) if QUEUE_DIR.exists() else 0)
    error_case_size.set(len(list(ERROR_DIR.glob("*.json"))) if ERROR_DIR.exists() else 0)

    current_alert = data.get("alert_type") or "monitor"
    for t in ALERT_TYPES:
        alert_type_gauge.labels(type=t).set(1 if current_alert == t else 0)

    current_policy = data.get("policy") or "unknown"
    for p in POLICIES:
        policy_gauge.labels(policy=p).set(1 if current_policy == p else 0)

    # Reset host gauges for known hosts in latest spread report.
    host_counts = spread.get("host_event_counts", {}) or {}
    host_event_gauge.clear()
    for host, count in list(host_counts.items())[:100]:
        host_event_gauge.labels(host=str(host)).set(safe_float(count, 0))

    # Evidence gauges
    evidence = data.get("dataset_evidence", {}) or {}
    dataset_evidence_gauge.clear()
    for r in evidence.get("top_matches", [])[:5]:
        ds = str(r.get("dataset_source") or "unknown")
        label = str(r.get("label") or "unknown")
        dataset_evidence_gauge.labels(dataset_source=ds, label=label).set(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9108)
    parser.add_argument("--poll", type=float, default=2.0)
    args = parser.parse_args()

    start_http_server(args.port)
    print(f"[GRAFANA EXPORTER] Prometheus metrics at http://localhost:{args.port}/metrics")
    print(f"[GRAFANA EXPORTER] Reading {STATUS_FILE}")

    while True:
        update_metrics()
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
