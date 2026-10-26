"""Count windows explicitly; expose unsupported rates/latencies as null."""
import numpy as np


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def detection_metrics(rows, predictions):
    if not rows or len(rows) != len(predictions):
        raise ValueError("Need nonempty aligned windows and predictions")
    if any(row.get("label") not in {"benign", "known_ransomware_like"} for row in rows):
        raise ValueError("Metrics require explicit reviewed or scripted labels")
    truth = np.array([row["label"] == "known_ransomware_like" for row in rows])
    pred = np.asarray(predictions, dtype=bool)
    tp, fp = int((truth & pred).sum()), int((~truth & pred).sum())
    tn, fn = int((~truth & ~pred).sum()), int((truth & ~pred).sum())
    # Every alerting window is one alert. No undocumented episode deduplication.
    result = {"windows": len(rows), "true_positive": tp, "false_positive": fp,
              "true_negative": tn, "false_negative": fn, "precision": ratio(tp, tp + fp),
              "recall": ratio(tp, tp + fn), "f1": ratio(2 * tp, 2 * tp + fp + fn),
              "fpr": ratio(fp, fp + tn), "benign_endpoint_hours": None,
              "simulated_benign_endpoint_hours": None,
              "false_alerts_per_endpoint_hour": None,
              "false_alerts_per_simulated_endpoint_hour": None,
              "detection_latency_seconds": None, "simulated_detection_latency_seconds": None,
              "latency_note": "Unavailable without valid per-endpoint window exposure and explicit attack onset"}
    valid_time = all(row.get("time_basis") in {"observed", "simulated"}
                     and row.get("endpoint_id") and row.get("session_id")
                     and isinstance(row.get("window_start_seconds"), (int, float))
                     and isinstance(row.get("window_end_seconds"), (int, float))
                     and np.isfinite([row["window_start_seconds"], row["window_end_seconds"]]).all()
                     and row["window_end_seconds"] > row["window_start_seconds"] for row in rows)
    if not valid_time or len({row["time_basis"] for row in rows}) != 1:
        return result
    # Refuse overlapping exposure on an endpoint, even across sessions.
    endpoints = {}
    for row in rows:
        endpoints.setdefault(row["endpoint_id"], []).append((row["window_start_seconds"], row["window_end_seconds"]))
    if any(any(b[0] < a[1] for a, b in zip(sorted(intervals), sorted(intervals)[1:])) for intervals in endpoints.values()):
        raise ValueError("Overlapping endpoint windows cannot be counted as independent exposure")
    hours = sum(row["window_end_seconds"] - row["window_start_seconds"] for row, positive in zip(rows, truth) if not positive) / 3600
    simulated = rows[0]["time_basis"] == "simulated"
    result["simulated_benign_endpoint_hours" if simulated else "benign_endpoint_hours"] = hours
    result["false_alerts_per_simulated_endpoint_hour" if simulated else "false_alerts_per_endpoint_hour"] = ratio(fp, hours)
    sessions = {}
    for row, hit in zip(rows, pred):
        sessions.setdefault(row["session_id"], []).append((row, bool(hit)))
    latencies, leads, missed, eligible = [], [], 0, 0
    for values in sessions.values():
        if not any(row["label"] == "known_ransomware_like" for row, _ in values):
            continue
        onsets = {row.get("attack_onset_seconds") for row, _ in values}
        impacts = {row.get("impact_onset_seconds") for row, _ in values}
        if len(onsets) != 1 or None in onsets:
            continue
        onset = next(iter(onsets))
        if not isinstance(onset, (float, int)) or not np.isfinite(onset):
            continue
        # Whole session must be observed from onset; don't hide left censoring.
        intervals = sorted((row["window_start_seconds"], row["window_end_seconds"]) for row, _ in values)
        if not any(start <= onset < end for start, end in intervals):
            continue
        if any(right[0] > left[1] for left, right in zip(intervals, intervals[1:]) if right[1] > onset):
            continue
        eligible += 1
        detections = [row["window_end_seconds"] for row, hit in values
                      if hit and row["window_start_seconds"] >= onset]
        if not detections:
            missed += 1
            continue
        first = min(detections)
        latencies.append(first - onset)
        if len(impacts) == 1 and None not in impacts:
            impact = next(iter(impacts))
            if isinstance(impact, (float, int)) and np.isfinite(impact):
                leads.append(impact - first)
    latency = {"eligible_attack_sessions": eligible, "detected_sessions": len(latencies), "missed_sessions": missed,
               "session_detection_rate": ratio(len(latencies), eligible),
               "median_among_detected": float(np.median(latencies)) if latencies else None,
               "p95_among_detected": float(np.quantile(latencies, .95)) if latencies else None,
               "median_impact_lead_among_detected": float(np.median(leads)) if leads else None,
               "resolution": "window end; missed sessions are censored, never assigned zero latency"}
    if eligible:
        result["simulated_detection_latency_seconds" if simulated else "detection_latency_seconds"] = latency
        result["latency_note"] = "Pre-onset warnings are false positives, not successful attack detections"
    return result
