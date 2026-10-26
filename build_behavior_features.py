"""Aggregate safe CSV events without mixing sessions or leaking their labels into features."""
from pathlib import Path
import pandas as pd
import numpy as np

INPUT = Path("data/raw/behavior/safe_behavior.csv")
OUTPUT = Path("data/processed/behavior/behavior_features.csv")
WINDOW_SIZE = 50


def build_features(df, window_size=WINDOW_SIZE):
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    df = df.copy()
    required = {"event_index", "timestamp", "event_type", "object", "process_name", "bytes_written", "entropy_before", "entropy_after", "label", "family"}
    missing = required - set(df)
    if missing or df.empty:
        raise ValueError(f"Missing behavior columns or empty dataset: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    if df["timestamp"].isna().any():
        raise ValueError("Missing timestamps")
    if df["label"].isna().any() or not df["label"].isin(["benign", "ransomware"]).all():
        raise ValueError("Behavior labels must be explicit benign or ransomware")
    for column in ["event_index", "bytes_written", "entropy_before", "entropy_after"]:
        df[column] = pd.to_numeric(df[column], errors="raise")
        if df[column].isna().any() or not np.isfinite(df[column]).all() or (df[column] < 0).any():
            raise ValueError(f"Invalid nonnegative measurement: {column}")
    if (df[["entropy_before", "entropy_after"]] > 8).any().any():
        raise ValueError("Byte entropy must be in [0, 8]")
    if "session_id" not in df:
        df["session_id"] = "legacy_single_session"
    if df["session_id"].isna().any() or df["session_id"].astype(str).str.strip().eq("").any():
        raise ValueError("session_id must identify every event")
    if df.duplicated(["session_id", "event_index"]).any() or (df["event_index"] % 1 != 0).any():
        raise ValueError("Event indices must be unique integers within a session")
    df = df.sort_values("event_index").reset_index(drop=True)
    df["window_id"] = df["event_index"] // window_size

    rows = []

    for (session_id, window_id), g in df.groupby(["session_id", "window_id"]):
        row = {
            "window_id": int(window_id),
            "session_id": session_id,
            "first_event_index": int(g["event_index"].min()),
            "last_event_index": int(g["event_index"].max()),
            "event_count": len(g),
            "file_read_count": int((g["event_type"] == "file_read").sum()),
            "file_write_count": int((g["event_type"] == "file_write").sum()),
            "file_rename_count": int((g["event_type"] == "file_rename").sum()),
            "file_delete_count": int((g["event_type"] == "file_delete").sum()),
            "registry_set_count": int((g["event_type"] == "registry_set").sum()),
            "process_create_count": int((g["event_type"] == "process_create").sum()),
            "service_stop_count": int((g["event_type"] == "service_stop").sum()),
            "shadow_copy_delete_count": int((g["event_type"] == "shadow_copy_delete").sum()),
            "dns_query_count": int((g["event_type"] == "dns_query").sum()),
            "network_connect_count": int((g["event_type"] == "network_connect").sum()),
            "unique_object_count": int(g["object"].nunique()),
            "unique_process_count": int(g["process_name"].nunique()),
            "total_bytes_written": int(g["bytes_written"].fillna(0).sum()),
            "avg_entropy_before": float(g["entropy_before"].fillna(0).mean()),
            "avg_entropy_after": float(g["entropy_after"].fillna(0).mean()),
            "entropy_delta": float((g["entropy_after"].fillna(0) - g["entropy_before"].fillna(0)).clip(lower=0).mean()),
            "affected_file_events": int(g["event_type"].isin(["file_write", "file_rename", "file_delete"]).sum()),
            "label": "ransomware" if (g["label"] == "ransomware").any() else "benign",
            "family": "unknown"  # Simulation family metadata must not determine the label
        }

        row["rename_burst_score"] = row["file_rename_count"] / max(row["event_count"], 1)
        row["write_burst_score"] = row["file_write_count"] / max(row["event_count"], 1)
        row["high_entropy_flag"] = 1 if row["entropy_delta"] > 2.0 else 0

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    if not INPUT.is_file():
        raise SystemExit(f"Missing dataset: {INPUT}; run generate_safe_behavior_logs.py first")
    try:
        features = build_features(pd.read_csv(INPUT))
    except (ValueError, KeyError) as exc:
        raise SystemExit(f"Invalid behavior dataset: {exc}") from exc
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT, index=False)
    print(f"Created {OUTPUT}: {len(features)} windows (simulation only)")


if __name__ == "__main__":
    main()
