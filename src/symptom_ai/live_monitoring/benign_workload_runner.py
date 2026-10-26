from pathlib import Path
from datetime import datetime
import argparse
import json
import random
import time
import uuid

EVENT_LOG = Path("data/custom_sandbox/events/events.jsonl")
STOP_SIGNAL = Path("data/custom_sandbox/control/STOP_SIGNAL.json")

HOSTS = [
    "host_hr",
    "host_finance",
    "host_accounting",
    "host_shared_drive",
]

FOLDERS = [
    "documents",
    "spreadsheets",
    "contracts",
    "archives",
]


def now_iso():
    return datetime.now().isoformat(timespec="milliseconds")


def random_path(host):
    folder = random.choice(FOLDERS)
    file_id = random.randint(1, 30)

    return (
        f"data/custom_sandbox/lab_files/"
        f"{host}/{folder}/business_file_{file_id}.txt"
    )


def reset_state():
    EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    STOP_SIGNAL.parent.mkdir(parents=True, exist_ok=True)

    if EVENT_LOG.exists():
        EVENT_LOG.unlink()

    if STOP_SIGNAL.exists():
        STOP_SIGNAL.unlink()


def append_event(event):
    if STOP_SIGNAL.exists():
        raise SystemExit(
            "[SAFE BENIGN WORKLOAD] STOP_SIGNAL found. "
            "Stopping scenario safely."
        )

    EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)

    with EVENT_LOG.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event) + "\n")


def workload_profile(scenario):
    profiles = {
        "backup_like": {
            "event_types": [
                "file_read",
                "file_write",
                "storage_write_spike",
                "backup_catalog_update",
            ],
            "process_name": "backup_agent",
            "context": "scheduled_backup_job",
            "host_pool": ["host_shared_drive", "host_finance"],
        },
        "cloud_sync_like": {
            "event_types": [
                "file_read",
                "file_write",
                "sync_metadata_update",
                "sync_upload",
            ],
            "process_name": "cloud_sync_client",
            "context": "cloud_synchronization",
            "host_pool": ["host_hr", "host_finance"],
        },
        "compression_like": {
            "event_types": [
                "file_read",
                "file_write",
                "archive_chunk_write",
                "archive_finalize",
            ],
            "process_name": "zip_archiver",
            "context": "bulk_archive_creation",
            "host_pool": ["host_accounting", "host_shared_drive"],
        },
        "software_update_like": {
            "event_types": [
                "file_write",
                "storage_write_spike",
                "installer_download",
                "signed_update_install",
                "service_restart",
            ],
            "process_name": "trusted_updater",
            "context": "software_update",
            "host_pool": ["host_hr", "host_finance"],
        },
    }

    return profiles[scenario]


def run_workload(scenario, duration, delay):
    profile = workload_profile(scenario)

    scenario_id = (
        f"{scenario}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    print(f"[SAFE BENIGN WORKLOAD] Scenario: {scenario}")
    print(f"[SAFE BENIGN WORKLOAD] Scenario ID: {scenario_id}")

    event_index = 0

    for _ in range(duration):
        event_count = random.randint(4, 6)

        for _ in range(event_count):
            event_index += 1

            host = random.choice(profile["host_pool"])
            event_type = random.choice(profile["event_types"])

            event = {
                "event_id": str(uuid.uuid4()),
                "timestamp": now_iso(),
                "scenario": scenario_id,
                "scenario_type": scenario,
                "scenario_event_index": event_index,
                "event_type": event_type,
                "host": host,
                "path": random_path(host),
                "extra": {
                    "business_context": profile["context"],
                    "process_name": profile["process_name"],
                    "expected_behavior": True,
                    "benign_workload": True,
                },
                "safety_note": (
                    "Safe benign workload simulation only. "
                    "No real backup, sync, compression, update, "
                    "or destructive action is executed."
                ),
            }

            append_event(event)

        print(
            f"[SAFE BENIGN WORKLOAD] "
            f"Generated {event_count} events. Total={event_index}"
        )

        time.sleep(delay)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenario",
        choices=[
            "backup_like",
            "cloud_sync_like",
            "compression_like",
            "software_update_like",
        ],
        required=True,
    )

    parser.add_argument("--duration", type=int, default=8)
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--reset-log", action="store_true")

    args = parser.parse_args()

    if args.reset_log:
        reset_state()
        print("[SAFE BENIGN WORKLOAD] Reset event log and STOP_SIGNAL.")

    run_workload(
        scenario=args.scenario,
        duration=args.duration,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
