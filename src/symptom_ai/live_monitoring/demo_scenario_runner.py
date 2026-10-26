from pathlib import Path
from datetime import datetime
import argparse
import json
import random
import time
import uuid


EVENT_DIR = Path("data/custom_sandbox/events")
EVENT_LOG = EVENT_DIR / "events.jsonl"

CONTROL_DIR = Path("data/custom_sandbox/control")
STOP_SIGNAL = CONTROL_DIR / "STOP_SIGNAL.json"

LAB_DIR = Path("data/custom_sandbox/lab_files")

HOSTS = ["host_hr", "host_finance", "host_accounting", "host_shared_drive"]
FOLDERS = ["documents", "spreadsheets", "contracts", "archives"]

# Counts events inside each safe simulation scenario.
# Used later to measure events-before-alert and detection lead time.
SCENARIO_EVENT_COUNTERS = {}


def now_iso():
    return datetime.now().isoformat(timespec="milliseconds")


def ensure_dirs():
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    LAB_DIR.mkdir(parents=True, exist_ok=True)

    for host in HOSTS:
        for folder in FOLDERS:
            d = LAB_DIR / host / folder
            d.mkdir(parents=True, exist_ok=True)
            for i in range(1, 8):
                f = d / f"business_file_{i}.txt"
                if not f.exists():
                    f.write_text(
                        f"Safe demo file {i} for {host}/{folder}\n",
                        encoding="utf-8"
                    )


def reset_demo_state():
    if EVENT_LOG.exists():
        EVENT_LOG.unlink()
    if STOP_SIGNAL.exists():
        STOP_SIGNAL.unlink()


def should_stop():
    return STOP_SIGNAL.exists()


def append_event(event):
    # Stop immediately if watcher already requested containment.
    if should_stop():
        raise SystemExit("[SAFE DEMO] Stop signal detected before writing next event. Demo stopped.")

    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    with EVENT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def random_path(host=None):
    host = host or random.choice(HOSTS)
    folder = random.choice(FOLDERS)
    idx = random.randint(1, 7)
    return LAB_DIR / host / folder / f"business_file_{idx}.txt"


def emit(scenario_id, scenario_type, event_type, host, path=None, extra=None):
    scenario_event_index = SCENARIO_EVENT_COUNTERS.get(scenario_id, 0) + 1
    SCENARIO_EVENT_COUNTERS[scenario_id] = scenario_event_index

    event = {
        "event_id": str(uuid.uuid4()),
        "timestamp": now_iso(),
        "scenario": scenario_id,
        "scenario_type": scenario_type,
        "scenario_event_index": scenario_event_index,
        "event_type": event_type,
        "host": host,
        "path": str(path) if path else None,
        "extra": extra or {},
        "safety_note": "Safe simulation only. No real ransomware or destructive action is executed."
    }
    append_event(event)
    return event


def print_activity(message):
    print(f"[SAFE DEMO] {message}")


def run_benign(duration, delay):
    """
    Kịch bản 1:
    Tác vụ bình thường: đọc file, lưu file nhẹ, mở ứng dụng văn phòng.
    Kỳ vọng: không chặn, monitor_only.
    """
    scenario_id = f"benign_normal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print_activity(f"Scenario 1 started: normal business activity")
    print_activity(f"Scenario ID: {scenario_id}")

    benign_events = [
        "normal_file_read",
        "normal_file_save",
        "office_app_open",
        "spreadsheet_edit",
        "document_preview",
        "browser_cache_write",
    ]

    for _ in range(duration):
        if should_stop():
            print_activity("Stop signal detected. Normal activity stopped.")
            break

        event_count = random.randint(2, 4)

        for _ in range(event_count):
            host = random.choice(HOSTS[:2])
            event_type = random.choice(benign_events)
            emit(
                scenario_id,
                "benign_normal",
                event_type,
                host,
                random_path(host),
                {
                    "business_context": "normal_user_activity",
                    "expected_behavior": True,
                }
            )

        print_activity(f"normal background activity: +{event_count} events")
        time.sleep(delay)


def run_unusual(duration, delay):
    """
    Kịch bản 2:
    Tác vụ khác lạ nhưng chưa có impact: process lạ, API/network ít.
    Kỳ vọng: có thể cảnh báo sớm, nhưng không xem là nhiễm ransomware.
    """
    scenario_id = f"unusual_non_impact_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print_activity(f"Scenario 2 started: unusual but non-impact activity")
    print_activity(f"Scenario ID: {scenario_id}")

    unusual_events = [
        "rare_process",
        "api_behavior",
        "network_beacon",
        "memory_anomaly",
        "anti_vm",
    ]

    for i in range(duration):
        if should_stop():
            print_activity("Stop signal detected. Unusual activity stopped.")
            break

        # Ít event để không tạo file impact.
        event_count = random.randint(1, 3)

        for _ in range(event_count):
            host = random.choice(HOSTS[:2])
            event_type = random.choice(unusual_events)
            emit(
                scenario_id,
                "unusual_non_impact",
                event_type,
                host,
                random_path(host),
                {
                    "suspicious_but_no_file_impact": True,
                    "memory_score": round(random.uniform(0.25, 0.55), 3)
                    if event_type == "memory_anomaly" else 0.0,
                }
            )

        print_activity(f"unusual background activity: +{event_count} events")
        time.sleep(delay)


def run_known_progressive(duration, delay):
    """
    Kịch bản 3:
    Ransomware lúc đầu lạ, sau đó xuất hiện pattern model/risk engine đã học:
    high entropy write, file write, rename, extension change.
    Kỳ vọng: phát hiện và chặn bằng protective_lockdown/isolate_and_backup.
    """
    scenario_id = f"known_progressive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print_activity(f"Scenario 3 started: unknown early behavior then known ransomware-like impact")
    print_activity(f"Scenario ID: {scenario_id}")

    for i in range(duration):
        if should_stop():
            print_activity("Stop signal detected. Known-progressive scenario stopped.")
            break

        progress = (i + 1) / duration

        if progress < 0.35:
            events = ["rare_process", "memory_anomaly", "api_behavior"]
            event_count = random.randint(2, 4)
            host_pool = HOSTS[:1]

        elif progress < 0.65:
            events = ["rare_process", "memory_anomaly", "process_injection_suspected", "network_beacon", "api_behavior"]
            event_count = random.randint(4, 6)
            host_pool = HOSTS[:2]

        else:
            events = [
                "file_write",
                "storage_write_spike",
                "high_entropy_write",
                "file_rename",
                "extension_change",
                "ransom_note_created",
            ]
            event_count = random.randint(6, 9)
            host_pool = HOSTS[:3]

        for _ in range(event_count):
            host = random.choice(host_pool)
            event_type = random.choice(events)
            emit(
                scenario_id,
                "known_progressive_ransomware",
                event_type,
                host,
                random_path(host),
                {
                    "entropy_score": round(random.uniform(0.72, 0.98), 3)
                    if event_type == "high_entropy_write" else 0.0,
                    "new_extension": ".simlocked"
                    if event_type in {"file_rename", "extension_change"} else "",
                    "expected_detection": "known_ransomware_like_after_impact",
                }
            )

        print_activity(f"background activity observed: +{event_count} events")
        time.sleep(delay)


def run_novel_zero_day(duration, delay):
    """
    Kịch bản 4:
    Ransomware lạ từ đầu đến cuối.
    Lần đầu: watcher nên coi là missed_detection_infected và lưu learning case.
    Lần chạy lại: watcher dùng learned behavior signatures để phát hiện và chặn.
    """
    scenario_id = f"novel_zero_day_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print_activity(f"Scenario 4 started: novel zero-day ransomware-like behavior")
    print_activity(f"Scenario ID: {scenario_id}")

    for i in range(duration):
        if should_stop():
            print_activity("Stop signal detected. Novel scenario stopped.")
            break

        progress = (i + 1) / duration

        if progress < 0.40:
            # Lạ nhưng chưa giống pattern đã học.
            events = [
                "low_noise_file_scan",
                "rare_process",
                "custom_protocol_ping",
                "strange_temp_write",
            ]
            event_count = random.randint(2, 4)
            host_pool = HOSTS[:1]

        elif progress < 0.75:
            # Bắt đầu có hành vi khó phân loại.
            events = [
                "low_noise_file_scan",
                "custom_protocol_ping",
                "stealth_file_touch",
                "stealth_metadata_change",
                "strange_temp_write",
            ]
            event_count = random.randint(4, 6)
            host_pool = HOSTS[:2]

        else:
            # Có impact nhưng dùng event type lạ để model cũ chưa biết.
            events = [
                "stealth_file_lock",
                "stealth_content_scramble",
                "stealth_extension_mutation",
                "stealth_note_marker",
                "stealth_file_touch",
            ]
            event_count = random.randint(6, 9)
            host_pool = HOSTS[:4]

        for _ in range(event_count):
            host = random.choice(host_pool)
            event_type = random.choice(events)
            emit(
                scenario_id,
                "novel_zero_day_ransomware",
                event_type,
                host,
                random_path(host),
                {
                    "novel_behavior": True,
                    "intended_first_run": "missed_detection_infected",
                    "intended_second_run": "learned_unknown_detected",
                    "new_extension": ".zday"
                    if event_type in {"stealth_file_lock", "stealth_extension_mutation"} else "",
                }
            )

        print_activity(f"background activity observed: +{event_count} events")
        time.sleep(delay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=["benign", "unusual", "known_progressive", "novel_zero_day"],
        required=True,
    )
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--reset-log", action="store_true")

    args = parser.parse_args()

    ensure_dirs()

    if args.reset_log:
        reset_demo_state()
        print_activity("Reset event log and stop signal")

    if args.scenario == "benign":
        run_benign(args.duration, args.delay)
    elif args.scenario == "unusual":
        run_unusual(args.duration, args.delay)
    elif args.scenario == "known_progressive":
        run_known_progressive(args.duration, args.delay)
    elif args.scenario == "novel_zero_day":
        run_novel_zero_day(args.duration, args.delay)


if __name__ == "__main__":
    main()
