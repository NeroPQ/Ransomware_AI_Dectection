from pathlib import Path
import argparse
import json
import sys

import symptom_ai.live_monitoring.live_log_ai_watcher as watcher


DEFAULT_QUEUE_DIR = Path("data/learning_queue")
DEFAULT_SIGNATURE_FILE = (
    Path("data/learning_queue/learned_behavior_signatures.json")
)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: Invalid JSON in {path}: {exc}") from exc


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + ".tmp")

    temp_path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )

    temp_path.replace(path)


def resolve_case_path(value, queue_dir):
    direct_path = Path(value)

    if direct_path.exists():
        return direct_path

    queue_path = queue_dir / value

    if queue_path.exists():
        return queue_path

    raise SystemExit(
        "ERROR: Learning case was not found. "
        f"Checked: {direct_path} and {queue_path}"
    )


def list_cases(queue_dir):
    case_files = sorted(queue_dir.glob("learning_case_*.json"))

    if not case_files:
        print("No learning cases found.")
        return

    print("=== PENDING / REVIEWED LEARNING CASES ===")

    for case_path in case_files:
        case = read_json(case_path)

        print(
            f"{case_path.name} | "
            f"status={case.get('learning_status')} | "
            f"decision={case.get('analyst_decision')} | "
            f"saved_at={case.get('saved_at')}"
        )


def review_case(
    case_path,
    decision,
    analyst,
    note,
    signature_file,
):
    case = read_json(case_path)

    if case.get("learning_status") != "pending_review":
        raise SystemExit(
            "ERROR: Only cases with learning_status='pending_review' "
            "can be reviewed."
        )

    window_id = case.get("window_id")

    if not window_id:
        raise SystemExit("ERROR: Learning case has no window_id.")

    reviewed_at = watcher.now_iso()

    case["analyst_decision"] = decision
    case["reviewed_by"] = analyst
    case["reviewed_at"] = reviewed_at
    case["review_note"] = note

    if decision == "approve":
        events = case.get("events", [])
        damage = case.get("damage", {})

        if not events:
            raise SystemExit(
                "ERROR: Cannot approve a case without events."
            )

        watcher.LEARNED_SIGNATURES = signature_file

        signature_path = watcher.save_learned_signature(
            window_id=window_id,
            events=events,
            damage=damage,
            approved_by=analyst,
            review_note=note,
            source_case_file=str(case_path),
        )

        case["learning_status"] = "approved"
        case["signature_status"] = "approved_signature_created"
        case["learned_signature_file"] = signature_path

        write_json(case_path, case)

        print("APPROVED")
        print("Case:", case_path)
        print("Signature file:", signature_path)

    else:
        case["learning_status"] = "rejected"
        case["signature_status"] = "not_created"
        case["learned_signature_file"] = None

        write_json(case_path, case)

        print("REJECTED")
        print("Case:", case_path)
        print("No learned signature was created.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyst approval workflow for safe-sandbox "
            "learning cases."
        )
    )

    parser.add_argument(
        "--queue-dir",
        default=str(DEFAULT_QUEUE_DIR),
    )

    parser.add_argument(
        "--signature-file",
        default=str(DEFAULT_SIGNATURE_FILE),
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List learning cases and review status.",
    )

    parser.add_argument(
        "--case-file",
        help="Case filename or full path to review.",
    )

    parser.add_argument(
        "--decision",
        choices=["approve", "reject"],
        help="Analyst decision for one pending case.",
    )

    parser.add_argument(
        "--analyst",
        default="analyst_demo",
        help="Analyst identifier stored in the review audit trail.",
    )

    parser.add_argument(
        "--note",
        default="",
        help="Short analyst justification for the decision.",
    )

    args = parser.parse_args()

    queue_dir = Path(args.queue_dir)
    signature_file = Path(args.signature_file)

    if args.list:
        list_cases(queue_dir)
        return

    if not args.case_file:
        parser.error("--case-file is required unless --list is used.")

    if not args.decision:
        parser.error("--decision is required when reviewing a case.")

    case_path = resolve_case_path(args.case_file, queue_dir)

    review_case(
        case_path=case_path,
        decision=args.decision,
        analyst=args.analyst,
        note=args.note,
        signature_file=signature_file,
    )


if __name__ == "__main__":
    main()
