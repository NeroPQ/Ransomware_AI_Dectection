"""Compatibility CLI for the shared evidence-bound analyst review workflow."""
from pathlib import Path
import argparse
import json

from symptom_ai.auto_learning.review import is_approved, review_case as record_review

DEFAULT_QUEUE_DIR = Path("data/learning_queue")
DEFAULT_SIGNATURE_FILE = DEFAULT_QUEUE_DIR / "learned_behavior_signatures.json"


def resolve_case_path(value, queue_dir):
    for path in (Path(value), Path(queue_dir) / value):
        if path.is_file():
            return path
    raise ValueError(f"Learning case not found: {value}")


def list_cases(queue_dir):
    for path in sorted(Path(queue_dir).glob("learning_case_*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        print(f"{path.name} | status={case.get('learning_status')} | valid_approval={is_approved(case)} | review={case.get('review')}")


def review_case(case_path, decision, analyst, note, signature_file=None):
    """PR #1 entry point; the case review is the sole signature authority.

    signature_file is retained for callers, but no separate signature database
    is written. Legacy approval flags must be reviewed again with evidence notes.
    """
    labels = {"approve": "known_ransomware_like", "benign": "benign", "reject": "rejected"}
    if decision not in labels:
        raise ValueError("Choose approve, benign, or reject")
    case = json.loads(Path(case_path).read_text(encoding="utf-8"))
    if decision == "approve" and not case.get("events"):
        raise ValueError("Behavior signature approval requires observed events")
    return record_review(case_path, labels[decision], analyst, note)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", type=Path, default=DEFAULT_QUEUE_DIR)
    parser.add_argument("--signature-file", type=Path, help="Deprecated compatibility option; reviews live in case files")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case-file")
    parser.add_argument("--decision", choices=["approve", "benign", "reject"])
    parser.add_argument("--analyst")
    parser.add_argument("--note")
    args = parser.parse_args()
    if args.list:
        list_cases(args.queue_dir)
        return
    if not all((args.case_file, args.decision, args.analyst, args.note)):
        parser.error("Review requires --case-file, --decision, --analyst and --note")
    try:
        path = resolve_case_path(args.case_file, args.queue_dir)
        review_case(path, args.decision, args.analyst, args.note, args.signature_file)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Review failed: {exc}\n")
    print(f"Review saved in {path}. Approved positive cases provide behavior signatures; promotion remains manual.")


if __name__ == "__main__":
    main()
