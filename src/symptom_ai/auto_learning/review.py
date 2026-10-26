"""Local analyst review: detection outputs never grant their own approval."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

from symptom_ai.dataset_ingestion.validation import VALID_LABELS


def case_digest(case):
    evidence = {key: case.get(key) for key in ("symptoms", "events", "damage")}
    return hashlib.sha256(json.dumps(evidence, sort_keys=True, allow_nan=False).encode()).hexdigest()


def is_approved(case):
    if not isinstance(case, dict) or not isinstance(case.get("symptoms"), dict):
        return False
    review = case.get("review")
    if not isinstance(review, dict):
        return False
    try:
        return (case.get("learning_status") == "approved"
                and review.get("confirmed_label") in VALID_LABELS
                and isinstance(review.get("reviewer"), str) and bool(review["reviewer"].strip())
                and isinstance(review.get("notes"), str) and bool(review["notes"].strip())
                and review.get("evidence_sha256") == case_digest(case))
    except (ValueError, TypeError):
        return False


def review_case(path, label, reviewer, notes):
    if label not in VALID_LABELS | {"rejected"} or not reviewer.strip() or not notes.strip():
        raise ValueError("Review requires an explicit label/rejection, reviewer and evidence notes")
    path = Path(path)
    case = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(case, dict) or not isinstance(case.get("symptoms"), dict):
        raise ValueError("Not a learning case")
    case["learning_status"] = "rejected" if label == "rejected" else "approved"
    case["review"] = {"confirmed_label": label, "reviewer": reviewer.strip(), "notes": notes.strip(),
                      "reviewed_at": datetime.now(timezone.utc).isoformat(), "evidence_sha256": case_digest(case)}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(case, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--label", required=True, choices=sorted(VALID_LABELS | {"rejected"}))
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--notes", required=True)
    args = parser.parse_args()
    review_case(args.case, args.label, args.reviewer, args.notes)
    print("Review saved. Dataset/model promotion remains manual.")


if __name__ == "__main__":
    main()
