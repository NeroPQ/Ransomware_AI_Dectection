"""Build reviewed training candidates without replacing production data/models."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import shutil
import subprocess
import sys
from uuid import uuid4

import pandas as pd

from symptom_ai.auto_learning.review import is_approved
from symptom_ai.models.evaluation import feature_matrix

BASE_DATASET = Path("data/symptom_labels/unified_symptom_dataset.csv")
QUEUE_DIR = Path("data/learning_queue")
MODEL_DIR = Path("models")
VERSIONS_DIR = MODEL_DIR / "versions"
REPORT_DIR = Path("reports/auto_learning")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:8]


def load_learning_cases():
    cases = []
    for path in sorted(QUEUE_DIR.glob("learning_case_*.json")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(case, dict) and is_approved(case):
                cases.append({**case, "_case_file": str(path)})
        except (ValueError, OSError) as exc:
            print(f"Skipping invalid case {path.name}: {type(exc).__name__}")
    return cases


def cases_to_dataframe(cases, base_columns):
    rows = []
    for case in cases:
        if not is_approved(case):
            continue
        row = {column: 0.0 for column in base_columns}
        for key, value in case["symptoms"].items():
            if key in row:
                row[key] = value
        # All cases from this unverified live collection share a source group.
        row.update({"sample_id": "reviewed_" + case["review"]["evidence_sha256"],
                    "dataset_source": "Reviewed_Live_Simulation", "family": "unknown",
                    "behavior_type": "reviewed_behavior", "collection_type": "live_log_watcher",
                    "platform": "simulation", "is_simulated": 1, "is_real_malware_executed": 0,
                    "label": case["review"]["confirmed_label"], "response_policy": "analyst_review"})
        rows.append(row)
    frame = pd.DataFrame(rows).drop_duplicates(subset=["sample_id"]) if rows else pd.DataFrame(columns=base_columns)
    if not frame.empty:
        feature_matrix(frame)  # malformed approved observations still fail validation
    return frame


def archive_current_models(version_name):
    version_dir = VERSIONS_DIR / version_name
    version_dir.mkdir(parents=True, exist_ok=False)
    for path in list(MODEL_DIR.glob("*.joblib")) + list(MODEL_DIR.glob("*.json")):
        shutil.copy2(path, version_dir / path.name)
    return version_dir


def main():
    if not BASE_DATASET.is_file():
        raise SystemExit(f"Missing base dataset: {BASE_DATASET}")
    cases = load_learning_cases()
    if not cases:
        print("No approved cases; no retraining performed.")
        return
    base = pd.read_csv(BASE_DATASET)
    if "sample_id" not in base or base["sample_id"].isna().any() or base["sample_id"].duplicated().any():
        raise SystemExit("Reviewed retraining requires stable, nonmissing, unique sample_id values in the base dataset")
    new_rows = cases_to_dataframe(cases, list(base.columns))
    final = pd.concat([base, new_rows], ignore_index=True).drop_duplicates(subset=["sample_id"])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    version_name = f"candidate_{stamp()}"
    candidate = VERSIONS_DIR / version_name
    candidate.mkdir(parents=True, exist_ok=False)
    dataset = candidate / "reviewed_training_dataset.csv"
    final.to_csv(dataset, index=False)
    backup = archive_current_models(f"before_{version_name}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "symptom_ai.models.train_symptom_models", "--dataset", str(dataset.resolve()),
           "--model-dir", str((candidate / "models").resolve()), "--report-dir", str((candidate / "reports").resolve())]
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=3600)
        (candidate / "training.log").write_text(result.stdout + "\nSTDERR:\n" + result.stderr, encoding="utf-8")
        status = "candidate_ready_for_review" if result.returncode == 0 else "candidate_failed"
        returncode = result.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        status, returncode = "candidate_failed", 1
        (candidate / "training.log").write_text(type(exc).__name__, encoding="utf-8")
    summary = {"status": status, "approved_cases": len(cases), "candidate": str(candidate),
               "archived_previous_models": str(backup), "production_modified": False,
               "promotion": "Manual review required; active dataset and models were never replaced"}
    (candidate / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if returncode:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
