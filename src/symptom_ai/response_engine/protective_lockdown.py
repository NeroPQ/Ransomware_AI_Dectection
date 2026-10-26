from pathlib import Path
from datetime import datetime
import shutil
import json
from uuid import uuid4


def ensure_demo_protected_docs(protected_dir: str = "data/protected_docs") -> None:
    protected_path = Path(protected_dir)
    protected_path.mkdir(parents=True, exist_ok=True)

    for i in range(1, 4):
        f = protected_path / f"important_report_{i}.txt"
        if not f.exists() and not f.is_symlink():
            f.write_text(
                f"Important protected demo document {i}\n",
                encoding="utf-8"
            )


def emergency_backup(
    protected_dir: str = "data/protected_docs",
    backup_root: str = "data/emergency_backup"
) -> str:
    protected_path = Path(protected_dir)
    backup_path = Path(backup_root)

    protected_path.mkdir(parents=True, exist_ok=True)
    backup_path.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:8]
    current_backup = backup_path / f"backup_{stamp}"
    current_backup.mkdir(parents=True, exist_ok=True)

    for item in protected_path.glob("*"):
        if item.is_file() and not item.is_symlink():
            shutil.copy2(item, current_backup / item.name)

    return str(current_backup)


def lock_protected_files_demo(protected_dir: str = "data/protected_docs") -> list:
    """
    Demo-only lock:
    records which lab files would be locked, without changing permissions or contents.
    """
    protected_path = Path(protected_dir)
    protected_path.mkdir(parents=True, exist_ok=True)

    locked_files = []

    for item in protected_path.glob("*"):
        if item.is_file() and not item.is_symlink():
            locked_files.append(str(item))

    return locked_files


def unlock_protected_files_demo(protected_dir: str = "data/protected_docs") -> list:
    protected_path = Path(protected_dir)
    unlocked_files = []

    if protected_path.exists():
        for item in protected_path.glob("*"):
            if item.is_file() and not item.is_symlink():
                unlocked_files.append(str(item))

    return unlocked_files


def store_unknown_case(
    case: dict,
    unknown_case_dir: str = "data/unknown_cases"
) -> str:
    out_dir = Path(unknown_case_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:8]
    out_file = out_dir / f"unknown_case_{stamp}.json"

    out_file.write_text(json.dumps({**case, "learning_status": "pending_review", "confirmed_label": None}, indent=2), encoding="utf-8")

    return str(out_file)


def run_protective_lockdown(case: dict) -> dict:
    ensure_demo_protected_docs()

    backup_path = emergency_backup()
    locked_files = lock_protected_files_demo()
    unknown_case_path = store_unknown_case(case)

    return {
        "lockdown_mode": True,
        "simulated_containment_executed": True,
        "containment_executed": False,
        "response_mode": "simulation",
        "backup_path": backup_path,
        "locked_files_demo": locked_files,
        "unknown_case_path": unknown_case_path,
        "note": "Simulation recorded; no permissions changed, process stopped, or host isolated."
    }
