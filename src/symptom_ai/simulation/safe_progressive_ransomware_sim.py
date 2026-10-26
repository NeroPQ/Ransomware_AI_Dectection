import os
from pathlib import Path
from datetime import datetime
import json
import time
import urllib.request


BASE = Path("data/simulated_enterprise")
REPORT_DIR = Path("reports/incidents")
UNKNOWN_DIR = Path("data/unknown_cases")

API_URL = os.getenv("SYMPTOM_AI_API_BASE_URL", "http://localhost:8000").rstrip("/") + "/respond"

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
]


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def prepare_lab_files(files_per_folder=6):
    BASE.mkdir(parents=True, exist_ok=True)

    for host in HOSTS:
        for folder in FOLDERS:
            d = BASE / host / folder
            d.mkdir(parents=True, exist_ok=True)

            for i in range(1, files_per_folder + 1):
                f = d / f"business_file_{i}.txt"
                if not f.exists() and not f.is_symlink():
                    f.write_text(
                        f"Safe demo business document {i} for {host}/{folder}\n",
                        encoding="utf-8"
                    )


def list_demo_files():
    return sorted([
        p for p in BASE.rglob("*")
        if p.is_file() and not p.is_symlink() and p.name.startswith("business_file_")
        and not p.name.endswith(".simlocked")
        and not p.name.endswith(".ransom_note_demo.txt")
    ])


def call_ai(symptoms):
    payload = json.dumps({"symptoms": symptoms}).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json", "X-API-Key": os.getenv("SYMPTOM_AI_API_KEY", "")},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {
            "api_error": str(e),
            "note": "Make sure FastAPI is running on localhost:8000"
        }


def is_dummy_lab_file(path):
    path = Path(path)
    try:
        return (not path.is_symlink() and path.is_file()
                and path.resolve().is_relative_to(BASE.resolve())
                and path.name.startswith("business_file_") and path.suffix == ".txt"
                and path.read_text(encoding="utf-8").startswith("Safe demo business document"))
    except (OSError, UnicodeError):
        return False


def simulate_touch_files(target_files, max_files):
    touched = []
    for path in target_files[:max_files]:
        if is_dummy_lab_file(path):
            content = path.read_text(encoding="utf-8")
            path.write_text(content + "[safe-sim-touch]\n", encoding="utf-8")
            touched.append(str(path))
    return touched


def simulate_safe_lock_files(target_files, max_files):
    """Write separate lab markers; retain every original dummy document."""
    locked = []
    for path in target_files[:max_files]:
        if not is_dummy_lab_file(path):
            continue
        marker_path = path.with_suffix(path.suffix + ".simlocked")
        if marker_path.is_symlink():
            continue
        if not marker_path.exists():
            marker_path.write_text("SAFE SIMULATION ONLY\nOriginal dummy content is preserved.\n", encoding="utf-8")
        locked.append(str(marker_path))
    return locked


def write_demo_note(host_dir):
    host_dir = Path(host_dir)
    if not host_dir.resolve().is_relative_to(BASE.resolve()):
        raise ValueError("Demo notes must remain inside the lab directory")
    note = host_dir / "READ_ME_ransom_note_demo.txt"
    if note.is_symlink():
        raise ValueError("Demo note cannot be a symlink")
    if not note.exists():
        note.write_text("SAFE RANSOMWARE NOTE SIMULATION ONLY.\nNo real ransomware was executed.\n", encoding="utf-8")
    return str(note)


def build_known_symptoms(stage):
    symptoms = {
        "file_write_burst": 0.10,
        "file_read_burst": 0.10,
        "file_rename_burst": 0.00,
        "high_entropy_write": 0.00,
        "mass_file_modification": 0.00,
        "suspicious_extension_change": 0.00,
        "suspicious_process_spawn": 0.10,
        "multi_directory_impact": 0.00,
        "user_document_impact_high": 0.00,
    }

    if stage >= 2:
        symptoms.update({
            "file_write_burst": 0.45,
            "file_read_burst": 0.35,
            "suspicious_process_spawn": 0.40,
            "multi_directory_impact": 0.30,
        })

    if stage >= 3:
        symptoms.update({
            "file_write_burst": 0.86,
            "file_rename_burst": 0.82,
            "high_entropy_write": 0.84,
            "mass_file_modification": 0.82,
            "suspicious_extension_change": 0.78,
            "file_api_usage": 0.80,
            "large_number_of_user_files_touched": 0.78,
            "user_document_impact_high": 0.82,
            "multi_directory_impact": 0.82,
            "ransom_note_created": 0.55,
        })

    if stage >= 4:
        symptoms.update({
            "file_write_burst": 0.92,
            "file_rename_burst": 0.90,
            "high_entropy_write": 0.93,
            "mass_file_modification": 0.92,
            "suspicious_extension_change": 0.90,
            "ransom_note_created": 0.85,
            "user_document_impact_high": 0.90,
            "multi_directory_impact": 0.90,
        })

    return symptoms


def build_unknown_symptoms(stage):
    symptoms = {
        "memory_access_spike": 0.20,
        "storage_write_spike": 0.20,
        "rare_process_name": 0.20,
        "novel_symptom_combination": 0.20,
        "file_write_burst": 0.10,
        "high_entropy_write": 0.10,
    }

    if stage >= 2:
        symptoms.update({
            "memory_access_spike": 0.70,
            "storage_write_spike": 0.68,
            "rare_process_name": 0.70,
            "process_tree_anomaly": 0.65,
            "novel_symptom_combination": 0.75,
            "file_write_burst": 0.45,
        })

    if stage >= 3:
        symptoms.update({
            "memory_access_spike": 0.85,
            "storage_write_spike": 0.85,
            "rare_process_name": 0.88,
            "novel_symptom_combination": 0.90,
            "partial_ransomware_match": 0.55,
            "file_write_burst": 0.70,
            "high_entropy_write": 0.75,
            "mass_file_modification": 0.60,
        })

    if stage >= 4:
        symptoms.update({
            "unknown_high_risk": 0.95,
            "analyst_review_required": 0.90,
            "retraining_candidate": 0.95,
            "storage_write_spike": 0.92,
            "memory_access_spike": 0.92,
            "file_write_burst": 0.82,
            "high_entropy_write": 0.88,
            "mass_file_modification": 0.82,
            "suspicious_extension_change": 0.70,
        })

    return symptoms


def summarize_damage():
    locked = list(BASE.rglob("*.simlocked"))
    notes = list(BASE.rglob("*ransom_note_demo.txt"))
    all_files = list(BASE.rglob("*"))

    impacted_hosts = set()
    impacted_dirs = set()

    for p in locked:
        try:
            rel = p.relative_to(BASE)
            impacted_hosts.add(rel.parts[0])
            impacted_dirs.add(str(p.parent))
        except Exception:
            pass

    return {
        "impacted_hosts": sorted(impacted_hosts),
        "impacted_host_count": len(impacted_hosts),
        "impacted_directory_count": len(impacted_dirs),
        "simulated_locked_file_count": len(locked),
        "ransom_note_demo_count": len(notes),
        "total_lab_file_count": len([p for p in all_files if p.is_file()]),
    }


def save_unknown_learning_case(incident_id, stage, symptoms, ai_result, damage):
    UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)

    out = UNKNOWN_DIR / f"learned_unknown_behavior_{incident_id}_stage_{stage}.json"

    case = {
        "incident_id": incident_id,
        "saved_at": now(),
        "reason": "Safe simulated ransomware reached file-locking stage but behavior is treated as unknown/retraining candidate.",
        "symptoms": symptoms,
        "ai_result": ai_result,
        "damage_summary": damage,
        "learning_status": "pending_review",
        "confirmed_label": None,
        "safety_note": "No real ransomware was executed. Only dummy files inside data/simulated_enterprise were modified."
    }

    out.write_text(json.dumps(case, indent=2), encoding="utf-8")
    return str(out)


def run_simulation(scenario="known", delay=1.0):
    incident_id = f"{scenario}_{stamp()}"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    prepare_lab_files()

    timeline = []
    stopped = False
    stop_reason = None
    unknown_learning_file = None

    stages = [
        {
            "stage": 1,
            "name": "Initial suspicious file activity",
            "host_limit": 1,
            "files_to_touch": 3,
            "files_to_lock": 0,
        },
        {
            "stage": 2,
            "name": "Wider file modification across one host",
            "host_limit": 1,
            "files_to_touch": 10,
            "files_to_lock": 0,
        },
        {
            "stage": 3,
            "name": "Multi-directory impact",
            "host_limit": 2,
            "files_to_touch": 20,
            "files_to_lock": 6,
        },
        {
            "stage": 4,
            "name": "Simulated encryption impact expands",
            "host_limit": 4,
            "files_to_touch": 40,
            "files_to_lock": 16,
        },
    ]

    for s in stages:
        stage = s["stage"]

        active_hosts = HOSTS[:s["host_limit"]]
        target_files = []

        for host in active_hosts:
            target_files.extend([
                p for p in (BASE / host).rglob("*.txt")
                if is_dummy_lab_file(p)
            ])

        touched = simulate_touch_files(target_files, s["files_to_touch"])
        locked = simulate_safe_lock_files(target_files, s["files_to_lock"])

        notes = []
        if locked:
            for host in active_hosts:
                notes.append(write_demo_note(BASE / host))

        if scenario == "known":
            symptoms = build_known_symptoms(stage)
        else:
            symptoms = build_unknown_symptoms(stage)

        ai_result = call_ai(symptoms)
        damage = summarize_damage()

        event = {
            "time": now(),
            "stage": stage,
            "stage_name": s["name"],
            "scenario": scenario,
            "active_hosts": active_hosts,
            "touched_files_count": len(touched),
            "simulated_locked_files_count": len(locked),
            "notes_created_count": len(notes),
            "symptoms": symptoms,
            "ai_result": ai_result,
            "damage_summary": damage,
        }

        timeline.append(event)

        policy = ai_result.get("response", {}).get("policy")
        severity = ai_result.get("response", {}).get("severity")
        risk = ai_result.get("risk_score")
        predicted = ai_result.get("predicted_label")
        unknown_risk = ai_result.get("unknown_risk")

        print("\n" + "=" * 80)
        print(f"[{now()}] Stage {stage}: {s['name']}")
        print(f"Scenario: {scenario}")
        print(f"Predicted: {predicted}")
        print(f"Risk: {risk}")
        print(f"Unknown risk: {unknown_risk}")
        print(f"Severity: {severity}")
        print(f"Policy: {policy}")
        print(f"Damage: {damage}")

        if policy in {"isolate_and_backup", "protective_lockdown"} and stage >= 3:
            stopped = True
            stop_reason = f"AI response policy triggered: {policy}"
            print(f"[!] Simulation stopped: {stop_reason}")
            break

        if scenario == "unknown" and damage["simulated_locked_file_count"] > 0:
            unknown_learning_file = save_unknown_learning_case(
                incident_id=incident_id,
                stage=stage,
                symptoms=symptoms,
                ai_result=ai_result,
                damage=damage
            )
            print(f"[!] Unknown behavior queued for retraining: {unknown_learning_file}")

        time.sleep(delay)

    final_damage = summarize_damage()

    report = {
        "incident_id": incident_id,
        "scenario": scenario,
        "started_at": timeline[0]["time"] if timeline else now(),
        "ended_at": now(),
        "stopped": stopped,
        "stop_reason": stop_reason,
        "unknown_learning_file": unknown_learning_file,
        "final_damage_summary": final_damage,
        "timeline": timeline,
        "safety_note": "Safe simulation only. No real ransomware, no real encryption, no network propagation."
    }

    out = REPORT_DIR / f"incident_report_{incident_id}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"[+] Incident report saved: {out}")
    print(f"[+] Final damage summary: {final_damage}")

    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=["known", "unknown"],
        default="known"
    )
    parser.add_argument("--delay", type=float, default=1.0)

    args = parser.parse_args()
    run_simulation(scenario=args.scenario, delay=args.delay)
