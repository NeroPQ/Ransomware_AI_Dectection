"""Run safe demonstrations and PR #1 evaluators in a disposable synthetic lab.

Requires requirements-dev.txt. Output metrics validate plumbing only. No existing
models, reports, datasets, watcher state, or external services are modified.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO))
from test_foundation import synthetic_samples
from test_integration import family_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    summary = {"scope": "Synthetic interface verification only; no accuracy/generalization claims", "checks": []}
    with tempfile.TemporaryDirectory(prefix="ransomware-integration-") as directory:
        root = Path(directory)
        env = os.environ.copy()
        env.update(PYTHONPATH=str(REPO / "src") + os.pathsep + str(REPO), MPLBACKEND="Agg")
        # Limit scientific worker pools in the disposable verification subprocesses.
        env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        def run(label, *cmd, expected=0, cwd=None):
            with (root / "verification.log").open("a") as log:
                result = subprocess.run(list(map(str, cmd)), cwd=cwd or root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            if (result.returncode == 0) != (expected == 0):
                raise RuntimeError(f"{label}: exit {result.returncode}\n{(root / 'verification.log').read_text()[-7000:]}")
            summary["checks"].append({"check": label, "exit": result.returncode, "expected_failure": expected != 0})
            print(f"PASS {label} (exit {result.returncode})", flush=True)
        def module(label, name, *options, expected=0):
            run(label, sys.executable, "-m", name, *options, expected=expected)
        for script in ["generate_safe_behavior_logs.py", "build_behavior_features.py"]:
            run(script, sys.executable, REPO / script)
        run("single-session training refusal", sys.executable, REPO / "train_behavior_models.py", expected=1)
        run("demonstration behavior training", sys.executable, REPO / "train_behavior_models.py", "--allow-row-split")
        run("MITRE output", sys.executable, REPO / "mitre_mapping.py")
        for name in ["safe_demo", "unknown_demo"]:
            module(name, "symptom_ai." + name)
        for scenario in ["benign", "unusual", "known_progressive", "novel_zero_day"]:
            module(f"safe scenario {scenario}", "symptom_ai.live_monitoring.demo_scenario_runner",
                   "--scenario", scenario, "--duration", 4, "--delay", 0, "--reset-log")
        # Missing source data must fail explicitly; don't report external evaluation as run.
        module("missing real dataset refusal", "symptom_ai.models.prepare_heldout_family_v2", "--family", "Akira", expected=1)
        family_fixture().to_csv(root / "family_fixture.csv", index=False)
        for family in ["Akira", "LockBit"]:
            module(f"prepare {family}", "symptom_ai.models.prepare_heldout_family_v2", "--family", family,
                   "--dataset", "family_fixture.csv", "--train-per-class", 100, "--test-per-class", 30)
            module(f"evaluate {family}", "symptom_ai.models.evaluate_heldout_family_v2", "--family", family)
        module("held-out summary", "symptom_ai.models.build_heldout_family_summary")
        module("benign-only calibration", "symptom_ai.models.evaluate_isolation_forest_benign_only", "--fit-benign-rows", 60)
        for script in ["draw_figure_4_2.py", "draw_figure_4_3.py"]:
            run(script, sys.executable, REPO / script)
        dataset = root / "data/symptom_labels/unified_symptom_dataset.csv"
        dataset.parent.mkdir(parents=True, exist_ok=True)
        active_fixture = synthetic_samples()
        for column in ("behavior_type", "collection_type", "platform"):
            active_fixture[column] = "synthetic_verification"
        active_fixture.to_csv(dataset, index=False)
        module("symptom candidate training", "symptom_ai.models.train_symptom_models")
        module("training references", "symptom_ai.models.build_training_reference")
        module("prediction demo with calibrated shadow", "symptom_ai.models.predict_symptom_case")
        # Start only our own isolated localhost API, with its own nonsecret test key.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env["SYMPTOM_AI_API_BASE_URL"] = f"http://127.0.0.1:{port}"
        env["SYMPTOM_AI_API_KEY"] = "disposable-verification-key"
        with (root / "api.log").open("w") as api_log:
            process = subprocess.Popen([sys.executable, "-m", "uvicorn", "symptom_ai.api.main:app", "--host", "127.0.0.1", "--port", str(port)], cwd=root, env=env, stdout=api_log, stderr=subprocess.STDOUT)
            try:
                for _ in range(100):
                    if process.poll() is not None:
                        raise RuntimeError((root / "api.log").read_text())
                    try:
                        with urllib.request.urlopen(env["SYMPTOM_AI_API_BASE_URL"] + "/health", timeout=1):
                            break
                    except OSError:
                        time.sleep(.1)
                else:
                    raise RuntimeError("Disposable API failed to become ready")
                for case, truth in [("known_progressive", "ransomware"), ("novel_zero_day", "ransomware"), ("software_update_like", "benign")]:
                    path = REPO / "reports/evaluation/ablation_inputs" / f"{case}_events.jsonl"
                    module(f"ablation {case}", "symptom_ai.live_monitoring.ablation_evaluator", "--events", path, "--truth", truth, "--case-id", case)
                module("ablation summary", "symptom_ai.live_monitoring.build_ablation_summary")
                for scenario in ["backup_like", "cloud_sync_like", "compression_like", "software_update_like"]:
                    module(f"benign generator {scenario}", "symptom_ai.live_monitoring.benign_workload_runner", "--scenario", scenario, "--duration", 8, "--delay", 0, "--reset-log")
                    module(f"benign benchmark {scenario}", "symptom_ai.live_monitoring.ablation_evaluator", "--truth", "benign", "--case-id", scenario, "--output-dir", "reports/benign_verification")
                for scenario in ["known", "unknown"]:
                    lab = root / f"progressive_{scenario}"
                    lab.mkdir()
                    run(f"progressive {scenario}", sys.executable, "-m", "symptom_ai.simulation.safe_progressive_ransomware_sim", "--scenario", scenario, "--delay", 0, cwd=lab)
                    originals = list((lab / "data/simulated_enterprise").rglob("business_file_*.txt"))
                    assert len(originals) == 72 and all(path.read_text().startswith("Safe demo business document") for path in originals)
                # Invoke one watcher window over actual HTTP prediction and authenticated explanation.
                code = "from symptom_ai.live_monitoring import live_log_ai_watcher as w; events=[{'event_type':'high_entropy_write'}]*4+[{'event_type':'extension_change'}]*4+[{'event_type':'file_write'}]*8+[{'event_type':'file_rename'}]*6; a=w.evaluate_window(events,1); assert a['unknown_detector_shadow']['status']=='available',a; assert not a['containment_executed']; assert a['simulated_stop_requested']; assert a['dataset_evidence']['status']=='ok',a"
                run("watcher over HTTP with shadow and explanation", sys.executable, "-c", code)
                run("timeline helper", "bash", REPO / "check_demo_result.sh")
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        # Preserve only the small actual benchmark outcomes in the optional summary.
        summary["benign_workload_results"] = {path.stem: json.loads(path.read_text())["method_summaries"]
                                              for path in (root / "reports/benign_verification").glob("*.json")}
        summary["ablation_results"] = json.loads((root / "reports/evaluation/ablation_summary.json").read_text())
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Passed {len(summary['checks'])} isolated checks. All metrics use synthetic fixtures.")


if __name__ == "__main__":
    main()
