"""Prepare, freeze and evaluate an offline synthetic detection-quality baseline."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import joblib
import numpy as np

from symptom_ai.benchmark.sessions import generate_sessions, session_partitions, frame_from_windows, PROFILES, GENERATOR_VERSION
from symptom_ai.benchmark.detectors import fit_development, score_windows, summarize, validation_diagnostics, METHODS
from symptom_ai.benchmark.sources import inventory, reviewed_workflow, sha256, REPO
from symptom_ai.models.evaluation import feature_matrix

PROTOCOL_VERSION = 1


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def code_fingerprint():
    files = list((REPO / "src/symptom_ai/benchmark").glob("*.py")) + [REPO / "src/symptom_ai" / name for name in (
        "models/evaluation.py", "models/train_symptom_models.py", "response_engine/response_policy.py",
        "live_monitoring/live_log_ai_watcher.py", "live_monitoring/ablation_evaluator.py",
        "live_monitoring/benign_workload_runner.py")]
    hashes = {str(path.relative_to(REPO)): sha256(path) for path in sorted(files)}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def versions():
    return {"python": platform.python_version(), **{name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "xgboost", "joblib")}}


def write_rows(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def load_manifest(root):
    manifest = json.loads((root / "split_manifest.json").read_text())
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported benchmark protocol")
    if manifest["implementation_sha256"] != code_fingerprint():
        raise ValueError("Implementation changed; create a new versioned benchmark rather than reusing final holdout")
    for name in ("train", "validation", "test"):
        if sha256(root / f"{name}.jsonl") != manifest["partitions"][name]["sha256"]:
            raise ValueError(f"Benchmark partition changed: {name}")
    return manifest


def read_rows(root, name):
    return [json.loads(line) for line in (root / f"{name}.jsonl").read_text().splitlines()]


def prepare(root, seed=20261026, sessions_per_profile=30, windows_per_session=8, window_seconds=60):
    root = Path(root)
    if root.exists():
        raise ValueError("Output exists; use a new directory. Final holdouts must not be overwritten")
    rows = generate_sessions(seed, sessions_per_profile, windows_per_session, window_seconds)
    parts, relationships = session_partitions(rows, seed)
    root.mkdir(parents=True)
    manifest = {"protocol_version": PROTOCOL_VERSION, "source_kind": "synthetic_only",
                "generator_version": GENERATOR_VERSION, "implementation_sha256": code_fingerprint(),
                "configuration": {"seed": seed, "sessions_per_profile": sessions_per_profile,
                                  "windows_per_session": windows_per_session, "window_seconds": window_seconds},
                "partitioning": relationships, "partitions": {}, "external_sources": inventory(),
                "protocol": "60/20/20 group split; both development partitions precede final model selection; profiles shared across splits",
                "safety": "Offline log strings and models only; no response APIs, stop files or learned signatures"}
    for name, partition in parts.items():
        write_rows(root / f"{name}.jsonl", partition)
        manifest["partitions"][name] = {"sha256": sha256(root / f"{name}.jsonl"), "windows": len(partition),
                                         "sessions": sorted({row["session_id"] for row in partition}),
                                         "endpoints": sorted({row["endpoint_id"] for row in partition}),
                                         "profiles": dict(Counter(row["profile"] for row in partition)),
                                         "label_counts": dict(Counter(row["label"] for row in partition))}
    write_json(root / "split_manifest.json", manifest)
    return manifest


def calibrate(root):
    root = Path(root)
    if (root / "frozen.json").exists() or (root / "report.json").exists():
        raise ValueError("Calibration already frozen; do not retune against this holdout")
    manifest = load_manifest(root)
    train, validation = read_rows(root, "train"), read_rows(root, "validation")
    bundle, development, X, V = fit_development(train, validation)
    diagnostics = validation_diagnostics(bundle, train, validation, X, V)
    write_json(root / "validation_diagnostics.json", diagnostics)
    joblib.dump(bundle, root / "frozen_models.joblib")
    frozen = {"protocol_version": PROTOCOL_VERSION, "implementation_sha256": code_fingerprint(), "versions": versions(),
              "manifest_sha256": sha256(root / "split_manifest.json"), "model_sha256": sha256(root / "frozen_models.joblib"),
              "validation_diagnostics_sha256": sha256(root / "validation_diagnostics.json"),
              "development": development, "feature_columns": bundle["feature_columns"],
              "test_features_used_for_fitting_or_calibration": False,
              "selection_and_calibration": "fixed RF/XGB specifications; validation macro-F1 selection; benign validation nominal 5% IF FPR; unchanged hybrid rules"}
    write_json(root / "frozen.json", frozen)
    return frozen


def clustered_intervals(rows, outputs, method, seed, replicates=500):
    """Resample sessions, not correlated windows. Conditional on this generator."""
    sessions = sorted({row["session_id"] for row in rows})
    counts = []
    for session in sessions:
        sample = [(row["label"] == "known_ransomware_like", output[method])
                  for row, output in zip(rows, outputs) if row["session_id"] == session]
        counts.append([sum(t and p for t, p in sample), sum(not t and p for t, p in sample),
                       sum(not t and not p for t, p in sample), sum(t and not p for t, p in sample)])
    rng = np.random.default_rng(seed)
    draws = np.array(counts)[rng.integers(0, len(sessions), size=(replicates, len(sessions)))].sum(axis=1)
    tp, fp, tn, fn = draws.T
    intervals = {}
    for metric, num, den in (("precision", tp, tp + fp), ("recall", tp, tp + fn),
                              ("f1", 2 * tp, 2 * tp + fp + fn), ("fpr", fp, fp + tn)):
        valid = den > 0
        intervals[metric] = {"lower": float(np.quantile(num[valid] / den[valid], .025)) if valid.any() else None,
                             "upper": float(np.quantile(num[valid] / den[valid], .975)) if valid.any() else None,
                             "valid_replicates": int(valid.sum())}
    return {"unit": "session", "replicates": replicates, "seed": seed, "percentile_95": intervals,
            "limitation": "Conditional on shared synthetic templates; not enterprise uncertainty or independent-window intervals"}


def evaluate(root):
    root = Path(root)
    if (root / "report.json").exists() or (root / "test_predictions.jsonl").exists():
        raise ValueError("Final test already consumed; report is immutable")
    manifest = load_manifest(root)
    frozen = json.loads((root / "frozen.json").read_text())
    if (frozen["manifest_sha256"] != sha256(root / "split_manifest.json")
            or frozen["model_sha256"] != sha256(root / "frozen_models.joblib")
            or frozen["validation_diagnostics_sha256"] != sha256(root / "validation_diagnostics.json")
            or frozen["implementation_sha256"] != code_fingerprint() or frozen["versions"] != versions()):
        raise ValueError("Frozen artifacts, data, code or dependencies changed; evaluation refused")
    bundle = joblib.load(root / "frozen_models.joblib")  # trusted local, checksum-bound only
    rows = read_rows(root, "test")
    outputs = score_windows(bundle, rows)
    methods = summarize(rows, outputs)
    for index, method in enumerate(METHODS):
        methods[method]["session_bootstrap"] = clustered_intervals(rows, outputs, method, manifest["configuration"]["seed"] + index)
    profiles = {}
    for profile in PROFILES:
        pairs = [(row, output) for row, output in zip(rows, outputs) if row["profile"] == profile]
        profiles[profile] = summarize([row for row, _ in pairs], [output for _, output in pairs])
    report = {"title": "Synthetic detection-quality baseline; no real-world performance claim",
              "protocol_version": PROTOCOL_VERSION, "source_kind": "synthetic_only", "versions": versions(),
              "configuration": manifest["configuration"], "implementation_sha256": manifest["implementation_sha256"],
              "split_manifest_sha256": sha256(root / "split_manifest.json"), "frozen_decisions_sha256": sha256(root / "frozen.json"),
              "validation_diagnostics_sha256": frozen["validation_diagnostics_sha256"],
              "partitions": manifest["partitions"], "partitioning": manifest["partitioning"],
              "external_sources": manifest["external_sources"], "feature_columns": bundle["feature_columns"],
              "development": frozen["development"], "test_results": methods, "per_profile": profiles,
              "method_definitions": {"rule_only": "Existing impact-count rule, independent of both models",
                                     "classifier_only": "Validation-selected RF or XGBoost class prediction",
                                     "anomaly_only": "Benign-only IF decision_function < 0 at frozen validation threshold",
                                     "hybrid": "Existing watcher actionable alert set; impact-only review is excluded"},
              "test_false_alarm_alert_types": dict(Counter(out["hybrid_alert"] for row, out in zip(rows, outputs) if row["label"] == "benign" and out["hybrid"])),
              "test_containment_recommendations": sum(out["containment_recommended"] for out in outputs),
              "actions_executed": 0,
              "limitations": ["All observations, endpoints, timestamps and labels are generated; no captured endpoint exposure",
                              "Independent random sessions share templates and a generator; no new family/source/site generalization",
                              "Positive labels reflect scripted phase onset, not verified malware or fileless telemetry",
                              "Window counts are correlated within sessions; intervals resample sessions",
                              "Real endpoint-hour and latency metrics unavailable; simulated equivalents are explicitly named",
                              "No live artifacts loaded, no reviewed signatures, no response actions or threshold changes deployed",
                              "Known grouping cannot exclude near duplicates or unspecified real-world dependencies",
                              "Once a final result is viewed it is exploratory for future changes; a future protocol needs new independent data"]}
    predictions = [{**{key: row[key] for key in ("sample_id", "session_id", "endpoint_id", "profile", "label", "window_start_seconds", "window_end_seconds")}, **out}
                   for row, out in zip(rows, outputs)]
    write_rows(root / "test_predictions.jsonl", predictions)
    report["test_predictions_sha256"] = sha256(root / "test_predictions.jsonl")
    write_json(root / "report.json", report)
    render_report(root, report)
    return report


def render_report(root, report):
    lines = ["# Detection-quality baseline", "", report["title"], "",
             "Protocol: prepare session holdouts → fit/train and diagnose on development data → freeze → evaluate test once.", "",
             "| Method | TP | FP | TN | FN | Precision | Recall | F1 | FPR | False alerts / simulated benign endpoint-hour |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def show(value):
        return "n/a" if value is None else f"{value:.4f}"
    for name, row in report["test_results"].items():
        values = [name] + [str(row[key]) for key in ("true_positive", "false_positive", "true_negative", "false_negative")] + [show(row[key]) for key in ("precision", "recall", "f1", "fpr", "false_alerts_per_simulated_endpoint_hour")]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "Real endpoint-hour and real latency fields are null. Rates above use simulated exposure and one alert per alerting window; no deduplication.", "",
              "## Provenance", "", f"Generator configuration: `{json.dumps(report['configuration'], sort_keys=True)}`.",
              f"Implementation SHA-256: `{report['implementation_sha256']}`.",
              f"Split manifest SHA-256: `{report['split_manifest_sha256']}`.",
              f"Frozen decisions SHA-256: `{report['frozen_decisions_sha256']}`.",
              "Captured external datasets were inventoried; none is automatically considered reviewed.", "",
              "## Limits", ""] + ["- " + item for item in report["limitations"]]
    (Path(root) / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "calibrate", "evaluate", "run", "reviewed-sources"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--sessions-per-profile", type=int, default=30)
    parser.add_argument("--windows-per-session", type=int, default=8)
    parser.add_argument("--window-seconds", type=int, default=60)
    parser.add_argument("--review-manifest", type=Path)
    args = parser.parse_args()
    try:
        if args.command in {"prepare", "run"}:
            prepare(args.output, args.seed, args.sessions_per_profile, args.windows_per_session, args.window_seconds)
        if args.command in {"calibrate", "run"}:
            calibrate(args.output)
        if args.command in {"evaluate", "run"}:
            report = evaluate(args.output)
            print(json.dumps({name: {key: metrics[key] for key in ("precision", "recall", "f1", "fpr")} for name, metrics in report["test_results"].items()}, indent=2))
        if args.command == "reviewed-sources":
            if not args.review_manifest:
                raise ValueError("reviewed-sources requires --review-manifest")
            reviewed_workflow(args.review_manifest, args.output)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Benchmark refused: {exc}\n")
    print(f"{args.command} completed: {args.output}")


if __name__ == "__main__":
    main()
