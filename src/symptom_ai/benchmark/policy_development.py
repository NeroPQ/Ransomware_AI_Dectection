"""Task 03 development/validation comparison; there is deliberately no test split.

Offline synthetic observations only. Does not open any Task 02 files or artifacts.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random

import joblib
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from symptom_ai.benchmark.detectors import fit_development, decoded_predictions
from symptom_ai.benchmark.quality_metrics import detection_metrics
from symptom_ai.benchmark.sessions import session_seed, frame_from_windows
from symptom_ai.models.evaluation import feature_matrix, related_groups
from symptom_ai.response_engine.evidence_policy import ALTERNATIVES, decide_policy, detector_evidence
from symptom_ai.telemetry.behavior import analyze_events

BENIGN = ("bulk_file_operations", "backup", "compression", "software_update", "cloud_sync", "build_tools", "administration", "network_polling")
POSITIVE = ("destructive_with_note", "destructive_with_recovery_tamper", "ambiguous_impact")
SEED = 310026


def make_event(operation, *, session="fixture", index=0, window=0, path=None, indicators=(), entropy=None, confidence=.95):
    event = {"schema_version": 1,
             "source": {"kind": "synthetic", "provider": "policy_development_v1", "event_id": operation,
                        "record_id": f"{session}:{window}:{index}"},
             "host": f"endpoint-{session}", "process": {"entity_id": f"process-{session}", "pid": 100, "image": "synthetic-placeholder"},
             "timestamp": (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=window * 60 + index % 50)).isoformat(),
             "operation": operation, "observation_confidence": confidence,
             "indicators": [{"kind": name, "reference": f"observation:{session}:{window}:{index}:{name}", "confidence": confidence}
                            for name in indicators]}
    if path is not None:
        event["path"] = path
    if operation == "file_rename":
        event["destination_path"] = path + ".renamed"
    if entropy is not None:
        event["entropy"] = {"object_type": "file_content", "unit": "bits_per_byte", "after": entropy,
                            "before": 4.0, "sample_bytes": 4096, "method": "shannon_byte_histogram"}
    return event


def generate(seed=SEED, sessions=30):
    if sessions < 6:
        raise ValueError("At least six independent sessions per profile required")
    rows = []
    profiles = {
        "bulk_file_operations": ["file_read", "file_touch", "file_rename", "file_modify"],
        "backup": ["file_read", "file_create", "file_touch", "file_modify"],
        "compression": ["file_read", "file_create", "file_modify"],
        "software_update": ["file_create", "file_modify", "file_rename", "process_create"],
        "cloud_sync": ["file_read", "file_modify", "file_rename", "network_poll"],
        "build_tools": ["file_create", "file_modify", "file_rename", "process_create"],
        "administration": ["file_read", "file_touch", "file_delete", "process_create", "network_connect"],
        "network_polling": ["network_poll", "network_connect"],
    }
    for profile in BENIGN + POSITIVE:
        for number in range(sessions):
            session = f"{profile}-{number:04d}"
            rng = random.Random(session_seed(seed, profile, number))
            rate = rng.uniform(.6, 1.6)
            pool = rng.randint(8, 100)
            directory_pool = rng.randint(1, 30)
            for window in range(8):
                attack = profile in POSITIVE and window >= 2
                impact = attack and window >= 4
                types = profiles.get(profile, ["file_read", "file_modify", "network_poll", "process_create"])
                if attack:
                    types = ["file_modify", "file_extension_change", "network_connect", "file_read"]
                if impact:
                    types = ["file_scramble", "ransom_note"] if profile != "destructive_with_recovery_tamper" else ["file_scramble", "recovery_change"]
                events = []
                for index in range(max(8, round(rng.uniform(20, 48) * rate))):
                    op = rng.choice(types)
                    path = f"C:\\lab\\directory-{rng.randrange(directory_pool)}\\file-{rng.randrange(pool)}.txt" if op.startswith("file_") or op == "ransom_note" else None
                    indicators = []
                    if impact and profile != "ambiguous_impact":
                        indicators = {"file_scramble": ["destructive_content"], "ransom_note": ["ransom_note"],
                                      "recovery_change": ["recovery_tamper"]}.get(op, [])
                    # Ambiguous impact deliberately has no external corroboration.
                    entropy = rng.uniform(7.2, 7.99) if op in {"file_modify", "file_create", "file_scramble"} and (attack or profile in {"compression", "backup"}) else None
                    event = make_event(op, session=session, window=window, index=index, path=path,
                                       indicators=indicators, entropy=entropy)
                    event["timestamp"] = (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=window * 60 + rng.uniform(0, 59.999))).isoformat()
                    events.append(event)
                symptoms, damage = analyze_events(events)
                rows.append({"sample_id": f"{session}:{window}", "session_id": session, "endpoint_id": f"endpoint-{session}",
                             "profile": profile, "window_index": window, "family": "unknown", "source_file": session,
                             "label": "known_ransomware_like" if attack else "benign", "time_basis": "simulated",
                             "window_start_seconds": window * 60, "window_end_seconds": (window + 1) * 60,
                             "attack_onset_seconds": 120 if profile in POSITIVE else None,
                             "impact_onset_seconds": 240 if profile in POSITIVE else None,
                             "events": events, "symptoms": symptoms, "damage": damage})
    return rows


def partitions(rows, seed=SEED):
    frame = frame_from_windows(rows)
    groups, relationships = related_groups(frame, feature_matrix(frame))
    profiles = set(frame["profile"])
    for train, validation in GroupShuffleSplit(n_splits=256, random_state=seed, test_size=.3).split(frame, groups=groups):
        if set(frame.iloc[train]["profile"]) == profiles == set(frame.iloc[validation]["profile"]):
            info = {"scope": "development_and_validation_only", "related_groups": relationships,
                    "session_ids": {name: sorted({rows[i]["session_id"] for i in indices})
                                    for name, indices in (("development", train), ("validation", validation))},
                    "group_ids": {"development": sorted(set(groups[train].tolist())), "validation": sorted(set(groups[validation].tolist()))}}
            return [rows[i] for i in train], [rows[i] for i in validation], info
    raise ValueError("Not enough independent groups; never weaken grouping")


def score(rows, bundle):
    X = feature_matrix(frame_from_windows(rows))[bundle["feature_columns"]]
    predicted = decoded_predictions(bundle["classifier"], X)
    anomaly = bundle["anomaly"].decision_function(X)
    outputs = []
    for row, prediction, decision in zip(rows, predicted, anomaly):
        ai = {"predicted_label": prediction, "isolation_forest_prediction": -1 if decision < 0 else 1,
              "isolation_forest_raw_score": float(decision), "model_feature_semantics_version": 3,
              "input_feature_semantics_version": 3}
        evidence = detector_evidence(row["symptoms"], row["damage"], ai)
        policies = {name: decide_policy(row["symptoms"], row["damage"], evidence, name) for name in ALTERNATIVES}
        outputs.append({"sample_id": row["sample_id"], "detector_evidence": evidence,
                        "policies": {name: {key: policy[key] for key in ("stage", "containment_recommended", "containment_executed")}
                                     for name, policy in policies.items()}})
    return outputs


def summarize(rows, outputs):
    methods = {f"{name}_only": detection_metrics(rows, [bool(out["detector_evidence"][name]["hit"]) for out in outputs])
               for name in ("rule", "classifier", "anomaly")}
    policies = {}
    for name in ALTERNATIVES:
        stages = [out["policies"][name]["stage"] for out in outputs]
        policies[name] = {
            "analyst_alert_or_containment": detection_metrics(rows, [stage in {"analyst_alert", "containment_recommendation"} for stage in stages]),
            "containment_recommendation": detection_metrics(rows, [stage == "containment_recommendation" for stage in stages]),
            "stage_counts": dict(Counter(stages)),
            "benign_stage_counts": dict(Counter(stage for stage, row in zip(stages, rows) if row["label"] == "benign")),
            "per_profile": {profile: {"windows": sum(row["profile"] == profile for row in rows),
                                      "stage_counts": dict(Counter(stage for row, stage in zip(rows, stages) if row["profile"] == profile))}
                            for profile in BENIGN + POSITIVE}}
    return {"detectors": methods, "policies": policies}


def run(root, seed=SEED, sessions=30):
    root = Path(root)
    if root.exists():
        raise ValueError("Output already exists; use a new development run")
    development, validation, split = partitions(generate(seed, sessions), seed)
    root.mkdir(parents=True)
    def write(name, value):
        (root / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    # Freeze each development comparison before scoring. These policies have
    # been iterated on development/validation, never the retired final holdout.
    source_root = Path(__file__).resolve().parents[1]
    implementation_files = [path for folder in ("telemetry", "response_engine", "benchmark")
                            for path in sorted((source_root / folder).glob("*.py"))]
    implementation_files += [source_root / name for name in ("models/train_symptom_models.py", "models/evaluation.py")]
    implementation = {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in implementation_files}
    protocol = {"implementation_sha256": implementation, "version": 1, "seed": seed, "sessions_per_profile": sessions, "windows_per_session": 8,
                "alternatives": list(ALTERNATIVES), "default": "evidence_balanced",
                "selection": "Semantic default, not a score-selected winner; policies developed on development/validation with a generic-volume context invariant; no independent final estimate",
                "test_partition": None, "task02_holdout_accessed": False, "split": split,
                "source": "synthetic_policy_generator_v1; no external endpoint telemetry",
                "safety": "Offline pure policy evaluation; no containment APIs, stop signals or endpoint operations"}
    write("protocol.json", protocol)
    bundle, fitting, _, _ = fit_development(development, validation)
    fitting["hybrid_policy"] = "v1 evidence policy; all alternatives, no test selection"
    bundle["feature_semantics_version"] = 3
    joblib.dump(bundle, root / "development_models.joblib")
    outputs = {name: score(rows, bundle) for name, rows in (("development", development), ("validation", validation))}
    report = {"protocol": protocol, "fitting": fitting,
              "partitions": {name: {"windows": len(rows), "sessions": len({row['session_id'] for row in rows}),
                                    "labels": dict(Counter(row["label"] for row in rows)), "results": summarize(rows, outputs[name])}
                             for name, rows in (("development", development), ("validation", validation))},
              "limitations": ["Synthetic dependent templates, labels and corroborating indicators are scripted",
                              "Validation participates in classifier selection and IF calibration; no independent final performance estimate",
                              "Task 02 published final holdout is retired and never read",
                              "No genuine Sysmon, observation confidence calibration or real endpoint-hour/latency measurements",
                              "Corroborating references assert separate observations, not independent trusted sensors",
                              "No production model promotion, response execution, process termination or host isolation"]}
    for name, rows in (("development", development), ("validation", validation)):
        with (root / f"{name}.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        write(f"{name}_predictions.json", outputs[name])
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.iterdir() if path.is_file()}
    source_root = Path(__file__).resolve().parents[1]
    hashes.update({str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for folder in ("telemetry", "response_engine") for path in sorted((source_root / folder).glob("*.py"))})
    report["sha256"] = hashes
    from symptom_ai.benchmark.baseline import versions
    report["versions"] = versions()
    write("report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sessions-per-profile", type=int, default=30)
    args = parser.parse_args()
    report = run(args.output, args.seed, args.sessions_per_profile)
    print(json.dumps({"output": str(args.output), "validation_windows": report["partitions"]["validation"]["windows"], "final_test": None}))


if __name__ == "__main__":
    main()
