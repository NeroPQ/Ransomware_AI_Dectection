"""Deterministic synthetic sessions, not captured endpoint telemetry.

Each session uses a separate RNG/endpoint and latent activity rate. Times are
simulated exposure, never the generator's execution time. All paths are strings.
"""
from collections import Counter
import hashlib
import random

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from symptom_ai.live_monitoring.benign_workload_runner import workload_profile
from symptom_ai.benchmark.legacy_v2 import aggregate_symptoms, is_rule_detected
from symptom_ai.models.evaluation import feature_matrix, related_groups

BENIGN_PROFILES = ("backup_like", "compression_like", "software_update_like", "cloud_sync_like",
                   "development_tools", "routine_administration")
ATTACK_PROFILES = ("known_progressive", "novel_impact", "mixed_activity")
PROFILES = BENIGN_PROFILES + ATTACK_PROFILES
GENERATOR_VERSION = 1


def session_seed(seed, profile, number):
    return int.from_bytes(hashlib.sha256(f"{seed}:{profile}:{number}".encode()).digest()[:8], "big")


def generate_sessions(seed=20261026, sessions_per_profile=30, windows_per_session=8, window_seconds=60):
    if sessions_per_profile < 6 or windows_per_session < 6 or window_seconds <= 0:
        raise ValueError("Need >=6 sessions/profile, >=6 windows/session and positive window duration")
    windows = []
    for profile in PROFILES:
        for session in range(sessions_per_profile):
            rng = random.Random(session_seed(seed, profile, session))
            identity = f"{profile}-{session:04d}"
            rate = rng.uniform(.65, 1.5)
            path_pool = rng.randint(6, 45)
            for index in range(windows_per_session):
                attack = profile in ATTACK_PROFILES and index >= 2
                impact = attack and index >= 4
                if profile in BENIGN_PROFILES:
                    types = workload_profile(profile)["event_types"]
                    entropy = (.65, .96) if profile == "compression_like" else (.03, .55)
                else:
                    types = ["file_read", "file_write", "process_spawn", "api_behavior"]
                    entropy = (.05, .4)
                    if attack and not impact:
                        types += ["rare_process", "memory_anomaly", "network_beacon"]
                        entropy = (.3, .65)
                    if impact:
                        types = (["stealth_file_lock", "stealth_content_scramble", "stealth_extension_mutation", "strange_temp_write"]
                                 if profile == "novel_impact" else
                                 ["file_write", "file_rename", "extension_change", "high_entropy_write", "ransom_note_created"])
                        if profile == "mixed_activity":
                            types += ["file_read", "archive_chunk_write", "api_behavior", "file_write"] * 2
                        entropy = (.65, .99)
                count = max(4, round(rng.uniform(14, 38) * rate))
                events = []
                for event_index in range(count):
                    name = rng.choice(types)
                    # One endpoint per session; reads and legitimate writes can share
                    # ransomware-like counts. Context/labels never enter features.
                    events.append({"event_type": name, "host": f"endpoint-{identity}",
                                   "path": f"lab/{identity}/directory-{rng.randrange(3)}/file-{rng.randrange(path_pool)}.txt",
                                   "scenario": identity, "scenario_event_index": index * 1000 + event_index + 1,
                                   "extra": {"entropy_score": rng.uniform(*entropy),
                                             "memory_score": rng.uniform(.01, .2) if name != "memory_anomaly" else rng.uniform(.5, .95)}})
                symptoms, damage = aggregate_symptoms(events)
                windows.append({"sample_id": f"{identity}-window-{index:03d}", "session_id": identity,
                                "endpoint_id": f"endpoint-{identity}", "profile": profile, "window_index": index,
                                "dataset_source": "synthetic_quality_generator_v1", "family": "unknown",
                                "label": "known_ransomware_like" if attack else "benign",
                                "time_basis": "simulated", "window_start_seconds": index * window_seconds,
                                "window_end_seconds": (index + 1) * window_seconds,
                                "attack_onset_seconds": 2 * window_seconds if profile in ATTACK_PROFILES else None,
                                "impact_onset_seconds": 4 * window_seconds if profile in ATTACK_PROFILES else None,
                                "events": events, "symptoms": symptoms, "damage": damage,
                                "rule_hit": is_rule_detected(damage)})
    return windows


def frame_from_windows(windows):
    return pd.DataFrame([{**{key: value for key, value in row.items() if key not in {"events", "symptoms", "damage"}},
                          **row["symptoms"]} for row in windows])


def session_partitions(windows, seed):
    """Within-generator holdout only; endpoint/session/exact duplicates stay together.

    Template/source sharing is explicit. This is not cross-source or family
    generalization. Stratification seeks profile coverage, never detector scores.
    """
    frame = frame_from_windows(windows)
    relationships = frame.drop(columns="dataset_source").assign(source_file=frame["endpoint_id"])
    groups, info = related_groups(relationships, feature_matrix(frame))
    wanted = set(frame["profile"])
    def split(indices, split_seed, fraction):
        splitter = GroupShuffleSplit(n_splits=256, random_state=split_seed, test_size=fraction)
        for left, right in splitter.split(indices, groups=groups[indices]):
            a, b = indices[left], indices[right]
            if set(frame.iloc[a]["profile"]) == wanted == set(frame.iloc[b]["profile"]):
                return a, b
        raise ValueError("Insufficient independent groups with every profile; increase sessions, never weaken relationships")
    import numpy as np
    development, test = split(np.arange(len(frame)), seed, .2)
    train, validation = split(development, seed + 1, .25)
    parts = {name: [windows[int(index)] for index in indices] for name, indices in
             (("train", train), ("validation", validation), ("test", test))}
    info.update(scope="within_synthetic_generator", endpoint_relationship="source_file=endpoint_id",
                source_grouping_excluded=True, split_seed=seed, validation_seed=seed + 1,
                group_ids={name: sorted(set(groups[indices].tolist())) for name, indices in
                           (("train", train), ("validation", validation), ("test", test))})
    return parts, info
