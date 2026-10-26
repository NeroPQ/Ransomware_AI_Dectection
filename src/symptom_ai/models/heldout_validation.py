"""Observation and relationship checks for the within-source family protocol.

Unlike default cross-source training this explicitly evaluates within one source.
Sessions, files, sample IDs, families and exact observations remain disjoint.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

from symptom_ai.models.evaluation import feature_matrix, related_groups
from symptom_ai.dataset_ingestion.validation import VALID_LABELS


def validate_partitions(partitions):
    frames = list(partitions.values())
    for frame in frames:
        required = {"sample_id", "dataset_source", "family", "label"}
        if not required <= set(frame):
            raise ValueError(f"Missing required columns: {sorted(required - set(frame))}")
        if frame.empty or not frame["label"].isin(VALID_LABELS).all():
            raise ValueError("Partitions require nonempty, explicitly verified labels")
        ids = frame["sample_id"]
        if ids.isna().any() or ids.astype(str).str.strip().eq("").any() or ids.duplicated().any():
            raise ValueError("Partitions require unique nonmissing sample IDs")
    columns = list(feature_matrix(frames[0]).columns)
    if any(list(feature_matrix(frame).columns) != columns for frame in frames[1:]):
        raise ValueError("Partition observation feature names/order differ")
    combined = pd.concat(frames, ignore_index=True)
    if combined["sample_id"].duplicated().any():
        raise ValueError("Sample-ID leakage between partitions")
    groups, _ = related_groups(combined.drop(columns="dataset_source"), feature_matrix(combined))
    offset = 0
    seen = set()
    for frame in frames:
        current = set(groups[offset:offset + len(frame)])
        if seen & current:
            raise ValueError("Session/file/family/duplicate observation leakage between partitions")
        seen.update(current)
        offset += len(frame)
    return {"relationship_overlap": 0, "exact_observation_overlap": 0,
            "scope": "within_source_heldout_family", "source_grouping_excluded": True,
            "limitation": "Within-source, selected class counts; no cross-source or natural-prevalence efficacy claim"}


def load_family_split(run_dir):
    """Read only schema-2 splits bound to the preparation manifest."""
    root = Path(run_dir)
    manifest = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("feature_schema_version") != 2 or manifest.get("normalization") != "fixed_nonnegative_saturation_v1":
        raise ValueError("Legacy splits require re-ingestion and schema-2 preparation")
    parts = {}
    for name in ("train_positive", "train_benign", "test_positive", "test_benign"):
        path = root / "data" / f"{name}.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("files_sha256", {}).get(name):
            raise ValueError(f"Split file differs from preparation manifest: {name}")
        parts[name] = pd.read_csv(path)
    return parts
