"""Explicit labels and deterministic, batch-independent feature normalization."""
import re

import numpy as np
import pandas as pd

VALID_LABELS = {"benign", "known_ransomware_like"}
BENIGN = {"benign", "normal", "goodware", "good", "clean"}
FAMILIES = {"wannacry", "ryuk", "lockbit", "conti", "darkside", "cerber",
            "sodinokibi", "teslacrypt", "gandcrab"}


def text_label(value):
    text = str(value).strip().lower()
    tokens = set(re.findall(r"[a-z0-9_]+", text))
    benign = bool(tokens & BENIGN)
    ransomware = bool(tokens & (FAMILIES | {"ransom", "ransomware", "known_ransomware_like", "ransomware_like"}))
    if tokens & {"not", "non", "unknown", "unverified"}:
        return "unknown"
    if benign == ransomware:  # neither, or conflicting evidence
        return "unknown"
    return "benign" if benign else "known_ransomware_like"


def binary_labels(values, *, one_is_benign=False):
    """Only use for a dataset with a verified binary label convention."""
    values = pd.to_numeric(values, errors="coerce")
    mapping = {0: "benign", 1: "known_ransomware_like"}
    if one_is_benign:
        mapping = {1: "benign", 0: "known_ransomware_like"}
    return values.map(mapping).fillna("unknown")


def norm01(values):
    """Fixed saturating transform for nonnegative raw measurements; no fitted extrema.

    Missing observations become zero. This is a proxy, not a probability or a
    substitute for a dataset-specific unit/scale specification.
    """
    values = pd.Series(values) if np.isscalar(values) else values
    values = pd.to_numeric(values, errors="coerce").fillna(0)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Measurements must be finite and nonnegative")
    return values / (1.0 + values)


def validate_labels(frame):
    if "label" not in frame:
        raise ValueError("Dataset requires an explicit label column")
    invalid = ~frame["label"].isin(VALID_LABELS)
    if invalid.any():
        counts = frame.loc[invalid, "label"].fillna("<missing>").value_counts().to_dict()
        raise ValueError(f"Unverified or unsupported labels: {counts}; review before training")
    if frame.empty or frame["label"].nunique() < 2:
        raise ValueError("Training requires nonempty benign and ransomware classes")


def combine_frames(frames):
    if not frames:
        raise ValueError("No datasets to unify; run a parser with an available dataset first")
    for frame in frames:
        if "label" not in frame or frame["label"].isna().any():
            raise ValueError("Each dataset needs labels; missing labels cannot be filled with zero")
    # Numeric symptoms may default to absent; metadata must remain missing.
    from symptom_ai.dataset_ingestion.ingest_all_available_datasets import SYMPTOMS
    final = pd.concat(frames, ignore_index=True)
    columns = [c for c in SYMPTOMS if c in final]
    final[columns] = final[columns].fillna(0.0)
    return final
