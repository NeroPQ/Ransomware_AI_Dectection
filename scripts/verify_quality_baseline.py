"""Verify raw synthetic feature ingestion → unification → training → API loading.

This contract fixture is not genuine reviewed source data or an efficacy study.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from symptom_ai.api import main as api
from symptom_ai.benchmark.sources import reviewed_workflow, sha256


def create_source_fixture(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(882)
    sources = []
    for index in range(15):
        rows = []
        for sample in range(20):
            positive = sample % 2
            rows.append({"sample_id": f"source-{index}-sample-{sample}", "dataset_source": f"fixture_source_{index}",
                         "family": f"fixture_family_{index}", "label": "known_ransomware_like" if positive else "benign",
                         "file_write_burst": float(rng.uniform(5, 50) if positive else rng.uniform(0, 4)),
                         "high_entropy_write": float(rng.uniform(0, 8)), "analyst_review_required": positive,
                         "behavior_type": "synthetic_fixture", "collection_type": "synthetic_fixture", "platform": "simulation",
                         "is_simulated": 1, "is_real_malware_executed": 0, "response_policy": "analyst_review"})
        path = root / f"raw_source_{index}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        sources.append({"path": path.name, "sha256": sha256(path), "adapter": "raw_observation_fixture",
                        "observation_columns": ["file_write_burst", "high_entropy_write"],
                        "provenance": "Deterministic generated raw measurements; seed 882, not captured telemetry",
                        "label_definition": "Alternating scripted classes, solely for ingestion/training contract tests"})
    manifest = root / "review_manifest.json"
    manifest.write_text(json.dumps({"schema_version": 2, "normalization": "fixed_nonnegative_saturation_v1",
                                    "source_kind": "synthetic_contract_fixture", "reviewer": "fixture_contract_test",
                                    "reviewed_at": "2026-10-26", "review_notes": "Fixture labels and transform only; no real-world review/accuracy asserted",
                                    "sources": sources}, indent=2))
    return manifest


def verify(root):
    root = Path(root)
    manifest = create_source_fixture(root / "sources")
    report = reviewed_workflow(manifest, root / "workflow")
    candidate = root / "workflow/candidate_models"
    with patch.object(api, "MODEL_DIR", candidate):
        response = TestClient(api.app).post("/predict", json={"symptoms": {"file_write_burst": .1, "high_entropy_write": .1}})
    if response.status_code != 200 or response.json()["response"]["containment_executed"]:
        raise RuntimeError(f"Candidate inference failed: {response.status_code}")
    if report["rows"] != 300 or report["source_kind"] != "synthetic_contract_fixture":
        raise RuntimeError("Fixture provenance mismatch")
    if "analyst_review_required" in report["feature_columns"]:
        raise RuntimeError("Outcome flag leaked into training")
    print("PASS: 15 raw fixture CSVs → fixed transform → unified 300 rows → grouped RF/XGB/IF training → compatible API inference")
    print("No genuine external dataset was used and no active artifacts were changed.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        verify(args.output)
    else:
        with tempfile.TemporaryDirectory(prefix="quality-ingestion-") as directory:
            verify(directory)


if __name__ == "__main__":
    main()
