"""Feature-only source inventory and explicit local review manifests."""
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from symptom_ai.dataset_ingestion.validation import VALID_LABELS, combine_frames, norm01
from symptom_ai.models.evaluation import feature_matrix
from symptom_ai.models.train_symptom_models import train_models

REPO = Path(__file__).resolve().parents[3]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory():
    registry = REPO / "data/raw/dataset_registry.json"
    items = json.loads(registry.read_text())["datasets"]
    return {"registry_sha256": sha256(registry), "sources": [
        {"name": item["name"], "path": item["path"], "source_url": item.get("source_url"),
         "present": (REPO / item["path"]).exists(),
         "review_status": "not_reviewed_by_this_benchmark" if (REPO / item["path"]).exists() else "unavailable"}
        for item in items],
        "policy": "Presence is not approval. No downloads, binary execution or automatic dataset admission."}


def reviewed_workflow(manifest_path, output):
    """Validate/ingest reviewed CSV sources and run the actual foundation trainer.

    Manifest identity is a local assertion, not authenticated analyst identity.
    No active repository datasets/artifacts are changed.
    """
    path, output = Path(manifest_path), Path(output)
    if output.exists():
        raise ValueError("Source workflow output already exists; choose a new directory")
    manifest = json.loads(path.read_text())
    if not all(isinstance(manifest.get(key), str) and manifest[key].strip()
               for key in ("reviewer", "reviewed_at", "review_notes")):
        raise ValueError("Source manifest requires reviewer, reviewed_at and review_notes")
    if manifest.get("schema_version") != 2 or manifest.get("normalization") != "fixed_nonnegative_saturation_v1":
        raise ValueError("Source schema 2 and fixed transform review required")
    kind = manifest.get("source_kind")
    if kind not in {"captured_feature_data", "synthetic_contract_fixture"}:
        raise ValueError("Declare captured_feature_data or synthetic_contract_fixture; never mix them")
    frames, provenance = [], []
    for source in manifest.get("sources", []):
        source_path = path.parent / source["path"]
        if source_path.suffix.lower() != ".csv" or sha256(source_path) != source.get("sha256"):
            raise ValueError("Reviewed source must be a checksum-bound CSV")
        if not source.get("provenance") or not source.get("label_definition"):
            raise ValueError("Each source requires provenance and reviewed label_definition")
        frame = pd.read_csv(source_path)
        if source.get("adapter") == "raw_observation_fixture":
            if kind != "synthetic_contract_fixture":
                raise ValueError("Fixture adapter cannot ingest genuine source data")
            for column in source["observation_columns"]:
                frame[column] = norm01(pd.to_numeric(frame[column], errors="raise"))
        elif source.get("adapter") != "normalized_symptoms_v2":
            raise ValueError("Use reviewed normalized_symptoms_v2; source-specific raw adapters need separate schema review")
        required = {"dataset_source", "sample_id", "label"}
        if frame.empty or not required <= set(frame) or not frame["label"].isin(VALID_LABELS).all():
            raise ValueError("Nonempty reviewed source needs explicit valid labels, IDs and provenance")
        feature_matrix(frame)
        frame["source_file"] = str(source_path.resolve())
        frames.append(frame)
        provenance.append({"path": source["path"], "sha256": source["sha256"], "adapter": source["adapter"],
                           "rows": len(frame), "label_counts": frame["label"].value_counts().to_dict(),
                           "provenance": source["provenance"], "label_definition": source["label_definition"]})
    if not frames:
        raise ValueError("No reviewed sources supplied")
    unified = combine_frames(frames)
    if unified["sample_id"].isna().any() or unified["sample_id"].duplicated().any():
        raise ValueError("Reviewed sample IDs must be unique and nonmissing")
    output.mkdir(parents=True)
    dataset = output / "unified_symptom_dataset.csv"
    unified.to_csv(dataset, index=False)
    report = {"source_kind": kind, "review_manifest_sha256": sha256(path), "reviewer": manifest["reviewer"],
              "reviewed_at": manifest["reviewed_at"], "review_notes": manifest["review_notes"],
              "sources": provenance, "unified_sha256": sha256(dataset), "rows": len(unified),
              "feature_columns": list(feature_matrix(unified).columns), "production_modified": False}
    try:
        models, metadata, training = train_models(unified)
    except ValueError as exc:
        report.update(status="training_refused", reason=str(exc))
        (output / "source_workflow.json").write_text(json.dumps(report, indent=2))
        raise
    model_dir = output / "candidate_models"
    model_dir.mkdir()
    names = {"random_forest": "random_forest_classifier.joblib", "xgboost": "xgboost_classifier.joblib",
             "isolation_forest": "unknown_behavior_detector.joblib"}
    for name, model in models.items():
        joblib.dump(model, model_dir / names[name])
    primary = models["xgboost"] if metadata["primary_classifier"].startswith("xgboost") else models["random_forest"]
    joblib.dump(primary, model_dir / "ransomware_symptom_classifier.joblib")
    metadata.update(dataset_sha256=sha256(dataset), review_manifest_sha256=sha256(path))
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    report.update(status="candidate_trained_not_promoted", training=training,
                  performance_scope="Synthetic contract verification only" if kind == "synthetic_contract_fixture" else
                  "Reviewed feature representation only; temporal endpoint metrics unavailable without exposure/onset")
    (output / "source_workflow.json").write_text(json.dumps(report, indent=2))
    return report
