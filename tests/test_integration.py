"""PR #1 evaluation/telemetry integrated with PR #2 review and safety contracts.

All model data is synthetic. Tests exercise interfaces, never model efficacy.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from test_foundation import temporary_cwd, corroborated_events
from symptom_ai.api import main as api
from symptom_ai.auto_learning import auto_retraining_pipeline as learning
from symptom_ai.auto_learning.review import is_approved
from symptom_ai.live_monitoring import analyst_review, ablation_evaluator as ablation
from symptom_ai.live_monitoring import live_log_ai_watcher as watcher
from symptom_ai.models import evaluate_heldout_family_v2 as heldout
from symptom_ai.models import evaluate_isolation_forest_benign_only as offline_if
from symptom_ai.models import unknown_detector_shadow as shadow
from symptom_ai.models.evaluation import feature_matrix, fit_benign_detector, calibrate_benign_threshold
from symptom_ai.models.heldout_validation import validate_partitions, load_family_split
from symptom_ai.models.prepare_heldout_family_v2 import prepare_split


def family_fixture():
    rng = np.random.default_rng(105)
    rows = []
    for family, count in [("benign", 600), ("Akira", 100), ("LockBit", 100), ("Other", 100)]:
        for index in range(count):
            key = f"{family}_{index}"
            rows.append({"sample_id": key, "session_id": key, "dataset_source": "CSU_Ransomware_Data",
                         "family": family, "label": "benign" if family == "benign" else "known_ransomware_like",
                         "file_write_burst": float(rng.uniform(0, .4) if family == "benign" else rng.uniform(.5, 1)),
                         "high_entropy_write": float(rng.random()), "analyst_review_required": int(family != "benign"),
                         "scenario_type": index, "response_policy": index})
    return pd.DataFrame(rows)


def call_main(module, *args):
    with patch.object(sys, "argv", [module.__name__, *map(str, args)]), contextlib.redirect_stdout(io.StringIO()):
        module.main()


class ReviewIntegrationTests(unittest.TestCase):
    def test_legacy_cli_approval_drives_signature_and_training_and_rejection_revokes_both(self):
        with temporary_cwd():
            watcher.QUEUE_DIR.mkdir(parents=True)
            events = [{"event_type": "extension_change", "scenario_type": "old"}] * 4 + [{"event_type": "high_entropy_write"}] * 4
            symptoms, damage = watcher.aggregate_symptoms(events)
            path = Path(watcher.save_learning_case("shared", events, symptoms, damage, {}, "fixture"))
            self.assertFalse(learning.load_learning_cases())
            analyst_review.review_case(path, "approve", "fixture reviewer", "Reviewed synthetic evidence", Path("legacy.json"))
            self.assertTrue(is_approved(json.loads(path.read_text())))
            self.assertEqual(len(learning.load_learning_cases()), 1)
            self.assertFalse(Path("legacy.json").exists())
            renamed = [{**event, "scenario_type": "different"} for event in events]
            self.assertIsNotNone(watcher.match_learned_signature(renamed))
            analyst_review.review_case(path, "reject", "fixture reviewer", "Revoke fixture approval")
            self.assertIsNone(watcher.match_learned_signature(events))
            self.assertFalse(learning.load_learning_cases())

    def test_benign_review_trains_without_positive_signature(self):
        with temporary_cwd():
            watcher.QUEUE_DIR.mkdir(parents=True)
            events = [{"event_type": "file_write"}]
            symptoms, damage = watcher.aggregate_symptoms(events)
            path = Path(watcher.save_learning_case("benign", events, symptoms, damage, {}, "fixture"))
            call_main(analyst_review, "--case-file", path, "--decision", "benign", "--analyst", "reviewer", "--note", "Benign fixture")
            cases = learning.load_learning_cases()
            self.assertEqual(cases[0]["review"]["confirmed_label"], "benign")
            self.assertFalse(watcher.load_learned_signatures())

    def test_blank_notes_and_legacy_flags_do_not_approve(self):
        with temporary_cwd():
            watcher.QUEUE_DIR.mkdir(parents=True)
            path = Path(watcher.save_learning_case("legacy", [{"event_type": "extension_change"}], {}, {}, {}, "fixture"))
            with self.assertRaises(ValueError):
                analyst_review.review_case(path, "approve", "reviewer", "")
            case = json.loads(path.read_text())
            case.update(learning_status="approved", analyst_decision="approve", approved_by="legacy")
            path.write_text(json.dumps(case))
            self.assertFalse(is_approved(case))
            self.assertFalse(watcher.load_learned_signatures())
            self.assertFalse(learning.load_learning_cases())

    def test_evidence_tampering_invalidates_both_consumers(self):
        with temporary_cwd():
            watcher.QUEUE_DIR.mkdir(parents=True)
            events = [{"event_type": "extension_change"}] * 4 + [{"event_type": "high_entropy_write"}] * 4
            symptoms, damage = watcher.aggregate_symptoms(events)
            path = Path(watcher.save_learning_case("tamper", events, symptoms, damage, {}, "fixture"))
            analyst_review.review_case(path, "approve", "reviewer", "Positive fixture")
            case = json.loads(path.read_text())
            case["events"].append({"event_type": "file_write"})
            path.write_text(json.dumps(case))
            self.assertFalse(watcher.load_learned_signatures())
            self.assertFalse(learning.load_learning_cases())


class WatcherIntegrationTests(unittest.TestCase):
    def evaluate(self, events, result):
        for directory in [watcher.REPORT_DIR, watcher.QUEUE_DIR, watcher.CONTROL_DIR, watcher.ERROR_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        with patch.object(watcher, "call_ai", return_value=result), patch.object(watcher, "call_explain", return_value={}), contextlib.redirect_stdout(io.StringIO()):
            return watcher.evaluate_window(events, 1)

    def test_early_warning_then_impact_tracks_timing_and_truthful_stop(self):
        with temporary_cwd(), patch.dict(watcher.SCENARIO_EARLY_DETECTION_STATE, {}, clear=True):
            def events(names, start):
                return [{"event_type": name, "scenario": "timing", "scenario_event_index": i,
                         "path": f"document-{i}"} for i, name in enumerate(names, start)]
            result = {"predicted_label": "known_ransomware_like", "response": {"policy": "early_warning"},
                      "unknown_detector_shadow": {"status": "available", "changes_unknown_risk": False},
                      "model_feature_semantics_version": 3, "input_feature_semantics_version": 3, "isolation_forest_prediction": -1}
            first = self.evaluate(events(["extension_change"] * 4, 1), result)
            self.assertEqual(first["alert_type"], "analyst_alert")
            self.assertFalse(first["simulated_stop_requested"])
            self.assertEqual(first["unknown_detector_shadow"], result["unknown_detector_shadow"])
            result["response"]["policy"] = "protective_lockdown"
            second = self.evaluate(corroborated_events(start=5, scenario="timing"), result)
            self.assertEqual(second["early_detection_metrics"]["detection_lead_events"], 1)
            self.assertTrue(second["simulated_stop_requested"])
            self.assertTrue(second["sandbox_safety_stop"])
            self.assertFalse(second["containment_triggered"])
            self.assertFalse(second["containment_executed"])
            self.assertFalse(second["simulated_containment_executed"])
            self.assertEqual(second["alert_schema_version"], 3)
            self.assertEqual(json.loads(watcher.STOP_SIGNAL.read_text())["containment_executed"], False)

    def test_api_failure_and_impact_only_never_request_stop(self):
        for result in [{"api_error": "fixture", "response": {"policy": "protective_lockdown"}},
                       {"response": {"policy": "monitor_only"}}]:
            with self.subTest(result=result), temporary_cwd():
                alert = self.evaluate([{"event_type": "stealth_file_lock"}] * 4, result)
                self.assertFalse(alert["simulated_stop_requested"])
                self.assertIsNone(alert["early_detection_metrics"]["first_alert_event_index"])
                self.assertFalse(watcher.STOP_SIGNAL.exists())

    def test_ablation_impact_review_is_not_successful_detection(self):
        with temporary_cwd():
            events = [{"event_type": "stealth_file_lock", "scenario_event_index": i} for i in range(1, 5)]
            Path("events.jsonl").write_text("\n".join(map(json.dumps, events)))
            with patch.object(ablation, "call_ai", return_value={"predicted_label": "benign", "response": {"policy": "monitor_only"}}):
                call_main(ablation, "--events", "events.jsonl", "--truth", "ransomware", "--case-id", "impact", "--window-events", 4)
            result = json.loads(Path("reports/evaluation/ablation_runs/impact.json").read_text())
            self.assertFalse(result["windows"][0]["full_hybrid_detected"])
            self.assertEqual(result["windows"][0]["full_hybrid_alert_type"], "evidence_collection")
            self.assertFalse(watcher.STOP_SIGNAL.exists())

    def test_reviewed_match_preserves_evidence_without_overriding_containment_gate(self):
        with temporary_cwd():
            watcher.QUEUE_DIR.mkdir(parents=True)
            events = [{"event_type": "extension_change"}] * 4 + [{"event_type": "high_entropy_write"}] * 4
            symptoms, damage = watcher.aggregate_symptoms(events)
            path = Path(watcher.save_learning_case("positive", events, symptoms, damage, {}, "fixture"))
            analyst_review.review_case(path, "approve", "reviewer", "Synthetic reviewed signature")
            alert = self.evaluate(events, {"predicted_label": "benign", "response": {"policy": "monitor_only"}})
            self.assertEqual(alert["alert_type"], "analyst_alert")
            self.assertTrue(alert["detector_evidence"]["reviewed_signature"]["hit"])
            self.assertFalse(alert["simulated_stop_requested"])
            self.assertFalse(alert["containment_executed"])
            self.assertFalse(alert["containment_triggered"])


class EvaluationIntegrationTests(unittest.TestCase):
    def test_split_manifest_rejects_legacy_or_changed_csv(self):
        with temporary_cwd():
            family_fixture().to_csv("fixture.csv", index=False)
            prepare_split("fixture.csv", "runs", "Akira", "CSU_Ransomware_Data", 100, 30)
            root = Path("runs/heldout_CSU_Ransomware_Data_Akira")
            self.assertEqual(len(load_family_split(root)), 4)
            path = root / "data/train_benign.csv"
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "manifest"):
                load_family_split(root)
            (root / "split_manifest.json").write_text('{"feature_schema_version": 1}')
            with self.assertRaisesRegex(ValueError, "Legacy"):
                load_family_split(root)

    def test_available_anomalous_shadow_cannot_change_active_risk_or_policy(self):
        classifier = Mock(classes_=np.array(["benign"]))
        classifier.predict.return_value = ["benign"]
        classifier.predict_proba.return_value = [[1.]]
        active = Mock()
        active.decision_function.return_value = [0.]
        active.predict.return_value = [1]
        metadata = {"feature_columns": ["file_write_burst"], "labels": ["benign"]}
        telemetry = {"mode": "benign_calibrated_active_with_calibrated_shadow",
                     "calibrated_shadow": {"status": "available", "is_anomalous": True, "changes_unknown_risk": False}}
        with patch.object(api, "load_artifacts", return_value=(classifier, active, metadata)), patch.object(api, "evaluate_unknown_detector_shadow", return_value=telemetry):
            result = api.infer(api.SymptomRequest(symptoms={"file_write_burst": .1}))
        self.assertEqual(result["unknown_risk"], "low")
        self.assertEqual(result["response"]["policy"], "monitor_only")
        self.assertTrue(result["unknown_detector_shadow"]["is_anomalous"])

    def test_holdout_rejects_missing_columns_in_either_partition(self):
        df = family_fixture()
        for train, test in [(df.drop(columns="family"), df), (df, df.drop(columns="sample_id"))]:
            with self.assertRaisesRegex(SystemExit, "Missing required columns"):
                heldout.validate_split(train, test, "Akira", "CSU_Ransomware_Data")

    def test_partitions_reject_sessions_duplicates_invalid_ranges_and_unknown_labels(self):
        df = family_fixture().iloc[:10].copy()
        train, test = df.iloc[:5].copy(), df.iloc[5:].copy()
        for kind in ["session", "duplicate", "range", "label", "missing_id", "feature_names"]:
            bad = test.copy()
            if kind == "session":
                bad.loc[5, "session_id"] = train.iloc[0]["session_id"]
            elif kind == "duplicate":
                bad.loc[5, ["file_write_burst", "high_entropy_write"]] = train.iloc[0][["file_write_burst", "high_entropy_write"]]
            elif kind == "range":
                bad.loc[5, "file_write_burst"] = 2
            elif kind == "label":
                bad.loc[5, "label"] = "unknown"
            elif kind == "missing_id":
                bad.loc[5, "sample_id"] = None
            else:
                bad = bad.drop(columns="high_entropy_write")
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_partitions({"train": train, "test": bad})

    def test_shared_calibration_equality_is_inlier(self):
        model = Mock()
        model.score_samples.return_value = np.array([-.4, -.4, -.4])
        model.predict.return_value = np.array([1, 1, 1])
        info = calibrate_benign_threshold(model, pd.DataFrame({"x": [1, 2, 3]}))
        self.assertEqual(info["threshold"], -.4)
        self.assertEqual(offline_if.detection_rate(np.array([.4, .4]), .4), 0)

    def test_full_family_evaluators_calibration_shadow_and_api(self):
        with temporary_cwd() as root:
            df = family_fixture()
            df.to_csv("dataset.csv", index=False)
            for family in ["Akira", "LockBit"]:
                manifest = prepare_split("dataset.csv", "reports/evaluation_v2", family, "CSU_Ransomware_Data", 100, 30)
                self.assertEqual(manifest["split_integrity"]["relationship_overlap"], 0)
                call_main(heldout, "--family", family)
                report = json.loads(Path(f"reports/evaluation_v2/heldout_CSU_Ransomware_Data_{family}/heldout_evaluation.json").read_text())
                self.assertEqual(report["feature_columns"], ["file_write_burst", "high_entropy_write"])
                self.assertEqual(report["feature_schema_version"], 2)
            call_main(offline_if, "--fit-benign-rows", 60)
            report = json.loads(shadow.CALIBRATION_REPORT.read_text())
            self.assertEqual(len(report["threshold_results"]), 4)
            self.assertEqual(report["dataset_sizes"]["benign_fit"], 60)
            self.assertEqual(report["dataset_sizes"]["benign_calibration"], 40)
            # The API's active IF uses the same calibrator. Shadow is telemetry only.
            benign = feature_matrix(df.iloc[:100])
            active, _ = fit_benign_detector(benign.iloc[:60], pd.Series(["benign"] * 60), benign.iloc[60:], pd.Series(["benign"] * 40))
            X = benign.iloc[[0]]
            result = shadow.evaluate_unknown_detector_shadow(X, active)
            self.assertEqual(result["calibrated_shadow"]["status"], "available", result)
            metadata = {"feature_columns": list(X.columns), "labels": ["benign"], "id_to_label": {}, "feature_schema_version": 2}
            clf = Mock(classes_=np.array(["benign"]))
            clf.predict.return_value = np.array(["benign"])
            clf.predict_proba.return_value = np.array([[1.]])
            with patch.object(api, "load_artifacts", return_value=(clf, active, metadata)):
                response = TestClient(api.app).post("/predict", json={"symptoms": X.iloc[0].to_dict()})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["unknown_detector_shadow"]["status"], "available")
            self.assertFalse(response.json()["response"]["containment_executed"])
            # Swapped same-width feature names and mismatched report/artifact fail closed.
            renamed = X.rename(columns={"file_write_burst": "file_read_burst"})
            mock_active = Mock()
            mock_active.decision_function.return_value = [0.]
            mock_active.predict.return_value = [1]
            bad = shadow.evaluate_unknown_detector_shadow(renamed, mock_active)
            self.assertEqual(bad["calibrated_shadow"]["status"], "unavailable")
            report["artifact_sha256"] = "wrong"
            shadow.CALIBRATION_REPORT.write_text(json.dumps(report))
            self.assertEqual(shadow.evaluate_unknown_detector_shadow(X, active)["calibrated_shadow"]["status"], "unavailable")

    def test_legacy_shadow_report_is_unavailable_without_changing_active_risk(self):
        with temporary_cwd():
            shadow.CALIBRATION_REPORT.parent.mkdir(parents=True)
            shadow.CALIBRATION_REPORT.write_text('{"threshold_results": []}')
            active = Mock()
            active.decision_function.return_value = [0.]
            active.predict.return_value = [1]
            result = shadow.evaluate_unknown_detector_shadow(pd.DataFrame({"file_write_burst": [.1]}), active)
            self.assertEqual(result["calibrated_shadow"]["status"], "unavailable")
            self.assertIn("schema 2", result["calibrated_shadow"]["reason"])
            self.assertFalse(result["calibrated_shadow"]["changes_unknown_risk"])
            self.assertEqual(result["active"]["prediction"], 1)


if __name__ == "__main__":
    unittest.main()
