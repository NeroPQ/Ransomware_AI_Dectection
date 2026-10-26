"""Synthetic pipeline/regression tests. These do not establish detection accuracy."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from build_behavior_features import build_features
from symptom_ai.api import main as api
from symptom_ai.auto_learning import auto_retraining_pipeline as learning
from symptom_ai.auto_learning.review import review_case, is_approved
from symptom_ai.dataset_ingestion.validation import norm01, text_label, binary_labels, combine_frames, validate_labels
from symptom_ai.dataset_ingestion.parser_csu_ransomware import parse_one_file, detect_label
from symptom_ai.dataset_ingestion.parser_ransomset import get_label
from symptom_ai.live_monitoring import live_log_ai_watcher as watcher
from symptom_ai.models.evaluation import feature_matrix, related_groups, split_dataset, metrics, fit_benign_detector
from symptom_ai.models.train_symptom_models import train_models
from symptom_ai.models.risk_scorer import calculate_symptom_risk
from symptom_ai.response_engine.response_policy import recommend_response
from symptom_ai.response_engine.protective_lockdown import run_protective_lockdown, lock_protected_files_demo


@contextlib.contextmanager
def temporary_cwd():
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as directory:
        try:
            os.chdir(directory)
            yield Path(directory)
        finally:
            os.chdir(previous)


def synthetic_samples():
    rng = np.random.default_rng(19)
    rows = []
    for source in range(15):
        for i in range(20):
            malignant = i % 2
            rows.append({"dataset_source": f"fixture_source_{source}", "family": f"fixture_family_{source}",
                         "sample_id": f"fixture_{source}_{i}", "label": "known_ransomware_like" if malignant else "benign",
                         "file_write_burst": float(rng.uniform(0.5, 1) if malignant else rng.uniform(0, 0.4)),
                         "high_entropy_write": float(rng.random()),
                         "analyst_review_required": malignant, "scenario_type": malignant,
                         "response_policy": malignant, "is_simulated": 1})
    return pd.DataFrame(rows)


def corroborated_events(start=1, scenario=None):
    from symptom_ai.benchmark.policy_development import make_event
    events = [make_event("file_scramble", index=i, path=f"docs/dir-{i % 2}/file-{i}.txt",
                         indicators=["destructive_content"], entropy=7.8) for i in range(5)]
    events.append(make_event("ransom_note", index=5, path="docs/note.txt", indicators=["ransom_note"]))
    for index, event in enumerate(events, start):
        event["scenario_event_index"] = index
        event["scenario"] = scenario
    return events


def synthetic_events():
    return [{"event_index": i, "timestamp": "2026-01-01T00:00:00", "event_type": event,
             "object": f"fixture_{i}", "process_name": "fixture", "bytes_written": 100,
             "entropy_before": 2.0, "entropy_after": 6.0, "label": "benign", "family": "benign",
             "session_id": session}
            for session in ("session_a", "session_b") for i, event in enumerate(["file_read", "file_write"])]


class FeaturesAndDataTests(unittest.TestCase):
    def test_normalization_does_not_depend_on_other_samples(self):
        self.assertEqual(norm01(pd.Series([5.0])).iloc[0], norm01(pd.Series([5.0, 1000])).iloc[0])
        self.assertEqual(norm01(0).iloc[0], 0)
        self.assertEqual(norm01(pd.Series([None])).iloc[0], 0)
        with self.assertRaises(ValueError):
            norm01(pd.Series([np.inf]))

    def test_unknown_labels_never_become_ransomware(self):
        for value in [None, "", "NaN", "malware", "trojan", "unrecognized", "benign ransomware", "not ransomware", "non-ransomware", "unknown_ransomware_candidate"]:
            self.assertEqual(text_label(value), "unknown")
        self.assertEqual(text_label("ransomware"), "known_ransomware_like")
        self.assertEqual(text_label("benign"), "benign")
        self.assertEqual(binary_labels(pd.Series([0, 1, None, 7])).tolist(), ["benign", "known_ransomware_like", "unknown", "unknown"])
        self.assertEqual(get_label(pd.Series({"score_binary": None})), "unknown")
        self.assertEqual(get_label(pd.Series({"score_binary": 1, "class": "benign"})), "unknown")
        self.assertEqual(detect_label({}), "unknown")
        self.assertEqual(detect_label({"target-class": 1, "Ware Type": "benign"}), "unknown")

    def test_training_rejects_unknown_or_missing_labels(self):
        frame = synthetic_samples()
        for invalid in [None, "unknown", "trojan"]:
            frame.loc[0, "label"] = invalid
            with self.assertRaisesRegex(ValueError, "Unverified"):
                validate_labels(frame)

    def test_metadata_and_outcomes_are_not_features(self):
        X = feature_matrix(synthetic_samples())
        self.assertEqual(set(X), {"file_write_burst", "high_entropy_write"})

    def test_feature_ranges_and_missingness(self):
        frame = synthetic_samples()
        frame.loc[0, "file_write_burst"] = np.nan
        self.assertEqual(feature_matrix(frame).iloc[0]["file_write_burst"], 0)
        for value in [np.inf, -1, 1.1, "invalid"]:
            frame["file_write_burst"] = frame["file_write_burst"].astype(object)
            frame.loc[0, "file_write_burst"] = value
            with self.assertRaises((ValueError, TypeError)):
                feature_matrix(frame)

    def test_sessions_never_merge_during_feature_extraction(self):
        result = build_features(pd.DataFrame(synthetic_events()), window_size=50)
        self.assertEqual(len(result), 2)
        self.assertTrue((result["event_count"] == 2).all())
        self.assertTrue((result["entropy_delta"] == 4).all())
        self.assertTrue((result["file_write_count"] == 1).all())

    def test_invalid_behavior_labels_and_indices_rejected(self):
        events = pd.DataFrame(synthetic_events())
        events.loc[0, "label"] = "unknown"
        with self.assertRaisesRegex(ValueError, "labels"):
            build_features(events)
        events = pd.DataFrame(synthetic_events())
        events.loc[1, "event_index"] = 0
        with self.assertRaisesRegex(ValueError, "indices"):
            build_features(events)
        with self.assertRaisesRegex(ValueError, "Missing behavior"):
            build_features(pd.DataFrame())

    def test_combination_preserves_missing_metadata(self):
        with self.assertRaisesRegex(ValueError, "No datasets"):
            combine_frames([])
        with self.assertRaisesRegex(ValueError, "labels"):
            combine_frames([pd.DataFrame({"file_write_burst": [0]})])
        combined = combine_frames([pd.DataFrame({"label": ["benign"], "family": ["example"]}),
                                   pd.DataFrame({"label": ["unknown"], "file_write_burst": [0.1]})])
        self.assertTrue(pd.isna(combined.loc[1, "family"]))
        self.assertEqual(combined.loc[0, "file_write_burst"], 0)

    def test_sparse_csu_schema_does_not_crash_or_invent_extension_impact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv"
            pd.DataFrame({"target-class": [0, None]}).to_csv(path, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                frame = parse_one_file(path)
            self.assertEqual(frame["label"].tolist(), ["benign", "unknown"])
            self.assertTrue((frame["suspicious_extension_change"] == 0).all())


class EvaluationTests(unittest.TestCase):
    def test_grouped_holdouts_are_reproducible_and_disjoint(self):
        frame = synthetic_samples()
        X = feature_matrix(frame)
        a = split_dataset(frame, X)
        b = split_dataset(frame, X)
        for first, second in zip(a[:3], b[:3]):
            np.testing.assert_array_equal(first, second)
        for column in ("dataset_source", "family", "sample_id"):
            partitions = [set(frame.iloc[indices][column]) for indices in a[:3]]
            self.assertFalse(partitions[0] & partitions[1])
            self.assertFalse(partitions[0] & partitions[2])
            self.assertFalse(partitions[1] & partitions[2])

    def test_related_families_and_duplicates_join_across_sources(self):
        frame = synthetic_samples()
        frame.loc[20, "family"] = frame.loc[0, "family"]
        frame.loc[40, ["file_write_burst", "high_entropy_write"]] = frame.loc[0, ["file_write_burst", "high_entropy_write"]]
        groups, _ = related_groups(frame, feature_matrix(frame))
        self.assertEqual(groups[0], groups[20])
        self.assertEqual(groups[0], groups[40])

    def test_missing_metadata_and_single_session_fail_explicitly(self):
        frame = synthetic_samples().drop(columns=["dataset_source", "family", "sample_id"])
        with self.assertRaisesRegex(ValueError, "metadata unavailable"):
            split_dataset(frame, feature_matrix(frame))
        frame["session_id"] = "one_session"
        with self.assertRaisesRegex(ValueError, "Insufficient independent"):
            split_dataset(frame, feature_matrix(frame))
        with self.assertWarns(UserWarning):
            train, validation, test, info = split_dataset(frame, feature_matrix(frame), allow_row_split=True)
        self.assertTrue(info["allow_row_split"])
        self.assertEqual(len(train) + len(validation) + len(test), len(frame))

    def test_metrics_include_false_positive_rate_and_per_class(self):
        result = metrics(["benign", "benign", "known_ransomware_like", "known_ransomware_like"],
                         ["benign", "known_ransomware_like", "benign", "known_ransomware_like"],
                         ["benign", "known_ransomware_like"])
        self.assertEqual(result["false_positive_rate"], 0.5)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(result["classification_report"]["known_ransomware_like"]["f1-score"], 0.5)
        missing = metrics(["known_ransomware_like"], ["known_ransomware_like"], ["benign", "known_ransomware_like"])
        self.assertIsNone(missing["false_positive_rate"])

    def test_isolation_forest_benign_only_training_and_calibration(self):
        X = pd.DataFrame({"file_write_burst": np.linspace(0, 1, 80)})
        y = pd.Series(["benign"] * 40 + ["known_ransomware_like"] * 40)
        model, report = fit_benign_detector(X, y, X.iloc[:20], y.iloc[:20])
        self.assertEqual(report["training_rows"], 40)
        self.assertEqual(model.offset_, report["threshold"])
        self.assertLessEqual(report["observed_validation_fpr"], 0.1)
        with self.assertRaisesRegex(ValueError, "Benign-only"):
            fit_benign_detector(X, pd.Series(["known_ransomware_like"] * 80), X, y)

    def test_existing_models_train_and_serialize_on_synthetic_fixtures(self):
        models, metadata, report = train_models(synthetic_samples())
        self.assertEqual(set(models), {"random_forest", "xgboost", "isolation_forest"})
        self.assertEqual(report["selection_metric"], "validation_macro_f1")
        self.assertEqual(metadata["feature_schema_version"], 2)
        self.assertEqual(set(report["validation"]), {"random_forest", "xgboost"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.joblib"
            joblib.dump(models["isolation_forest"], path)
            restored = joblib.load(path)
            self.assertEqual(restored.offset_, models["isolation_forest"].offset_)
            model_dir = Path(directory)
            primary = models["xgboost"] if metadata["primary_classifier"].startswith("xgboost") else models["random_forest"]
            joblib.dump(primary, model_dir / metadata["primary_classifier"])
            joblib.dump(models["isolation_forest"], model_dir / "unknown_behavior_detector.joblib")
            (model_dir / "model_metadata.json").write_text(json.dumps(metadata))
            with patch.object(api, "MODEL_DIR", model_dir):
                client = TestClient(api.app)
                response = client.post("/predict", json={"symptoms": {"file_write_burst": .1, "high_entropy_write": .1}})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(set(response.json()["prediction_probabilities"]), set(metadata["labels"]))
                self.assertFalse(response.json()["response"]["containment_executed"])



class DetectionSafetyTests(unittest.TestCase):
    def test_benign_and_isolated_anomaly_do_not_recommend_containment(self):
        self.assertEqual(recommend_response({})["policy"], "monitor_only")
        result = recommend_response({"file_write_burst": 0.8}, "high")
        self.assertEqual(result["policy"], "monitor_only")
        self.assertEqual(result["legacy_unknown_risk"], "high")
        self.assertFalse(result["containment_recommended"])
        self.assertFalse(result["containment_executed"])

    def test_correlated_untyped_file_evidence_requires_review_before_containment(self):
        symptoms = {"file_write_burst": 0.9, "high_entropy_write": 0.9}
        self.assertEqual(recommend_response(symptoms)["stage"], "suspicious_activity")
        self.assertFalse(recommend_response(symptoms, "high")["containment_recommended"])
        self.assertFalse(recommend_response({"backup_disable_attempt": 0.9})["containment_recommended"])

    def test_risk_uses_fixed_denominator(self):
        self.assertLess(calculate_symptom_risk({"file_write_burst": 1.0}), 0.2)
        self.assertGreaterEqual(calculate_symptom_risk({"file_write_burst": .9, "file_rename_burst": .9, "high_entropy_write": .9}), .85)
        with self.assertRaises(ValueError):
            calculate_symptom_risk({"file_write_burst": float("nan")})

    def test_impact_and_recommendation_are_not_execution(self):
        damage = {"has_file_impact": True}
        self.assertEqual(watcher.decide_alert_type({}, damage, "protective_lockdown", "high"), "evidence_collection")
        self.assertEqual(watcher.decide_alert_type({}, damage, "monitor_only", "low"), "evidence_collection")
        self.assertEqual(watcher.decide_alert_type({"api_error": "unavailable"}, damage, None, None), "api_error_needs_review")

    def test_explicit_entropy_events_are_not_masked_by_zero_extra(self):
        symptoms, _ = watcher.aggregate_symptoms([{"event_type": "high_entropy_write"}] * 4)
        self.assertEqual(symptoms["high_entropy_write"], 1)

    def test_partial_jsonl_record_is_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text('{"event_type":"file_read"}\n{"event_type":')
            events, offset = watcher.read_new_lines(path, 0)
            self.assertEqual(len(events), 1)
            with path.open("a") as stream:
                stream.write('"file_write"}\n')
            events, _ = watcher.read_new_lines(path, offset)
            self.assertEqual(events, [{"event_type": "file_write"}])

    def test_progressive_simulation_markers_preserve_originals_and_reject_outside_files(self):
        from symptom_ai.simulation import safe_progressive_ransomware_sim as simulation
        with temporary_cwd() as directory:
            simulation.prepare_lab_files(files_per_folder=1)
            path = simulation.list_demo_files()[0]
            original = path.read_bytes()
            marker_paths = simulation.simulate_safe_lock_files([path], 1)
            self.assertEqual(len(marker_paths), 1)
            self.assertEqual(path.read_bytes(), original)
            outside = directory / "business_file_outside.txt"
            outside.write_text("Safe demo business document outside")
            link = path.parent / "business_file_symlink.txt"
            link.symlink_to(outside)
            self.assertEqual(simulation.simulate_safe_lock_files([outside, link], 2), [])
            self.assertEqual(simulation.simulate_touch_files([outside, link], 2), [])
            self.assertEqual(outside.read_text(), "Safe demo business document outside")

    def test_safe_lockdown_preserves_contents_modes_and_symlink_target(self):
        with temporary_cwd() as directory:
            target = directory / "outside.txt"
            target.write_text("preserve")
            target.chmod(0o640)
            docs = Path("data/protected_docs")
            docs.mkdir(parents=True)
            (docs / "link.txt").symlink_to(target)
            result = run_protective_lockdown({"symptoms": {}})
            self.assertEqual(target.read_text(), "preserve")
            self.assertEqual(target.stat().st_mode & 0o777, 0o640)
            self.assertNotIn(str(docs / "link.txt"), lock_protected_files_demo())
            self.assertFalse(result["containment_executed"])
            self.assertTrue(result["simulated_containment_executed"])
            saved = json.loads(Path(result["unknown_case_path"]).read_text())
            self.assertEqual(saved["learning_status"], "pending_review")


class LearningSafetyTests(unittest.TestCase):
    def test_malformed_review_and_empty_reviewer_are_not_approved(self):
        for case in [None, [], {"symptoms": {}, "review": []},
                     {"symptoms": {}, "learning_status": "approved", "review": {"reviewer": None, "notes": None}}]:
            self.assertFalse(is_approved(case))

    def test_retraining_success_still_requires_manual_promotion(self):
        with temporary_cwd():
            learning.BASE_DATASET.parent.mkdir(parents=True)
            synthetic_samples().to_csv(learning.BASE_DATASET, index=False)
            learning.QUEUE_DIR.mkdir(parents=True)
            path = learning.QUEUE_DIR / "learning_case_fixture.json"
            path.write_text(json.dumps({"symptoms": {"file_write_burst": .1}, "events": []}))
            review_case(path, "benign", "test reviewer", "Synthetic benign fixture")
            learning.MODEL_DIR.mkdir(exist_ok=True)
            active = learning.MODEL_DIR / "fixture.joblib"
            active.write_bytes(b"fixture model")
            original = learning.BASE_DATASET.read_bytes()
            fake = Mock(returncode=0, stdout="fixture success", stderr="")
            with patch.object(learning.subprocess, "run", return_value=fake), contextlib.redirect_stdout(io.StringIO()):
                learning.main()
            self.assertEqual(learning.BASE_DATASET.read_bytes(), original)
            self.assertEqual(active.read_bytes(), b"fixture model")
            summary = json.loads(next(learning.VERSIONS_DIR.glob("candidate_*/summary.json")).read_text())
            self.assertEqual(summary["status"], "candidate_ready_for_review")
            self.assertFalse(summary["production_modified"])

    def test_unreviewed_legacy_and_rejected_cases_do_not_train(self):
        for case in [{"symptoms": {}}, {"symptoms": {}, "label": "known_ransomware_like", "learning_status": "queued_for_auto_retraining"},
                     {"symptoms": {}, "learning_status": "rejected"}]:
            self.assertFalse(is_approved(case))
            self.assertTrue(learning.cases_to_dataframe([case], ["file_write_burst", "label"]).empty)

    def test_approval_accepts_benign_and_invalidates_on_evidence_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "learning_case_fixture.json"
            path.write_text(json.dumps({"symptoms": {"file_write_burst": .1}, "events": [], "learning_status": "pending_review"}))
            approved = review_case(path, "benign", "test reviewer", "Synthetic benign fixture")
            self.assertTrue(is_approved(approved))
            rows = learning.cases_to_dataframe([approved], ["file_write_burst", "label"])
            self.assertEqual(rows.iloc[0]["label"], "benign")
            approved["symptoms"]["file_write_burst"] = .9
            self.assertFalse(is_approved(approved))
            rejected = review_case(path, "rejected", "test reviewer", "Synthetic rejected fixture")
            self.assertFalse(is_approved(rejected))
            with self.assertRaises(ValueError):
                review_case(path, "known_ransomware_like", "", "")

    def test_scenario_name_and_common_events_cannot_match_legacy_signature(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(watcher, "QUEUE_DIR", Path(directory)):
            (Path(directory) / "learned_behavior_signatures.json").write_text(json.dumps([{"scenario_types": ["novel_zero_day"], "event_types": ["file_read", "file_write", "process_create"]}]))
            events = [{"event_type": name, "scenario_type": "novel_zero_day"} for name in ["file_read", "file_write", "process_create"]]
            self.assertIsNone(watcher.match_learned_signature(events))

    def test_approved_signature_uses_observations_and_accepts_renamed_scenario(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(watcher, "QUEUE_DIR", Path(directory)):
            events = [{"event_type": "extension_change", "scenario_type": "fixture"}] * 4 + [{"event_type": "high_entropy_write"}] * 4
            symptoms, damage = watcher.aggregate_symptoms(events)
            case = Path(watcher.save_learning_case("fixture", events, symptoms, damage, {}, "fixture"))
            self.assertIsNone(watcher.match_learned_signature(events))
            review_case(case, "known_ransomware_like", "test reviewer", "Synthetic positive fixture")
            renamed = [{**event, "scenario_type": "unrelated_name"} for event in events]
            self.assertIsNotNone(watcher.match_learned_signature(renamed))
            self.assertIsNone(watcher.match_learned_signature([{"event_type": "file_rename"}] * 20))

    def test_retraining_failure_preserves_base_and_active_models_and_environment(self):
        with temporary_cwd():
            learning.BASE_DATASET.parent.mkdir(parents=True)
            base = synthetic_samples()
            base.to_csv(learning.BASE_DATASET, index=False)
            learning.QUEUE_DIR.mkdir(parents=True)
            case = learning.QUEUE_DIR / "learning_case_fixture.json"
            case.write_text(json.dumps({"symptoms": {"file_write_burst": .9}, "events": []}))
            review_case(case, "known_ransomware_like", "test reviewer", "Synthetic positive fixture")
            learning.MODEL_DIR.mkdir(exist_ok=True)
            active = learning.MODEL_DIR / "fixture.joblib"
            active.write_bytes(b"fixture model")
            original = learning.BASE_DATASET.read_bytes()
            fake = Mock(returncode=1, stdout="fixture failure", stderr="fixture stderr")
            with patch.object(learning.subprocess, "run", return_value=fake) as run, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    learning.main()
            self.assertEqual(learning.BASE_DATASET.read_bytes(), original)
            self.assertEqual(active.read_bytes(), b"fixture model")
            self.assertEqual(run.call_args.kwargs["env"]["PATH"], os.environ["PATH"])
            self.assertEqual(run.call_args.kwargs["cwd"], learning.PROJECT_ROOT)
            summaries = list(learning.VERSIONS_DIR.glob("candidate_*/summary.json"))
            self.assertEqual(json.loads(summaries[0].read_text())["status"], "candidate_failed")

    def test_watcher_stop_request_does_not_report_executed_containment(self):
        with temporary_cwd(), patch.object(watcher, "call_ai", return_value={"response": {"policy": "protective_lockdown"}, "unknown_risk": "high"}), patch.object(watcher, "call_explain", return_value={}):
            for directory in [watcher.QUEUE_DIR, watcher.REPORT_DIR, watcher.CONTROL_DIR]:
                directory.mkdir(parents=True, exist_ok=True)
            with contextlib.redirect_stdout(io.StringIO()):
                alert = watcher.evaluate_window(corroborated_events(), 1)
            self.assertTrue(alert["simulated_stop_requested"])
            self.assertTrue(alert["containment_recommended"])
            self.assertFalse(alert["containment_executed"])
            self.assertFalse(alert["simulated_containment_executed"])
            first_signal = watcher.STOP_SIGNAL.read_bytes()
            watcher.write_stop_signal("second", "fixture", "fixture", {})
            self.assertEqual(watcher.STOP_SIGNAL.read_bytes(), first_signal)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    def test_invalid_inputs_return_422(self):
        for symptoms in [{}, {"not_a_symptom": .2}, {"file_write_burst": -1}, {"file_write_burst": 2},
                         {"file_write_burst": "0.5"}, {"file_write_burst": True}, {"file_write_burst": None}]:
            with self.subTest(symptoms=symptoms):
                self.assertEqual(self.client.post("/predict", json={"symptoms": symptoms}).status_code, 422)
        response = self.client.post("/predict", content='{"symptoms":{"file_write_burst": NaN}}', headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 422)

    def test_sensitive_routes_fail_closed_without_key(self):
        with patch.dict(os.environ, {"SYMPTOM_AI_API_KEY": ""}):
            self.assertEqual(self.client.post("/respond", json={"symptoms": {"file_write_burst": .1}}).status_code, 503)
        with patch.dict(os.environ, {"SYMPTOM_AI_API_KEY": "test-only-placeholder"}):
            for method, route in [("post", "/respond"), ("post", "/prepare-demo-docs"), ("post", "/unlock-demo"), ("post", "/explain"), ("get", "/cases/unknown"), ("get", "/backups")]:
                self.assertEqual(getattr(self.client, method)(route).status_code, 401)
            with patch.object(api, "ensure_demo_protected_docs") as prepare:
                self.assertEqual(self.client.post("/prepare-demo-docs", headers={"X-API-Key": "test-only-placeholder"}).status_code, 200)
                prepare.assert_called_once()

    def test_missing_models_return_503_and_health_remains_available(self):
        with patch.object(api, "MODEL_DIR", Path("/missing-fixture-models")):
            response = self.client.post("/predict", json={"symptoms": {"file_write_burst": .1}})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(self.client.get("/model/info").status_code, 503)
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_corrupt_and_legacy_metadata_fail_with_503(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(api, "MODEL_DIR", Path(directory)):
            path = Path(directory) / "model_metadata.json"
            for content in ["corrupt", "[]", '{"feature_schema_version": 1}', '{"feature_schema_version": 2, "feature_columns": ["label"]}']:
                path.write_text(content)
                response = self.client.post("/predict", json={"symptoms": {"file_write_burst": .1}})
                self.assertEqual(response.status_code, 503)

    def test_unicode_invalid_key_returns_401(self):
        with patch.dict(os.environ, {"SYMPTOM_AI_API_KEY": "test-only-placeholder"}):
            request = api.require_api_key
            with self.assertRaises(api.HTTPException) as error:
                request("invalid-unicode-\u2603")
            self.assertEqual(error.exception.status_code, 401)

    def test_numeric_prediction_and_probability_labels(self):
        metadata = {"labels": ["benign", "known_ransomware_like"], "id_to_label": {"0": "benign", "1": "known_ransomware_like"}, "feature_columns": ["file_write_burst"]}
        classifier = Mock(classes_=np.array([0, 1]))
        classifier.predict.return_value = np.array([0])
        classifier.predict_proba.return_value = np.array([[.8, .2]])
        anomaly = Mock()
        anomaly.decision_function.return_value = np.array([-.1])
        anomaly.predict.return_value = np.array([-1])
        with patch.object(api, "load_artifacts", return_value=(classifier, anomaly, metadata)):
            result = self.client.post("/predict", json={"symptoms": {"file_write_burst": .1}}).json()
        self.assertEqual(result["predicted_label"], "benign")
        self.assertEqual(result["prediction_probabilities"], {"benign": .8, "known_ransomware_like": .2})
        self.assertFalse(result["response"]["containment_recommended"])


if __name__ == "__main__":
    unittest.main()
