"""Behavior and policy regressions over legitimate, nonexecuting observations."""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from symptom_ai.api import main as api
from symptom_ai.telemetry.schema import TelemetryEvent
from symptom_ai.telemetry.behavior import analyze_events
from symptom_ai.response_engine.evidence_policy import ALTERNATIVES, decide_policy, detector_evidence
from symptom_ai.benchmark.policy_development import make_event, generate, partitions, run
from symptom_ai.live_monitoring import live_log_ai_watcher as watcher
from test_foundation import temporary_cwd


def destructive_events():
    events = [make_event("file_scramble", index=i, path=f"C:\\docs\\dir-{i % 2}\\file-{i}.txt",
                         indicators=["destructive_content"], entropy=7.8) for i in range(5)]
    events.append(make_event("ransom_note", index=5, path="C:\\docs\\note.txt", indicators=["ransom_note"]))
    return events


def compatible_ai(classifier=True, anomaly=True):
    return {"model_feature_semantics_version": 3, "input_feature_semantics_version": 3,
            "predicted_label": "known_ransomware_like" if classifier else "benign",
            "isolation_forest_prediction": -1 if anomaly else 1, "isolation_forest_raw_score": -.2 if anomaly else .2}


class TelemetrySchemaTests(unittest.TestCase):
    def test_schema_round_trip_includes_source_identity_and_aware_time(self):
        event = make_event("file_read", path="C:\\docs\\a.txt")
        self.assertEqual(TelemetryEvent.model_validate(event).model_dump(mode="json"), {**event, "timestamp": event["timestamp"].replace("+00:00", "Z"), "received_at": None,
                         "destination_path": None, "entropy": None, "scenario": None, "scenario_event_index": None,
                         "process": {**event["process"], "parent_entity_id": None}})

    def test_invalid_version_time_confidence_pid_and_unknown_fields_rejected(self):
        for change in ({"schema_version": 2}, {"schema_version": True}, {"timestamp": "2026-01-01T00:00:00"},
                       {"observation_confidence": 1.1}, {"observation_confidence": True}, {"unexpected": "field"}, {"path": None}):
            event = {**make_event("file_read", path="a.txt"), **change}
            with self.subTest(change=change), self.assertRaises(ValidationError):
                TelemetryEvent.model_validate(event)
        event = make_event("network_poll"); event["process"]["pid"] = True
        with self.assertRaises(ValidationError):
            TelemetryEvent.model_validate(event)

    def test_entropy_requires_units_sample_size_object_and_write_operation(self):
        event = make_event("file_modify", path="a.txt", entropy=7.9)
        for change in ({"unit": "normalized"}, {"sample_bytes": 8}, {"object_type": "memory_region"}, {"after": float("nan")}):
            copy = deepcopy(event); copy["entropy"].update(change)
            with self.subTest(change=change), self.assertRaises(ValidationError):
                TelemetryEvent.model_validate(copy)
        event["operation"] = "file_read"
        with self.assertRaises(ValidationError):
            TelemetryEvent.model_validate(event)

    def test_rename_destination_and_indicator_operation_alignment(self):
        event = make_event("file_rename", path="a.txt"); event.pop("destination_path")
        with self.assertRaises(ValidationError):
            TelemetryEvent.model_validate(event)
        with self.assertRaises(ValidationError):
            TelemetryEvent.model_validate(make_event("network_poll", indicators=["destructive_content"]))


class BehaviorSemanticsTests(unittest.TestCase):
    def test_many_reads_touches_and_renames_do_not_create_impact(self):
        for operation in ("file_read", "file_touch", "file_rename"):
            with self.subTest(operation=operation):
                events = [make_event(operation, index=i, path=f"C:\\data\\directory-{i}\\file.txt") for i in range(40)]
                symptoms, damage = analyze_events(events)
                self.assertEqual(damage["observed_directory_count"], 40)
                self.assertEqual(symptoms["multi_directory_impact"], 0)
                self.assertEqual(symptoms["mass_file_modification"], 0)
                self.assertFalse(damage["has_file_impact"])
                self.assertEqual(decide_policy(symptoms, damage)["stage"], "monitor")

    def test_impact_counts_directories_not_paths_and_normalizes_windows_case(self):
        events = [make_event("file_scramble", index=i, path=f"C:\\Docs\\one\\file-{i}.txt") for i in range(30)]
        events.append(make_event("file_scramble", index=31, path="c:\\docs\\ONE\\other.txt"))
        symptoms, damage = analyze_events(events)
        self.assertEqual(damage["impacted_directory_count"], 1)
        self.assertAlmostEqual(symptoms["multi_directory_impact"], 1 / 15)
        events.append(make_event("file_scramble", index=32, path="C:\\docs\\two\\other.txt"))
        self.assertAlmostEqual(analyze_events(events)[0]["multi_directory_impact"], 2 / 15)

    def test_legacy_network_polling_never_confirms_c2_or_exfil(self):
        symptoms, damage = analyze_events([{"event_type": "network_beacon", "extra": {"entropy_score": .99}}] * 50)
        self.assertEqual(symptoms["c2_beaconing"], 0)
        self.assertEqual(symptoms["data_exfiltration_pattern"], 0)
        self.assertEqual(symptoms["high_entropy_write"], 0)
        self.assertEqual(decide_policy(symptoms, damage)["stage"], "monitor")

    def test_polling_indicators_are_separate_observations(self):
        event = make_event("network_poll", indicators=["suspected_beacon"])
        symptoms, damage = analyze_events([event])
        self.assertEqual(symptoms["c2_beaconing"], 0)
        self.assertEqual(decide_policy(symptoms, damage)["stage"], "suspicious_activity")
        confirmed = make_event("network_connect", indicators=["confirmed_c2", "confirmed_exfiltration"])
        symptoms, damage = analyze_events([confirmed])
        self.assertEqual(symptoms["c2_beaconing"], 1)
        self.assertEqual(symptoms["data_exfiltration_pattern"], 1)
        self.assertFalse(damage["has_file_impact"])
        self.assertTrue(damage["evidence_categories"]["possible_impact"])
        self.assertFalse(decide_policy(symptoms, damage)["containment_recommended"])

    def test_compression_entropy_alone_is_suspicious_not_destruction(self):
        events = [make_event("file_modify", index=i, path=f"archives/{i}.zip", entropy=7.99) for i in range(40)]
        symptoms, damage = analyze_events(events)
        self.assertGreater(symptoms["high_entropy_write"], .9)
        self.assertFalse(damage["has_file_impact"])
        self.assertEqual(decide_policy(symptoms, damage)["stage"], "suspicious_activity")

    def test_low_confidence_entropy_and_untyped_extra_are_not_measured_entropy(self):
        event = make_event("file_modify", path="a.txt", entropy=7.99, confidence=.3)
        self.assertEqual(analyze_events([event])[0]["high_entropy_write"], 0)
        self.assertEqual(analyze_events([{"event_type": "file_write", "extra": {"entropy_score": .99}}])[0]["high_entropy_write"], 0)

    def test_memory_entropy_does_not_become_file_entropy(self):
        event = make_event("memory_observation")
        event["entropy"] = {"object_type": "memory_region", "unit": "bits_per_byte", "after": 7.9,
                            "sample_bytes": 4096, "method": "shannon_byte_histogram"}
        symptoms, _ = analyze_events([event])
        self.assertEqual(symptoms["high_entropy_write"], 0)
        self.assertGreater(symptoms["memory_entropy_region_high"], .9)

    def test_duplicate_records_do_not_inflate_evidence_and_horizon_is_bounded(self):
        event = make_event("file_scramble", path="a.txt")
        _, damage = analyze_events([event] * 10)
        self.assertEqual(damage["event_count"], 1)
        self.assertEqual(damage["duplicate_records_ignored"], 9)
        later = make_event("file_scramble", index=1, window=6, path="b.txt")
        with self.assertRaisesRegex(ValueError, "horizon"):
            analyze_events([event, later])


class EvidencePolicyTests(unittest.TestCase):
    def test_one_generic_volume_feature_never_escalates(self):
        for name in ("file_write_burst", "file_read_burst", "file_rename_burst", "mass_file_modification", "network_api_usage"):
            self.assertEqual(decide_policy({name: 1.0})["stage"], "monitor")
            evidence = detector_evidence({name: 1.0}, ai_result=compatible_ai())
            for alternative in ALTERNATIVES:
                self.assertEqual(decide_policy({name: 1.0}, detectors=evidence, alternative=alternative)["stage"], "suspicious_activity")

    def test_impact_review_preserves_individual_positive_detectors(self):
        symptoms, damage = analyze_events([make_event("file_lock", path="a.txt")])
        evidence = detector_evidence(symptoms, damage, compatible_ai())
        snapshot = deepcopy(evidence)
        decision = decide_policy(symptoms, damage, evidence)
        self.assertEqual(decision["stage"], "analyst_alert")
        self.assertTrue(decision["detector_evidence"]["classifier"]["hit"])
        self.assertTrue(decision["detector_evidence"]["anomaly"]["hit"])
        self.assertFalse(decision["detector_evidence"]["rule"]["hit"])
        self.assertEqual(evidence, snapshot)
        self.assertFalse(decision["containment_recommended"])

    def test_legacy_models_are_preserved_as_shadow_evidence(self):
        ai = compatible_ai(); ai["model_feature_semantics_version"] = 2
        evidence = detector_evidence({}, ai_result=ai)
        decision = decide_policy({}, detectors=evidence)
        self.assertTrue(evidence["classifier"]["hit"])
        self.assertFalse(evidence["classifier"]["usable_for_policy"])
        self.assertEqual(decision["stage"], "monitor")

    def test_isolated_model_hit_collects_evidence_model_consensus_alerts(self):
        for cls, iso in ((True, False), (False, True)):
            evidence = detector_evidence({}, ai_result=compatible_ai(cls, iso))
            self.assertEqual(decide_policy({}, detectors=evidence)["stage"], "suspicious_activity")
        evidence = detector_evidence({}, ai_result=compatible_ai())
        self.assertEqual(decide_policy({}, detectors=evidence)["stage"], "suspicious_activity")
        self.assertEqual(decide_policy({"high_entropy_write": .9}, detectors=evidence)["stage"], "analyst_alert")
        self.assertFalse(decide_policy({}, detectors=evidence)["containment_recommended"])

    def test_corroborated_detection_recommends_but_never_executes_containment(self):
        symptoms, damage = analyze_events(destructive_events())
        for alternative in ALTERNATIVES:
            response = decide_policy(symptoms, damage, alternative=alternative)
            self.assertEqual(response["stage"], "containment_recommendation")
            self.assertTrue(response["containment_recommended"])
            self.assertFalse(response["containment_executed"])
            self.assertTrue(response["evidence_categories"]["corroborated_high_confidence_detection"])

    def test_cross_process_host_low_confidence_and_one_reference_prevent_corroboration(self):
        for change in ("process", "host", "confidence", "reference"):
            events = destructive_events()
            if change == "process": events[-1]["process"]["entity_id"] = "other"
            if change == "host": events[-1]["host"] = "other"
            if change == "confidence": events[-1]["observation_confidence"] = .2
            if change == "reference":
                for event in events:
                    for item in event["indicators"]: item["reference"] = "same-observation"
            symptoms, damage = analyze_events(events)
            with self.subTest(change=change):
                self.assertFalse(decide_policy(symptoms, damage)["containment_recommended"])

    def test_reviewed_signature_stays_evidence_not_containment_override(self):
        symptoms, damage = analyze_events([make_event("file_lock", path="a.txt")])
        evidence = detector_evidence(symptoms, damage, learned_match={"matched": True})
        response = decide_policy(symptoms, damage, evidence)
        self.assertEqual(response["stage"], "analyst_alert")
        self.assertTrue(response["detector_evidence"]["reviewed_signature"]["hit"])
        self.assertFalse(response["containment_recommended"])
        nonmatch = detector_evidence(symptoms, damage, learned_match={"matched": False})
        self.assertFalse(nonmatch["reviewed_signature"]["hit"])
        self.assertEqual(decide_policy(symptoms, damage, nonmatch)["stage"], "suspicious_activity")

    def test_safe_watcher_only_requests_cooperative_stop_for_synthetic_corroboration(self):
        for kind in ("synthetic", "sysmon"):
            events = destructive_events()
            for event in events: event["source"]["kind"] = kind
            with self.subTest(kind=kind), temporary_cwd():
                for folder in (watcher.QUEUE_DIR, watcher.REPORT_DIR, watcher.CONTROL_DIR, watcher.ERROR_DIR): folder.mkdir(parents=True, exist_ok=True)
                with patch.object(watcher, "call_ai", return_value=compatible_ai()), patch.object(watcher, "call_explain", return_value={}), contextlib.redirect_stdout(io.StringIO()):
                    alert = watcher.evaluate_window(events, 1)
                self.assertEqual(alert["simulated_stop_requested"], kind == "synthetic")
                self.assertFalse(alert["containment_executed"])
                self.assertFalse(alert["simulated_containment_executed"])
                self.assertTrue(alert["detector_evidence"]["classifier"]["hit"])
                self.assertEqual(alert["alert_schema_version"], 3)

    def test_event_api_returns_evidence_when_model_unavailable(self):
        with patch.object(api, "infer", side_effect=api.HTTPException(status_code=503, detail="fixture")):
            result = TestClient(api.app).post("/events/evaluate", json={"events": destructive_events()})
        self.assertEqual(result.status_code, 200)
        self.assertTrue(result.json()["response"]["containment_recommended"])
        self.assertFalse(result.json()["detector_evidence"]["classifier"]["available"])
        self.assertFalse(result.json()["response"]["containment_executed"])

    def test_api_model_compatibility_gate_and_window_validation(self):
        from unittest.mock import Mock
        import numpy as np
        classifier = Mock(classes_=np.array(["known_ransomware_like"]))
        classifier.predict.return_value = ["known_ransomware_like"]
        classifier.predict_proba.return_value = [[1.0]]
        anomaly = Mock()
        anomaly.predict.return_value = [-1]
        anomaly.decision_function.return_value = [-.2]
        metadata = {"feature_columns": ["file_write_burst"], "feature_semantics_version": 3}
        shadow = {"mode": "fixture", "calibrated_shadow": {"status": "unavailable"}}
        with patch.object(api, "load_artifacts", return_value=(classifier, anomaly, metadata)), patch.object(api, "evaluate_unknown_detector_shadow", return_value=shadow):
            for version in (2, 3):
                result = api.infer(api.SymptomRequest(symptoms={"file_write_burst": .8, "high_entropy_write": .9}, feature_semantics_version=version))
                self.assertEqual(result["detector_evidence"]["classifier"]["usable_for_policy"], version == 3)
                self.assertEqual(result["response"]["stage"], "analyst_alert" if version == 3 else "suspicious_activity")
        events = [make_event("network_poll"), make_event("network_poll", window=6)]
        result = TestClient(api.app).post("/events/evaluate", json={"events": events})
        self.assertEqual(result.status_code, 422)

    def test_conflicting_duplicate_source_records_fail_closed(self):
        event = make_event("file_read", path="a.txt")
        different = {**event, "path": "b.txt"}
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            analyze_events([event, different])

    def test_feature_version_tampering_invalidates_review_and_mixing_is_rejected(self):
        from symptom_ai.auto_learning.review import review_case, is_approved
        from symptom_ai.auto_learning.auto_retraining_pipeline import cases_to_dataframe
        from symptom_ai.models.train_symptom_models import dataset_semantics_version
        import pandas as pd
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.json"
            path.write_text(json.dumps({"symptoms": {"file_write_burst": .2}, "events": [], "feature_semantics_version": 3}))
            case = review_case(path, "benign", "reviewer", "Typed synthetic observations")
            self.assertTrue(is_approved(case))
            with self.assertRaisesRegex(ValueError, "semantics differ"):
                cases_to_dataframe([case], ["file_write_burst", "label"])
            self.assertEqual(len(cases_to_dataframe([case], ["file_write_burst", "label"], 3)), 1)
            case["feature_semantics_version"] = 2
            self.assertFalse(is_approved(case))
        with self.assertRaisesRegex(ValueError, "Mixed"):
            dataset_semantics_version(pd.DataFrame({"feature_semantics_version": [2, 3]}))

    def test_typed_reviewed_signature_uses_shared_approval_record(self):
        from symptom_ai.auto_learning.review import review_case
        events = destructive_events()
        with tempfile.TemporaryDirectory() as directory, patch.object(watcher, "QUEUE_DIR", Path(directory)):
            symptoms, damage = analyze_events(events)
            path = watcher.save_learning_case("typed", events, symptoms, damage, {}, "fixture")
            self.assertIsNone(watcher.match_learned_signature(events))
            review_case(path, "known_ransomware_like", "reviewer", "Synthetic corroborated behavior")
            self.assertIsNotNone(watcher.match_learned_signature(events))
            review_case(path, "rejected", "reviewer", "Revoke approval")
            self.assertIsNone(watcher.match_learned_signature(events))



class PolicyDevelopmentTests(unittest.TestCase):
    def test_multiple_independent_sessions_and_legitimate_scenarios(self):
        rows = generate(seed=7, sessions=6)
        again = generate(seed=7, sessions=6)
        self.assertEqual(rows, again)
        benign = [row for row in rows if row["profile"] in {"bulk_file_operations", "backup", "compression", "build_tools", "administration", "network_polling"}]
        for row in benign:
            response = decide_policy(row["symptoms"], row["damage"])
            self.assertIn(response["stage"], ("monitor", "suspicious_activity"))
            self.assertFalse(response["containment_recommended"])
        development, validation, info = partitions(rows, seed=7)
        self.assertFalse(set(info["session_ids"]["development"]) & set(info["session_ids"]["validation"]))
        self.assertFalse(set(info["group_ids"]["development"]) & set(info["group_ids"]["validation"]))
        self.assertEqual(len(development) + len(validation), len(rows))

    def test_protocol_never_calls_task02_partition_reader_or_final_evaluator(self):
        with tempfile.TemporaryDirectory() as folder, patch("symptom_ai.benchmark.baseline.read_rows", side_effect=AssertionError("Retired holdout")), patch("symptom_ai.benchmark.baseline.evaluate", side_effect=AssertionError("No final evaluation")):
            root = Path(folder) / "run"
            report = run(root, seed=19, sessions=6)
            self.assertIsNone(report["protocol"]["test_partition"])
            self.assertFalse(report["protocol"]["task02_holdout_accessed"])
            self.assertEqual(set(report["partitions"]), {"development", "validation"})
            self.assertTrue((root / "protocol.json").exists())
            self.assertFalse((root / "test.jsonl").exists())
            with self.assertRaisesRegex(ValueError, "already exists"):
                run(root, seed=19, sessions=6)


if __name__ == "__main__":
    unittest.main()
