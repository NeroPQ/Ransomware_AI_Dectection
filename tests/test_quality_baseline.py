"""Protocol correctness tests; these fixtures do not establish detector efficacy."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from symptom_ai.benchmark import baseline
from symptom_ai.benchmark.sessions import generate_sessions, session_partitions, frame_from_windows, BENIGN_PROFILES
from symptom_ai.benchmark.detectors import score_windows, METHODS
from symptom_ai.benchmark.quality_metrics import detection_metrics
from symptom_ai.benchmark.sources import reviewed_workflow
from symptom_ai.models.evaluation import feature_matrix
from symptom_ai.live_monitoring.live_log_ai_watcher import aggregate_symptoms
from symptom_ai.response_engine.response_policy import recommend_response

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_quality_baseline import create_source_fixture


def metric_rows(labels, time_basis="observed"):
    return [{"label": label, "session_id": "session", "endpoint_id": "endpoint", "time_basis": time_basis,
             "window_start_seconds": index * 60, "window_end_seconds": (index + 1) * 60,
             "attack_onset_seconds": 120, "impact_onset_seconds": 240}
            for index, label in enumerate(labels)]


class MetricContractTests(unittest.TestCase):
    def test_counts_rates_and_latency_use_window_ends_and_exclude_pre_onset_alarms(self):
        rows = metric_rows(["benign", "benign", "known_ransomware_like", "known_ransomware_like"])
        result = detection_metrics(rows, [True, False, False, True])
        self.assertEqual([result[key] for key in ("true_positive", "false_positive", "true_negative", "false_negative")], [1, 1, 1, 1])
        self.assertEqual(result["precision"], .5)
        self.assertEqual(result["recall"], .5)
        self.assertEqual(result["f1"], .5)
        self.assertEqual(result["fpr"], .5)
        self.assertEqual(result["false_alerts_per_endpoint_hour"], 30)
        self.assertEqual(result["detection_latency_seconds"]["median_among_detected"], 120)
        self.assertIsNone(result["false_alerts_per_simulated_endpoint_hour"])

    def test_synthetic_times_never_become_real_endpoint_hours(self):
        result = detection_metrics(metric_rows(["benign", "known_ransomware_like"], "simulated"), [True, False])
        self.assertEqual(result["false_alerts_per_simulated_endpoint_hour"], 60)
        self.assertIsNone(result["false_alerts_per_endpoint_hour"])
        self.assertIsNone(result["detection_latency_seconds"])

    def test_missing_time_or_onset_is_not_fabricated(self):
        rows = [{"label": "benign"}, {"label": "known_ransomware_like"}]
        result = detection_metrics(rows, [False, True])
        self.assertIsNone(result["false_alerts_per_endpoint_hour"])
        self.assertIsNone(result["detection_latency_seconds"])
        self.assertIsNone(detection_metrics(rows[:1], [False])["recall"])
        self.assertIsNone(detection_metrics(rows[:1], [False])["precision"])

    def test_overlapping_endpoint_exposure_is_rejected(self):
        rows = metric_rows(["benign", "benign"])
        rows[1]["window_start_seconds"] = 30
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            detection_metrics(rows, [True, True])

    def test_unknown_labels_and_gaps_cannot_create_supported_metrics(self):
        with self.assertRaisesRegex(ValueError, "labels"):
            detection_metrics([{"label": "unknown"}], [True])
        rows = metric_rows(["benign", "benign", "known_ransomware_like", "known_ransomware_like"])
        rows[3]["window_start_seconds"] = 200
        result = detection_metrics(rows, [False, False, False, True])
        self.assertIsNone(result["detection_latency_seconds"])

    def test_missed_attacks_remain_censored_not_zero_latency(self):
        rows = metric_rows(["benign", "benign", "known_ransomware_like", "known_ransomware_like"])
        result = detection_metrics(rows, [True, True, False, False])
        latency = result["detection_latency_seconds"]
        self.assertEqual(latency["missed_sessions"], 1)
        self.assertIsNone(latency["median_among_detected"])
        self.assertEqual(latency["detected_sessions"], 0)


class SessionContractTests(unittest.TestCase):
    def test_generator_reproducible_and_all_six_workloads_have_multiple_sessions_and_windows(self):
        first = generate_sessions(seed=17, sessions_per_profile=6)
        self.assertEqual(first, generate_sessions(seed=17, sessions_per_profile=6))
        self.assertNotEqual(first[0]["events"], generate_sessions(seed=18, sessions_per_profile=6)[0]["events"])
        for profile in BENIGN_PROFILES:
            sessions = {row["session_id"] for row in first if row["profile"] == profile}
            self.assertEqual(len(sessions), 6)
            for session in sessions:
                self.assertEqual(sum(row["session_id"] == session for row in first), 8)

    def test_session_endpoint_and_duplicate_observations_do_not_cross_partitions(self):
        rows = generate_sessions(seed=29, sessions_per_profile=12)
        # Exact observations duplicated across otherwise distinct sessions must
        # connect the entire sessions, not just the duplicated windows.
        rows[8]["symptoms"] = dict(rows[0]["symptoms"])
        parts, info = session_partitions(rows, 29)
        memberships = {row["session_id"]: name for name, part in parts.items() for row in part}
        self.assertEqual(memberships[rows[0]["session_id"]], memberships[rows[8]["session_id"]])
        for key in ("session_id", "endpoint_id"):
            groups = [{row[key] for row in part} for part in parts.values()]
            for index, group in enumerate(groups):
                for other in groups[index + 1:]:
                    self.assertFalse(group & other)
        signatures = [{tuple(values) for values in feature_matrix(frame_from_windows(part)).to_numpy()} for part in parts.values()]
        self.assertFalse(signatures[0] & signatures[1])
        self.assertFalse(signatures[0] & signatures[2])
        self.assertFalse(signatures[1] & signatures[2])
        self.assertTrue(info["source_grouping_excluded"])

    def test_metadata_and_target_flags_are_excluded(self):
        rows = generate_sessions(seed=29, sessions_per_profile=6)
        frame = frame_from_windows(rows)
        X = feature_matrix(frame)
        self.assertFalse(set(X) & {"profile", "window_index", "rule_hit", "label", "attack_onset_seconds", "analyst_review_required"})

    def test_known_benign_path_cardinality_triggers_warning_without_models(self):
        events = [{"event_type": "file_read", "host": "host", "path": f"one_directory/{i}.txt"} for i in range(15)]
        symptoms, _ = aggregate_symptoms(events)
        self.assertEqual(symptoms["multi_directory_impact"], 1)
        self.assertEqual(recommend_response(symptoms, "low")["policy"], "early_warning")


class FrozenProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name) / "experiment"
        baseline.prepare(cls.root, seed=41, sessions_per_profile=12)
        original = baseline.read_rows
        reads = []
        def read(root, name):
            reads.append(name)
            return original(root, name)
        with patch.object(baseline, "read_rows", side_effect=read):
            cls.frozen = baseline.calibrate(cls.root)
        cls.development_reads = reads

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_calibration_reads_only_train_and_validation(self):
        self.assertEqual(self.development_reads, ["train", "validation"])
        self.assertFalse(self.frozen["test_features_used_for_fitting_or_calibration"])
        with self.assertRaisesRegex(ValueError, "already frozen"):
            baseline.calibrate(self.root)

    def test_tampered_model_or_inputs_fail_before_final_scoring(self):
        path = self.root / "test.jsonl"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "partition changed"):
                baseline.evaluate(self.root)
        finally:
            path.write_bytes(original)
        path = self.root / "frozen_models.joblib"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"changed")
            with self.assertRaisesRegex(ValueError, "Frozen artifacts"):
                baseline.evaluate(self.root)
        finally:
            path.write_bytes(original)

    def test_final_evaluation_is_separate_and_refuses_overwrite(self):
        # Copy the frozen experiment because other tests verify its unused holdout.
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "copy"
            shutil.copytree(self.root, root)
            result = baseline.evaluate(root)
            self.assertEqual(set(result["test_results"]), set(METHODS))
            self.assertEqual(result["actions_executed"], 0)
            for metrics in result["test_results"].values():
                self.assertIsNone(metrics["false_alerts_per_endpoint_hour"])
                self.assertEqual(metrics["session_bootstrap"]["unit"], "session")
            with self.assertRaisesRegex(ValueError, "already consumed"):
                baseline.evaluate(root)


class ReviewedSourceTests(unittest.TestCase):
    def test_missing_review_or_changed_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = create_source_fixture(root / "sources")
            manifest = json.loads(path.read_text())
            manifest["reviewer"] = ""
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "reviewer"):
                reviewed_workflow(path, root / "output")
            manifest["reviewer"] = "fixture reviewer"
            manifest["sources"][0]["sha256"] = "wrong"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "checksum"):
                reviewed_workflow(path, root / "output")

    def test_fixture_adapter_cannot_be_used_for_genuine_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = create_source_fixture(root / "sources")
            manifest = json.loads(path.read_text())
            manifest["source_kind"] = "captured_feature_data"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Fixture adapter"):
                reviewed_workflow(path, root / "output")


if __name__ == "__main__":
    unittest.main()
