from pathlib import Path
import tempfile
import unittest

from microduck_project.common import ROOT, read_json, write_json, sha256, upstream_lock, digest
from microduck_project.config import load_experiment
from microduck_project.runtime import checkpoint_metadata, validate_resume
from microduck_project.metrics import TrackingMetrics, aggregate, command_at
from microduck_project.artifacts import compare


class MetricsTests(unittest.TestCase):
    def test_rmse_uses_all_samples(self):
        m = TrackingMetrics()
        m.add((0.2, 0, 0.5), (0.1, 0, 0.3))
        m.add((0, 0, 0), (0.1, 0, 0.2))
        self.assertAlmostEqual(m.result()["vx_rmse_m_s"], 0.1)
        self.assertAlmostEqual(m.result()["yaw_rate_rmse_rad_s"], 0.2)
        self.assertAlmostEqual(m.result()["stop_speed_rms_m_s"], 0.1)

    def test_empty_is_unknown_not_zero(self):
        self.assertIsNone(TrackingMetrics().result()["xy_rmse_m_s"])

    def test_nonfinite_measurement_rejected(self):
        with self.assertRaises(ValueError):
            TrackingMetrics().add((0, 0, 0), (float("nan"), 0, 0))

    def test_failed_episode_stays_in_denominator(self):
        result = aggregate([{"success": True, "reason": "arrived", "duration_seconds": 4},
                            {"success": False, "reason": "fall", "duration_seconds": 1}])
        self.assertEqual(result["success_rate"], 0.5)
        self.assertEqual(result["fall_rate"], 0.5)

    def test_command_boundary(self):
        segments = [{"seconds": 1, "twist": [0.1, 0, 0]}, {"seconds": 2, "twist": [0, 0, 0.5]}]
        self.assertEqual(command_at(segments, 1), (0, 0, 0.5))
        self.assertEqual(command_at(segments, 3), (0, 0, 0))


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.exp = load_experiment(ROOT / "configs/experiments/smoke.json")
        self.meta = {"schema_version": 1, "upstream_commit": upstream_lock()["commit"],
                     "experiment": self.exp, "completed_iterations": 5, "num_steps_per_env": 24,
                     "total_env_steps": 5 * 24 * self.exp["num_envs"], "source_sha256": "source",
                     "configuration_sha256": "cfg", "lock_sha256": "lock"}

    def test_atomic_json_and_checkpoint_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "final.pt"
            path.write_bytes(b"test-fixture-not-a-real-trained-model")
            self.meta["checkpoint_sha256"] = sha256(path)
            write_json(path.with_suffix(".json"), self.meta)
            self.assertEqual(checkpoint_metadata(path)["completed_iterations"], 5)
            self.assertFalse(path.with_suffix(".json.tmp").exists())
            path.write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                checkpoint_metadata(path)

    def test_resume_rejects_changed_code_or_condition(self):
        new = dict(self.exp, total_iterations=10)
        validate_resume(self.meta, new, "cfg", "source", "lock")
        for contract, source, lock in (("different", "source", "lock"), ("cfg", "different", "lock"), ("cfg", "source", "different")):
            with self.assertRaises(ValueError):
                validate_resume(self.meta, new, contract, source, lock)
        new["num_envs"] = 64
        with self.assertRaises(ValueError):
            validate_resume(self.meta, new, "cfg", "source", "lock")

    def test_resume_total_is_not_additional(self):
        with self.assertRaises(ValueError):
            validate_resume(self.meta, self.exp, "cfg", "source", "lock")

    def test_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "invalid.json"
            with self.assertRaises(ValueError):
                write_json(p, {"value": float("nan")})
            p.write_text('{"x": NaN}')
            with self.assertRaises(ValueError):
                read_json(p)


class ComparisonTests(unittest.TestCase):
    def reports(self):
        scenario = read_json(ROOT / "configs/evaluation.json")
        result = []
        for name in ("baseline", "low_speed"):
            cfg = load_experiment(ROOT / f"configs/experiments/{name}.json")
            result.append({"status": "completed", "mode": "walk", "split": "final", "scenario": scenario,
                "scenario_sha256": digest(scenario), "evaluation_source_sha256": "test", "evaluation_settings": {},
                "seeds": scenario["final_seeds"],
                "episodes": [{"seed": s, "success": s != 200, "reason": "fall" if s == 200 else "completed", "duration_seconds": 30} for s in scenario["final_seeds"]],
                "checkpoint": {"experiment": cfg, "completed_iterations": 4000, "total_env_steps": 4000*24*1024,
                    "num_steps_per_env": 24, "upstream_commit": "test", "source_sha256": "source", "lock_sha256": "lock",
                    "initialization": "scratch", "wall_seconds_total": 100}})
        return result

    def check(self, reports):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, data in zip(("a", "b"), reports):
                write_json(root / f"{name}.json", data)
            return compare(root / "a.json", root / "b.json", root / "out")

    def test_shared_scenarios_accepted(self):
        rows = self.check(self.reports())
        self.assertEqual(rows[0]["fall_rate"], 0.1)

    def test_missing_failure_rejected(self):
        reports = self.reports()
        reports[0]["episodes"].pop(0)
        with self.assertRaises(ValueError):
            self.check(reports)

    def test_mismatched_steps_rejected(self):
        reports = self.reports()
        reports[1]["checkpoint"]["total_env_steps"] += 24
        with self.assertRaises(ValueError):
            self.check(reports)

    def test_selection_metrics_cannot_be_final(self):
        reports = self.reports()
        reports[0]["split"] = "selection"
        with self.assertRaises(ValueError):
            self.check(reports)

    def test_scenario_change_rejected(self):
        reports = self.reports()
        reports[0]["scenario_sha256"] = "tampered"
        with self.assertRaises(ValueError):
            self.check(reports)
