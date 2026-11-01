from pathlib import Path
import tempfile
import unittest

from microduck_project.common import ROOT, digest, read_json, sha256, upstream_lock, write_json
from scripts.check_m0 import check_m0


class M0RecordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.evaluation = self.root / "video_inference_fix"
        self.evaluation.mkdir()
        self.baseline = self.root / "baseline.json"
        write_json(self.baseline, read_json(ROOT / "configs/experiments/baseline.json"))
        experiment = read_json(ROOT / "configs/experiments/smoke.json")
        gpu = {"name": "test fixture; no actual GPU"}
        write_json(self.root / "environment.json", {"gpu_validation": "passed",
            "gpu_probe": {"physics_probe": "passed", "gpu": gpu}})
        initial_hash = None
        for stage, iterations in (("initial", 5), ("resumed", 10)):
            directory = self.root / stage
            directory.mkdir()
            model = directory / "final.pt"
            model.write_bytes(f"{stage}: not a real model".encode())
            model_hash = sha256(model)
            steps = iterations * 24 * experiment["num_envs"]
            meta = {"schema_version": 1, "checkpoint_sha256": model_hash,
                "upstream_commit": upstream_lock()["commit"], "lock_sha256": "fixture-lock",
                "source_sha256": "fixture-source", "configuration_sha256": "fixture-config",
                "io_contract": {}, "experiment": dict(experiment, total_iterations=iterations),
                "completed_iterations": iterations, "total_env_steps": steps, "num_steps_per_env": 24,
                "parent_checkpoint_sha256": initial_hash}
            write_json(model.with_suffix(".json"), meta)
            write_json(directory / "run.json", {"status": "completed", "completed_iterations": iterations,
                "total_env_steps": steps, "checkpoint_sha256": model_hash, "gpu": gpu,
                "initialization": "scratch" if stage == "initial" else "resume",
                "parent_checkpoint_sha256": initial_hash, "resume_verified_before_learning": stage == "resumed",
                "restored_completed_iterations": 0 if stage == "initial" else 5,
                "wall_seconds_this_process": 20, "warm_iteration_mean_seconds": 2.5})
            if stage == "initial":
                initial_hash = model_hash
        video = self.evaluation / "seed_200.mp4"
        video.write_bytes(b"video fixture; not an actual MP4")
        scenario = read_json(ROOT / "configs/evaluation.json")
        write_json(self.evaluation / "evaluation.json", {"status": "completed", "split": "smoke",
            "checkpoint": meta, "scenario": scenario, "scenario_sha256": digest(scenario), "seeds": [200],
            "episodes": [{"seed": 200, "success": False, "reason": "fall",
                "video": {"file": video.name, "decode": "passed", "sha256": sha256(video), "duration_seconds": 1.5}}]})
        self.onnx = self.root / "smoke_export.onnx"
        self.onnx.write_bytes(b"ONNX fixture; not a real model")
        write_json(self.onnx.with_suffix(".parity.json"), {"status": "passed", "samples": 768,
            "checkpoint_sha256": model_hash, "onnx_sha256": sha256(self.onnx)})
        # Recovered recording/export must not require rewriting the failed original run.
        write_json(self.root / "smoke.json", {"status": "failed"})

    def check(self, price=None):
        return check_m0(self.root, self.evaluation, self.onnx, self.baseline, price)

    def test_separate_recovery_passes_without_claiming_walk_success_or_launch(self):
        result = self.check()
        self.assertEqual(result["status"], "passed_record_checks")
        self.assertFalse(result["training_started"])
        self.assertEqual(result["baseline"]["total_env_steps"], 98_304_000)
        self.assertIsNone(result["estimate"]["gpu_compute_cost"])
        self.assertAlmostEqual(result["estimate"]["training_hours"], 4000 * 2.5 / 3600)
        self.assertAlmostEqual(self.check(3)["estimate"]["gpu_compute_cost"], 4000 * 2.5 / 3600 * 3)
        self.assertEqual(read_json(self.root / "smoke.json")["status"], "failed")

    def test_requires_resume_evidence(self):
        path = self.root / "resumed/run.json"
        original = read_json(path)
        for change in ({"resume_verified_before_learning": False}, {"restored_completed_iterations": 4},
                       {"parent_checkpoint_sha256": "different"}, {"total_env_steps": 1}):
            with self.subTest(change=change):
                write_json(path, dict(original, **change))
                with self.assertRaises(ValueError):
                    self.check()
        write_json(path, original)

    def test_rejects_modified_model_video_or_onnx(self):
        for path in (self.root / "resumed/final.pt", self.evaluation / "seed_200.mp4", self.onnx):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    self.check()
                path.write_bytes(original)

    def test_requires_passed_parity_for_the_same_checkpoint(self):
        path = self.onnx.with_suffix(".parity.json")
        original = read_json(path)
        for change in ({"status": "failed"}, {"samples": 0}, {"checkpoint_sha256": "other-model"}):
            with self.subTest(change=change):
                write_json(path, dict(original, **change))
                with self.assertRaises(ValueError):
                    self.check()
        write_json(path, original)

    def test_estimate_requires_matching_parallelism_and_measured_time(self):
        original = read_json(self.baseline)
        write_json(self.baseline, dict(original, num_envs=2048))
        with self.assertRaises(ValueError):
            self.check()
        write_json(self.baseline, original)
        path = self.root / "initial/run.json"
        record = read_json(path)
        write_json(path, dict(record, warm_iteration_mean_seconds=0))
        with self.assertRaises(ValueError):
            self.check()
