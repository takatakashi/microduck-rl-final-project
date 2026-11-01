"""Exercise the evaluation loop with simulated physics, never robot training.

Use actual CPU PyTorch tensors when PyTorch is installed (e.g. on Brev).
The standard-library-only suite uses a small inference-tensor fixture instead.
"""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from microduck_project import evaluation
from microduck_project.common import ROOT, read_json, write_json
from microduck_project.controller import Observation

try:
    import torch as real_torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    real_torch = None


class InferenceTensorFixture:
    def __init__(self, backend):
        self.backend = backend
        self.inference = backend.is_inference_mode_enabled()

    def zero_(self):
        if self.inference and not self.backend.is_inference_mode_enabled():
            raise RuntimeError("Inplace update to inference tensor outside InferenceMode is not allowed.")
        return self

    def all(self):
        return True


class TorchModeFixture:
    def __init__(self):
        self.enabled = False

    @contextmanager
    def inference_mode(self):
        previous = self.enabled
        self.enabled = True
        try:
            yield
        finally:
            self.enabled = previous

    def is_inference_mode_enabled(self):
        return self.enabled

    def zeros(self, shape):
        return InferenceTensorFixture(self)

    def isfinite(self, tensor):
        return tensor


class SimulationFixture:
    step_dt = 0.1

    def __init__(self, torch, *, at_goal=False, fail_policy=False):
        self.torch = torch
        self.cache = torch.zeros((1, 3))
        self.at_goal, self.fail_policy = at_goal, fail_policy
        self.calls, self.seeds = [], []
        self.steps = 0
        self.closed = False

    def touch(self, name):
        self.calls.append((name, self.torch.is_inference_mode_enabled()))
        self.cache.zero_()

    def reset(self, *, seed):
        self.touch("reset")
        self.seeds.append(seed)

    def command(self, env, twist):
        self.touch("command")

    def observe(self, env):
        self.touch("observation")
        return Observation(0.6 if self.at_goal else 0.0, 0, 0, 0, 0, 0), (0, 0, 0)

    def get_observations(self):
        self.touch("get_observations")
        return {"actor": self.cache}

    def policy(self, obs):
        self.touch("policy")
        if self.fail_policy:
            raise ValueError("test policy failure")
        return self.torch.zeros((1, 14))

    def step(self, actions):
        self.touch("step")
        # mjlab also replaces its cached observation during a physics step.
        self.cache = self.torch.zeros((1, 3))
        self.steps += 1
        return None, None, [SimpleNamespace(item=lambda: False)], {}

    def render(self):
        self.touch("render")
        return object()  # Consumed by the movie fixture, not a real video frame.

    def movie(self, path, **kwargs):
        return SimpleNamespace(path=path,
            append=lambda *args: self.touch("movie_append"),
            close=lambda: self.touch("movie_close"))

    def close(self):
        self.closed = True
        self.touch("close")


class EvaluationInferenceTests(unittest.TestCase):
    def setUp(self):
        self.torch = real_torch if real_torch is not None else TorchModeFixture()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "evaluation"
        self.scenario = read_json(ROOT / "configs/evaluation.json")
        self.scenario["segments"] = [{"seconds": 0.3, "twist": [0.1, 0, 0]}]
        self.scenario_path = Path(self.temp.name) / "scenario.json"
        write_json(self.scenario_path, self.scenario)

    def run_evaluation(self, world, *, video=False, mode="walk", smoke=False):
        ranges = SimpleNamespace(lin_vel_x=(-0.4, 0.4), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-1, 1))
        config = SimpleNamespace(commands={"twist": SimpleNamespace(ranges=ranges)})
        metadata = {"experiment": {"condition": "test-fixture", "seed": 42}}
        with patch.dict(sys.modules, {"torch": self.torch}), patch.multiple(
            evaluation,
            load_for_inference=lambda *args, **kwargs: (world, world, None, world.policy, metadata),
            task_configs=lambda *args: (config, None, None, None),
            source_identity=lambda: {"source_sha256": "test-fixture"},
            set_command=world.command,
            observation=world.observe,
            VideoWriter=world.movie,
            validate_video=lambda *args: {"fixture": "no real recording"},
        ):
            return evaluation.evaluate(Path("fixture.pt"), self.scenario_path, self.output,
                                       split="selection", mode=mode, video=video, smoke=smoke)

    def assert_mode_restored(self, world):
        self.assertTrue(world.closed)
        self.assertFalse(self.torch.is_inference_mode_enabled())
        self.assertTrue(world.calls)
        self.assertTrue(all(enabled for _, enabled in world.calls), world.calls)

    def test_repeated_steps_and_seed_resets_share_inference_mode(self):
        world = SimulationFixture(self.torch)
        result = self.run_evaluation(world)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(world.seeds, self.scenario["selection_seeds"])
        self.assertEqual(world.steps, 9)
        self.assert_mode_restored(world)

    def test_video_render_and_cleanup_share_inference_mode(self):
        world = SimulationFixture(self.torch)
        result = self.run_evaluation(world, video=True, smoke=True)
        self.assertEqual(result["status"], "completed")
        names = [name for name, _ in world.calls]
        self.assertEqual(names.count("render"), world.steps + 1)
        self.assertEqual(names.count("movie_close"), 1)
        self.assert_mode_restored(world)

    def test_goal_controller_and_stopping_share_inference_mode(self):
        world = SimulationFixture(self.torch, at_goal=True)
        result = self.run_evaluation(world, mode="goal", smoke=True)
        self.assertEqual(result["episodes"][0]["reason"], "arrived")
        self.assertGreaterEqual(result["episodes"][0]["held_seconds"], 2)
        self.assert_mode_restored(world)

    def test_exception_records_failure_and_closes_environment(self):
        world = SimulationFixture(self.torch, fail_policy=True)
        with self.assertRaisesRegex(ValueError, "test policy failure"):
            self.run_evaluation(world, video=True, smoke=True)
        report = read_json(self.output / "evaluation.json")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"], "test policy failure")
        self.assert_mode_restored(world)
