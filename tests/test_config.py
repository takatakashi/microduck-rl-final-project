from copy import deepcopy
import unittest
from types import SimpleNamespace

from microduck_project.common import ROOT, read_json
from microduck_project.config import apply_change, assert_fair_pair, load_experiment, resume_key, validate_experiment
from microduck_project.evaluation import validate_scenario


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.base = load_experiment(ROOT / "configs/experiments/baseline.json")
        self.variant = load_experiment(ROOT / "configs/experiments/low_speed.json")

    def test_fair_defaults(self):
        assert_fair_pair(self.base, self.variant)

    def test_reject_extra_setting(self):
        self.base["imaginary_reward"] = 3
        with self.assertRaises(ValueError):
            validate_experiment(self.base)

    def test_reject_invalid_range(self):
        self.variant["change"]["value"] = [0.2, -0.2]
        with self.assertRaises(ValueError):
            validate_experiment(self.variant)

    def test_reject_wrong_path(self):
        self.variant["change"]["path"] = "nonexistent.setting"
        with self.assertRaises(ValueError):
            validate_experiment(self.variant)

    def test_only_one_real_leaf_changes(self):
        env = SimpleNamespace(commands={"twist": SimpleNamespace(ranges=SimpleNamespace(
            lin_vel_x=(-0.4, 0.4), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-1, 1)))}, rewards={"sentinel": 10})
        original = deepcopy(env)
        self.assertEqual(apply_change(env, self.base), [])
        self.assertEqual(vars(env), vars(original))
        diff = apply_change(env, self.variant)
        self.assertEqual(len(diff), 1)
        self.assertEqual(env.commands["twist"].ranges.lin_vel_x, (-0.2, 0.2))
        self.assertEqual(env.commands["twist"].ranges.lin_vel_y, (-0.3, 0.3))
        self.assertEqual(env.rewards, original.rewards)
        with self.assertRaises(ValueError):
            apply_change(env, self.variant)  # Not silently applied to an unexpected source.

    def test_unfair_training_steps_rejected(self):
        self.variant["total_iterations"] += 1
        with self.assertRaises(ValueError):
            assert_fair_pair(self.base, self.variant)

    def test_unfair_parallelism_rejected(self):
        self.variant["num_envs"] *= 2
        with self.assertRaises(ValueError):
            assert_fair_pair(self.base, self.variant)

    def test_resume_allows_only_run_controls(self):
        updated = deepcopy(self.base)
        updated.update(total_iterations=5000, save_interval=10, name="resume")
        self.assertEqual(resume_key(self.base), resume_key(updated))
        updated["seed"] += 1
        self.assertNotEqual(resume_key(self.base), resume_key(updated))

    def test_final_seeds_disjoint_from_selection(self):
        scenario = read_json(ROOT / "configs/evaluation.json")
        validate_scenario(scenario)
        scenario["final_seeds"][0] = scenario["selection_seeds"][0]
        with self.assertRaises(ValueError):
            validate_scenario(scenario)
