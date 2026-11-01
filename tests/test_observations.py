from types import SimpleNamespace
import unittest

from microduck_project.observations import command_slices, refresh_command_observations


class Values:
    """Minimal array fixture, without installing GPU/numerical dependencies."""
    def __init__(self, data):
        self.data = list(data)
        self.shape = (1, len(data))

    def clone(self):
        return Values(self.data)

    def __setitem__(self, key, value):
        rows, columns = key
        self.data[columns] = value.data


class Manager:
    def __init__(self):
        # Deliberately different order and proprioception size: no fixed 48D offset.
        self.active_terms = {"actor": ["sensors", "head", "velocity", "body"]}
        self.group_obs_concatenate = {"actor": True}
        self.group_obs_term_dim = {"actor": [(2,), (4,), (3,), (6,)]}
        self.cache = {"actor": Values([9, 8] + [0] * 13)}
        self.configs = {}
        for name, command in (("sensors", None), ("head", "head_pose"), ("velocity", "twist"), ("body", "body_pose")):
            self.configs[name] = SimpleNamespace(params={"command_name": command}, history_length=0,
                delay_max_lag=0, noise=None, clip=None, scale=None,
                func=lambda env, command_name: Values(env.commands[command_name]))

    def get_term_cfg(self, group, name):
        return self.configs[name]

    def compute(self):
        return self.cache


class ObservationTests(unittest.TestCase):
    def test_resolve_registered_order(self):
        self.assertEqual(command_slices(Manager()), [("actor", "head", 2, 6), ("actor", "velocity", 6, 9), ("actor", "body", 9, 15)])

    def test_command_update_reaches_cached_observation_without_sensor_recompute(self):
        manager = Manager()
        env = SimpleNamespace(num_envs=1, observation_manager=manager,
                              commands={"twist": [0.2, 0, 0.5], "head_pose": [0]*4, "body_pose": [0]*6})
        refresh_command_observations(env)
        self.assertEqual(manager.compute()["actor"].data[:2], [9, 8])
        self.assertEqual(manager.compute()["actor"].data[6:9], [0.2, 0, 0.5])

    def test_delay_or_history_contract_change_fails(self):
        manager = Manager()
        manager.configs["velocity"].delay_max_lag = 1
        with self.assertRaises(ValueError):
            command_slices(manager)

    def test_missing_command_fails(self):
        manager = Manager()
        manager.configs["body"].params.clear()
        with self.assertRaises(ValueError):
            command_slices(manager)
