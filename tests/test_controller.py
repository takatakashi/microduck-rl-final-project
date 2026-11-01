import math
import unittest

from microduck_project.controller import Criteria, GoalController, Observation, State


def obs(x=0, y=0, yaw=0, speed=0, yaw_rate=0, tilt=0, fallen=False):
    return Observation(x, y, yaw, speed, yaw_rate, tilt, fallen)


class GoalTests(unittest.TestCase):
    def controller(self, goal=(1, 0)):
        return GoalController(goal, max_forward=0.2, max_yaw_rate=1.0)

    def test_turn_before_translation(self):
        decision = self.controller((0, 1)).update(obs(), 0)
        self.assertEqual(decision.state, State.ALIGN)
        self.assertEqual(decision.twist[:2], (0, 0))
        self.assertEqual(decision.twist[2], 1.0)

    def test_approach_is_clamped_to_training_range(self):
        self.assertEqual(self.controller().update(obs(), 0).twist, (0.2, 0, 0))

    def test_two_seconds_stop_required(self):
        c = self.controller()
        self.assertEqual(c.update(obs(x=0.9), 0).state, State.HOLD)
        self.assertEqual(c.update(obs(x=0.9), 1.99).state, State.HOLD)
        self.assertEqual(c.update(obs(x=0.9), 2).state, State.SUCCESS)

    def test_pass_through_not_arrival(self):
        c = self.controller()
        c.update(obs(x=0.9, speed=0.3), 0)
        self.assertEqual(c.update(obs(x=0.9, speed=0.3), 3).state, State.HOLD)

    def test_spinning_not_stopped(self):
        c = self.controller()
        c.update(obs(x=1, yaw_rate=0.5), 0)
        self.assertEqual(c.update(obs(x=1, yaw_rate=0.5), 3).state, State.HOLD)

    def test_hold_resets_when_exiting_radius(self):
        c = self.controller()
        c.update(obs(x=1), 0)
        c.update(obs(x=0.7), 1.9)
        self.assertEqual(c.update(obs(x=1), 2.1).held_seconds, 0)
        self.assertEqual(c.update(obs(x=1), 4.0).state, State.HOLD)

    def test_hold_resets_when_speed_increases(self):
        c = self.controller()
        c.update(obs(x=1), 0)
        c.update(obs(x=1, speed=0.2), 1.5)
        c.update(obs(x=1), 2)
        self.assertEqual(c.update(obs(x=1), 3).state, State.HOLD)

    def test_failures_zero_commands(self):
        for measurement, expected in ((obs(tilt=1.1), "fall"), (obs(fallen=True), "fall"),
                                      (obs(speed=5), "abnormal_velocity"), (obs(yaw=float("nan")), "non_finite_state")):
            with self.subTest(expected=expected):
                decision = self.controller().update(measurement, 0)
                self.assertEqual(decision.reason, expected)
                self.assertEqual(decision.state, State.FAILED)
                self.assertEqual(decision.twist, (0, 0, 0))

    def test_fall_takes_precedence_over_arrival(self):
        c = self.controller()
        c.update(obs(x=1), 0)
        self.assertEqual(c.update(obs(x=1, tilt=2), 2).reason, "fall")

    def test_timeout(self):
        self.assertEqual(self.controller().update(obs(), 30).reason, "timeout")

    def test_late_arrival_cannot_override_timeout(self):
        c = self.controller()
        c.update(obs(x=1), 29)
        self.assertEqual(c.update(obs(x=1), 31).reason, "timeout")

    def test_angle_wrap(self):
        c = self.controller((-1, -0.01))
        decision = c.update(obs(yaw=math.pi - 0.01), 0)
        self.assertGreater(decision.twist[0], 0)
        self.assertLess(abs(decision.twist[2]), 0.1)

    def test_failure_latches(self):
        c = self.controller()
        c.update(obs(fallen=True), 0)
        self.assertEqual(c.update(obs(x=1), 3).state, State.FAILED)

    def test_hysteresis(self):
        c = self.controller()
        c.update(obs(yaw=0.1), 0)
        self.assertEqual(c.update(obs(yaw=0.3), 1).state, State.APPROACH)
        self.assertEqual(c.update(obs(yaw=0.6), 2).state, State.ALIGN)

    def test_invalid_config_and_time(self):
        with self.assertRaises(ValueError):
            Criteria(radius_m=-0.1)
        with self.assertRaises(ValueError):
            self.controller((float("nan"), 0))
        c = self.controller()
        c.update(obs(), 1)
        with self.assertRaises(ValueError):
            c.update(obs(), 0.5)
