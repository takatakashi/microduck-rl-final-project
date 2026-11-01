"""Rule-based navigation. No policy learning and no simulator state writes."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math


class State(StrEnum):
    ALIGN = "align"
    APPROACH = "approach"
    HOLD = "hold"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass(frozen=True)
class Observation:
    x: float
    y: float
    yaw: float
    speed: float
    yaw_rate: float
    tilt: float
    fallen: bool = False


@dataclass(frozen=True)
class Criteria:
    radius_m: float = 0.15
    hold_seconds: float = 2.0
    stop_speed_m_s: float = 0.05
    stop_yaw_rate_rad_s: float = 0.15
    timeout_seconds: float = 30.0
    max_tilt_rad: float = 1.0
    max_speed_m_s: float = 3.0
    max_yaw_rate_rad_s: float = 12.0
    align_tolerance_rad: float = 0.2
    realign_threshold_rad: float = 0.5
    distance_gain: float = 0.8
    heading_gain: float = 1.5

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in vars(self).values()):
            raise ValueError("制御基準は有限の正値が必要です")
        if self.realign_threshold_rad <= self.align_tolerance_rad:
            raise ValueError("再整列の閾値は整列完了の閾値より大きくしてください")


@dataclass(frozen=True)
class Decision:
    state: State
    twist: tuple[float, float, float]
    distance: float | None
    held_seconds: float
    reason: str | None


class GoalController:
    def __init__(self, goal, *, max_forward, max_yaw_rate, criteria=None):
        self.goal = tuple(goal)
        if len(self.goal) != 2 or not all(math.isfinite(x) for x in self.goal):
            raise ValueError("目標は有限の2次元座標です")
        if not all(math.isfinite(x) and x > 0 for x in (max_forward, max_yaw_rate)):
            raise ValueError("指令上限は有限の正値です")
        self.max_forward, self.max_yaw_rate = max_forward, max_yaw_rate
        self.criteria = criteria or Criteria()
        self.state, self.reason = State.ALIGN, None
        self.last_time = None
        self.hold_since = None
        self.held_seconds = 0.0

    def update(self, obs, elapsed):
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("時刻は有限の非負値です")
        if self.last_time is not None and elapsed < self.last_time:
            raise ValueError("時刻を巻き戻す場合はcontrollerを再作成してください")
        self.last_time = elapsed
        finite = all(math.isfinite(v) for k, v in vars(obs).items() if k != "fallen")
        distance = math.hypot(self.goal[0] - obs.x, self.goal[1] - obs.y) if finite else None
        c = self.criteria
        if self.state in (State.SUCCESS, State.FAILED):
            return Decision(self.state, (0.0, 0.0, 0.0), distance, self.held_seconds, self.reason)
        failure = None
        if not finite:
            failure = "non_finite_state"
        elif obs.fallen or abs(obs.tilt) >= c.max_tilt_rad:
            failure = "fall"
        elif obs.speed < 0 or obs.speed > c.max_speed_m_s or abs(obs.yaw_rate) > c.max_yaw_rate_rad_s:
            failure = "abnormal_velocity"
        elif elapsed > c.timeout_seconds:
            failure = "timeout"
        if failure:
            self.state, self.reason = State.FAILED, failure
            return Decision(self.state, (0.0, 0.0, 0.0), distance, self.held_seconds, self.reason)
        stopped = obs.speed <= c.stop_speed_m_s and abs(obs.yaw_rate) <= c.stop_yaw_rate_rad_s
        if distance <= c.radius_m:
            self.state = State.HOLD
            if stopped:
                if self.hold_since is None:
                    self.hold_since = elapsed
                self.held_seconds = elapsed - self.hold_since
                if self.held_seconds + 1e-9 >= c.hold_seconds:
                    self.state = State.SUCCESS
            else:
                self.hold_since, self.held_seconds = None, 0.0
            twist = (0.0, 0.0, 0.0)
        else:
            self.hold_since, self.held_seconds = None, 0.0
            heading = math.atan2(self.goal[1] - obs.y, self.goal[0] - obs.x) - obs.yaw
            heading = math.atan2(math.sin(heading), math.cos(heading))
            if self.state == State.APPROACH and abs(heading) <= c.realign_threshold_rad:
                self.state = State.APPROACH
            else:
                self.state = State.APPROACH if abs(heading) <= c.align_tolerance_rad else State.ALIGN
            angular = max(-self.max_yaw_rate, min(self.max_yaw_rate, c.heading_gain * heading))
            forward = min(self.max_forward, c.distance_gain * distance) if self.state == State.APPROACH else 0.0
            twist = (forward, 0.0, angular)
        if elapsed >= c.timeout_seconds and self.state != State.SUCCESS:
            self.state, self.reason, twist = State.FAILED, "timeout", (0.0, 0.0, 0.0)
        return Decision(self.state, twist, distance, self.held_seconds, self.reason)
