from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean


@dataclass
class TrackingMetrics:
    samples: int = 0
    x_squared: float = 0.0
    y_squared: float = 0.0
    yaw_squared: float = 0.0
    stop_samples: int = 0
    stop_speed_squared: float = 0.0

    def add(self, command, measured):
        if len(command) != 3 or len(measured) != 3 or not all(math.isfinite(x) for x in (*command, *measured)):
            raise ValueError("追従指標には有限の3次元速度が必要です")
        self.samples += 1
        self.x_squared += (command[0] - measured[0]) ** 2
        self.y_squared += (command[1] - measured[1]) ** 2
        self.yaw_squared += (command[2] - measured[2]) ** 2
        if command == (0.0, 0.0, 0.0) or command == [0.0, 0.0, 0.0]:
            self.stop_samples += 1
            self.stop_speed_squared += measured[0] ** 2 + measured[1] ** 2

    def result(self):
        n = self.samples
        return {"samples": n, "vx_rmse_m_s": math.sqrt(self.x_squared / n) if n else None,
                "vy_rmse_m_s": math.sqrt(self.y_squared / n) if n else None,
                "xy_rmse_m_s": math.sqrt((self.x_squared + self.y_squared) / n) if n else None,
                "yaw_rate_rmse_rad_s": math.sqrt(self.yaw_squared / n) if n else None,
                "stop_speed_rms_m_s": math.sqrt(self.stop_speed_squared / self.stop_samples) if self.stop_samples else None}


def aggregate(episodes):
    if not episodes:
        raise ValueError("評価エピソードがありません")
    n = len(episodes)
    result = {"episodes": n, "successes": sum(x["success"] for x in episodes),
              "success_rate": sum(x["success"] for x in episodes) / n,
              "fall_rate": sum(x["reason"] == "fall" for x in episodes) / n,
              "mean_duration_seconds": mean(x["duration_seconds"] for x in episodes),
              "failures": {reason: sum(x["reason"] == reason for x in episodes)
                           for reason in sorted({x["reason"] for x in episodes if not x["success"]})}}
    for key in ("vx_rmse_m_s", "xy_rmse_m_s", "yaw_rate_rmse_rad_s", "stop_speed_rms_m_s", "arrival_error_m"):
        values = [x[key] for x in episodes if x.get(key) is not None]
        result["mean_" + key] = mean(values) if values else None
    result["metric_aggregation"] = "equal weight per episode; errors measured only until termination, report alongside survival/falls"
    return result


def command_at(segments, elapsed):
    if elapsed < 0 or not math.isfinite(elapsed):
        raise ValueError("elapsed must be finite and non-negative")
    end = 0.0
    for segment in segments:
        end += segment["seconds"]
        if elapsed < end - 1e-9:
            return tuple(segment["twist"])
    return (0.0, 0.0, 0.0)

