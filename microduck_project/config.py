from __future__ import annotations

from copy import deepcopy
import math

from .common import read_json, digest

CHANGE_PATH = "commands.twist.ranges.lin_vel_x"
REQUIRED = {"name", "condition", "seed", "num_envs", "total_iterations", "save_interval", "checkpoint_seconds", "initialization", "change"}


def load_experiment(path):
    return validate_experiment(read_json(path))


def validate_experiment(cfg):
    if set(cfg) - REQUIRED - {"hypothesis"} or REQUIRED - set(cfg):
        raise ValueError("実験設定のキーが不足、または不明なキーがあります")
    if cfg["condition"] not in {"baseline", "low_speed"}:
        raise ValueError("未知の比較条件")
    if cfg["initialization"] != "scratch":
        raise ValueError("この比較は両条件をscratchから学習します。再開は --resume を使用")
    for key in ("num_envs", "total_iterations", "save_interval", "checkpoint_seconds"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(cfg["seed"]) is not int or cfg["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    change = cfg["change"]
    if cfg["condition"] == "baseline":
        if change is not None:
            raise ValueError("baseline must keep the upstream environment unchanged")
    else:
        if not isinstance(change, dict) or set(change) != {"path", "expected", "value"}:
            raise ValueError("変更は一つの範囲指定だけが必要です")
        if change["path"] != CHANGE_PATH or change["expected"] != [-0.4, 0.4]:
            raise ValueError("採用上流に存在する前後速度範囲だけを変更できます")
        value = change["value"]
        if (not isinstance(value, list) or len(value) != 2
                or not all(type(x) in (int, float) and math.isfinite(x) for x in value)
                or not (-0.4 <= value[0] < 0 < value[1] <= 0.4)
                or value == change["expected"]):
            raise ValueError("変更範囲は上流内でゼロを含む、異なる有限の範囲にしてください")
    return deepcopy(cfg)


def apply_change(env_cfg, experiment):
    """Mutate only the verified leaf; unknown keys fail before training."""
    change = experiment["change"]
    if change is None:
        return []
    ranges = env_cfg.commands["twist"].ranges
    before = tuple(ranges.lin_vel_x)
    if before != tuple(change["expected"]):
        raise ValueError(f"Upstream range changed: {before}")
    ranges.lin_vel_x = tuple(change["value"])
    return [{"path": CHANGE_PATH, "before": list(before), "after": list(ranges.lin_vel_x)}]


def resume_key(cfg):
    return digest({k: v for k, v in cfg.items() if k not in {
        "name", "hypothesis", "total_iterations", "save_interval", "checkpoint_seconds"}})


def assert_fair_pair(a, b):
    validate_experiment(a)
    validate_experiment(b)
    if {a["condition"], b["condition"]} != {"baseline", "low_speed"}:
        raise ValueError("baselineとlow_speedを一つずつ指定してください")
    for key in ("seed", "num_envs", "total_iterations", "initialization", "save_interval", "checkpoint_seconds"):
        if a[key] != b[key]:
            raise ValueError(f"公平比較の設定が不一致: {key}")

