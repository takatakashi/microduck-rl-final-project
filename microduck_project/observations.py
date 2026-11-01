"""Synchronize command terms in mjlab's cached observation, preserving sensor history.

mjlab 1.3.0 compute(update_history=False) returns a cache. Recomputing every term
would append to delay buffers twice. Resolve command slices from the manager's
actual layout and evaluate only those official terms after changing a command.
"""
from __future__ import annotations

import math

COMMANDS = {"twist", "head_pose", "body_pose"}


def command_slices(manager):
    result = []
    for group, names in manager.active_terms.items():
        if not manager.group_obs_concatenate[group]:
            raise ValueError("採用した公式タスクは連結観測を前提とします")
        offset = 0
        found = set()
        for name, shape in zip(names, manager.group_obs_term_dim[group], strict=True):
            cfg = manager.get_term_cfg(group, name)
            size = math.prod(shape)
            command = cfg.params.get("command_name")
            if command in COMMANDS:
                if cfg.history_length or cfg.delay_max_lag or cfg.noise is not None:
                    raise ValueError("指令観測に履歴・遅延・ノイズが追加されています。互換性の再確認が必要です")
                result.append((group, name, offset, offset + size))
                found.add(command)
            offset += size
        if found != COMMANDS:
            raise ValueError(f"{group}に公式の3種の指令観測がそろっていません")
    return result


def refresh_command_observations(env):
    manager = env.observation_manager
    observations = manager.compute()  # Public cache access; does not advance history.
    for group, name, start, stop in command_slices(manager):
        cfg = manager.get_term_cfg(group, name)
        value = cfg.func(env, **cfg.params).clone()
        if cfg.clip is not None:
            value = value.clamp(*cfg.clip)
        if cfg.scale is not None:
            value = value * cfg.scale
        if tuple(value.shape) != (env.num_envs, stop - start):
            raise ValueError("公式指令観測のshapeが登録されたlayoutと不一致")
        observations[group][:, start:stop] = value

