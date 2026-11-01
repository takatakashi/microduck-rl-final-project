"""Thin adapter to the pinned official mjlab task and PPO runner."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from .common import jsonable, digest, sha256, read_json, upstream_lock
from .config import apply_change, resume_key, validate_experiment


def task_configs(experiment):
    import mjlab.tasks  # noqa: F401; discovers the official entry point
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    task = upstream_lock()["task"]
    env_cfg = load_env_cfg(task)  # Training config also used as the evaluation base.
    agent_cfg = load_rl_cfg(task)
    changes = apply_change(env_cfg, experiment)
    env_cfg.scene.num_envs = experiment["num_envs"]
    env_cfg.seed = experiment["seed"]
    agent_cfg.seed = experiment["seed"]
    agent_cfg.logger = "tensorboard"
    agent_cfg.upload_model = False
    agent_cfg.save_interval = experiment["save_interval"]
    agent_cfg.max_iterations = experiment["total_iterations"]
    agent_cfg.experiment_name = "coursework_" + experiment["condition"]
    agent_cfg.run_name = experiment["name"]
    runner_cls = load_runner_cls(task)
    if runner_cls is None:
        raise RuntimeError("公式Microduck runnerが登録されていません")
    if agent_cfg.num_steps_per_env != 24:
        raise RuntimeError("上流のrollout長が変わっています")
    return env_cfg, agent_cfg, runner_cls, changes


def configuration_contract(env_cfg, agent_cfg):
    agent = deepcopy(jsonable(agent_cfg))
    for key in ("max_iterations", "save_interval", "run_name", "resume", "load_run", "load_checkpoint"):
        agent.pop(key, None)
    return digest({"env": jsonable(env_cfg), "agent": agent})


def io_contract(env, agent_cfg):
    from mjlab.rl.exporter_utils import get_base_metadata
    actor_shape = list(env.observation_manager.group_obs_dim["actor"])
    action_count = env.action_manager.total_action_dim
    if actor_shape != [61] or action_count != 14:
        raise RuntimeError(f"公式歩行モデルの入出力が不一致: actor={actor_shape}, actions={action_count}")
    return {"actor_shape": actor_shape, "action_shape": [action_count], "dtype": "float32",
            "actor_terms": list(env.observation_manager.active_terms["actor"]),
            "step_dt": env.step_dt, "physics_dt": env.physics_dt,
            "normalization": agent_cfg.actor.obs_normalization,
            "metadata": jsonable(get_base_metadata(env, run_path=None)),
            "observation_source": "official observation_manager; no custom layout reconstruction",
            "command_frame": "body frame twist vx, vy, yaw rate; head/body commands zero during evaluation"}


def checkpoint_metadata(path):
    path = Path(path)
    meta = read_json(path.with_suffix(".json"))
    if meta.get("schema_version") != 1 or meta.get("checkpoint_sha256") != sha256(path):
        raise ValueError("チェックポイントのSHA-256またはsidecar schemaが不一致")
    if meta["upstream_commit"] != upstream_lock()["commit"]:
        raise ValueError("チェックポイントの上流SHAが不一致")
    validate_experiment(meta["experiment"])
    if meta["completed_iterations"] * meta["num_steps_per_env"] * meta["experiment"]["num_envs"] != meta["total_env_steps"]:
        raise ValueError("チェックポイントの環境ステップ数が不整合")
    return meta


def validate_resume(meta, experiment, contract, source_sha, lock_sha):
    if resume_key(meta["experiment"]) != resume_key(experiment):
        raise ValueError("条件、seed、並列数、初期化方法が異なるcheckpointからは再開できません")
    if meta["configuration_sha256"] != contract:
        raise ValueError("観測・行動・機体を含む実効設定がcheckpointと不一致")
    if meta["source_sha256"] != source_sha or meta["lock_sha256"] != lock_sha:
        raise ValueError("自作コードまたは依存lockが異なります。実験の互換性を確認してください")
    if experiment["total_iterations"] <= meta["completed_iterations"]:
        raise ValueError("total_iterationsは保存済みiteration数より大きくしてください（追加数ではなく累計）")


def load_for_inference(checkpoint, *, render=False, env_transform=None):
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from .environment import require_gpu
    require_gpu()
    meta = checkpoint_metadata(checkpoint)
    cfg, agent_cfg, runner_cls, _ = task_configs(meta["experiment"])
    if configuration_contract(cfg, agent_cfg) != meta["configuration_sha256"]:
        raise ValueError("checkpointと再構築した学習設定が一致しません")
    cfg.scene.num_envs = 1
    cfg.auto_reset = False
    if env_transform is not None:
        env_transform(cfg)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array" if render else None)
    try:
        wrapper = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = runner_cls(wrapper, asdict(agent_cfg), device="cuda:0")
        runner.load(str(checkpoint), map_location="cuda:0")
        policy = runner.get_inference_policy(device="cuda:0")
        if not torch.isfinite(env.scene["robot"].data.root_link_pos_w).all():
            raise RuntimeError("初期状態にNaN/Infがあります")
        return env, wrapper, runner, policy, meta
    except BaseException:
        env.close()
        raise
