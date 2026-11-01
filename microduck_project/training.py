from __future__ import annotations

import csv
import os
from pathlib import Path
import random
import time
from dataclasses import asdict

from .common import ROOT, sha256, source_identity, upstream_lock, utc_now, write_json
from .config import load_experiment, validate_experiment
from .environment import inventory, require_gpu
from .runtime import checkpoint_metadata, configuration_contract, io_contract, task_configs, validate_resume


def train(config, output, *, resume=None, total_iterations=None, num_envs=None):
    experiment = load_experiment(config)
    if total_iterations is not None:
        experiment["total_iterations"] = total_iterations
    if num_envs is not None:
        experiment["num_envs"] = num_envs
    validate_experiment(experiment)
    if os.environ.get("MICRODUCK_WARM_START"):
        raise ValueError("MICRODUCK_WARM_STARTはscratch比較・通常再開では使用しません")
    if output.exists():
        raise ValueError("出力先が既に存在します。再開も別の出力ディレクトリを指定してください")
    gpu = require_gpu()
    import numpy as np
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from mjlab.utils.os import dump_yaml
    configure_torch_backends()
    cfg, agent_cfg, official_runner, changes = task_configs(experiment)
    contract = configuration_contract(cfg, agent_cfg)
    source = source_identity()
    previous = checkpoint_metadata(resume) if resume else None
    if previous:
        validate_resume(previous, experiment, contract, source["source_sha256"], gpu["lock_sha256"])
    output.mkdir(parents=True)
    started = time.perf_counter()
    record = {"schema_version": 1, "status": "running", "started_at": utc_now(),
              "experiment": experiment, "changes": changes, "upstream": upstream_lock(),
              "source": source, "environment": inventory(), "gpu": gpu,
              "configuration_sha256": contract, "initialization": "resume" if resume else "scratch",
              "parent_checkpoint_sha256": previous["checkpoint_sha256"] if previous else None,
              "restored_completed_iterations": previous["completed_iterations"] if previous else 0,
              "limitations": ["Simulator state, actuator history and per-environment RNG state are not restored; bitwise trajectory identity is not promised."]}
    write_json(output / "run.json", record)
    write_json(output / "experiment.json", experiment)
    write_json(output / "source.json", source)
    dump_yaml(output / "params/env.yaml", asdict(cfg))
    dump_yaml(output / "params/agent.yaml", asdict(agent_cfg))
    env = None
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
        wrapper = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        record["environment_build_seconds_including_compilation"] = time.perf_counter() - started
        record["io_contract"] = io_contract(env, agent_cfg)
        write_json(output / "run.json", record)

        class RecordedRunner(official_runner):
            def save(self, path, infos=None):
                path = Path(path)
                steps = self.env.unwrapped.common_step_counter
                nsteps = self.cfg["num_steps_per_env"]
                if steps % nsteps:
                    raise RuntimeError("更新境界以外ではcheckpointを保存しません")
                completed = steps // nsteps
                rng = {"python": random.getstate(), "numpy": np.random.get_state(),
                       "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}
                infos = {**(infos or {}), "coursework_rng": rng,
                         "coursework_completed_iterations": completed}
                temporary = path.with_name(path.stem + ".partial.pt")
                super().save(str(temporary), infos=infos)
                os.replace(temporary, path)
                wall = time.perf_counter() - started
                sidecar = {"schema_version": 1, "checkpoint_sha256": sha256(path),
                           "upstream_commit": upstream_lock()["commit"],
                           "source_sha256": source["source_sha256"], "lock_sha256": gpu["lock_sha256"],
                           "configuration_sha256": contract, "io_contract": record["io_contract"],
                           "experiment": experiment, "completed_iterations": completed,
                           "num_steps_per_env": nsteps, "total_env_steps": steps * env.num_envs,
                           "wall_seconds_this_process": wall,
                           "wall_seconds_total": wall + (previous["wall_seconds_total"] if previous else 0),
                           "initialization": "scratch", "parent_checkpoint_sha256": record["parent_checkpoint_sha256"]}
                write_json(path.with_suffix(".json"), sidecar)
                write_json(output / "latest.json", {"checkpoint": path.name, **sidecar})

        runner = RecordedRunner(wrapper, asdict(agent_cfg), str(output), "cuda:0")
        completed = 0
        if previous:
            infos = runner.load(str(resume), map_location="cuda:0")
            completed = previous["completed_iterations"]
            if env.common_step_counter != completed * agent_cfg.num_steps_per_env:
                raise RuntimeError("common_step_counterが保存値から復元されていません")
            if infos.get("coursework_completed_iterations") != completed:
                raise RuntimeError("checkpoint内部の完了iteration数とsidecarが不一致")
            # rsl-rl 5.0.1 stores the last loop index, not the next iteration index.
            runner.current_learning_iteration = completed
            rng = infos["coursework_rng"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu())
            torch.cuda.set_rng_state_all([x.cpu() for x in rng["cuda"]])
            record["resume_verified_before_learning"] = True
            write_json(output / "run.json", record)

        with (output / "iterations.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ["completed_iterations", "total_env_steps", "collect_seconds", "learn_seconds", "wall_seconds", "learning_rate", "losses"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            original_log = runner.logger.log
            last_save = time.perf_counter()
            measured_iteration_times = []

            def logged(**kwargs):
                nonlocal last_save
                original_log(**kwargs)
                wall = time.perf_counter() - started
                iteration = kwargs["it"] + 1
                import json
                losses = {k: float(v) for k, v in kwargs["loss_dict"].items()}
                writer.writerow({"completed_iterations": iteration,
                                 "total_env_steps": env.common_step_counter * env.num_envs,
                                 "collect_seconds": kwargs["collect_time"], "learn_seconds": kwargs["learn_time"],
                                 "wall_seconds": wall, "learning_rate": kwargs["learning_rate"],
                                 "losses": json.dumps(losses, allow_nan=False)})
                stream.flush()
                elapsed = kwargs["collect_time"] + kwargs["learn_time"]
                measured_iteration_times.append(elapsed)
                if len(measured_iteration_times) == 1:
                    record["first_iteration_seconds_including_compilation"] = elapsed
                if time.perf_counter() - last_save >= experiment["checkpoint_seconds"]:
                    runner.save(str(output / f"timed_{iteration}.pt"))
                    last_save = time.perf_counter()

            runner.logger.log = logged
            runner.learn(num_learning_iterations=experiment["total_iterations"] - completed,
                         init_at_random_ep_len=previous is None)
            # Stable name for downstream scripts; state saved after actual PPO updates.
            runner.save(str(output / "final.pt"))
            final = checkpoint_metadata(output / "final.pt")
            if final["completed_iterations"] != experiment["total_iterations"]:
                raise RuntimeError("要求した累計iteration数と実際の保存状態が一致しません")
            record.update(status="completed", final_checkpoint="final.pt", **{
                "completed_iterations": final["completed_iterations"],
                "total_env_steps": final["total_env_steps"],
                "checkpoint_sha256": final["checkpoint_sha256"]})
            warm = measured_iteration_times[1:]
            if warm:
                mean = sum(warm) / len(warm)
                record["warm_iteration_mean_seconds"] = mean
                record["recommended_save_interval_for_300_seconds"] = max(1, int(300 / mean))
    except BaseException as exc:
        record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        record["finished_at"] = utc_now()
        record["wall_seconds_this_process"] = time.perf_counter() - started
        write_json(output / "run.json", record)
        if env is not None:
            env.close()
        with (ROOT / "docs/progress.md").open("a", encoding="utf-8") as progress:
            progress.write(f"\n- {record['finished_at']} 学習 `{experiment['name']}`: {record['status']}。詳細はローカル outputs 内のrun.json。\n")
    return record
