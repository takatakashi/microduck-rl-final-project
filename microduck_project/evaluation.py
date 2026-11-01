from __future__ import annotations

import csv
from dataclasses import asdict
import math

from .common import digest, jsonable, read_json, write_json, utc_now, source_identity
from .controller import Criteria, GoalController, Observation, State
from .metrics import TrackingMetrics, aggregate, command_at
from .runtime import load_for_inference, task_configs
from .video import VideoWriter, validate_video


def validate_scenario(scenario):
    if scenario.get("schema_version") != 1:
        raise ValueError("Unsupported evaluation schema")
    groups = [scenario[key] for key in ("selection_seeds", "final_seeds", "goal_seeds")]
    for seeds in groups:
        if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError("評価seedは重複しない非負整数です")
    if set(groups[0]) & set(groups[1]):
        raise ValueError("モデル選択用と最終評価用のseedを分けてください")
    for segment in scenario["segments"]:
        if not math.isfinite(segment["seconds"]) or segment["seconds"] <= 0:
            raise ValueError("指令区間の長さは有限の正値です")
        if len(segment["twist"]) != 3 or not all(math.isfinite(x) for x in segment["twist"]):
            raise ValueError("指令は有限の3次元速度です")
    if not scenario["segments"]:
        raise ValueError("指令列がありません")
    if len(scenario["goal"]) != 2 or not all(math.isfinite(x) for x in scenario["goal"]):
        raise ValueError("目標座標が不正です")
    Criteria(**scenario["goal_criteria"])
    return scenario


def evaluation_transform(scenario, mode):
    def transform(cfg):
        # Same physical randomization and initial-state distribution in both conditions.
        # Freeze at the initial DR ranges (no curriculum and no external velocity pushes).
        cfg.curriculum = {}
        cfg.events.pop("push_robot", None)
        for observation in cfg.observations.values():
            observation.enable_corruption = False
        cfg.episode_length_s = max(sum(s["seconds"] for s in scenario["segments"]), scenario["goal_criteria"]["timeout_seconds"]) + 2.0
        for command in cfg.commands.values():
            command.resampling_time_range = (1.0e9, 1.0e9)
            command.debug_vis = False
        twist = cfg.commands["twist"]
        twist.heading_command = False
        twist.ranges.heading = None
        for name in ("rel_standing_envs", "rel_heading_envs", "rel_world_envs", "rel_forward_envs", "rel_turn_in_place_envs"):
            if hasattr(twist, name):
                setattr(twist, name, 0.0)
        for key in ("head_pose", "body_pose"):
            cfg.commands[key].ranges = tuple((0.0, 0.0) for _ in cfg.commands[key].ranges)
        if mode == "goal":
            prior = cfg.scene.spec_fn
            def add_goal(spec):
                import mujoco
                if prior is not None:
                    prior(spec)
                # A visual site has no mass/contact and cannot move the robot.
                spec.worldbody.add_site(name="coursework_goal", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    pos=[*scenario["goal"], 0.003], size=[scenario["goal_criteria"]["radius_m"], 0.002, 0.002],
                    rgba=[0.1, 0.8, 0.3, 0.5])
            cfg.scene.spec_fn = add_goal
    return transform


def set_command(env, twist):
    import torch
    # Only command tensors are changed, never root/joint state or actuator targets.
    term = env.command_manager.get_term("twist")
    value = torch.tensor(twist, dtype=torch.float32, device=env.device)
    term.command[:] = value
    term.vel_command_w[:] = value
    for name in ("is_standing_env", "is_heading_env", "is_world_env", "is_forward_env"):
        getattr(term, name).fill_(False)
    for name in ("head_pose", "body_pose"):
        env.command_manager.get_command(name).zero_()
    from .observations import refresh_command_observations
    refresh_command_observations(env)


def observation(env):
    data = env.scene["robot"].data
    pos = data.root_link_pos_w[0].tolist()
    velocity = data.root_link_lin_vel_b[0].tolist()
    angular = data.root_link_ang_vel_b[0].tolist()
    world_velocity = data.root_link_lin_vel_w[0].tolist()
    gravity = data.projected_gravity_b[0].tolist()
    yaw = data.heading_w[0].item()
    tilt = math.acos(max(-1.0, min(1.0, -gravity[2]))) if math.isfinite(gravity[2]) else float("nan")
    # Planar world speed is used for stop criteria; body velocity for tracking.
    return Observation(pos[0], pos[1], yaw, math.hypot(*world_velocity[:2]), angular[2], tilt), (velocity[0], velocity[1], angular[2])


def classify_state(obs, criteria):
    if not all(math.isfinite(v) for k, v in asdict(obs).items() if k != "fallen"):
        return "non_finite_state"
    if obs.fallen or obs.tilt >= criteria.max_tilt_rad:
        return "fall"
    if obs.speed < 0 or obs.speed > criteria.max_speed_m_s or abs(obs.yaw_rate) > criteria.max_yaw_rate_rad_s:
        return "abnormal_velocity"
    return None


def evaluate(checkpoint, scenario_path, output, *, mode="walk", split="final", video=False, smoke=False):
    if output.exists():
        raise ValueError("評価出力先は新しいディレクトリを指定してください")
    scenario = validate_scenario(read_json(scenario_path))
    seeds = scenario["goal_seeds"] if mode == "goal" else scenario[f"{split}_seeds"]
    if smoke:
        seeds = seeds[:1]
    elif (mode == "walk" and split == "final" and len(seeds) < 10) or (mode == "goal" and len(seeds) < 3):
        raise ValueError("最終歩行評価は10試行以上、目標到達は3試行以上を指定してください")
    import torch
    env, wrapper, runner, policy, meta = load_for_inference(checkpoint, render=video,
        env_transform=evaluation_transform(scenario, mode))
    output.mkdir(parents=True)
    base_cfg, _, _, _ = task_configs(meta["experiment"])
    ranges = base_cfg.commands["twist"].ranges
    limits = (ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z)
    report = None
    # Physics steps can cache inference tensors. Keep resets, commands,
    # observations, rendering and cleanup in the same inference context.
    with torch.inference_mode():
        try:
            for segment in scenario["segments"]:
                if any(not lo <= cmd <= hi for cmd, (lo, hi) in zip(segment["twist"], limits)):
                    raise ValueError("評価指令が学習した範囲外です")
            criteria = Criteria(**scenario["goal_criteria"])
            report = {"schema_version": 1, "status": "running", "created_at": utc_now(), "mode": mode,
                      "split": "smoke" if smoke else split, "scenario": scenario, "scenario_sha256": digest(scenario),
                      "checkpoint": meta, "evaluation_source_sha256": source_identity()["source_sha256"],
                      "seeds": seeds, "episodes": [],
                      "evaluation_settings": {"noise": False, "pushes": False, "curriculum": False,
                          "randomization": "upstream startup/reset distribution at initial curriculum ranges",
                          "startup_seed": meta["experiment"]["seed"], "auto_reset": False,
                          "position_source": "simulator ground truth", "head_body_commands": "zero",
                          "goal_coordinate_frame": "fixed world XY, specified before the episode"}}
            write_json(output / "evaluation.json", report)
            for seed in seeds:
                env.reset(seed=seed)
                controller = GoalController(scenario["goal"], max_forward=limits[0][1],
                    max_yaw_rate=min(-limits[2][0], limits[2][1]), criteria=criteria)
                tracking = TrackingMetrics()
                duration = criteria.timeout_seconds if mode == "goal" else sum(s["seconds"] for s in scenario["segments"])
                if smoke:
                    duration = min(duration, 3.0)
                nsteps = math.ceil(duration / env.step_dt)
                movie = VideoWriter(output / f"seed_{seed}.mp4", fps=1.0 / env.step_dt) if video else None
                state = "walk"
                success, reason, held = False, "timeout", 0.0
                elapsed = 0.0
                initial, _ = observation(env)
                end = initial
                terminal_terms = []
                try:
                    if movie:
                        movie.append(env.render(), f"{meta['experiment']['condition']} | {mode} | seed {seed}\nEPISODE START / reset | 1x speed")
                    with (output / f"seed_{seed}.csv").open("w", newline="", encoding="utf-8") as stream:
                        fields = ["seed", "time", "state", "vx_command", "vy_command", "yaw_command", "vx", "vy", "yaw_rate", "x", "y", "yaw", "speed", "tilt", "distance", "held_seconds", "reason"]
                        writer = csv.DictWriter(stream, fieldnames=fields)
                        writer.writeheader()
                        for step in range(nsteps):
                            before, _ = observation(env)
                            failure = classify_state(before, criteria)
                            if failure:
                                reason = failure
                                break
                            if mode == "goal":
                                decision = controller.update(before, elapsed)
                                state, command, held = decision.state, decision.twist, decision.held_seconds
                                if state in (State.SUCCESS, State.FAILED):
                                    success, reason = state == State.SUCCESS, decision.reason or "arrived"
                                    break
                            else:
                                command = command_at(scenario["segments"], elapsed)
                            set_command(env, command)
                            obs = wrapper.get_observations()
                            if not all(torch.isfinite(t).all() for t in obs.values()):
                                reason = "non_finite_observation"
                                break
                            actions = policy(obs)
                            if not torch.isfinite(actions).all():
                                reason = "non_finite_action"
                                break
                            _, _, done, extras = wrapper.step(actions)
                            elapsed = (step + 1) * env.step_dt
                            end, measured = observation(env)
                            reason_now = classify_state(end, criteria)
                            if reason_now is None:
                                tracking.add(command, measured)
                            if done[0].item():
                                manager = env.termination_manager
                                terminal_terms = [name for name in manager.active_terms if manager.get_term(name)[0].item()]
                                reason_now = reason_now or "environment_termination"
                                if "nan_state" in terminal_terms:
                                    reason_now = "non_finite_state"
                            if mode == "goal" and reason_now is None:
                                decision = controller.update(end, elapsed)
                                state, held = decision.state, decision.held_seconds
                                if state == State.FAILED:
                                    reason_now = decision.reason
                                elif state == State.SUCCESS:
                                    success, reason = True, "arrived"
                            if mode == "walk" and step == nsteps - 1 and reason_now is None:
                                success, reason, state = True, "completed", "completed"
                            distance = math.hypot(end.x - scenario["goal"][0], end.y - scenario["goal"][1])
                            row = dict(seed=seed, time=elapsed, state=state,
                                vx_command=command[0], vy_command=command[1], yaw_command=command[2],
                                vx=measured[0], vy=measured[1], yaw_rate=measured[2], x=end.x, y=end.y,
                                yaw=end.yaw, speed=end.speed, tilt=end.tilt, distance=distance,
                                held_seconds=held, reason=reason_now or (reason if success else ""))
                            # CSV preserves numerical failure evidence; JSON summaries use null for undefined errors.
                            writer.writerow(row)
                            if movie:
                                frame = env.render()
                                if frame is None:
                                    raise RuntimeError("オフスクリーン画像が生成されません")
                                movie.append(frame, f"{meta['experiment']['condition']} | {mode} | seed {seed} | {elapsed:.2f}s | {state}\n{reason_now or ('SUCCESS' if success else 'running')} | 1x speed")
                            if reason_now:
                                success, reason = False, reason_now
                                break
                            if success:
                                break
                        else:
                            success, reason = (True, "completed") if mode == "walk" else (False, "timeout")
                finally:
                    if movie:
                        movie.close()
                arrival_error = math.hypot(end.x - scenario["goal"][0], end.y - scenario["goal"][1])
                episode = {"seed": seed, "success": success, "reason": reason, "terminal_terms": terminal_terms,
                           "duration_seconds": elapsed, "initial_state": jsonable(asdict(initial)),
                           "final_state": jsonable(asdict(end)), "held_seconds": held,
                           "arrival_error_m": arrival_error if mode == "goal" and math.isfinite(arrival_error) else None,
                           "arrival_time_seconds": elapsed if mode == "goal" and success else None, **tracking.result()}
                if movie:
                    episode["video"] = validate_video(movie.path)
                report["episodes"].append(episode)
                write_json(output / "evaluation.json", report)
            report["status"] = "completed"
            report["summary"] = aggregate(report["episodes"])
            write_json(output / "evaluation.json", report)
            return report
        except BaseException as exc:
            if report is not None:
                report.update(status="failed", error=str(exc))
                write_json(output / "evaluation.json", report)
            raise
        finally:
            env.close()
