"""Goal experiment switching two self-trained policies; no training or state teleport.

Run from project root: python3 docs/diagnostics/hybrid_goal.py
The original single-policy evaluation remains unchanged. This experiment records
its own script hash, checkpoints and unchanged arrival/stop criteria.
"""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import argparse
import copy
import csv
import json
import math
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class HybridDecision:
    policy: str
    state: str
    twist: tuple
    distance: float | None
    held_seconds: float
    success: bool = False
    reason: str | None = None


class SwitchingController:
    def __init__(self, goal, *, max_forward, max_yaw_rate, criteria=None,
                 switch_radius=0.12, resume_radius=0.18):
        # The system Python can be 3.10. Import StrEnum-dependent code only
        # after main() has dispatched into the pinned Python 3.12 environment.
        from microduck_project.controller import Criteria, GoalController
        self.criteria = criteria or Criteria()
        if not 0 < switch_radius < self.criteria.radius_m < resume_radius < math.inf:
            raise ValueError('切替半径 < 成功半径 < 再移動半径を指定してください')
        self.goal = tuple(goal)
        self.walk = GoalController(goal, max_forward=max_forward, max_yaw_rate=max_yaw_rate,
                                   criteria=replace(self.criteria, radius_m=switch_radius))
        self.switch_radius, self.resume_radius = switch_radius, resume_radius
        self.policy = 'walk'
        self.hold_since = None
        self.terminal = None

    def update(self, obs, elapsed):
        from microduck_project.controller import State
        nav = self.walk.update(obs, elapsed)
        if self.terminal is not None:
            return self.terminal
        distance = nav.distance
        held = 0.0
        if nav.state == State.FAILED and nav.reason != 'timeout':
            self.terminal = HybridDecision(self.policy, 'failed', (0.,0.,0.), distance, held,
                                           reason=nav.reason)
            return self.terminal
        if distance <= self.switch_radius:
            self.policy = 'stand'
        elif distance > self.resume_radius:
            self.policy = 'walk'
        c = self.criteria
        stopped = obs.speed <= c.stop_speed_m_s and abs(obs.yaw_rate) <= c.stop_yaw_rate_rad_s
        if distance <= c.radius_m and stopped:
            if self.hold_since is None:
                self.hold_since = elapsed
            held = elapsed - self.hold_since
        else:
            self.hold_since = None
        if held + 1e-9 >= c.hold_seconds and elapsed <= c.timeout_seconds:
            self.terminal = HybridDecision(self.policy, 'success', (0.,0.,0.), distance, held,
                                           success=True, reason='arrived')
        elif elapsed >= c.timeout_seconds:
            self.terminal = HybridDecision(self.policy, 'failed', (0.,0.,0.), distance, held,
                                           reason='timeout')
        if self.terminal is not None:
            return self.terminal
        if self.policy == 'stand':
            return HybridDecision('stand', 'hold' if stopped else 'settle', (0.,0.,0.), distance, held)
        return HybridDecision('walk', str(nav.state), nav.twist, distance, held)


def validate_pair(walk, stand):
    from microduck_project.config import resume_key
    for key in ('upstream_commit', 'lock_sha256', 'source_sha256', 'configuration_sha256', 'io_contract'):
        if walk[key] != stand[key]:
            raise ValueError(f'切替モデルの互換性が不一致: {key}')
    if resume_key(walk['experiment']) != resume_key(stand['experiment']):
        raise ValueError('同一の標準学習から得たモデルを指定してください')


def select_goal_seeds(scenario, validation, *, fast_validation=False):
    if validation and fast_validation:
        raise ValueError('元の方式と速度変更版の検証seedは別に指定してください')
    if not validation and not fast_validation:
        return list(scenario['goal_seeds'])
    seeds = list(range(500,510)) if fast_validation else list(range(400,410))
    used = set(scenario['goal_seeds']) | set(scenario['selection_seeds']) | set(scenario['final_seeds'])
    if used.intersection(seeds):
        raise ValueError('追加検証seedが既存の評価seedと重複しています')
    return seeds


def experiment_scenario(scenario, *, validation=False, fast_approach=False, fast_validation=False):
    if sum((validation,fast_approach,fast_validation)) > 1:
        raise ValueError('速度変更の開発実験と元の方式の最終検証は分けてください')
    result = copy.deepcopy(scenario)
    result['goal_seeds'] = select_goal_seeds(scenario, validation, fast_validation=fast_validation)
    if fast_approach or fast_validation:
        if result['goal_criteria']['distance_gain'] != 0.8:
            raise ValueError('速度比較の基準distance_gainは0.8です')
        result['goal_criteria']['distance_gain'] = 1.2
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--walk-checkpoint', type=Path, default=ROOT/'outputs/baseline/model_1000.pt')
    parser.add_argument('--stand-checkpoint', type=Path, default=ROOT/'outputs/baseline/final.pt')
    parser.add_argument('--output', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--validation', action='store_true', help='元の方式をseed 400〜409の10試行で検証')
    mode.add_argument('--fast-approach', action='store_true', help='減速を遅らせる開発実験: distance_gain 0.8→1.2、seed 300〜302')
    mode.add_argument('--fast-validation', action='store_true', help='速度変更版を固定し、未使用seed 500〜509の10試行で検証')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    fast = args.fast_approach or args.fast_validation
    if args.output is None:
        name = ('hybrid_goal_fast_validation' if args.fast_validation else
                ('hybrid_goal_fast_pilot' if args.fast_approach else
                 ('hybrid_goal_validation' if args.validation else 'hybrid_goal_pilot')))
        args.output = ROOT/'outputs'/name
    if args.output.exists():
        raise ValueError('出力先は新しいディレクトリを指定してください')
    if not args.worker:
        from microduck_project.common import verify_upstream
        python = verify_upstream()/'.venv/bin/python'
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        for key,value in {'MUJOCO_GL':'egl','CUDA_VISIBLE_DEVICES':'0','WANDB_MODE':'disabled',
                          'XDG_CACHE_HOME':str(ROOT/'.cache')}.items():
            env.setdefault(key,value)
        os.execve(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:], '--worker'], env)

    import torch
    from microduck_project.controller import Criteria
    from microduck_project.common import read_json, write_json, digest, sha256, source_identity
    from microduck_project.runtime import checkpoint_metadata, load_for_inference, task_configs
    from microduck_project.evaluation import evaluation_transform, set_command, observation, classify_state
    from microduck_project.metrics import TrackingMetrics, aggregate
    from microduck_project.video import VideoWriter, validate_video
    scenario = read_json(ROOT/'configs/evaluation.json')
    original_scenario_sha = digest(scenario)
    development_seeds = list(scenario['goal_seeds'])
    scenario = experiment_scenario(scenario,validation=args.validation,fast_approach=args.fast_approach,
                                   fast_validation=args.fast_validation)
    walk_meta, stand_meta = checkpoint_metadata(args.walk_checkpoint), checkpoint_metadata(args.stand_checkpoint)
    validate_pair(walk_meta, stand_meta)
    args.output.mkdir(parents=True)
    report = {'status':'running', 'mode':'hybrid_goal',
              'split':('held_out_speed_validation' if args.fast_validation else
                       ('development_speed_comparison' if args.fast_approach else
                        ('held_out_validation' if args.validation else 'development_after_single_policy_final'))),
              'controller_variant':'fast_approach' if fast else 'original',
              'controller_change':({'path':'goal_criteria.distance_gain','before':0.8,'after':1.2} if fast else None),
              'seeds':scenario['goal_seeds'], 'scenario':scenario, 'scenario_sha256':digest(scenario),
              'original_scenario_sha256':original_scenario_sha,'development_seeds':development_seeds,
              'walk_checkpoint':walk_meta,'stand_checkpoint':stand_meta,'training_executed':False,
              'experiment_script_sha256':sha256(Path(__file__)), 'evaluation_source':source_identity(),
              'switch_radius_m':0.12,'resume_walk_radius_m':0.18,
              'limitations':[('Fixed fast approach evaluated on predeclared seeds 500-509; only their first evaluation is held out. Do not compare mean duration against other seeds as a paired speed test.' if args.fast_validation else
                              ('Speed experiment uses development seeds 300-302; held-out performance is not established.' if args.fast_approach else
                              ('Fixed controller evaluated on predeclared seeds 400-409; only their first evaluation is held out.' if args.validation else
                               'Same three goal seeds reused for development after inspecting the single-policy failures.'))),
                             'Policy switching is a rule, not a newly learned controller. Both checkpoints come from one training seed.',
                             'No interpolation of policy outputs; transition stability must be verified in simulation.'],
              'episodes':[]}
    write_json(args.output/'evaluation.json',report)
    env = None
    try:
        env, wrapper, runner, policy, _ = load_for_inference(args.walk_checkpoint, render=True,
            env_transform=evaluation_transform(scenario,'goal'))
        walk_policy = copy.deepcopy(policy)
        runner.load(str(args.stand_checkpoint), map_location='cuda:0')
        stand_policy = runner.get_inference_policy(device='cuda:0')
        if any(p.training or p.obs_normalizer.training for p in (walk_policy,stand_policy)):
            raise RuntimeError('policyと正規化器がevalモードではありません')
        cfg,_,_,_ = task_configs(walk_meta['experiment'])
        limits = cfg.commands['twist'].ranges
        criteria = Criteria(**scenario['goal_criteria'])
        report['command_limits'] = {'forward_max_m_s':limits.lin_vel_x[1],
                                    'yaw_rate_abs_max_rad_s':min(-limits.ang_vel_z[0],limits.ang_vel_z[1])}
        label = 'hybrid fast approach' if fast else 'hybrid goal'
        with torch.inference_mode():
            for seed in scenario['goal_seeds']:
                env.reset(seed=seed)
                controller = SwitchingController(scenario['goal'], max_forward=limits.lin_vel_x[1],
                    max_yaw_rate=min(-limits.ang_vel_z[0],limits.ang_vel_z[1]), criteria=criteria)
                initial,_ = observation(env)
                decision = controller.update(initial,0.)
                end,elapsed,reason,success = initial,0.,'timeout',False
                switches = []
                tracking = TrackingMetrics()
                movie = VideoWriter(args.output/f'seed_{seed}.mp4', fps=1/env.step_dt)
                try:
                    movie.append(env.render(),f'{label} | seed {seed} | reset | 1x speed')
                    with (args.output/f'seed_{seed}.csv').open('w',newline='') as stream:
                        fields=['time','policy_used','next_policy','state','vx_command','vy_command','yaw_command',
                                'vx','vy','yaw_rate','x','y','yaw','speed','tilt','distance','held_seconds','reason']
                        writer = csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                        for step in range(math.ceil(criteria.timeout_seconds/env.step_dt)):
                            if decision.reason:
                                success,reason=decision.success,decision.reason
                                break
                            used=decision.policy
                            command=decision.twist
                            set_command(env,command)
                            obs=wrapper.get_observations()
                            if not all(torch.isfinite(v).all() for v in obs.values()):
                                reason='non_finite_observation';break
                            actions=(walk_policy if used=='walk' else stand_policy)(obs)
                            if not torch.isfinite(actions).all():
                                reason='non_finite_action';break
                            _,_,done,_=wrapper.step(actions)
                            elapsed=(step+1)*env.step_dt
                            end,velocity=observation(env)
                            failure=classify_state(end,criteria)
                            if failure is None and done[0].item():
                                failure='environment_termination'
                            if failure is None:
                                tracking.add(command,velocity)
                            decision=controller.update(end,elapsed)
                            if decision.policy!=used:
                                switches.append({'time':elapsed,'from':used,'to':decision.policy,'distance':decision.distance})
                            reason_now=failure or decision.reason
                            writer.writerow(dict(time=elapsed,policy_used=used,next_policy=decision.policy,state=decision.state,
                                vx_command=command[0],vy_command=command[1],yaw_command=command[2],vx=velocity[0],vy=velocity[1],
                                yaw_rate=velocity[2],x=end.x,y=end.y,yaw=end.yaw,speed=end.speed,tilt=end.tilt,
                                distance=decision.distance,held_seconds=decision.held_seconds,reason=reason_now or ''))
                            movie.append(env.render(),f'{label} | seed {seed} | {elapsed:.2f}s | used {used}\n{decision.state} | {reason_now or "running"} | 1x speed')
                            if reason_now:
                                reason=reason_now;success=decision.success and failure is None;break
                finally:
                    movie.close()
                report['episodes'].append({'seed':seed,'success':success,'reason':reason,'duration_seconds':elapsed,
                    'initial_state':asdict(initial),'final_state':asdict(end),'held_seconds':decision.held_seconds,
                    'arrival_error_m':decision.distance,'arrival_time_seconds':elapsed if success else None,
                    'switches':switches,'video':validate_video(movie.path),**tracking.result()})
                write_json(args.output/'evaluation.json',report)
        report.update(status='completed',summary=aggregate(report['episodes']))
    except BaseException as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        if env is not None:
            with torch.inference_mode():
                env.close()
        write_json(args.output/'evaluation.json',report)
    print(json.dumps({'status':report['status'],'summary':report['summary']},indent=2))


if __name__=='__main__':
    main()
