"""Read-only checkpoint diagnosis; deterministic and stochastic rollouts, no training.

Kept outside the training source hash so existing experiments remain resumable.
Use: python3 docs/diagnostics/probe_policy.py
"""
from pathlib import Path
import argparse
import csv
import inspect
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def policy_contract(policy):
    groups = policy.obs_groups
    return {'class': type(policy).__qualname__,
            'is_recurrent': bool(policy.is_recurrent),
            'obs_groups_type': type(groups).__name__,
            'obs_groups': list(groups) if isinstance(groups, (list, tuple)) else repr(groups),
            'compatible': not policy.is_recurrent and isinstance(groups, (list, tuple))
                          and tuple(groups) == ('actor',)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'outputs/baseline/final.pt')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/baseline_policy_probe')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('出力先は新しいディレクトリを指定してください')
    if not args.worker:
        from microduck_project.common import verify_upstream
        python = verify_upstream() / '.venv/bin/python'
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        for key, value in {'MUJOCO_GL': 'egl', 'CUDA_VISIBLE_DEVICES': '0',
                           'WANDB_MODE': 'disabled', 'XDG_CACHE_HOME': str(ROOT / '.cache')}.items():
            env.setdefault(key, value)
        os.execve(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:], '--worker'], env)

    import torch
    from microduck_project.common import read_json, write_json, sha256
    from microduck_project.runtime import load_for_inference
    from microduck_project.evaluation import evaluation_transform, set_command, observation, classify_state
    from microduck_project.observations import command_slices
    from microduck_project.controller import Criteria
    from microduck_project.metrics import command_at

    scenario = read_json(ROOT / 'configs/evaluation.json')
    args.output.mkdir(parents=True)
    report = {'status': 'running', 'purpose': 'diagnosis only, not final evaluation',
              'training_executed': False, 'seed': 100, 'scenario': scenario,
              'probe_script_sha256': sha256(Path(__file__)), 'rollouts': []}
    write_json(args.output / 'probe.json', report)
    try:
        for stochastic in (False, True):
            mode = 'stochastic' if stochastic else 'deterministic'
            env, wrapper, runner, policy, meta = load_for_inference(args.checkpoint,
                env_transform=evaluation_transform(scenario, 'walk'))
            with torch.inference_mode():
                try:
                    report['checkpoint_sha256'] = meta['checkpoint_sha256']
                    contract = policy_contract(policy)
                    report['policy_contract'] = contract
                    write_json(args.output / 'probe.json', report)
                    if policy.training or policy.obs_normalizer.training:
                        raise RuntimeError('診断時にpolicy/normalizerがevalモードではありません')
                    if not contract['compatible']:
                        raise RuntimeError(f'想定したMLP actorと異なります: {contract}')
                    env.reset(seed=100)
                    manager = env.observation_manager
                    slices = command_slices(manager)
                    _, name, start, stop = next(x for x in slices if x[0] == 'actor'
                        and manager.get_term_cfg(x[0], x[1]).params.get('command_name') == 'twist')
                    cfg = manager.get_term_cfg('actor', name)
                    if cfg.scale is not None or cfg.clip is not None or stop-start != 3:
                        raise RuntimeError('指令観測のscale/clip/shapeが想定外です')
                    normalizer_before = {k: v.clone() for k,v in policy.obs_normalizer.state_dict().items()}
                    report['policy'] = {'class': type(policy).__qualname__, 'repr': str(policy),
                        'actor_twist_slice': [start, stop], 'normalizer': {
                            k: v.cpu().tolist() for k,v in normalizer_before.items()}}
                    if not stochastic:
                        objects = {'command_class': type(env.command_manager.get_term('twist')),
                                   'observation_compute': type(manager).compute,
                                   'policy_forward': type(policy).forward}
                        with (args.output / 'runtime_sources.txt').open('w') as stream:
                            for label, obj in objects.items():
                                stream.write('\n# ' + label + '\n')
                                try:
                                    stream.write(inspect.getsource(obj))
                                except (OSError, TypeError) as exc:
                                    stream.write('Source unavailable: ' + str(exc))
                    initial, _ = observation(env)
                    record = {'mode': mode, 'initial_state': vars(initial), 'snapshots': [],
                              'max_policy_twist_error': 0.0, 'max_command_buffer_error': 0.0}
                    report['rollouts'].append(record)
                    rows = []
                    reason = 'completed'
                    nsteps = round(sum(s['seconds'] for s in scenario['segments']) / env.step_dt)
                    for step in range(nsteps):
                        before, _ = observation(env)
                        reason_now = classify_state(before, Criteria(**scenario['goal_criteria']))
                        if reason_now:
                            reason = reason_now
                            break
                        desired = command_at(scenario['segments'], step * env.step_dt)
                        set_command(env, desired)
                        obs = wrapper.get_observations()
                        if not all(torch.isfinite(value).all() for value in obs.values()):
                            raise RuntimeError('非有限の観測')
                        expected = torch.tensor(desired, device=env.device).reshape(1,3)
                        term = env.command_manager.get_term('twist')
                        obs_error = float((obs['actor'][:,start:stop] - expected).abs().max())
                        buffer_error = float((term.command - expected).abs().max())
                        record['max_policy_twist_error'] = max(record['max_policy_twist_error'],obs_error)
                        record['max_command_buffer_error'] = max(record['max_command_buffer_error'],buffer_error)
                        if obs_error > 1e-6 or buffer_error > 1e-6:
                            record['mismatch'] = {'time': step*env.step_dt, 'desired': desired,
                                'actor_twist': obs['actor'][:,start:stop].cpu().tolist(),
                                'command_buffer': term.command.cpu().tolist()}
                            raise RuntimeError('指定指令とpolicy入力が不一致です。probe.jsonを確認してください')
                        if step % 150 == 0:
                            responses = []
                            for test in [(0.,0.,0.),(.2,0.,0.),(0.,0.,.5)]:
                                copied = obs.clone()
                                copied['actor'][:,start:stop] = torch.tensor(test,device=env.device)
                                responses.append(policy(copied).clone())
                            record['snapshots'].append({'time':step*env.step_dt,
                                'actual_actor_twist':obs['actor'][:,start:stop].cpu().tolist(),
                                'action_change_from_zero_vx':float((responses[1]-responses[0]).abs().max()),
                                'action_change_from_zero_yaw':float((responses[2]-responses[0]).abs().max())})
                        actions = policy(obs, stochastic_output=stochastic)
                        if not torch.isfinite(actions).all():
                            raise RuntimeError('非有限のaction')
                        _, reward, done, extras = wrapper.step(actions)
                        end, velocity = observation(env)
                        row = {'time':(step+1)*env.step_dt,'vx_command':desired[0],
                            'yaw_command':desired[2],'vx':velocity[0],'vy':velocity[1],
                            'yaw_rate':velocity[2],'x':end.x,'y':end.y,'yaw':end.yaw,
                            'speed':end.speed,'tilt':end.tilt,'action_rms':float(actions.square().mean().sqrt())}
                        rows.append(row)
                        reason_now = classify_state(end, Criteria(**scenario['goal_criteria']))
                        if reason_now or done[0].item():
                            reason = reason_now or 'environment_termination'
                            break
                    if rows:
                        with (args.output / (mode+'.csv')).open('w',newline='') as stream:
                            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                    record['termination'] = reason
                    record['duration_seconds'] = len(rows)*env.step_dt
                    record['normalizer_unchanged'] = all(torch.equal(v,policy.obs_normalizer.state_dict()[k])
                                                        for k,v in normalizer_before.items())
                    record['segments'] = []
                    cursor=0.
                    for segment in scenario['segments']:
                        endtime=cursor+segment['seconds']
                        group=[r for r in rows if cursor < r['time'] <= endtime]
                        record['segments'].append({'from_s':cursor,'to_s':endtime,'samples':len(group),
                            'command':segment['twist'],**{key:sum(r[key] for r in group)/len(group) if group else None
                             for key in ('vx','yaw_rate','speed','action_rms')}})
                        cursor=endtime
                    write_json(args.output / 'probe.json', report)
                finally:
                    env.close()
        report['status']='completed'
    except BaseException as exc:
        report.update(status='failed',error=str(exc))
        raise
    finally:
        write_json(args.output / 'probe.json', report)
    print(json.dumps({'status':report['status'],'rollouts':report['rollouts']},indent=2))


if __name__ == '__main__':
    main()
