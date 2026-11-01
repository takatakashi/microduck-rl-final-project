from __future__ import annotations

from copy import deepcopy

from .common import read_json, sha256, write_json
from .runtime import checkpoint_metadata, load_for_inference, task_configs


def export_checkpoint(checkpoint, output, scenario_path):
    from .environment import require_gpu
    require_gpu()
    import numpy as np
    import onnxruntime as ort
    import torch
    from mjlab.tasks.registry import register_mjlab_task
    from mjlab_microduck.export import ExportConfig, run_export
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx
    from .evaluation import evaluation_transform, set_command, validate_scenario
    from .metrics import command_at
    if output.exists():
        raise ValueError("ONNX出力先が既に存在します")
    meta = checkpoint_metadata(checkpoint)
    env_cfg, agent_cfg, runner_cls, _ = task_configs(meta["experiment"])
    task = "PAI-Export-Microduck-" + meta["checkpoint_sha256"][:12]
    register_mjlab_task(task, env_cfg=env_cfg, play_env_cfg=deepcopy(env_cfg), rl_cfg=agent_cfg, runner_cls=runner_cls)
    # Mandatory official exporter: obs normalization and ordering remain upstream-owned.
    run_export(task, ExportConfig(checkpoint_file=str(checkpoint.resolve()),
                                 onnx_file=str(output.resolve()), num_envs=1, device="cuda:0"))
    # Remove absolute local paths from the exported metadata before sharing.
    import onnx
    model = onnx.load(str(output))
    for entry in model.metadata_props:
        if entry.key == "run_path":
            entry.value = "sha256:" + meta["checkpoint_sha256"]
    onnx.save(model, str(output))
    attach_metadata_to_onnx(str(output), {"checkpoint_sha256": meta["checkpoint_sha256"],
        "upstream_commit": meta["upstream_commit"], "configuration_sha256": meta["configuration_sha256"]})
    scenario = validate_scenario(read_json(scenario_path))
    env, wrapper, runner, policy, _ = load_for_inference(checkpoint, env_transform=evaluation_transform(scenario, "walk"))
    report = {"schema_version": 1, "status": "running", "checkpoint_sha256": meta["checkpoint_sha256"],
              "onnx_sha256": sha256(output), "io_contract": meta["io_contract"],
              "upstream_commit": meta["upstream_commit"], "samples": 0,
              "rtol": 1e-4, "atol": 1e-5, "max_absolute_error": 0.0,
              "limitations": "有限個の実観測での照合。全入力での一致や実機互換を保証しない。"}
    try:
        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
            raise ValueError("単一入力・単一出力の公式歩行ONNXだけを照合できます")
        input_meta = session.get_inputs()[0]
        report["onnx_input"] = {"name": input_meta.name, "shape": input_meta.shape, "type": input_meta.type}
        if input_meta.shape != [1, 61] or input_meta.type != "tensor(float)":
            raise ValueError("ONNXの観測shape/dtypeが公式61D float32と不一致")
        with torch.inference_mode():
            # Multiple seeds and commands use only model-selection scenarios.
            for seed in scenario["selection_seeds"]:
                env.reset(seed=seed)
                for step in range(256):
                    set_command(env, command_at(scenario["segments"], step * env.step_dt))
                    obs = wrapper.get_observations()
                    expected = policy(obs)
                    actual = session.run(None, {input_meta.name: obs["actor"].cpu().numpy()})[0]
                    expected_np = expected.cpu().numpy()
                    if actual.shape != expected_np.shape or not np.isfinite(actual).all() or not np.isfinite(expected_np).all():
                        raise ValueError("ONNX出力にshape不一致またはNaN/Infがあります")
                    error = float(np.abs(expected_np - actual).max())
                    report["max_absolute_error"] = max(report["max_absolute_error"], error)
                    report["samples"] += 1
                    if not np.allclose(expected_np, actual, rtol=report["rtol"], atol=report["atol"]):
                        raise ValueError(f"ONNX parity失敗: max absolute error={error}")
                    _, _, done, _ = wrapper.step(expected)
                    if done[0].item():
                        env.reset(seed=seed)
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        env.close()
        write_json(output.with_suffix(".parity.json"), report)
    return report
