from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
import sys
import time

from .common import ROOT, sha256, upstream_lock, verify_upstream, write_json, utc_now


def inventory():
    report = {"schema_version": 1, "recorded_at": utc_now(), "os": platform.system(),
              "architecture": platform.machine(), "python": platform.python_version(),
              "cpu_count": __import__("os").cpu_count(),
              "free_disk_gb": round(shutil.disk_usage(ROOT).free / 1e9, 2),
              "tools": {}, "dependencies": {}, "gpu_validation": "not_run",
              "persistent_workspace": "unverified", "hourly_price": None}
    for name, arguments in (("uv", ["--version"]), ("ffmpeg", ["-version"]), ("ffprobe", ["-version"]),
                            ("nvidia-smi", ["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])):
        executable = shutil.which(name)
        if not executable:
            report["tools"][name] = {"available": False}
            continue
        result = subprocess.run([executable, *arguments], capture_output=True, text=True, timeout=20)
        report["tools"][name] = {"available": result.returncode == 0,
                                  "output": result.stdout.splitlines()[:2]}
    for package in ("mjlab", "torch", "warp-lang", "mujoco", "mujoco-warp", "rsl-rl-lib", "onnxruntime", "better-actuator-models"):
        try:
            report["dependencies"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["dependencies"][package] = None
    return report


def require_gpu():
    if (platform.system(), platform.machine()) != ("Linux", "x86_64"):
        raise RuntimeError("学習はLinux x86_64のNVIDIA GPU環境で実行してください")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("採用環境にはPython 3.12が必要です")
    folder = verify_upstream()
    # Prevent accidentally importing a globally installed, different robot package.
    import mjlab_microduck
    package_folder = __import__("pathlib").Path(next(iter(mjlab_microduck.__path__))).resolve()
    if not package_folder.is_relative_to(folder):
        raise RuntimeError("mjlab_microduckが固定したvendorソースからimportされていません")
    for package, expected in upstream_lock()["expected_versions"].items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f"固定版と不一致: {package}. uv sync --locked を再確認してください")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA_VISIBLE_DEVICESでNVIDIA GPUを1枚だけ指定してください")
    result = (torch.tensor([1.0, 2.0], device="cuda:0") * 2).sum().item()
    if result != 6.0:
        raise RuntimeError("PyTorch CUDA計算検証に失敗しました")
    return {"name": torch.cuda.get_device_name(0),
            "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
            "torch_cuda": torch.version.cuda, "lock_sha256": sha256(folder / "uv.lock")}


def gpu_physics_probe():
    gpu = require_gpu()
    import mujoco
    import mujoco_warp as mjw
    import warp as wp
    import numpy as np
    wp.init()
    xml = '<mujoco><option timestep="0.002"/><worldbody><body pos="0 0 1"><freejoint/><geom type="sphere" size="0.05" mass="1"/></body></worldbody></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with wp.ScopedDevice("cuda:0"):
        wm = mjw.put_model(model)
        wd = mjw.put_data(model, data, nworld=1)
        started = time.perf_counter()
        mjw.step(wm, wd)
        wp.synchronize()
        first_step = time.perf_counter() - started
        started = time.perf_counter()
        for _ in range(20):
            mjw.step(wm, wd)
        wp.synchronize()
        warm_steps = time.perf_counter() - started
        state = wd.qpos.numpy()
        if not np.isfinite(state).all() or not (state[0, 2] < 1.0):
            raise RuntimeError("MuJoCo WarpのGPU物理ステップで落下を確認できません")
    return {"gpu": gpu, "first_step_seconds_including_compilation": first_step,
            "warm_20_steps_seconds": warm_steps, "end_z": float(state[0, 2]),
            "physics_probe": "passed", "training": "not_tested_by_this_probe"}


def doctor(output, gpu=False, persistent_workspace=None):
    report = inventory()
    if persistent_workspace:
        from pathlib import Path
        folder = Path(persistent_workspace).resolve(strict=True)
        if not ROOT.is_relative_to(folder):
            raise ValueError("プロジェクトは指定した永続化作業ディレクトリ内に置いてください")
        report["persistent_workspace"] = str(folder)
        report["persistence_basis"] = "operator_supplied; not verified against provider lifecycle"
    try:
        if gpu:
            if not report["tools"]["nvidia-smi"]["available"]:
                raise RuntimeError("nvidia-smiが実行できません")
            report["gpu_probe"] = gpu_physics_probe()
            report["gpu_validation"] = "passed"
    except Exception as exc:
        report["gpu_validation"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        write_json(output, report)
    return report
