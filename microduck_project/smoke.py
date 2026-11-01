from __future__ import annotations

import subprocess
import sys

from .common import ROOT, read_json, write_json, upstream_lock
from .runtime import checkpoint_metadata


def smoke(output, num_envs):
    if output.exists() or num_envs <= 0:
        raise ValueError("新しい出力先と正の並列数を指定してください")
    output = output.resolve()
    output.mkdir(parents=True)
    report = {"schema_version": 1, "status": "running", "upstream": upstream_lock(), "stages": []}
    def child(name, arguments):
        with (output / f"{name}.log").open("w", encoding="utf-8") as log:
            command = [sys.executable, "-u", "-m", "microduck_project.cli", *map(str, arguments)]
            print(f"[{name}] starting; log={name}.log", flush=True)
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        report["stages"].append({"name": name, "exit_code": result.returncode})
        write_json(output / "smoke.json", report)
        if result.returncode:
            raise RuntimeError(f"{name}に失敗しました。{name}.logを確認してください")
    try:
        child("doctor", ["doctor", "--gpu", "--output", output / "environment.json"])
        # Capture the official CLI help from the installed, locked dependency tree.
        with (output / "upstream-train-help.txt").open("w", encoding="utf-8") as help_log:
            subprocess.run([sys.executable, "-m", "mjlab.scripts.train", upstream_lock()["task"], "--help"],
                           cwd=ROOT, stdout=help_log, stderr=subprocess.STDOUT, check=True)
        common = ["--config", ROOT / "configs/experiments/smoke.json", "--num-envs", num_envs]
        child("initial", ["train", *common, "--output", output / "initial"])
        before = checkpoint_metadata(output / "initial/final.pt")
        child("resumed", ["train", *common, "--output", output / "resumed",
              "--resume", output / "initial/final.pt", "--total-iterations", "10"])
        after = checkpoint_metadata(output / "resumed/final.pt")
        resumed_run = read_json(output / "resumed/run.json")
        if not resumed_run.get("resume_verified_before_learning") or before["completed_iterations"] != 5 or after["completed_iterations"] != 10:
            raise RuntimeError("保存済みの学習ステップから進んだことを確認できません")
        if after["total_env_steps"] != 2 * before["total_env_steps"]:
            raise RuntimeError("再開後の環境ステップ数が不整合です")
        child("video", ["evaluate", "--checkpoint", output / "resumed/final.pt", "--smoke", "--video", "--output", output / "video"])
        child("onnx", ["export", "--checkpoint", output / "resumed/final.pt", "--output", output / "smoke.onnx"])
        report["status"] = "passed_automated_checks"
        report["visual_review"] = "required; open the MP4 and inspect the robot"
        report["walking_skill"] = "not_claimed; 10 iterations only validates the pipeline"
    except BaseException:
        report["status"] = "failed"
        raise
    finally:
        write_json(output / "smoke.json", report)
    return report
