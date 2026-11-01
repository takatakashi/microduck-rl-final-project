"""Check saved M0 evidence and estimate baseline time, without running a GPU job."""
from pathlib import Path
import argparse
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from microduck_project.common import digest, read_json, sha256, write_json
from microduck_project.config import load_experiment, resume_key
from microduck_project.runtime import checkpoint_metadata


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive_seconds(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            f"有効な実測時間がありません: {name}")
    return value


def check_m0(run_dir, evaluation_dir, onnx_file, baseline_config, hourly_price=None):
    if hourly_price is not None:
        require(math.isfinite(hourly_price) and hourly_price >= 0, "時間単価が不正です")
    baseline = load_experiment(baseline_config)
    require(baseline["condition"] == "baseline", "標準条件の設定を指定してください")
    environment = read_json(run_dir / "environment.json")
    require(environment["gpu_validation"] == "passed", "GPU診断が未完了です")
    require(environment["gpu_probe"]["physics_probe"] == "passed", "GPU物理probeが未完了です")
    records, metadata = {}, {}
    for stage, expected_iterations in (("initial", 5), ("resumed", 10)):
        record = read_json(run_dir / stage / "run.json")
        meta = checkpoint_metadata(run_dir / stage / "final.pt")
        require(record["status"] == "completed", f"{stage}の学習が未完了です")
        require(record["completed_iterations"] == meta["completed_iterations"] == expected_iterations,
                f"{stage}の完了iteration数が不一致です")
        require(record["total_env_steps"] == meta["total_env_steps"], f"{stage}の総ステップが不一致です")
        require(record["checkpoint_sha256"] == meta["checkpoint_sha256"], f"{stage}のモデルが不一致です")
        require(resume_key(meta["experiment"]) == resume_key(baseline),
                f"{stage}と標準条件のseed・並列数・学習条件が異なります")
        positive_seconds(record["wall_seconds_this_process"], f"{stage}.wall_seconds_this_process")
        positive_seconds(record["warm_iteration_mean_seconds"], f"{stage}.warm_iteration_mean_seconds")
        records[stage], metadata[stage] = record, meta
    initial, resumed = metadata["initial"], metadata["resumed"]
    restored = records["resumed"]
    require(records["initial"]["initialization"] == "scratch" and restored["initialization"] == "resume",
            "scratch学習→別プロセスで再開という記録になっていません")
    require(restored["resume_verified_before_learning"] is True and restored["restored_completed_iterations"] == 5,
            "保存した5 iterationからの復元が未確認です")
    require(restored["parent_checkpoint_sha256"] == initial["checkpoint_sha256"]
            and resumed["parent_checkpoint_sha256"] == initial["checkpoint_sha256"],
            "再開元のチェックポイントが不一致です")
    require(resumed["total_env_steps"] == 2 * initial["total_env_steps"], "再開後の総ステップが不整合です")
    for key in ("upstream_commit", "lock_sha256", "source_sha256", "configuration_sha256", "io_contract"):
        require(initial[key] == resumed[key], f"学習・再開間で変更があります: {key}")
    for stage, record in records.items():
        require(record["gpu"]["name"] == environment["gpu_probe"]["gpu"]["name"],
                f"診断と{stage}でGPUが異なります")
    assessment = read_json(evaluation_dir / "evaluation.json")
    require(assessment["status"] == "completed" and assessment["split"] == "smoke", "短時間評価が未完了です")
    require(assessment["checkpoint"]["checkpoint_sha256"] == resumed["checkpoint_sha256"],
            "録画評価のモデルが再開後のモデルと異なります")
    require(digest(assessment["scenario"]) == assessment["scenario_sha256"], "評価設定のハッシュが不一致です")
    require(len(assessment["episodes"]) > 0 and len(assessment["episodes"]) == len(assessment["seeds"]),
            "評価エピソードが不足しています")
    videos = []
    for episode, seed in zip(assessment["episodes"], assessment["seeds"], strict=True):
        require(episode["seed"] == seed, "評価seedが不一致です")
        video = episode["video"]
        video_path = (evaluation_dir / video["file"]).resolve()
        require(video_path.is_relative_to(evaluation_dir.resolve()), "動画の保存先が不正です")
        require(video["decode"] == "passed" and sha256(video_path) == video["sha256"],
                "動画の検証記録またはファイルハッシュが不一致です")
        videos.append({"file": video["file"], "duration_seconds": video["duration_seconds"], "decode": "passed"})
    parity = read_json(onnx_file.with_suffix(".parity.json"))
    require(parity["status"] == "passed" and parity["samples"] > 0, "ONNX出力照合が未完了です")
    require(parity["checkpoint_sha256"] == resumed["checkpoint_sha256"], "ONNX照合の元モデルが不一致です")
    require(parity["onnx_sha256"] == sha256(onnx_file), "ONNXのファイルハッシュが不一致です")
    warm = max(r["warm_iteration_mean_seconds"] for r in records.values())
    seconds = baseline["total_iterations"] * warm
    return {"status": "passed_record_checks", "gpu": records["resumed"]["gpu"],
        "initial_completed_iterations": 5, "resumed_completed_iterations": 10,
        "resume_verified_before_learning": True, "resumed_total_env_steps": resumed["total_env_steps"],
        "measured_seconds": {stage: {key: r.get(key) for key in (
            "wall_seconds_this_process", "environment_build_seconds_including_compilation",
            "first_iteration_seconds_including_compilation", "warm_iteration_mean_seconds")}
            for stage, r in records.items()},
        "videos": videos, "visual_review": "confirm separately; not checked by this script",
        "onnx_parity": parity["status"], "onnx_samples": parity["samples"],
        "baseline": {"initialization": "scratch", "num_envs": baseline["num_envs"],
            "iterations": baseline["total_iterations"],
            "total_env_steps": baseline["total_iterations"] * resumed["num_steps_per_env"] * baseline["num_envs"]},
        "estimate": {"basis": "slower warm iteration mean from the two short runs; same GPU and num_envs",
            "training_hours": seconds / 3600, "hourly_price": hourly_price,
            "gpu_compute_cost": None if hourly_price is None else seconds / 3600 * hourly_price,
            "excluded": ["setup", "compilation", "evaluation", "idle", "storage", "tax"],
            "limitation": "short-run estimate, not a guarantee"},
        "training_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/m0_enum_fix")
    parser.add_argument("--evaluation-dir", type=Path)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--baseline-config", type=Path, default=ROOT / "configs/experiments/baseline.json")
    parser.add_argument("--hourly-price", type=float, help="実際のGPU時間単価。未指定時は金額を推定しない")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check_m0(args.run_dir, args.evaluation_dir or args.run_dir / "video_inference_fix",
                      args.onnx or args.run_dir / "smoke_export.onnx", args.baseline_config, args.hourly_price)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        sys.exit(f"M0記録の確認に失敗しました: {exc}")
