from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .common import ROOT


def parser():
    cli = argparse.ArgumentParser(description="Microduck最終課題: 固定上流による学習・再開・比較・目標移動")
    commands = cli.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="OS/依存/GPU/物理の検証")
    doctor.add_argument("--gpu", action="store_true")
    doctor.add_argument("--persistent-workspace", type=Path)
    doctor.add_argument("--output", type=Path, default=ROOT / "outputs/environment.json")
    validate = commands.add_parser("validate", help="依存不要の実験・評価設定検証")
    validate.add_argument("--baseline", type=Path, default=ROOT / "configs/experiments/baseline.json")
    validate.add_argument("--variant", type=Path, default=ROOT / "configs/experiments/low_speed.json")
    validate.add_argument("--scenario", type=Path, default=ROOT / "configs/evaluation.json")
    train = commands.add_parser("train", help="公式PPOを学習。iteration数は累計")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--resume", type=Path)
    train.add_argument("--total-iterations", type=int)
    train.add_argument("--num-envs", type=int)
    smoke = commands.add_parser("smoke", help="GPU学習→別プロセスで再開→MP4→ONNX照合")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--num-envs", type=int, default=1024)
    evaluate = commands.add_parser("evaluate", help="共通seedの歩行評価または目標到達評価")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--scenario", type=Path, default=ROOT / "configs/evaluation.json")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--mode", choices=("walk", "goal"), default="walk")
    evaluate.add_argument("--split", choices=("selection", "final"), default="final")
    evaluate.add_argument("--video", action="store_true")
    evaluate.add_argument("--smoke", action="store_true", help="1 seed / 最長3秒。最終評価には使用不可")
    export = commands.add_parser("export", help="公式ONNX export + 元checkpointとの実観測での照合")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--scenario", type=Path, default=ROOT / "configs/evaluation.json")
    compare = commands.add_parser("compare", help="同条件を検証して比較CSV/JSONを生成")
    compare.add_argument("first", type=Path)
    compare.add_argument("second", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    video = commands.add_parser("check-video", help="MP4全体をデコードし長さを検証")
    video.add_argument("path", type=Path)
    video.add_argument("--submission", action="store_true", help="60〜180秒も検証")
    backup = commands.add_parser("backup", help="ソース・設定・実験をtar.gzへ保存しハッシュ照合")
    backup.add_argument("run_directory", type=Path)
    backup.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify-backup", help="移動したバックアップを展開せず検証")
    verify.add_argument("archive", type=Path)
    estimate = commands.add_parser("estimate", help="smoke実測から時間と計算料金を推定")
    estimate.add_argument("run_json", type=Path)
    estimate.add_argument("--iterations", type=int, required=True)
    estimate.add_argument("--hourly-price", type=float, required=True, help="実際の1 GPU時間単価（同一通貨）")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "doctor":
        from .environment import doctor
        result = doctor(args.output, args.gpu, args.persistent_workspace)
    elif args.command == "validate":
        from .config import load_experiment, assert_fair_pair
        from .evaluation import validate_scenario
        from .common import read_json
        assert_fair_pair(load_experiment(args.baseline), load_experiment(args.variant))
        validate_scenario(read_json(args.scenario))
        result = {"configuration_validation": "passed", "gpu_training": "not_executed"}
    elif args.command == "train":
        from .training import train
        result = train(args.config, args.output, resume=args.resume,
                       total_iterations=args.total_iterations, num_envs=args.num_envs)
        result = {"status": result["status"], "total_env_steps": result["total_env_steps"]}
    elif args.command == "smoke":
        from .smoke import smoke
        result = smoke(args.output, args.num_envs)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        result = evaluate(args.checkpoint, args.scenario, args.output, mode=args.mode,
                          split=args.split, video=args.video, smoke=args.smoke)
        result = {"status": result["status"], "summary": result["summary"]}
    elif args.command == "export":
        from .exporting import export_checkpoint
        result = export_checkpoint(args.checkpoint, args.output, args.scenario)
    elif args.command == "compare":
        from .artifacts import compare
        result = compare(args.first, args.second, args.output)
    elif args.command == "check-video":
        from .video import validate_video
        result = validate_video(args.path, submission=args.submission)
    elif args.command == "backup":
        from .artifacts import backup
        result = backup(args.run_directory, args.output)
    elif args.command == "verify-backup":
        from .artifacts import verify_backup
        result = verify_backup(args.archive)
    elif args.command == "estimate":
        from .artifacts import estimate
        result = estimate(args.run_json, args.iterations, args.hourly_price)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
