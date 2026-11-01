from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile

from .common import ROOT, digest, read_json, sha256, write_json
from .config import assert_fair_pair


def compare(first, second, output):
    reports = [read_json(first), read_json(second)]
    a, b = reports
    if any(r["status"] != "completed" for r in reports):
        raise ValueError("未完了の評価は比較できません")
    if any(r["mode"] != "walk" or r["split"] != "final" for r in reports):
        raise ValueError("比較表には最終歩行評価を使います。smoke・モデル選択評価は除外")
    for key in ("scenario_sha256", "seeds", "evaluation_settings", "evaluation_source_sha256"):
        if a[key] != b[key]:
            raise ValueError(f"評価条件が不一致: {key}")
    for r in reports:
        if len(r["episodes"]) < 10 or [x["seed"] for x in r["episodes"]] != r["seeds"]:
            raise ValueError("全seedの結果を含めてください。成功試行だけの比較は禁止です")
        if digest(r["scenario"]) != r["scenario_sha256"]:
            raise ValueError("評価シナリオのハッシュが不一致")
        if r["checkpoint"]["completed_iterations"] != r["checkpoint"]["experiment"]["total_iterations"]:
            raise ValueError("指定した総学習量まで学習していないcheckpointです")
    assert_fair_pair(a["checkpoint"]["experiment"], b["checkpoint"]["experiment"])
    for key in ("total_env_steps", "num_steps_per_env", "upstream_commit", "source_sha256", "lock_sha256", "initialization"):
        if a["checkpoint"][key] != b["checkpoint"][key]:
            raise ValueError(f"学習条件が不一致: {key}")
    if output.exists():
        raise ValueError("比較出力先が既に存在します")
    output.mkdir(parents=True)
    from .metrics import aggregate
    rows = [{"condition": r["checkpoint"]["experiment"]["condition"],
             "total_env_steps": r["checkpoint"]["total_env_steps"],
             "training_seconds": r["checkpoint"]["wall_seconds_total"],
             **aggregate(r["episodes"])} for r in reports]
    keys = ["condition", "total_env_steps", "training_seconds", "episodes", "fall_rate", "success_rate", "mean_duration_seconds", "mean_vx_rmse_m_s", "mean_xy_rmse_m_s", "mean_yaw_rate_rmse_rad_s", "mean_stop_speed_rms_m_s"]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "comparison.json", {"schema_version": 1, "rows": rows,
        "limitations": ["各条件の学習seedは1個。統計的な優位性は示さない。",
                        "早期転倒は誤差の計測時間を短縮する。RMSEだけで結論を出さない。"],
        "source_evaluations": [{"file": Path(p).name, "sha256": sha256(p)} for p in (first, second)]})
    return rows


def backup(run_directory, archive):
    """Create and verify a portable archive. Copy off-instance is a separate operation."""
    run_directory = Path(run_directory).resolve(strict=True)
    archive = Path(archive).resolve()
    if archive.exists():
        raise ValueError("バックアップ先が既に存在します")
    if archive.is_relative_to(run_directory):
        raise ValueError("バックアップは対象ディレクトリの外へ作成してください")
    files = []
    for file in sorted(run_directory.rglob("*")):
        if file.is_symlink():
            raise ValueError("バックアップ対象にsymlinkを含めないでください")
        if file.is_file():
            files.append((file, "run/" + file.relative_to(run_directory).as_posix()))
    for folder in ("microduck_project", "scripts", "configs", "tests", "docs", "licenses", "assets"):
        for file in sorted((ROOT / folder).rglob("*")):
            if file.is_file() and "__pycache__" not in file.parts and not file.is_symlink():
                files.append((file, "project/" + file.relative_to(ROOT).as_posix()))
    for name in ("README.md", "THIRD_PARTY_NOTICES.md", "pyproject.toml", ".python-version"):
        files.append((ROOT / name, "project/" + name))
    manifest = [{"path": name, "sha256": sha256(file), "size_bytes": file.stat().st_size} for file, name in files]
    payload = json.dumps({"schema_version": 1, "files": manifest}, indent=2).encode()
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        for file, name in files:
            info = tar.gettarinfo(str(file), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with file.open("rb") as stream:
                tar.addfile(info, stream)
        info = tarfile.TarInfo("backup_manifest.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    result = verify_backup(archive)
    result["external_copy"] = "not_verified; verify again after copying off the GPU instance"
    write_json(archive.with_suffix(archive.suffix + ".json"), result)
    return result


def verify_backup(archive):
    with tarfile.open(archive, "r:gz") as tar:
        manifest_file = tar.extractfile("backup_manifest.json")
        if manifest_file is None:
            raise ValueError("manifestがありません")
        manifest = json.load(manifest_file)
        for row in manifest["files"]:
            member = tar.getmember(row["path"])
            if not member.isfile() or member.size != row["size_bytes"]:
                raise ValueError("バックアップサイズ不一致")
            stream = tar.extractfile(member)
            h = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
            if h.hexdigest() != row["sha256"]:
                raise ValueError(f"バックアップハッシュ不一致: {row['path']}")
    return {"archive": Path(archive).name, "sha256": sha256(archive), "files": len(manifest["files"]), "verification": "passed"}


def estimate(run_path, target_iterations, hourly_price):
    run = read_json(run_path)
    warm = run.get("warm_iteration_mean_seconds")
    if not warm or target_iterations <= 0 or not math.isfinite(hourly_price) or hourly_price < 0:
        raise ValueError("完了したsmokeの定常iteration時間、正のiteration数、非負の実際の時間単価が必要です")
    seconds = target_iterations * warm
    return {"kind": "estimate", "training_seconds": seconds, "gpu_compute_cost": seconds / 3600 * hourly_price,
            "hourly_price": hourly_price, "excluded": ["setup", "compilation", "evaluation", "idle", "storage", "tax"],
            "basis": "measured warm iteration average; same GPU and num_envs assumed"}
