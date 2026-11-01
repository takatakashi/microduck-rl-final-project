from __future__ import annotations

import dataclasses
import enum
import hashlib
import functools
import json
import math
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def jsonable(value):
    if isinstance(value, enum.Enum):
        # Keep type, member and value stable for configuration/resume hashes.
        # Handle this before primitives: IntEnum/StrEnum also inherit int/str.
        return {"enum": f"{type(value).__module__}.{type(value).__qualname__}",
                "name": value.name, "value": jsonable(value.value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return value.relative_to(ROOT).as_posix() if value.is_relative_to(ROOT) else value.as_posix()
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if isinstance(value, functools.partial):
        return {"partial": jsonable(value.func), "args": jsonable(value.args), "keywords": jsonable(value.keywords)}
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, float) and not math.isfinite(value):
        return {"non_finite_config_value": str(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def digest(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def run(command, *, cwd=ROOT, capture=False):
    return subprocess.run([str(x) for x in command], cwd=cwd, check=True, text=True,
                          capture_output=capture)


def source_identity():
    files = []
    for folder in ("microduck_project", "scripts", "configs", "tests"):
        for p in sorted((ROOT / folder).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                files.append({"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p)})
    try:
        commit = run(["git", "rev-parse", "HEAD"], capture=True).stdout.strip()
        dirty = bool(run(["git", "status", "--porcelain"], capture=True).stdout.strip())
    except subprocess.CalledProcessError:
        commit, dirty = None, None
    return {"git_sha": commit, "git_dirty": dirty, "source_sha256": digest(files), "files": files}


def upstream_lock():
    return read_json(ROOT / "configs/upstream.json")


def verify_upstream():
    lock = upstream_lock()
    folder = ROOT / lock["directory"]
    if not (folder / ".git").is_dir():
        raise RuntimeError("上流未取得です。python3 scripts/bootstrap.py を先に実行してください。")
    actual = run(["git", "rev-parse", "HEAD"], cwd=folder, capture=True).stdout.strip()
    if actual != lock["commit"]:
        raise RuntimeError(f"上流SHAが固定値と不一致: {actual}")
    if run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=folder, capture=True).stdout.strip():
        raise RuntimeError("上流の追跡ファイルに変更があります。固定ソースを復元してから実行してください。")
    return folder

