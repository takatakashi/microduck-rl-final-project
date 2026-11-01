"""Fetch the pinned upstream without changing its pyproject.toml or lockfile."""
from pathlib import Path
import argparse
import os
import platform
import subprocess
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from microduck_project.common import ROOT, run, sha256, upstream_lock, verify_upstream, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-only", action="store_true", help="依存をインストールせずソースだけ取得")
    args = parser.parse_args()
    lock = upstream_lock()
    folder = ROOT / lock["directory"]
    if not args.fetch_only and (platform.system(), platform.machine()) != ("Linux", "x86_64"):
        raise RuntimeError("依存構築・学習対象はBrevのLinux x86_64です。ソース取得のみは --fetch-only")
    if folder.exists():
        verify_upstream()  # Never discard an existing checkout, even a partial fetch.
    else:
        folder.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".microduck-fetch-", dir=folder.parent))
        try:
            run(["git", "init", staging])
            run(["git", "remote", "add", "origin", lock["repository"]], cwd=staging)
            run(["git", "fetch", "--depth", "1", "origin", lock["commit"]], cwd=staging)
            run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=staging)
            staging.rename(folder)
        finally:
            if staging.exists():
                shutil.rmtree(staging)  # Only the temporary directory created above.
    verify_upstream()
    for filename in ("pyproject.toml", "uv.lock", "LICENSE", "AGENTS.md"):
        if not (folder / filename).is_file():
            raise RuntimeError(f"上流の必要ファイルがありません: {filename}")
    files = []
    tracked = run(["git", "ls-files", "-z"], cwd=folder, capture=True).stdout.split("\0")
    for name in sorted(filter(None, tracked)):
        p = folder / name
        # Hash all upstream source and assets, including BAM configs and XML.
        if p.is_file():
            is_mesh = p.suffix.lower() == ".stl" and "/assets/" in name
            files.append({"path": name, "sha256": sha256(p), "size_bytes": p.stat().st_size,
                          "license": "CC-BY-NC-SA-4.0" if is_mesh else "see upstream LICENSE and file headers",
                          "source": f"https://github.com/pollen-robotics/microduck_rl/blob/{lock['commit']}/{name}"})
    write_json(ROOT / "assets/manifest.json", {
        "schema_version": 1, "status": "fetched_and_hashed", "upstream": lock,
        "upstream_model_notice_present": (folder / "LICENSE-MODELS.md").exists(),
        "model_license_clarification": "licenses/UPSTREAM_MODEL_NOTICE.md",
        "official_pretrained_weights": [], "files": files})
    if not args.fetch_only:
        os.environ.setdefault("UV_CACHE_DIR", str(ROOT / ".cache/uv"))
        os.environ.setdefault("UV_PYTHON_INSTALL_DIR", str(ROOT / ".cache/python"))
        os.environ.setdefault("UV_HTTP_TIMEOUT", "600")
        run(["uv", "sync", "--locked", "--python", "3.12"], cwd=folder)
        verify_upstream()
        print("構築完了。python3 scripts/run.py doctor --gpu を実行してください。")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        sys.exit(str(exc))
