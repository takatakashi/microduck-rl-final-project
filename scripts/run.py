"""Dispatch into the pinned upstream virtual environment, with local source on PYTHONPATH."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from microduck_project.common import verify_upstream
    from microduck_project.cli import main as cli
    arguments = sys.argv[1:]
    # Offline commands deliberately work without the large GPU dependency tree.
    gpu = bool(arguments and arguments[0] in {"train", "evaluate", "export", "smoke"})
    gpu = gpu or ("doctor" in arguments and "--gpu" in arguments)
    if gpu and not any(x in arguments for x in ("--help", "-h")):
        folder = verify_upstream()
        python = folder / ".venv/bin/python"
        if not python.is_file():
            raise RuntimeError("上流の仮想環境がありません。python3 scripts/bootstrap.py を実行してください")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT)
        env.setdefault("MUJOCO_GL", "egl")
        env.setdefault("WANDB_MODE", "disabled")
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        env.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
        os.execve(str(python), [str(python), "-m", "microduck_project.cli", *arguments], env)
    cli(arguments)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        sys.exit(str(exc))
