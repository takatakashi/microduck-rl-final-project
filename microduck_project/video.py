from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

from .common import sha256, write_json


class VideoWriter:
    def __init__(self, path, *, fps, width=640, height=480):
        if not shutil.which("ffmpeg"):
            raise RuntimeError("MP4録画にはffmpegが必要です")
        self.path, self.width, self.height = Path(path), width, height
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.path.with_suffix(".ffmpeg.log").open("wb")
        self.process = subprocess.Popen([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)
        ], stdin=subprocess.PIPE, stderr=self.log)

    def append(self, frame, caption):
        from PIL import Image, ImageDraw
        image = Image.fromarray(frame).convert("RGB").resize((self.width, self.height))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, self.width, 48), fill=(18, 24, 34))
        draw.text((12, 10), caption, fill=(255, 255, 255))
        assert self.process.stdin is not None
        self.process.stdin.write(image.tobytes())

    def close(self):
        try:
            if self.process.stdin:
                self.process.stdin.close()
            code = self.process.wait(timeout=60)
            if code:
                raise RuntimeError(f"FFmpeg失敗: 詳細は {self.path.name}.ffmpeg.log")
        finally:
            self.log.close()


def validate_video(path, *, submission=False):
    path = Path(path)
    result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                            check=True, capture_output=True, text=True)
    probe = json.loads(result.stdout)
    streams = [s for s in probe["streams"] if s["codec_type"] == "video"]
    if not streams:
        raise ValueError("動画ストリームがありません")
    duration = float(probe["format"]["duration"])
    if duration <= 0 or (submission and not 60 <= duration <= 180):
        raise ValueError(f"動画の長さが条件外です: {duration}秒")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
                   check=True, capture_output=True)
    report = {"file": path.name, "sha256": sha256(path), "duration_seconds": duration,
              "decode": "passed", "submission_duration": "passed" if submission else "not_checked",
              "visual_review": "required; decoding does not verify robot behavior or privacy"}
    write_json(path.with_suffix(".video.json"), report)
    return report
