from pathlib import Path
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
import unittest

from microduck_project.common import write_json, source_identity
from microduck_project.artifacts import backup, estimate, verify_backup
from microduck_project.video import validate_video


class ArtifactTests(unittest.TestCase):
    def test_backup_survives_copy_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            (run / "final.pt").write_bytes(b"fixture-only-not-a-trained-model")
            (run / "final.json").write_text('{"test": true}')
            archive = root / "archive.tar.gz"
            report = backup(run, archive)
            copied = root / "copied.tar.gz"
            shutil.copyfile(archive, copied)
            self.assertEqual(report["sha256"], verify_backup(copied)["sha256"])
            with tarfile.open(archive) as tar:
                self.assertTrue(all(not m.uname and not m.gname for m in tar.getmembers()))
                archived = json.load(tar.extractfile("backup_manifest.json"))
                file_hashes = {x["path"]: x["sha256"] for x in archived["files"]}
                for row in source_identity()["files"]:
                    self.assertEqual(file_hashes["project/" + row["path"]], row["sha256"])
            bad = root / "bad.tar.gz"
            with tarfile.open(archive) as source, tarfile.open(bad, "w:gz") as target:
                for member in source.getmembers():
                    stream = source.extractfile(member)
                    if member.name == "run/final.pt":
                        data = b"x" * member.size
                        stream = io.BytesIO(data)
                    target.addfile(member, stream)
            with self.assertRaises(ValueError):
                verify_backup(bad)

    def test_backup_never_includes_itself(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                backup(root, root / "archive.tar.gz")

    def test_cost_uses_measured_warm_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            write_json(path, {"warm_iteration_mean_seconds": 2.0})
            result = estimate(path, 1800, 3.0)
            self.assertEqual(result["training_seconds"], 3600)
            self.assertEqual(result["gpu_compute_cost"], 3.0)
            self.assertEqual(result["kind"], "estimate")
            with self.assertRaises(ValueError):
                estimate(path, 1800, float("nan"))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg unavailable")
    def test_real_video_decode_and_submission_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "synthetic-fixture.mp4"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                            "color=c=blue:s=160x120:r=10:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
                           check=True, capture_output=True)
            report = validate_video(video)
            self.assertEqual(report["decode"], "passed")
            self.assertAlmostEqual(report["duration_seconds"], 1.0, places=1)
            with self.assertRaises(ValueError):
                validate_video(video, submission=True)
