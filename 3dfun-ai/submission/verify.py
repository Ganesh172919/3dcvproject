"""Check the final course archive and its report, video, and recorded evidence."""
from __future__ import annotations

import argparse
import csv
from io import BytesIO, StringIO
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile

from imageio_ffmpeg import get_ffmpeg_exe
from pypdf import PdfReader

from .package import ARCHIVE, EVIDENCE_FILES, MAX_ARCHIVE_BYTES


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_archive(path: Path) -> None:
    require(path.stat().st_size < MAX_ARCHIVE_BYTES, "Archive exceeds the size limit.")
    with ZipFile(path) as archive:
        names = archive.namelist()
        require(archive.testzip() is None, "Archive contains corrupt data.")
        require(len(names) == len({name.casefold() for name in names}), "Archive contains duplicate paths.")
        for name in names:
            parts = PurePosixPath(name).parts
            require(len(parts) >= 2 and parts[0] == ARCHIVE and ".." not in parts
                    and "\\" not in name and ":" not in name, f"Unexpected archive entry: {name}")
            require(not any(part in {"data", "out", "__pycache__"} or part.startswith(".venv")
                            or (part.startswith(".env") and part != ".env.example") for part in parts)
                    and PurePosixPath(name).suffix.lower() not in {".onnx", ".pth", ".sqlite", ".db", ".pyc"},
                    f"Unexpected local data or weight file: {name}")

        def read(name: str) -> bytes:
            return archive.read(f"{ARCHIVE}/{name}")

        expected = {
            "report.pdf", "demo.mp4", "README.md", "requirements.txt",
            "sample_data/box.png", "sample_data/box.obj", "sample_data/sneaker.png",
            "sample_data/manifest.json", "code/app/main.py", "code/engine/run_recon.py",
            "code/training/pipeline.py", "code/training/evaluate_clouds.py",
            "code/web/public/index.html", "code/web/app/index.html",
            *{f"code/results/course17/{name}" for name in EVIDENCE_FILES},
        }
        require(expected <= {name.removeprefix(f"{ARCHIVE}/") for name in names}, "Missing required files.")

        reader = PdfReader(BytesIO(read("report.pdf")))
        require(len(reader.pages) == 4, "The course report must have four pages.")
        report = " ".join(page.extract_text() for page in reader.pages)
        require("Group 17" in report and "author contributions" in report.casefold(),
                "Missing report attribution.")

        mesh_runs = list(csv.DictReader(StringIO(read("code/results/raw/runs.csv").decode())))
        cohort = json.loads(read("code/results/dataset_metadata.json"))["scenes"]
        require(len(mesh_runs) == 3 * 3 * cohort["total"], "Recorded mesh benchmark is incomplete.")
        pointcloud_dir = "code/results/course17/"
        for suffix in ("", "_no_grad"):
            summary = json.loads(read(f"{pointcloud_dir}pointcloud{suffix}_summary.json"))
            rows = list(csv.DictReader(StringIO(read(f"{pointcloud_dir}pointcloud{suffix}_per_view.csv").decode())))
            require(len(rows) == summary["views"] == 8, "Recorded point-cloud run is incomplete.")
            for metric, score in summary["mean_by_mesh"].items():
                mean = sum(float(row[metric]) for row in rows) / len(rows)
                require(math.isclose(mean, score), f"Recorded per-view scores disagree on {metric}.")
                require(f"{score:.4f}" in report, f"Report is missing the recorded {metric} score.")

        with TemporaryDirectory(prefix="course-demo-verification-") as directory:
            video = Path(directory) / "demo.mp4"
            video.write_bytes(read("demo.mp4"))
            result = subprocess.run(
                [get_ffmpeg_exe(), "-v", "error", "-xerror", "-nostats", "-progress", "pipe:1",
                 "-i", str(video), "-map", "0:v:0", "-f", "null", "-"],
                check=True, capture_output=True, text=True,
            )
            timestamps = [int(line.partition("=")[2]) for line in result.stdout.splitlines()
                          if line.startswith("out_time_us=") and not line.endswith("=N/A")]
            require(timestamps and abs(timestamps[-1] / 1_000_000 - 84) < .25,
                    "The decoded course demo duration differs from 84 seconds.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    try:
        verify_archive(args.archive)
    except (OSError, ValueError, KeyError, BadZipFile, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Submission verification failed: {exc}\n")
    print("Verified archive, four-page report, benchmark rows, point-cloud scores, and 84-second demo.")


if __name__ == "__main__":
    main()
