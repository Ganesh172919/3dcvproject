"""Check the final course archive and its report, video, and recorded evidence."""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
from io import BytesIO, StringIO
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile

from imageio_ffmpeg import get_ffmpeg_exe
from pypdf import PdfReader

from .package import ARCHIVE, EVIDENCE_FILES, MAX_ARCHIVE_BYTES

REPORT_SECTIONS = (
    "i. introduction", "ii. problem statement", "iii. literature review", "iv. objectives",
    "v. methodology", "vi. results and discussion", "vii. conclusion", "references", "author contributions",
)
GROUP_IDS = ("23BDS024", "23BDS033", "23BDS011", "23BCS098", "23BDS075")


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
            "code/training/dataset_gen/split_dataset.py", "code/training/compare_cloud_runs.py",
            "code/web/public/index.html", "code/web/app/index.html",
            *{f"code/docs/handbook/{name}.md"
              for name in ("README", "architecture", "local-setup", "operations", "reproducibility")},
            *{f"code/results/course17/{name}" for name in EVIDENCE_FILES},
        }
        require(expected <= {name.removeprefix(f"{ARCHIVE}/") for name in names}, "Missing required files.")

        pdf = read("report.pdf")
        reader = PdfReader(BytesIO(pdf))
        require(4 <= len(reader.pages) <= 6, "The course report must have four to six pages.")
        preview = "code/web/public/report-preview/"
        preview_index = read(preview + "index.html").decode()
        require(sha256(pdf).hexdigest() in preview_index, "The browser preview is out of date.")
        for page in range(1, len(reader.pages) + 1):
            name = f"page-{page}.webp"
            require(name in preview_index and read(preview + name)[8:12] == b"WEBP",
                    f"Missing report preview page: {page}.")
        report = " ".join(page.extract_text() for page in reader.pages)
        first_page = reader.pages[0].extract_text()
        require("Group 17" in first_page and all(roll in first_page for roll in GROUP_IDS),
                "Missing Group 17 attribution in the author block.")
        # PDF extraction can put a space between the initial and the rest of a small-cap heading.
        normalized = re.sub(r"\s+", "", report.casefold())
        cursor = 0
        for heading in REPORT_SECTIONS:
            search = heading.replace(" ", "")
            position = normalized.find(search, cursor)
            require(position >= 0, f"Missing or out-of-order report section: {heading}.")
            cursor = position + len(search)

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
            require(timestamps and 60 <= timestamps[-1] / 1_000_000 <= 120,
                    "The decoded course demo must run for one to two minutes.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    try:
        verify_archive(args.archive)
    except (OSError, ValueError, KeyError, BadZipFile, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Submission verification failed: {exc}\n")
    print("Verified archive, report sections, benchmark rows, point-cloud scores, and demo playback length.")


if __name__ == "__main__":
    main()
