"""Build the Group 17 course ZIP from the source tree and recorded evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = "Group17_3dcv_project"
MAX_ARCHIVE_BYTES = 200_000_000
SOURCE_DIRS = (
    "app", "benchmarks", "docs", "engine", "paper", "scripts", "submission",
    "tests", "training", "web", "tools", "results/figures", "results/summary",
)
SKIP_DIRS = {
    "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules", "models",
    "checkpoints", "data", "out",
}
SOURCE_SUFFIXES = {
    ".bib", ".cls", ".css", ".csv", ".gif", ".glb", ".html", ".ini", ".js", ".json",
    ".md", ".png", ".ps1", ".py", ".sh", ".svg", ".tex", ".txt", ".webp",
    ".yaml", ".yml",
}
ROOT_FILES = (
    "AUTHORS.md", "LICENSE", "run.ps1", "run.sh", ".env.example", ".gitignore",
    "pytest.ini", "Makefile", "mkdocs.yml", "paper/main.pdf", "presentations/3DFUN-AI-Project.pptx",
    "results/environment.json",
    "results/dataset_metadata.json", "results/raw/runs.csv",
)
EVIDENCE_FILES = (
    "tinydepth.json", "tinydepth_no_grad.json", "geometry_calibration.json",
    "tinydepth_figure.png", "output_comparison.png", "pointcloud_summary.json",
    "pointcloud_per_view.csv", "pointcloud_no_grad_summary.json",
    "pointcloud_no_grad_per_view.csv", "duck_side.png",
)


def check_media(report: Path, demo: Path) -> None:
    with report.open("rb") as stream:
        if not stream.read(8).startswith(b"%PDF-"):
            raise ValueError(f"Expected a PDF report: {report}")
    with demo.open("rb") as stream:
        if stream.read(8)[4:] != b"ftyp":
            raise ValueError(f"Expected an MP4 video: {demo}")


def build_archive(report: Path, demo: Path, evidence: Path, output_dir: Path) -> Path:
    from benchmarks.make_dataset import write_synthetic_scene

    check_media(report, demo)
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / f"{ARCHIVE}.zip"
    with TemporaryDirectory(prefix="course-archive-", dir=output_dir) as directory:
        scratch = Path(directory)
        box_image, box_mesh = scratch / "box.png", scratch / "box.obj"
        write_synthetic_scene("box", box_image, box_mesh)
        draft = scratch / final.name
        with ZipFile(draft, "w", ZIP_DEFLATED, compresslevel=6) as archive:
            def put(path: Path, dest: str) -> None:
                archive.write(path, f"{ARCHIVE}/{dest}")

            put(report, "report.pdf")
            put(demo, "demo.mp4")
            put(ROOT / "submission/README.md", "README.md")
            archive.writestr(f"{ARCHIVE}/requirements.txt", "-r code/requirements.txt\n")

            for folder in SOURCE_DIRS:
                for path in sorted((ROOT / folder).rglob("*")):
                    rel = path.relative_to(ROOT)
                    if (path.is_file() and not path.is_symlink()
                            and not any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts)
                            and path.suffix.lower() in SOURCE_SUFFIXES):
                        put(path, f"code/{rel.as_posix()}")
            for path in sorted(ROOT.glob("requirements*.txt")):
                put(path, f"code/{path.name}")
            for name in ROOT_FILES:
                put(ROOT / name, f"code/{name}")
            put(ROOT / "README.md", "code/README.md")
            for name in EVIDENCE_FILES:
                put(evidence / name, f"code/results/course17/{name}")

            put(box_image, "sample_data/box.png")
            put(box_mesh, "sample_data/box.obj")
            put(ROOT / "engine/testdata/sneaker.png", "sample_data/sneaker.png")
            archive.writestr(f"{ARCHIVE}/sample_data/manifest.json", json.dumps(
                {"entries": [{"id": "synthetic/box", "image": "box.png", "gt_mesh": "box.obj"}]},
                indent=2,
            ))

        if draft.stat().st_size >= MAX_ARCHIVE_BYTES:
            raise ValueError(f"Archive exceeds the {MAX_ARCHIVE_BYTES // 1_000_000} MB limit.")
        with ZipFile(draft) as archive:
            names = archive.namelist()
            if len(names) != len({name.casefold() for name in names}) or archive.testzip() is not None:
                raise ValueError("Archive contains duplicate or corrupted entries.")
        draft.replace(final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "out/course17/report.pdf")
    parser.add_argument("--demo", type=Path, default=ROOT / "out/course17/demo.mp4")
    parser.add_argument("--evidence", type=Path, default=ROOT / "results/course17")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "out/course17")
    args = parser.parse_args()
    try:
        final = build_archive(args.report, args.demo, args.evidence, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Could not package the submission: {exc}\n")
    print(f"{final.resolve()} ({final.stat().st_size / 1_000_000:.2f} MB)")


if __name__ == "__main__":
    if not __package__:
        sys.path.insert(0, str(ROOT))
    main()
