"""Bundle the local source, benchmark evidence and academic deliverables."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "out" / "submission" / "3DFUN-AI-submission.zip"
DELIVERABLES = {"paper/main.pdf", "docs/report.pdf", "presentations/3DFUN-AI-Project.pptx"}
EXCLUDED_DIRS = {".git", "__pycache__", "node_modules"}
SECRETS = re.compile(
    rb"(?:AIza[0-9A-Za-z_-]{35}|sk-[0-9A-Za-z_-]{20,}|AKIA[A-Z0-9]{16}"
    rb"|ghp_[0-9A-Za-z]{20,}|BEGIN [A-Z ]*PRIVATE KEY)"
)
TEXT_TYPES = {".bib", ".csv", ".css", ".html", ".js", ".json", ".jsonl", ".log", ".md",
              ".ps1", ".py", ".sh", ".tex", ".txt", ".yaml", ".yml"}


def source_paths() -> set[str]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT
    )
    candidates = output.decode("utf-8").split("\0")
    paths = set()
    for name in filter(None, candidates):
        path = Path(name)
        if path.parts[0] in {"data", "out", "uploads"} or EXCLUDED_DIRS.intersection(path.parts):
            continue
        if any(part.startswith(".venv") for part in path.parts):
            continue
        if path.name.startswith(".env") and path.name != ".env.example":
            continue
        if path.suffix.lower() in {".onnx", ".sqlite", ".sqlite3", ".db", ".pyc"}:
            continue
        if name.startswith(("results/raw/meshes/", "benchmarks/data/_photocache/")):
            continue
        paths.add(name)
    return paths


def dataset_paths() -> set[str]:
    base = ROOT / "benchmarks" / "data"
    manifest = json.loads((base / "dataset.json").read_text(encoding="utf-8"))
    paths = {"benchmarks/data/dataset.json", "benchmarks/data/ATTRIBUTION.md"}
    for entry in manifest["entries"]:
        for key in ("image", "gt_mesh"):
            if not (name := entry.get(key)):
                continue
            path = (base / name).resolve().relative_to(base)
            paths.add(f"benchmarks/data/{path.as_posix()}")
    return paths


def package(destination: Path) -> None:
    verification = json.loads((ROOT / "results/verification.json").read_text(encoding="utf-8"))
    if any(item["status"] == "FAIL" for item in verification) or not any(
        item["status"] == "PASS" and item["check"] == "test suite passes" for item in verification
    ):
        raise SystemExit("Run a passing full submission verification before packaging.")

    required = DELIVERABLES | dataset_paths() | {"paper/main.log", "results/verification.json"}
    names = source_paths() | required
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, suffix=".zip", delete=False) as tmp:
        staging = Path(tmp.name)
    try:
        with ZipFile(staging, "w", ZIP_DEFLATED) as archive:
            for name in sorted(names):
                path = ROOT / name
                if not path.is_file() and name not in required:
                    continue  # a tracked file deleted from the working tree
                if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
                    raise ValueError(f"Unexpected path outside the project: {name}")
                if path.suffix.lower() in TEXT_TYPES or path.name == ".env.example":
                    if SECRETS.search(path.read_bytes()):
                        raise ValueError(f"Possible credential in {name}")
                info = f"3DFUN-AI/{name}"
                archive.write(path, info)
        with ZipFile(staging) as archive:
            if invalid := archive.testzip():
                raise ValueError(f"Corrupt archive entry: {invalid}")
        staging.replace(destination)
    finally:
        staging.unlink(missing_ok=True)
    with destination.open("rb") as file:
        checksum = hashlib.file_digest(file, "sha256").hexdigest()
    destination.with_suffix(".sha256").write_text(f"{checksum}  {destination.name}\n", encoding="ascii")
    print(f"{destination} ({destination.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    package(args.output.resolve())


if __name__ == "__main__":
    main()
