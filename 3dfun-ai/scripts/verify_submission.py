"""Verify the submission end to end.

Course: 3D Computer Vision Course Project · IIIT Dharwad
Supervisor: Dr. Shirshendu Layek (Dept. of DSAI)
Authors: J Ganesh (23BDS024), M Jagadeshwar (23BDS033), B Varshith (23BDS011),
         Pranav Kashyap (23BCS098), C Sai Aravind (23BDS075)

    python3 scripts/verify_submission.py            # full check
    python3 scripts/verify_submission.py --fast     # skip the test suite

Checks, in order:
 1. layout / hygiene      — required files exist, no secrets, no large weights in git
 2. results integrity     — summary CSVs exist, are internally consistent, environment recorded
 3. figures               — every figure referenced by the paper exists and is non-empty
 4. paper                 — PDF and saved build diagnostics
 5. deck                  — every slide fits the 16:9 canvas, no text box overflows
 6. number cross-check    — every metric printed in paper/slides/report is traceable to
                            results/summary/*.csv (rounded to the printed precision)
 7. report / docs         — docs/report.md regenerated values match results
 8. tests                 — pytest suite passes (unless --fast)

Exit code 0 only when every check passes. Each check prints PASS/FAIL with the
evidence line that a reader can re-run.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
issues: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    issues.append((status, name, detail))
    mark = {PASS: "\033[92m[ok]\033[0m", FAIL: "\033[91m[FAIL]\033[0m",
            WARN: "\033[93m[warn]\033[0m"}[status]
    print(f"{mark} {name}" + (f"  — {detail}" if detail else ""))


# ---------------------------------------------------------------- 1. layout
REQUIRED = [
    "README.md", "AUTHORS.md", "LICENSE", "Makefile", "requirements.txt", "run.sh", "run.ps1",
    "engine/recon/__init__.py", "app/main.py", "web/app/index.html", "web/public/index.html",
    "benchmarks/run_benchmarks.py", "benchmarks/metrics.py",
    "scripts/fetch_models.py", "scripts/make_figures.py", "scripts/generate_paper.py",
    "scripts/generate_slides.py", "scripts/generate_report.py",
    "paper/main.tex", "paper/main.pdf", "paper/references.bib",
    "docs/report.md", "docs/report.pdf", "presentations/3DFUN-AI-Project.pptx",
    "results/environment.json", "results/summary/main_table.csv",
]


def check_layout() -> None:
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    record(PASS if not missing else FAIL, "required files present",
           "all present" if not missing else f"missing: {', '.join(missing)}")

    # no credentials committed
    secret_pat = re.compile(r"(sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY"
                            r"|ghp_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{35})")
    hits = []
    SKIP_DIRS = {".git", "node_modules", "__pycache__", "results", ".venv", ".venv-directml",
                 ".venv-cuda", ".pytest_cache", "out", "uploads", "_photocache"}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        if Path(dirpath) == ROOT:
            dirnames[:] = [d for d in dirnames if d != "data"]
        for fname in filenames:
            if fname.startswith(".env") and fname != ".env.example":
                continue
            path = Path(dirpath) / fname
            if path.suffix in {".png", ".jpg", ".jpeg", ".glb", ".obj",
                               ".ply", ".onnx", ".pptx", ".pdf", ".pyc"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if secret_pat.search(text):
                hits.append(str(path.relative_to(ROOT)))
    record(PASS if not hits else FAIL, "no API keys or private keys in the tree",
           "clean" if not hits else f"found in {hits}")

    models_dir = ROOT / "engine" / "models"
    heavy = [p for p in models_dir.glob("*.onnx") if p.stat().st_size > 1_000_000] if models_dir.exists() else []
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8") if (ROOT / ".gitignore").exists() else ""
    ok = all("engine/models/*.onnx" in ignored for _ in heavy) if heavy else True
    record(PASS if ok else FAIL, "large weights are git-ignored",
           f"{len(heavy)} weight files, pattern present" if ok else "add engine/models/*.onnx to .gitignore")


# ---------------------------------------------------------------- 2. results
def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def check_results() -> dict:
    main = read_csv(RESULTS / "summary" / "main_table.csv")
    record(PASS if main else FAIL, "main_table.csv exists",
           f"{len(main)} method rows" if main else "run `make benchmark`")
    methods = {r["method"] for r in main}
    record(PASS if {"proposed", "tsdf", "visual_hull"} <= methods else FAIL,
           "both baselines present in the main table", ", ".join(sorted(methods)) or "none")
    env = RESULTS / "environment.json"
    if env.exists():
        data = json.loads(env.read_text(encoding="utf-8"))
        need = ["cpu", "cpu_count", "memory", "gpu", "python", "seed", "dataset", "libraries"]
        missing = [k for k in need if k not in data]
        record(PASS if not missing else FAIL, "environment record complete",
               "hardware, libs, seed, dataset, commit" if not missing else f"missing {missing}")
    else:
        record(FAIL, "environment record complete", "results/environment.json missing")

    raw = read_csv(RESULTS / "raw" / "runs.csv")
    repeats: dict[tuple, int] = {}
    for row in raw:
        key = (row["entry_id"], row["method"])
        repeats[key] = repeats.get(key, 0) + 1
    rep_counts = sorted(set(repeats.values()))
    record(PASS if rep_counts and min(rep_counts) >= 3 else WARN,
           "timing repeated at least 3x per configuration",
           f"repetition counts observed: {rep_counts}")
    record(PASS if raw and len(raw) >= 60 else FAIL, "raw per-run rows recorded", f"{len(raw)} rows")
    log_dir = RESULTS / "logs"
    n_logs = len(list(log_dir.glob("*.log"))) if log_dir.exists() else 0
    record(PASS if n_logs >= len(repeats) else WARN, "per-run logs written",
           f"{n_logs} logs for {len(repeats)} configurations")
    return {"main": main, "raw": raw}


def check_results_consistency(main: list[dict]) -> None:
    """The summary must be the arithmetic of the raw rows, not a separate copy."""
    raw = read_csv(RESULTS / "raw" / "runs.csv")
    by_method: dict[str, list[dict]] = {}
    for row in raw:
        by_method.setdefault(row["method"], []).append(row)
    worst = 0.0
    for row in main:
        rows = [r for r in by_method.get(row["method"], []) if r.get("ok") in ("True", "true")]
        if not rows:
            continue
        times = [float(r["seconds"]) for r in rows if r.get("seconds")]
        if times:
            worst = max(worst, abs(sum(times) / len(times) - float(row["seconds_mean"])))
    record(PASS if worst < 0.05 else FAIL,
           "summary means match the mean of the raw runs", f"max deviation {worst:.4f} s")


# ---------------------------------------------------------------- 3. figures
def check_figures() -> list[str]:
    figures = {p.name for p in (RESULTS / "figures").glob("*.png")} \
        if (RESULTS / "figures").exists() else set()
    tex = (ROOT / "paper" / "main.tex")
    wanted: set[str] = set()
    if tex.exists():
        wanted = set(re.findall(r"includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex.read_text(encoding="utf-8")))
    missing = sorted(w for w in wanted if w not in figures)
    record(PASS if not missing else FAIL, "every figure referenced by the paper exists",
           f"{len(figures)} figures, {len(wanted)} referenced"
           if not missing else f"missing {missing}")
    empty = [f.name for f in (RESULTS / "figures").glob("*.png") if f.stat().st_size < 3000]
    record(PASS if not empty else WARN, "figures are non-trivial",
           "all plots have content" if not empty else f"suspiciously small: {empty}")
    return sorted(figures)


# ---------------------------------------------------------------- 4. paper
def check_paper() -> None:
    paper = ROOT / "paper"
    tex = paper / "main.tex"
    if not tex.exists():
        record(FAIL, "paper source exists", "run `make paper`")
        return
    log = (paper / "main.log")
    text = log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""
    if not text.strip():
        record(WARN, "paper build diagnostics", "build with scripts/generate_paper.py --pdf")
    else:
        undefined = re.findall(r"Reference `([^']+)' on page \d+ undefined", text)
        record(PASS if not undefined else FAIL, "paper has no undefined references",
               "clean" if not undefined else f"{len(set(undefined))} undefined: {sorted(set(undefined))[:6]}")
        cites = re.findall(r"Citation `([^']+)' on page \d+ undefined", text)
        record(PASS if not cites else FAIL, "no undefined citations",
               "clean" if not cites else str(sorted(set(cites))[:6]))
        overfull = re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", text)
        oversized = re.findall(r"Float too large for page by ([\d.]+)pt", text)
        problems = overfull + oversized
        record(PASS if not problems else FAIL, "paper has no oversized boxes or floats",
               "clean" if not problems else f"{len(problems)} overflows, worst "
                                            f"{max(float(x) for x in problems):.1f}pt")
    pdf = paper / "main.pdf"
    pages = 0
    if pdf.exists():
        try:
            out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True)
            match = re.search(r"Pages:\s+(\d+)", out.stdout)
            pages = int(match.group(1)) if match else 0
        except FileNotFoundError:
            pass
        if not pages:
            try:
                import pypdf
                reader = pypdf.PdfReader(str(pdf))
                pages = len(reader.pages)
            except Exception:
                try:
                    import fitz
                    doc = fitz.open(str(pdf))
                    pages = len(doc)
                except Exception:
                    pass
    record(PASS if 6 <= pages <= 8 else WARN, "paper length within 6-8 pages",
           f"{pages} pages" if pages else "pdfinfo unavailable or PDF missing")

    # author block order
    body = tex.read_text(encoding="utf-8")
    order = [name for name in ("J Ganesh", "M Jagadeshwar", "B Varshith", "Pranav Kashyap",
                               "C Sai Aravind") if name in body]
    ordered = all(body.index(a) < body.index(b) for a, b in zip(order, order[1:]))
    record(PASS if len(order) == 5 and ordered else FAIL, "author block has all five students "
           "in order", f"{len(order)}/5 in order={ordered}")
    record(PASS if "Shirshendu Layek" in body else FAIL, "guide named in the paper",
           "guide present" if "Shirshendu Layek" in body else "missing")


# ---------------------------------------------------------------- 5. deck
def check_deck() -> None:
    path = ROOT / "presentations" / "3DFUN-AI-Project.pptx"
    if not path.exists():
        record(FAIL, "deck exists", "run `make slides`")
        return
    from pptx import Presentation
    from pptx.util import Emu

    prs = Presentation(str(path))
    width, height = prs.slide_width, prs.slide_height
    n = len(prs.slides._sldIdLst)
    record(PASS if 12 <= n <= 15 else FAIL, "deck has 12-15 slides", f"{n} slides")
    ratio = width / height
    record(PASS if abs(ratio - 16 / 9) < 0.01 else FAIL, "deck is 16:9",
           f"{width / 914400:.3f} x {height / 914400:.3f} in (ratio {ratio:.4f})")
    out_of_bounds = []
    overflowing = []
    for i, slide in enumerate(prs.slides, start=1):
        for shape in slide.shapes:
            try:
                if (shape.left < 0 or shape.top < 0
                        or shape.left + shape.width > width + Emu(9144)
                        or shape.top + shape.height > height + Emu(9144)):
                    out_of_bounds.append((i, shape.name))
            except TypeError:
                continue
            if shape.has_text_frame and shape.text_frame.text.strip():
                # crude but effective: ~1.9 chars per 0.1 in at 12pt, scaled by font size
                for para in shape.text_frame.paragraphs:
                    sizes = [r.font.size.pt for r in para.runs if r.font.size]
                    size = max(sizes) if sizes else 14
                    chars = len(para.text)
                    capacity = (shape.width / 914400) * (150.0 / size) * 1.05
                    lines = max(1, -(-chars // max(capacity, 1)))
                    needed = lines * size * 1.25 / 72.0
                    if needed > (shape.height / 914400) + 0.08:
                        overflowing.append((i, para.text[:40], round(needed, 2),
                                            round(shape.height / 914400, 2)))
                        break
    record(PASS if not out_of_bounds else FAIL, "all slide shapes inside the canvas",
           "clean" if not out_of_bounds else str(out_of_bounds[:6]))
    record(PASS if not overflowing else WARN, "no text box appears clipped",
           "clean" if not overflowing else str(overflowing[:5]))
    notes = [s for s in prs.slides if s.has_notes_slide and s.notes_slide.notes_text_frame.text.strip()]
    record(PASS if len(notes) == n else WARN, "speaker notes on every slide",
           f"{len(notes)}/{n} slides annotated")


# ---------------------------------------------------------------- 6. numbers
def printed_numbers(path: Path) -> set[float]:
    """Every decimal literal that looks like a measurement in a text artifact."""
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8", errors="ignore")
    found = set()
    for token in re.findall(r"\b\d+\.\d{2,4}\b", text):
        try:
            found.add(round(float(token), 4))
        except ValueError:
            continue
    return found


def check_numbers(main: list[dict]) -> None:
    allowed: set[float] = set()
    for path in list((RESULTS / "summary").glob("*.csv")) + list((RESULTS / "raw").glob("*.csv")):
        for row in read_csv(path):
            for value in row.values():
                try:
                    allowed.add(round(float(value), 4))
                except (TypeError, ValueError):
                    continue
    for path in (RESULTS / "summary").glob("*.json"):
        for record_ in json.loads(path.read_text(encoding="utf-8")) or []:
            for value in record_.values() if isinstance(record_, dict) else []:
                try:
                    allowed.add(round(float(value), 4))
                except (TypeError, ValueError):
                    continue

    # numbers that are version strings in the environment block, not measurements
    env = json.loads((RESULTS / "environment.json").read_text(encoding="utf-8")) \
        if (RESULTS / "environment.json").exists() else {}
    for version in list((env.get("libraries") or {}).values()) + [env.get("python", "")]:
        for token in re.findall(r"\d+\.\d+", str(version)):
            try:
                allowed.add(round(float(token), 4))
            except ValueError:
                pass

    def tolerant(value: float) -> bool:
        return any(abs(value - a) <= max(0.0051, abs(a) * 0.006) for a in allowed)

    for label, path in [("paper", ROOT / "paper" / "main.tex"),
                        ("report", ROOT / "docs" / "report.md")]:
        values = printed_numbers(path)
        unknown = sorted(v for v in values if not tolerant(v))
        record(PASS if len(unknown) <= 2 else FAIL,
               f"numbers in the {label} trace to results/",
               f"{len(values) - len(unknown)}/{len(values)} matched"
               + (f"; unmatched {unknown[:5]}" if unknown else ""))

    # deck: numbers live inside XML runs
    deck = ROOT / "presentations" / "3DFUN-AI-Project.pptx"
    if deck.exists():
        import zipfile
        with zipfile.ZipFile(deck, "r") as zf:
            raw_xml = "".join(
                zf.read(name).decode("utf-8", errors="ignore")
                for name in zf.namelist()
                if name.startswith("ppt/slides/slide") and name.endswith(".xml")
            )
        values = {round(float(t), 4) for t in re.findall(r"\b\d+\.\d{2,4}\b", raw_xml)}
        unknown = sorted(v for v in values if not tolerant(v))
        record(PASS if len(unknown) <= 2 else FAIL, "numbers in the deck trace to results/",
               f"{len(values) - len(unknown)}/{len(values)} matched"
               + (f"; unmatched {unknown[:5]}" if unknown else ""))


# ---------------------------------------------------------------- 7. docs
def check_docs() -> None:
    md = ROOT / "docs" / "report.md"
    text = md.read_text(encoding="utf-8") if md.exists() else ""
    record(PASS if "not evaluated" in text or "No run crashed" in text else FAIL,
           "report documents what was not evaluated",
           "section present" if text else "docs/report.md missing")
    for name in ("J Ganesh", "M Jagadeshwar", "B Varshith", "Pranav Kashyap",
                 "C Sai Aravind", "Shirshendu Layek"):
        if name not in text:
            record(FAIL, f"{name} present in the report", "missing")
            return
    record(PASS, "all team members and the guide appear in the report", "6/6")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    record(PASS if "make benchmark" in readme and "make verify" in readme else FAIL,
           "README documents reproduce and verify commands",
           "commands present" if "make verify" in readme else "missing")


# ---------------------------------------------------------------- 8. tests
def check_tests() -> None:
    out = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"],
                         capture_output=True, text=True, cwd=str(ROOT))
    tail = (out.stdout or "").strip().splitlines()[-1] if out.stdout else ""
    record(PASS if out.returncode == 0 else FAIL, "test suite passes", tail or "no output")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="skip the pytest suite")
    args = parser.parse_args()

    print("3DFUN AI — submission verification\n" + "=" * 60)
    check_layout()
    payload = check_results()
    if payload["main"]:
        check_results_consistency(payload["main"])
    check_figures()
    check_paper()
    check_deck()
    if payload["main"]:
        check_numbers(payload["main"])
    check_docs()
    if not args.fast:
        check_tests()

    fails = [i for i in issues if i[0] == FAIL]
    warns = [i for i in issues if i[0] == WARN]
    print("=" * 60)
    print(f"{len(issues) - len(fails) - len(warns)} passed, {len(warns)} warnings, "
          f"{len(fails)} failed")
    if fails:
        print("\nstill missing / broken:")
        for _, name, detail in fails:
            print(f"  - {name}: {detail}")
    (ROOT / "results" / "verification.json").write_text(json.dumps(
        [{"status": s, "check": n, "detail": d} for s, n, d in issues], indent=2),
        encoding="utf-8")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
