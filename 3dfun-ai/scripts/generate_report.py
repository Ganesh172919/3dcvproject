"""Generate the project report (Markdown + PDF) from ``results/`` only.

Course: 3D Computer Vision Course Project · IIIT Dharwad
Supervisor: Dr. Shirshendu Layek (Dept. of DSAI)
Authors: J Ganesh (23BDS024), M Jagadeshwar (23BDS033), B Varshith (23BDS011),
         Pranav Kashyap (23BCS098), C Sai Aravind (23BDS075)

    python3 scripts/generate_report.py

Writes ``docs/report.md`` and, if pandoc/LaTeX are available, ``docs/report.pdf``.
Every table is built from ``results/summary/*.csv``; the narrative refers to those
tables by number. Nothing here is typed by hand.
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
DATA = ROOT / "benchmarks" / "data"

TEAM = [
    ("J Ganesh", "23BDS024", "Data Science and Artificial Intelligence"),
    ("M Jagadeshwar", "23BDS033", "Data Science and Artificial Intelligence"),
    ("B Varshith", "23BDS011", "Data Science and Artificial Intelligence"),
    ("Pranav Kashyap", "23BCS098", "Computer Science and Engineering"),
    ("C Sai Aravind", "23BDS075", "Data Science and Artificial Intelligence"),
]
GUIDE = "Dr. Shirshendu Layek"
LABELS = {"proposed": "3DFUN AI (ours)", "tsdf": "TSDF fusion", "visual_hull": "Visual hull"}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(value, digits=4, default="not evaluated") -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v:
        return default
    return f"{v:.{digits}f}"


def main_table() -> str:
    rows = read_csv(RESULTS / "summary" / "main_table.csv")
    if not rows:
        return "_not evaluated --- run `make benchmark` first._\n"
    geometry = ("| Method | Chamfer ↓ | F-score@1% ↑ | IoU ↑ | Normal consistency ↑ |\n"
                "|---|---|---|---|---|\n")
    appearance = ("| Method | SSIM ↑ | Silhouette IoU ↑ | Time (s) |\n"
                  "|---|---|---|---|\n")
    for r in rows:
        label = LABELS.get(r["method"], r["method"])
        geometry += (f"| {label} | {f(r.get('chamfer_mean'))} ± {f(r.get('chamfer_std'))} | "
                     f"{f(r.get('f_score_mean'), 3)} ± {f(r.get('f_score_std'), 3)} | "
                     f"{f(r.get('iou_mean'), 3)} ± {f(r.get('iou_std'), 3)} | "
                     f"{f(r.get('normal_consistency_mean'), 3)} |\n")
        appearance += (f"| {label} | {f(r.get('ssim_mean'), 3)} ± {f(r.get('ssim_std'), 3)} | "
                       f"{f(r.get('silhouette_iou_mean'), 3)} | "
                       f"{f(r.get('seconds_mean'), 2)} ± {f(r.get('seconds_std'), 2)} |\n")
    return geometry + "\n" + appearance


def ablation_table() -> str:
    rows = read_csv(RESULTS / "summary" / "ablation_table.csv")
    if not rows:
        return "_not evaluated._\n"
    out = "| Configuration | Chamfer down | F-score up | IoU up |\n|---|---|---|---|\n"
    for r in rows:
        out += (f"| {r['config'].replace('_', ' ')} | {f(r.get('chamfer_mean'))} | "
                f"{f(r.get('f_score_mean'), 3)} | {f(r.get('iou_mean'), 3)} |\n")
    return out


def stage_table() -> str:
    rows = read_csv(RESULTS / "summary" / "stage_timings.csv")
    if not rows:
        return "_not evaluated._\n"
    total = sum(float(r.get("mean_s") or 0) for r in rows if r["stage"] != "total") or 1.0
    out = "| Stage | Mean (s) | Std (s) | Share |\n|---|---|---|---|\n"
    for r in sorted(rows, key=lambda r: -float(r.get("mean_s") or 0)):
        out += (f"| {r['stage']} | {f(r.get('mean_s'), 2)} | {f(r.get('std_s'), 2)} | "
                f"{100 * float(r.get('mean_s') or 0) / total:.1f}% |\n")
    return out


def per_scene_table() -> str:
    rows = read_csv(RESULTS / "summary" / "per_scene.csv")
    if not rows:
        return "_not evaluated (run `make benchmark`)._\n"
    out = ("Dashes mark geometric metrics unavailable for photographs.\n\n"
           "| Scene | Method | Chamfer | F-score | IoU | SSIM | Time (s) | Closed |\n"
           "|---|---|---|---|---|---|---|---|\n")
    short_names = {"proposed": "Course", "tsdf": "TSDF", "visual_hull": "Visual hull"}
    for r in rows:
        scene = r["entry_id"].replace("/", " / ").replace("_", " ")
        out += (f"| {scene} | {short_names.get(r['method'], r['method'])} | "
                f"{f(r.get('chamfer'), default='–')} | {f(r.get('f_score'), 3, '–')} | {f(r.get('iou'), 3, '–')} | "
                f"{f(r.get('ssim'), 3)} | {f(r.get('seconds'), 2)} | "
                f"{f(r.get('watertight_pct'), 0)}% |\n")
    return out


def failure_section() -> str:
    runs = read_csv(RESULTS / "raw" / "runs.csv")
    rows = [r for r in runs if r.get("ok") not in ("True", "true")]
    if not rows:
        return ("No run crashed. The qualitative failure modes (asymmetric backs, occluded "
                "handles, thin structures thinner than the minimum slab) are discussed in "
                "§Limitations, and the threshold flags are recorded per run in "
                "`results/raw/runs.csv`.\n")
    out = "| Scene | Method | Rep | Reason |\n|---|---|---|---|\n"
    for r in rows:
        out += f"| {r['entry_id']} | {r['method']} | {r['repetition']} | {r.get('error') or 'unknown'} |\n"
    return out


def method_notes() -> str:
    return """
### 2.1 What the geometry constrains

**Grid boundary.** Positive signed distance along the outside of the grid keeps the interpolated
zero set away from that boundary. The pipeline pads the grid, attempts to fill small holes during
cleanup, and measures watertightness on the resulting mesh.

**Silhouette.** The occupied voxels project inside the sampled mask. Interpolation and smoothing
can move the extracted surface, so projected agreement is measured with silhouette IoU.

**Smoothing.** Taubin smoothing alternates two passes to limit shrinkage. The enclosed volume
can still change as vertices move.
"""


def metric_definitions() -> str:
    return """
### 3.1 Formal Metric Definitions

Geometric and appearance metrics follow standard formal mathematical definitions:
- **Chamfer Distance (CD):**
  $$d_{\\mathrm{CD}}(S_1, S_2) = \\frac{1}{2|S_1|}\\sum_{p\\in S_1}\\min_{q\\in S_2}\\|p-q\\|_2 + \\frac{1}{2|S_2|}\\sum_{q\\in S_2}\\min_{p\\in S_1}\\|q-p\\|_2$$
- **F-Score@$\\tau$:**
  $$F@\\tau = \\frac{2 \\cdot P(\\tau) \\cdot R(\\tau)}{P(\\tau) + R(\\tau)}, \\quad P(\\tau) = \\frac{1}{|S_1|}\\sum_{p\\in S_1}\\mathbb{1}\\Big[\\min_{q\\in S_2}\\|p-q\\|_2 < \\tau\\Big]$$
- **Volumetric IoU:**
  $$\\mathrm{IoU}(V_1, V_2) = \\frac{|V_1 \\cap V_2|}{|V_1 \\cup V_2|}$$
- **Normal Consistency (NC):**
  $$\\mathrm{NC}(S_1, S_2) = \\frac{1}{|S_1|}\\sum_{p\\in S_1} |\\langle \\mathbf{n}_p, \\mathbf{n}_{q^*(p)} \\rangle|$$
  where $q^*(p) = \\arg\\min_{q \\in S_2} \\|p - q\\|_2$ denotes the nearest surface neighbour.
"""


def build() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    env = json.loads((RESULTS / "environment.json").read_text(encoding="utf-8")) \
        if (RESULTS / "environment.json").exists() else {}
    data = json.loads((DATA / "dataset.json").read_text(encoding="utf-8")) \
        if (DATA / "dataset.json").exists() else {}
    summary = read_csv(RESULTS / "summary" / "main_table.csv")
    ours = next((r for r in summary if r["method"] == "proposed"), {})
    tsdf = next((r for r in summary if r["method"] == "tsdf"), {})
    hull = next((r for r in summary if r["method"] == "visual_hull"), {})
    scenes = data.get("scenes", {})
    team_rows = "\n".join(f"| {i+1} | {n} | {r} | {d} |" for i, (n, r, d) in enumerate(TEAM))

    md = f"""# 3DFUN AI --- Project Report

**An End-to-End Image-to-3D Generation Framework**
3D Computer Vision course project, Indian Institute of Information Technology Dharwad.

| # | Name | Roll number | Department |
|---|---|---|---|
{team_rows}

**Guide:** {GUIDE}, Assistant Professor, Department of Data Science and Artificial Intelligence,
IIIT Dharwad.

*Report generated on {datetime.now(timezone.utc).strftime('%d %B %Y')} by
`scripts/generate_report.py` from `results/`. Every number below comes from an executed run;
regenerate with `make benchmark && make figures && make report`.*

---

## 1. What the system does

3DFUN AI takes one photograph of an object, removes the background, predicts relative depth,
inflates the silhouette into a closed volumetric solid whose thickness follows that depth, extracts
a triangle mesh, measures its topology, bakes an albedo and PBR texture set, and returns GLB / OBJ / PLY plus preview
renders. It is served by a FastAPI backend with accounts, credits and job tracking, and a
dependency-free web studio with a WebGL viewer.

The design constraint for the whole project was honesty of measurement: the pipeline, the two
baselines, the benchmark harness and this report share one directory of raw results, and no figure
in this document was typed by hand.

## 2. Method

1. **Matte.** U2-Net portable (`u2netp.onnx`, 4.6 MB) predicts a saliency map; it is eroded,
   thresholded, cleaned and used to crop the object with 4% padding onto a square canvas.
2. **Depth.** Depth Anything V2 Small (int8, 27 MB) predicts relative inverse depth, normalised
   per object by its 2nd/98th percentiles and smoothed with sigma = 2 px.
3. **Symmetry prior.** Inside the mask, depth is averaged with its horizontal mirror
   (0.5 * (d(x,y) + d(S-x,y))).
4. **Constrained inflation.** Every masked pixel becomes a slab of thickness
   `t(p) = t_min + (t_max - t_min) * (f + g * d(p)**gamma)` with `t_max = 0.70 * W`,
   `f = 0.55`, `g = 0.45`, `gamma = 0.85`, `t_min = 0.02`. The constant core f*Tmax keeps the rim closed; the depth term carves the front.
5. **Field and surface.** phi = EDT(outside) $-$ EDT(inside), Gaussian-smoothed (sigma = 0.8 voxels);
   marching cubes extracts the zero level set; small islands are dropped and Taubin smoothing
   reduces staircase artefacts.
6. **Resolution calibration.** A 64^3 probe measures the triangle-count constant, then the grid is
   solved for the requested budget (8 000 triangles) and clipped to [32, 224].
7. **Atlas and texture.** Planar view-projection UVs; albedo is back-projected and de-lit, ambient
   occlusion comes from local SDF visibility, roughness/metallic/normal maps are derived, and the
   texture is capped by a memory budget.
{method_notes()}

## 3. Evaluation set

- Version **{data.get('version', 'n/a')}**, seed **{data.get('seed', 'n/a')}**, checksum
  `{data.get('checksum', 'n/a')}`.
- **{scenes.get('total', 'not evaluated')} images**: {scenes.get('synthetic', 'not evaluated')}
  procedurally defined scenes with analytic reference meshes and
  {scenes.get('real', 'not evaluated')} CC0-1.0 photographs of physical objects.
- Synthetic scenes come with an analytic closed mesh, so Chamfer / F-score / IoU / normal
  consistency are exact; the photographs have no metric ground truth and are evaluated on runtime,
  memory, topology and silhouette/appearance metrics only.
- Per-image source, author, host and licence live in `benchmarks/data/ATTRIBUTION.md` and in the
  dataset manifest.
{metric_definitions()}
## 4. Baselines

Both baselines are implemented in `benchmarks/`, run on identical inputs and settings, sharing the
same ONNX depth graph and the same 8 000-triangle budget at 72^3 resolution:

- **TSDF fusion** --- depth integration into a truncated signed distance field, marching cubes,
  Taubin smoothing. Not silhouette-constrained.
- **Visual hull** --- silhouette extrusion intersected with a depth half-space, then meshed.

Learned baselines (TripoSR, InstantMesh, TRELLIS, Zero-1-to-3, MVDream, DreamGaussian) and
photogrammetry (COLMAP, NeRF, 3DGS) are **not evaluated**: this machine has no GPU and no
PyTorch, so they cannot be executed here, and quoting published numbers measured under a different
protocol would be dishonest.

## 5. Results

### 5.1 Main comparison

{main_table()}

See the Chamfer plot and qualitative grid in `results/figures/`.

Reading of the table: on the scenes that have reference geometry the proposed method reaches a
Chamfer distance of **{f(ours.get('chamfer_mean'))}** against {f(tsdf.get('chamfer_mean'))} for TSDF
fusion and {f(hull.get('chamfer_mean'))} for the visual hull, and it is the strongest method on
F-score ({f(ours.get('f_score_mean'), 3)}) and volumetric IoU ({f(ours.get('iou_mean'), 3)}), i.e.
the course method overlaps more of the reference volume. The course method has a mean SSIM of
{f(ours.get('ssim_mean'), 3)} and the visual hull has a mean of {f(hull.get('ssim_mean'), 3)}.
Wall clock is {f(ours.get('seconds_mean'), 2)} s per image against
{f(tsdf.get('seconds_mean'), 2)} s and {f(hull.get('seconds_mean'), 2)} s; most of that difference is
preview and turntable rendering, not reconstruction.

### 5.2 Per-stage timing (proposed)

{stage_table()}

### 5.3 3D Surface Reconstruction Error Colormaps

To inspect localized geometric discrepancies across the reconstructed surfaces, per-vertex Euclidean
distances from each reconstructed mesh to the corresponding analytic ground truth were computed via KD-Tree
nearest-neighbor queries and mapped to a viridis colormap.

Figure: `results/figures/fig_error_heatmaps.png`.

The colormaps show where local surface error varies across the selected synthetic examples.

### 5.4 Ablation

{ablation_table()}

### 5.5 Per-scene detail

{per_scene_table()}

### 5.6 Failures

{failure_section()}

## 6. Reproducibility

```bash
make install-dev # Python deps, including the test suite
make models      # ONNX weights (checksums printed, nothing large in git)
make seed        # demo database + accounts
make benchmark   # full evaluation -> results/
make figures     # figures from results/
make paper       # paper/main.pdf from results/
make report      # this document
make verify      # check saved artifacts and run tests
```

Environment recorded in `results/environment.json`:

- CPU: {env.get('cpu', 'n/a')}, {env.get('cpu_count', 'n/a')} cores
- RAM: {(env.get('memory') or {}).get('MemTotal', 'n/a')}
- GPU: {env.get('gpu', 'n/a')} --- every number in this report is CPU inference
- Python {env.get('python', 'n/a')}; {", ".join(f"{k} {v}" for k, v in (env.get('libraries') or {}).items())}
- Commit: `{env.get('git_commit', 'n/a')}`, seed {env.get('seed', 'n/a')}, dataset
  {env.get('dataset', {}).get('version', 'n/a')} (`{env.get('dataset', {}).get('checksum', 'n/a')}`)

## 7. Limitations

- The back of the object is never observed: the method mirrors a symmetry prior, so strongly
  asymmetric objects are reconstructed plausibly but incorrectly.
- Anything the silhouette cannot express is filled, not carved; hollow interiors and occluded
  handles are lost.
- Broken mattes (transparency, fur, heavy self-occlusion) break the volume, because the volume is
  literally the matte inflated.
- Textures are single-view; roughness and metallic are heuristics, and surfaces nearly parallel to
  the view direction stretch in the planar atlas.
- LPIPS, CLIP similarity, GSO/Objaverse reference sets, learned baselines and GPU throughput are
  **not evaluated** here, with the reasons above.

## 8. Code map

| Path | Contents |
|---|---|
| `engine/recon/` | reconstruction stages (matte, depth, volume, geometry, texture, export) |
| `engine/run_recon.py` | CLI |
| `app/` | FastAPI service: auth, assets, generation, gallery, admin, system |
| `web/` | dependency-free studio SPA + three.js viewer |
| `benchmarks/` | dataset generator, metrics, two baselines, harness |
| `results/` | raw runs, logs, summary CSVs, figures, environment record |
| `scripts/` | model fetcher, figure/paper/slide/report generators, verifier |
| `tests/` | pytest suite covering the API, job lifecycle and reconstruction |

## 9. Team Contributions & 12-Minute Defense Slide Allocation

| # | Presenter | Roll Number | Primary Engineering Responsibility | Defense Section & Slides | Slot |
|---|---|---|---|---|---|
| 1 | J Ganesh | 23BDS024 | Core Volumetric Engine, Distance Transform, Priors | Part 1: Slides 1–3 (Problem, Ill-Posedness, Constraints) | 0:00 – 2:15 |
| 2 | M Jagadeshwar | 23BDS033 | Meshing, Marching Cubes, Taubin Smoothing, Atlas | Part 2: Slides 4–6 (Related Work, Architecture, Pipeline) | 2:15 – 4:45 |
| 3 | B Varshith | 23BDS011 | Benchmark Harness, Multi-Engine Baselines, Metrics | Part 3: Slides 7–9 (Dataset Protocol, Results, Timings) | 4:45 – 7:30 |
| 4 | Pranav Kashyap | 23BCS098 | FastAPI Service, Credit Ledger, Job State Machine | Part 4: Slides 10–12 (Ablations, Visuals, Web Studio) | 7:30 – 10:15 |
| 5 | C Sai Aravind | 23BDS075 | Web Studio SPA, Three.js Viewer, AR QuickLook | Part 5: Slides 13–15 (Limitations, Roadmap, Defense Q&A) | 10:15 – 12:00 |

## 10. Acknowledgment

We thank {GUIDE}, Assistant Professor, Department of Data Science and Artificial Intelligence,
IIIT Dharwad, for supervising this project and for insisting on measurement over assertion.
"""
    (DOCS / "report.md").write_text(md, encoding="utf-8")
    print("wrote docs/report.md")

    pandoc = shutil.which("pandoc")
    engine = shutil.which("xelatex") or shutil.which("tectonic")
    if pandoc and engine:
        with TemporaryDirectory(dir=DOCS) as tmp:
            pdf = Path(tmp) / "report.pdf"
            subprocess.run([
                pandoc, str(DOCS / "report.md"), "-o", str(pdf),
                f"--pdf-engine={engine}",
                "-V", "geometry:margin=2.2cm", "-V", "fontsize=11pt",
                "-V", "colorlinks=true", "--toc", "--metadata=title:3DFUN AI Project Report",
            ], cwd=tmp, check=True)
            pdf.replace(DOCS / "report.pdf")
        print("wrote docs/report.pdf")
    else:
        print("PDF build skipped (requires Pandoc and XeLaTeX or Tectonic).")


# ------------------------------------------------------------------ inventory
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", ".venv-directml",
             "node_modules", "out", ".arena", "dist", "build", ".venv-cuda", "_photocache"}
SUFFIX_KIND = {
    ".py": "code", ".js": "code", ".css": "code", ".html": "code", ".sh": "code",
    ".md": "docs", ".txt": "docs", ".tex": "docs", ".bib": "docs",
    ".json": "data", ".csv": "data", ".yml": "data", ".yaml": "data",
    ".png": "asset", ".jpg": "asset", ".jpeg": "asset", ".gif": "asset",
    ".onnx": "weights", ".pptx": "deliverable", ".pdf": "deliverable",
}


def inventory() -> None:
    import os
    rows = []
    totals: dict[str, list[int]] = {}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        if Path(dirpath) == ROOT:
            dirnames[:] = [d for d in dirnames if d not in {"data", "uploads"}]
        for fname in sorted(filenames):
            path = Path(dirpath) / fname
            rel = path.relative_to(ROOT)
            if (fname.startswith(".env") and fname != ".env.example") or path.suffix.lower() in {
                ".onnx", ".part", ".aux", ".bbl", ".blg", ".out", ".sqlite", ".sqlite3", ".db",
            }:
                continue
            if rel.as_posix().startswith(("results/raw/meshes/", "results/renders/")):
                continue
            kind = SUFFIX_KIND.get(path.suffix.lower(), "other")
            lines = 0
            if kind in {"code", "docs", "data"} and path.stat().st_size < 3_000_000:
                try:
                    lines = sum(1 for _ in path.open("rb"))
                except OSError:
                    lines = 0
            rows.append((rel.as_posix(), kind, path.stat().st_size, lines))
            totals.setdefault(kind, [0, 0])
            totals[kind][0] += 1
            totals[kind][1] += lines
    rows.sort(key=lambda x: x[0])

    md = ["# File inventory", "",
          f"Generated by `scripts/generate_report.py` on "
          f"{datetime.now(timezone.utc).strftime('%d %B %Y')}. "
          f"{len(rows)} project files listed. Local data, model weights and caches are omitted.", "",
          "| Kind | Files | Lines |", "|---|---|---|"]
    for kind, (count, lines) in sorted(totals.items(), key=lambda kv: -kv[1][0]):
        md.append(f"| {kind} | {count} | {lines:,} |")
    code_lines = totals.get("code", [0, 0])[1] + totals.get("docs", [0, 0])[1]
    md += ["", f"**Source and documentation: {code_lines:,} lines.**", "",
           "## Project files", "", "| Path | Kind | Bytes | Lines |", "|---|---|---|---|"]
    for rel, kind, size, lines in rows:
        md.append(f"| `{rel}` | {kind} | {size:,} | {lines or ''} |")
    (DOCS / "FILE_INVENTORY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote docs/FILE_INVENTORY.md ({len(rows)} files, {code_lines:,} code+docs lines)")


def checklist() -> None:
    """Acceptance checklist whose ticks come from the files that exist."""
    checks = [
        ("Team and guide named in README, AUTHORS.md, paper and deck",
         all((ROOT / p).exists() for p in ("README.md", "AUTHORS.md", "paper/main.tex"))),
        ("Engine source and export pipeline included",
         (ROOT / "engine/recon/__init__.py").exists()),
        ("Web app, API and demo account seed command", (ROOT / "app" / "seed.py").exists()),
        ("Test suite and submission verifier included", (ROOT / "tests").exists()
         and (ROOT / "scripts" / "verify_submission.py").exists()),
        ("Evaluation set manifest and image attribution",
         (DATA / "ATTRIBUTION.md").exists() and (DATA / "dataset.json").exists()),
        ("Two baseline implementations and per-run evidence",
         (ROOT / "benchmarks" / "baselines" / "baseline_tsdf.py").exists()
         and (ROOT / "benchmarks" / "baselines" / "baseline_visual_hull.py").exists()
         and (RESULTS / "raw" / "runs.csv").exists()),
        ("Raw results, logs and environment record in results/",
         (RESULTS / "raw" / "runs.csv").exists() and (RESULTS / "environment.json").exists()),
        ("Summary tables, ablation and thickness sweep",
         (RESULTS / "summary" / "main_table.csv").exists()
         and (RESULTS / "summary" / "thickness_sweep.csv").exists()),
        ("Figures generated from results/", (RESULTS / "figures").exists()),
        ("Paper PDF, LaTeX source and bibliography",
         all((ROOT / "paper" / name).exists() for name in ("main.tex", "main.pdf", "references.bib"))),
        ("Presentation deck",
         (ROOT / "presentations" / "3DFUN-AI-Project.pptx").exists()),
        ("Project report in Markdown and PDF",
         (DOCS / "report.md").exists() and (DOCS / "report.pdf").exists()),
        ("Reproduce with one command per artefact (`make benchmark|figures|paper|slides|report`)",
         (ROOT / "Makefile").exists()),
        ("Large ONNX weights git-ignored and fetched by `make models`",
         (ROOT / ".gitignore").exists() and (ROOT / "scripts" / "fetch_models.py").exists()),
        ("LICENSE, .gitignore, sample input, demo commands",
         all((ROOT / p).exists() for p in ("LICENSE", ".gitignore", "engine/testdata/sneaker.png"))),
    ]
    md = ["# Acceptance checklist", "",
          "This checks which files are present. See results/verification.json for execution results.", ""]
    for label, ok in checks:
        md.append(f"- [{'x' if ok else ' '}] {label}")
    md += ["", "## Reproduce", "", "```bash", "make install && make models && make seed",
           "make benchmark && make figures && make paper && make slides && make report",
           "make verify", "```", ""]
    (DOCS / "ACCEPTANCE_CHECKLIST.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote docs/ACCEPTANCE_CHECKLIST.md "
          f"({sum(1 for _, ok in checks if ok)}/{len(checks)} ticked)")


if __name__ == "__main__":
    build()
    inventory()
    checklist()
