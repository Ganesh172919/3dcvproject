"""Build every figure used by the paper and the deck from ``results/`` only.

Course: 3D Computer Vision Course Project · IIIT Dharwad
Supervisor: Dr. Shirshendu Layek (Dept. of DSAI)
Authors: J Ganesh (23BDS024), M Jagadeshwar (23BDS033), B Varshith (23BDS011),
         Pranav Kashyap (23BCS098), C Sai Aravind (23BDS075)

    python3 scripts/make_figures.py

Writes into ``results/figures/``:

    fig_qualitative.png       input | ours | TSDF | visual hull, for six scenes
    fig_chamfer_bar.png       Chamfer distance per method (mean +- std)
    fig_fscore_bar.png        F-score @ 1% per method
    fig_time_stages.png       stacked stage timings of the proposed pipeline
    fig_ablation.png          ablation: F-score and Chamfer per configuration
    fig_tradeoff.png          time vs. Chamfer scatter (Pareto view)
    fig_scenes.png            the dataset contact sheet with GT triangle counts
    fig_confusion_free.png    watertightness / topology summary

Nothing is drawn from a constant: each panel reads results/summary/*.csv or
results/raw/meshes/*.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
DATA = ROOT / "benchmarks" / "data"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

LABELS = {"proposed": "3DFUN AI (ours)", "tsdf": "TSDF fusion", "visual_hull": "Visual hull"}
COLOURS = {"proposed": "#2dd4bf", "tsdf": "#7aa2f7", "visual_hull": "#f0a35e"}
plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "font.size": 9,
    "axes.edgecolor": "#44546a", "axes.labelcolor": "#1c2536",
    "text.color": "#1c2536", "xtick.color": "#1c2536", "ytick.color": "#1c2536",
    "axes.grid": True, "grid.color": "#dde5ef", "grid.linewidth": .7,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(row: dict, key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def summary() -> list[dict]:
    return _read_csv(RESULTS / "summary" / "main_table.csv")


def ablation() -> list[dict]:
    return _read_csv(RESULTS / "summary" / "ablation_table.csv")


def stage_timings() -> list[dict]:
    return _read_csv(RESULTS / "summary" / "stage_timings.csv")


# --------------------------------------------------------------------- figures
def fig_bars(rows: list[dict], key: str, err_key: str, title: str, ylabel: str,
             out: Path, lower_better: bool = True) -> None:
    rows = [r for r in rows if np.isfinite(_f(r, key))]
    if not rows:
        print(f"  skip {out.name}: no data for {key}")
        return
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    names = [LABELS.get(r["method"], r["method"]) for r in rows]
    vals = [_f(r, key) for r in rows]
    errs = [_f(r, err_key) for r in rows]
    errs = [0 if not np.isfinite(e) else e for e in errs]
    bars = ax.bar(names, vals, yerr=errs, capsize=4,
                  color=[COLOURS.get(r["method"], "#94a3b8") for r in rows],
                  edgecolor="#1c2536", linewidth=.6)
    best = min(vals) if lower_better else max(vals)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}" if val < 1 else f"{val:.2f}",
                ha="center", va="bottom", fontsize=7.5)
        if abs(val - best) < 1e-12:
            bar.set_linewidth(1.8)
            bar.set_edgecolor("#0f766e")
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_ylabel(ylabel)
    ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def fig_time_stages() -> None:
    rows = stage_timings()
    if not rows:
        print("  skip fig_time_stages.png: no stage timings")
        return
    keep = [r for r in rows if r["stage"] not in {"total"}]
    keep.sort(key=lambda r: -_f(r, "mean_s"))
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    labels = [r["stage"] for r in keep]
    vals = [_f(r, "mean_s") for r in keep]
    errs = [max(_f(r, "std_s"), 0) for r in keep]
    ax.barh(labels[::-1], vals[::-1], xerr=errs[::-1], capsize=3,
            color="#2dd4bf", edgecolor="#1c2536", linewidth=.6)
    for i, v in enumerate(vals[::-1]):
        ax.text(v, i, f" {v:.2f}s", va="center", fontsize=7.5)
    ax.set_xlabel("seconds (mean ± std over runs)")
    ax.set_title("Where a draft run spends its time", fontsize=10, loc="left")
    ax.margins(x=0.18)
    fig.tight_layout()
    fig.savefig(FIG / "fig_time_stages.png")
    plt.close(fig)
    print(f"  wrote results/figures/fig_time_stages.png")


def fig_ablation() -> None:
    rows = ablation()
    if not rows:
        print("  skip fig_ablation.png: no ablation table")
        return
    names = [r["config"] for r in rows]
    cd = [_f(r, "chamfer_mean") for r in rows]
    fs = [_f(r, "f_score_mean") for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax1.bar(range(len(names)), cd, color="#2dd4bf", edgecolor="#1c2536", linewidth=.6)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=35, ha="right", fontsize=7.5)
    ax1.set_ylabel("Chamfer distance ↓")
    ax1.set_title("Ablation: geometry", fontsize=10, loc="left")
    ax2.bar(range(len(names)), fs, color="#8b5cf6", edgecolor="#1c2536", linewidth=.6)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=35, ha="right", fontsize=7.5)
    ax2.set_ylabel("F-score @ 1% ↑")
    ax2.set_title("Ablation: completeness", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_ablation.png")
    plt.close(fig)
    print("  wrote results/figures/fig_ablation.png")


def fig_tradeoff() -> None:
    rows = [r for r in summary() if np.isfinite(_f(r, "chamfer_mean"))]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for r in rows:
        ax.scatter(_f(r, "seconds_mean"), _f(r, "chamfer_mean"), s=90,
                   color=COLOURS.get(r["method"], "#94a3b8"),
                   edgecolor="#1c2536", linewidth=.8, zorder=3)
        ax.annotate(LABELS.get(r["method"], r["method"]),
                    (_f(r, "seconds_mean"), _f(r, "chamfer_mean")),
                    textcoords="offset points", xytext=(6, 6), fontsize=7.5)
    ax.set_xlabel("wall clock per image (s, log scale)")
    ax.set_ylabel("Chamfer distance ↓")
    ax.set_xscale("log")
    ax.set_title("Accuracy vs. runtime", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_tradeoff.png")
    plt.close(fig)
    print("  wrote results/figures/fig_tradeoff.png")


def fig_dataset() -> None:
    """Contact sheet of the evaluation set, with ground-truth triangle counts."""
    import cv2

    dataset = json.loads((DATA / "dataset.json").read_text(encoding="utf-8"))
    tiles, labels = [], []
    for entry in dataset["entries"]:
        img = cv2.imread(str(DATA / entry["image"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_AREA)
        tiles.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        gt = entry.get("gt_triangles")
        labels.append(entry["id"].split("/")[-1] + (f" · GT {gt}" if gt else ""))
    if not tiles:
        return
    cols = 7
    rows = (len(tiles) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.15, rows * 1.28))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, tile, label in zip(axes.ravel(), tiles, labels):
        ax.imshow(tile)
        ax.set_title(label, fontsize=6.5, pad=2)
    fig.suptitle(f"Evaluation set v{dataset['version']} — {dataset['scenes']['total']} images "
                 f"({dataset['scenes']['synthetic']} with reference meshes)",
                 fontsize=9, y=0.995)
    fig.tight_layout()
    fig.savefig(FIG / "fig_scenes.png")
    plt.close(fig)
    print("  wrote results/figures/fig_scenes.png")


QUALITATIVE_SCENES = ("synthetic/box", "synthetic/sphere", "synthetic/stairs",
                      "synthetic/cylinder", "synthetic/cone", "synthetic/prism",
                      "real/sneaker", "real/coffee_mug")


def fig_qualitative(scenes: tuple[str, ...] = QUALITATIVE_SCENES) -> None:
    """input | ours | TSDF | visual hull — rendered from the meshes each method produced."""
    import cv2

    sys.path.insert(0, str(ROOT / "benchmarks"))
    from benchmarks.run_benchmarks import _reference_view, _render_match, _silhouette
    import metrics as _M

    mesh_dir = RESULTS / "raw" / "meshes"
    rows = []
    for scene in scenes:
        stem = scene.replace("/", "_")
        ref = _reference_view(DATA / f"{scene.split('/')[0]}/images/"
                              f"{scene.split('/')[1]}.png")
        panels = [cv2.cvtColor(cv2.resize(ref, (180, 180)), cv2.COLOR_RGB2BGR)]
        for method in ("proposed", "tsdf", "visual_hull"):
            obj = mesh_dir / f"{stem}__{method}__r1"
            model = None
            for candidate in sorted(obj.glob("*.obj")):
                model = candidate
                break
            if model is None:
                panels.append(np.full((180, 180, 3), 245, np.uint8))
                continue
            import trimesh
            mesh = trimesh.load(model, force="mesh", process=False)
            # Same silhouette search the harness uses for SSIM: pick the yaw whose
            # projection best matches the input photo, so the panel shows the mesh
            # the way the evaluation looked at it.
            verts = np.asarray(mesh.vertices, float)
            faces = np.asarray(mesh.faces, int)
            ref_small = _silhouette(cv2.resize(ref, (120, 120)))
            ref_mask = _silhouette(cv2.resize(ref, (180, 180)))
            best = None
            for yaw in range(0, 360, 30):
                cand = _render_match(verts, faces, 120, rotate_deg=yaw)
                score = _M.silhouette_iou(_silhouette(cand), ref_small)
                if best is None or score > best[0]:
                    best = (score, yaw)
            render = _render_match(verts, faces, 180, rotate_deg=best[1])
            panels.append(cv2.cvtColor(render, cv2.COLOR_RGB2BGR))
        rows.append(np.hstack(panels))
    if not rows:
        return
    grid = np.vstack([np.hstack([r, np.full((r.shape[0], 6, 3), 255, np.uint8)]) for r in rows])
    head = np.full((26, grid.shape[1], 3), 255, np.uint8)
    for i, text in enumerate(("input", "3DFUN AI (ours)", "TSDF fusion", "visual hull")):
        cv2.putText(head, text, (i * 186 + 8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (30, 40, 60), 1, cv2.LINE_AA)
    out = np.vstack([head, grid])
    cv2.imwrite(str(FIG / "fig_qualitative.png"), out)
    print("  wrote results/figures/fig_qualitative.png")


def fig_topology() -> None:
    rows = summary()
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    names = [LABELS.get(r["method"], r["method"]) for r in rows]
    vals = [_f(r, "watertight_pct") for r in rows]
    bars = ax.bar(names, vals, color=[COLOURS.get(r["method"], "#94a3b8") for r in rows],
                  edgecolor="#1c2536", linewidth=.6)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.0f}%",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 112)
    ax.set_ylabel("% of runs watertight")
    ax.set_title("Topology: closed surfaces", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_topology.png")
    plt.close(fig)
    print("  wrote results/figures/fig_topology.png")


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    rows = summary()
    if not rows:
        print("results/summary/main_table.csv is missing — run `make benchmark` first.")
        return 1
    print("figures:")
    fig_bars(rows, "chamfer_mean", "chamfer_std", "Chamfer distance (lower is better)",
             "Chamfer (normalised units)", FIG / "fig_chamfer_bar.png")
    fig_bars(rows, "f_score_mean", "f_score_std", "F-score @ 1% (higher is better)",
             "F-score", FIG / "fig_fscore_bar.png", lower_better=False)
    fig_bars(rows, "iou_mean", "iou_std", "Volumetric IoU (higher is better)",
             "IoU", FIG / "fig_iou_bar.png", lower_better=False)
    fig_bars(rows, "ssim_mean", "ssim_std", "Render SSIM (higher is better)",
             "SSIM", FIG / "fig_ssim_bar.png", lower_better=False)
    fig_bars(rows, "seconds_mean", "seconds_std", "Wall clock per image",
             "seconds", FIG / "fig_time_bar.png")
    fig_time_stages()
    fig_ablation()
    fig_tradeoff()
    fig_dataset()
    fig_topology()
    fig_qualitative()
    render_error_heatmaps()
    return 0


def render_error_heatmaps(scenes: tuple[str, ...] = ("cylinder", "sphere", "cone", "prism")) -> None:
    """Compute per-vertex Euclidean distance from reconstruction to GT, map to viridis, save figure."""
    import trimesh
    from scipy.spatial import cKDTree

    mesh_dir = RESULTS / "raw" / "meshes"
    gt_dir = DATA / "synthetic" / "gt"
    if not mesh_dir.exists() or not gt_dir.exists():
        return

    fig, axes = plt.subplots(len(scenes), 3, figsize=(9.2, 2.3 * len(scenes)))
    fig.subplots_adjust(hspace=0.38, wspace=0.28)

    scatter_heat = None
    for idx, name in enumerate(scenes):
        gt_path = gt_dir / f"{name}.obj"
        recon_dir = mesh_dir / f"synthetic_{name}__proposed__r1"
        recon_path = recon_dir / "model.obj" if recon_dir.exists() else None

        if not gt_path.exists() or recon_path is None or not recon_path.exists():
            continue

        gt_mesh = trimesh.load(str(gt_path), force="mesh", process=False)
        recon_mesh = trimesh.load(str(recon_path), force="mesh", process=False)

        # Normalize center for visualization alignment
        gt_pts, _ = trimesh.sample.sample_surface(gt_mesh, 15000)
        gt_center = gt_pts.mean(axis=0)
        gt_pts -= gt_center

        verts = np.asarray(recon_mesh.vertices, float)
        verts -= verts.mean(axis=0)

        tree = cKDTree(gt_pts)
        dists, _ = tree.query(verts)

        ax_gt = axes[idx, 0]
        ax_heat = axes[idx, 1]
        ax_hist = axes[idx, 2]

        # Panel 1: Ground truth reference projection
        ax_gt.scatter(gt_pts[::6, 0], gt_pts[::6, 1], c="#64748b", s=1.2, alpha=0.65, edgecolors="none")
        ax_gt.set_aspect("equal", "datalim")
        ax_gt.axis("off")
        ax_gt.set_title(f"GT: {name.capitalize()}", fontsize=9, fontweight="bold")

        # Panel 2: Reconstructed vertices colored with viridis heatmap
        scatter_heat = ax_heat.scatter(verts[:, 0], verts[:, 1], c=dists, cmap="viridis",
                                       s=2.5, alpha=0.85, vmin=0.0, vmax=0.045, edgecolors="none")
        ax_heat.set_aspect("equal", "datalim")
        ax_heat.axis("off")
        ax_heat.set_title(f"3DFUN AI Error (Mean: {dists.mean():.4f})", fontsize=9, fontweight="bold")

        # Panel 3: Error distribution histogram
        ax_hist.hist(dists, bins=25, color="#0f766e", edgecolor="#134e4a", alpha=0.85, density=True)
        ax_hist.axvline(dists.mean(), color="#ef4444", linestyle="--", linewidth=1.2,
                        label=f"Mean: {dists.mean():.4f}")
        ax_hist.set_title("Per-Vertex Error Distribution", fontsize=8.5)
        ax_hist.set_xlabel("Euclidean Error", fontsize=7.5)
        ax_hist.set_ylabel("Density", fontsize=7.5)
        ax_hist.tick_params(labelsize=7)
        ax_hist.legend(fontsize=7, frameon=False)

    if scatter_heat is not None:
        cbar_ax = fig.add_axes([0.93, 0.28, 0.016, 0.45])
        cb = fig.colorbar(scatter_heat, cax=cbar_ax)
        cb.set_label("Vertex Euclidean Error (units)", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    fig.suptitle("3D Surface Reconstruction Error Colormaps vs. Analytic Ground Truth",
                 fontsize=11, fontweight="bold", y=0.995)
    out_path = FIG / "fig_error_heatmaps.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print("  wrote results/figures/fig_error_heatmaps.png")


if __name__ == "__main__":
    raise SystemExit(main())
