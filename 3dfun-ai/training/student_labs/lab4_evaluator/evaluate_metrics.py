"""Lab 4: 3D Metric Evaluation Suite (Chamfer Distance, EMD, F-Score).

Course: 3D Computer Vision · Student Educational Laboratory
Theoretical Foundation:
1. Symmetric Unsquared Chamfer Distance (CD):
    CD(S_1, S_2) = 1/(2 |S_1|) sum_{x in S_1} min_{y in S_2} ||x - y||_2
                 + 1/(2 |S_2|) sum_{y in S_2} min_{x in S_1} ||x - y||_2

2. Precision, Recall & F-Score at threshold tau:
    Precision(tau) = 1/|S_1| sum_{x in S_1} [ min_{y in S_2} ||x - y|| < tau ]
    Recall(tau)    = 1/|S_2| sum_{y in S_2} [ min_{x in S_1} ||x - y|| < tau ]
    F1(tau)        = 2 * (Precision * Recall) / (Precision + Recall)

3. Earth Mover's Distance (EMD):
    EMD(S_1, S_2) = min_{phi: S_1 -> S_2} 1/|S_1| sum_{x in S_1} ||x - phi(x)||_2
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("lab4.evaluator")


def sample_mesh_surface(mesh_path: str | Path, num_points: int = 5000, seed: int | None = None) -> np.ndarray:
    """Sample a mesh surface or read the vertices of a point cloud."""
    p = Path(mesh_path)
    if not p.exists():
        raise FileNotFoundError(f"Mesh not found: {p}")

    import trimesh
    mesh = trimesh.load(str(p))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()

    if isinstance(mesh, trimesh.points.PointCloud):
        points = np.asarray(mesh.vertices)
        if len(points) > num_points:
            points = points[np.random.default_rng(seed).choice(len(points), num_points, replace=False)]
    elif isinstance(mesh, trimesh.Trimesh) and len(mesh.faces):
        points, _ = trimesh.sample.sample_surface(mesh, num_points, seed=seed)
    else:
        raise ValueError(f"{p.name} contains no usable mesh or point cloud.")
    if not len(points) or not np.isfinite(points).all():
        raise ValueError(f"{p.name} has empty or non-finite geometry.")
    return points.astype(np.float32)


def compute_chamfer_and_fscore(
    pts_pred: np.ndarray, pts_gt: np.ndarray, thresholds: list[float] = [0.01, 0.05]
) -> dict[str, Any]:
    """Computes unsquared symmetric Chamfer Distance and F-Scores at various thresholds."""
    tree_gt = cKDTree(pts_gt)
    tree_pred = cKDTree(pts_pred)

    # d_pred_to_gt: for each point in pred, distance to nearest point in gt
    d_pred_to_gt, _ = tree_gt.query(pts_pred, k=1)
    # d_gt_to_pred: for each point in gt, distance to nearest point in pred
    d_gt_to_pred, _ = tree_pred.query(pts_gt, k=1)

    # Unsquared Chamfer Distance
    cd_pred = float(np.mean(d_pred_to_gt))
    cd_gt = float(np.mean(d_gt_to_pred))
    cd_symmetric = float(0.5 * (cd_pred + cd_gt))

    # F-Scores
    fscores = {}
    for tau in thresholds:
        prec = float(np.mean(d_pred_to_gt < tau))
        rec = float(np.mean(d_gt_to_pred < tau))
        f1 = float(2.0 * prec * rec / max(prec + rec, 1e-8))
        fscores[f"tau_{tau}"] = {
            "threshold": tau,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f_score": round(f1, 4),
        }

    return {
        "chamfer_distance": {
            "pred_to_gt": round(cd_pred, 5),
            "gt_to_pred": round(cd_gt, 5),
            "symmetric": round(cd_symmetric, 5),
        },
        "f_scores": fscores,
    }


def compute_emd_approx(pts_1: np.ndarray, pts_2: np.ndarray, max_pts: int = 1000,
                       seed: int | None = None) -> float:
    """Approximate Earth Mover's Distance with equal-size point samples."""
    size = min(len(pts_1), len(pts_2), max_pts)
    rng = np.random.default_rng(seed)
    first = rng.choice(len(pts_1), size, replace=False)
    second = first if len(pts_1) == len(pts_2) else rng.choice(len(pts_2), size, replace=False)
    pts_1, pts_2 = pts_1[first], pts_2[second]

    # Fast 1-to-1 matching via scipy linear_sum_assignment
    from scipy.optimize import linear_sum_assignment
    dists = np.linalg.norm(pts_1[:, None, :] - pts_2[None, :, :], axis=-1)
    row_ind, col_ind = linear_sum_assignment(dists)
    emd = float(np.mean(dists[row_ind, col_ind]))
    return round(emd, 5)


def run_evaluation(
    pred_path: str | Path | None = None,
    gt_path: str | Path | None = None,
    out_dir: str | Path = "out/student_labs/lab4",
    num_points: int = 4000,
    seed: int = 17,
) -> dict[str, Any]:
    if not 16 <= num_points <= 100_000 or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Use 16–100,000 points and a seed between 0 and 2³²−1.")
    if (pred_path is None) != (gt_path is None):
        raise ValueError("Provide both a prediction and a reference.")
    rng = np.random.default_rng(seed)

    if pred_path is not None and gt_path is not None:
        log.info("Evaluating real meshes: %s vs %s", pred_path, gt_path)
        pts_pred = sample_mesh_surface(pred_path, num_points=num_points, seed=seed)
        pts_gt = sample_mesh_surface(gt_path, num_points=num_points, seed=seed)
    else:
        # 3D render evaluation demo: ground truth sphere vs noisy / slightly deformed predicted sphere
        log.info("Generating 3D render sphere comparison benchmark (GT vs Predicted with noise)")
        theta = rng.uniform(0, 2 * np.pi, num_points)
        phi = np.arccos(rng.uniform(-1, 1, num_points))
        r_gt = 0.7
        x_gt = r_gt * np.sin(phi) * np.cos(theta)
        y_gt = r_gt * np.sin(phi) * np.sin(theta)
        z_gt = r_gt * np.cos(phi)
        pts_gt = np.stack([x_gt, y_gt, z_gt], axis=1).astype(np.float32)

        # Predicted has 3% Gaussian noise and 5% ellipsoid elongation
        noise = rng.normal(0, 0.015, size=pts_gt.shape).astype(np.float32)
        pts_pred = pts_gt * np.array([1.04, 0.98, 1.0], dtype=np.float32) + noise

    results = compute_chamfer_and_fscore(pts_pred, pts_gt, thresholds=[0.01, 0.05])
    emd = compute_emd_approx(pts_pred, pts_gt, max_pts=800, seed=seed)
    results["earth_movers_distance"] = emd
    results["seed"] = seed
    results["point_counts"] = {"prediction": len(pts_pred), "reference": len(pts_gt)}

    log.info("Evaluation Complete:")
    log.info("  Chamfer Distance (Symmetric): %.5f", results["chamfer_distance"]["symmetric"])
    log.info("  F-Score (tau=0.01): %.4f (Prec: %.4f, Rec: %.4f)",
             results["f_scores"]["tau_0.01"]["f_score"],
             results["f_scores"]["tau_0.01"]["precision"],
             results["f_scores"]["tau_0.01"]["recall"])
    log.info("  F-Score (tau=0.05): %.4f (Prec: %.4f, Rec: %.4f)",
             results["f_scores"]["tau_0.05"]["f_score"],
             results["f_scores"]["tau_0.05"]["precision"],
             results["f_scores"]["tau_0.05"]["recall"])
    log.info("  Earth Mover's Distance: %.5f", emd)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "lab4_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab 4: 3D Metric Evaluation Suite")
    parser.add_argument("--pred", type=str, default=None, help="Path to predicted mesh (.obj, .ply, .stl, .glb)")
    parser.add_argument("--gt", type=str, default=None, help="Path to ground truth mesh")
    parser.add_argument("--num-points", type=int, default=4000, help="Number of surface points to sample")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out-dir", type=str, default="out/student_labs/lab4")
    args = parser.parse_args()
    try:
        run_evaluation(args.pred, args.gt, args.out_dir, num_points=args.num_points, seed=args.seed)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
