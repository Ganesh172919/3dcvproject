"""Evaluation metrics used by the benchmark harness.

Course: 3D Computer Vision Course Project · IIIT Dharwad
Supervisor: Dr. Shirshendu Layek (Dept. of DSAI)
Authors: J Ganesh (23BDS024), M Jagadeshwar (23BDS033), B Varshith (23BDS011),
         Pranav Kashyap (23BCS098), C Sai Aravind (23BDS075)

All metrics are implemented directly from their definitions; nothing is
estimated or inferred from a paper table.

* Chamfer distance            CD  = 0.5*(mean_{x∈A} min_{y∈B} ||x-y|| + mean_{y∈B} min_{x∈A} ||y-x||)
* F-score at a distance tau   F   = harmonic mean of precision/recall at tau
* Volumetric IoU              IoU = |A∩B| / |A∪B| on a shared voxel grid
* PSNR / SSIM                 on rendered comparisons of matched views
* Normal consistency (bonus)  mean |n_A · n_B| over nearest neighbours

Distances are reported in the normalised object frame (largest bounding-box
axis = 1.0) so models of different scale can be compared after isotropic
normalisation, which is the same normalisation both the proposed method and the
baselines apply.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------- sampling
def sample_surface(vertices: np.ndarray, faces: np.ndarray, count: int = 20000,
                   seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted surface samples plus per-sample face normals."""
    rng = np.random.default_rng(seed)
    tri = vertices[faces].astype(np.float64)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    total = area.sum()
    if total <= 0:
        raise ValueError("degenerate mesh: zero surface area")
    prob = area / total
    idx = rng.choice(len(faces), size=count, p=prob)
    u = rng.random(count)
    v = rng.random(count)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    picked = tri[idx]
    pts = (picked[:, 0] * (1 - u - v)[:, None] + picked[:, 1] * u[:, None]
           + picked[:, 2] * v[:, None])
    normals = cross[idx]
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(nlen, 1e-12)
    return pts.astype(np.float32), normals.astype(np.float32)


def normalize(vertices: np.ndarray) -> np.ndarray:
    """Centre at the bbox centre and scale the largest axis to 1.0."""
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    centre = (lo + hi) / 2.0
    extent = float((hi - lo).max())
    return ((vertices - centre) / max(extent, 1e-9)).astype(np.float64)


def ict_normalize(vertices: np.ndarray) -> np.ndarray:
    """Normalise and align to the principal axes (inertia tensor), sign-canonical."""
    v = normalize(vertices)
    cov = np.cov(v.T)
    w, vec = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    rot = vec[:, order]
    if np.linalg.det(rot) < 0:
        rot[:, -1] *= -1
    out = v @ rot
    # sign convention: largest |coordinate| on each axis is positive
    for axis in range(3):
        if abs(out[:, axis].min()) > abs(out[:, axis].max()):
            out[:, axis] *= -1
    return out


# ---------------------------------------------------------------------- metrics
@dataclass
class GeometryScores:
    chamfer: float
    chamfer_a_to_b: float
    chamfer_b_to_a: float
    f_score: float
    precision: float
    recall: float
    normal_consistency: float
    iou: float
    tau: float
    f_score_005: float = 0.0
    f_score_020: float = 0.0
    chamfer_p50: float = 0.0
    chamfer_p90: float = 0.0
    chamfer_p99: float = 0.0


def chamfer_percentiles(d_ab: np.ndarray, d_ba: np.ndarray,
                        percentiles: tuple[float, ...] = (50.0, 90.0, 99.0)) -> dict[int, float]:
    both = np.concatenate([d_ab, d_ba])
    return {int(p): float(np.percentile(both, p)) for p in percentiles}


def multi_threshold_f_scores(pred_pts: np.ndarray, gt_pts: np.ndarray,
                             thresholds: tuple[float, ...] = (0.005, 0.01, 0.02)) -> dict[float, float]:
    d_ab, _ = _nn_distance(pred_pts, gt_pts)
    d_ba, _ = _nn_distance(gt_pts, pred_pts)
    out = {}
    for tau in thresholds:
        prec = float((d_ab < tau).mean())
        rec = float((d_ba < tau).mean())
        score = 0.0 if (prec + rec) == 0 else float(2.0 * prec * rec / (prec + rec))
        out[tau] = score
    return out


def _nn_distance(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    tree = cKDTree(b)
    dist, idx = tree.query(a, k=1, workers=-1)
    return dist, idx


def chamfer_and_fscore(pred_pts: np.ndarray, gt_pts: np.ndarray,
                       tau: float = 0.01) -> tuple[float, float, float, float, float]:
    d_ab, _ = _nn_distance(pred_pts, gt_pts)
    d_ba, _ = _nn_distance(gt_pts, pred_pts)
    cd = 0.5 * (d_ab.mean() + d_ba.mean())
    precision = float((d_ab < tau).mean())
    recall = float((d_ba < tau).mean())
    f = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return cd, d_ab.mean(), d_ba.mean(), f, (precision, recall)  # type: ignore[return-value]


def f_score(pred_pts: np.ndarray, gt_pts: np.ndarray, tau: float = 0.01) -> tuple[float, float, float]:
    d_ab, _ = _nn_distance(pred_pts, gt_pts)
    d_ba, _ = _nn_distance(gt_pts, pred_pts)
    precision = float((d_ab < tau).mean())
    recall = float((d_ba < tau).mean())
    if precision + recall == 0:
        return 0.0, precision, recall
    return 2 * precision * recall / (precision + recall), precision, recall


def normal_consistency(pred_pts: np.ndarray, pred_n: np.ndarray,
                       gt_pts: np.ndarray, gt_n: np.ndarray) -> float:
    _, idx = _nn_distance(pred_pts, gt_pts)
    dots = np.abs(np.einsum("ij,ij->i", pred_n, gt_n[idx]))
    return float(dots.mean())


def voxel_iou(pred_pts: np.ndarray, gt_pts: np.ndarray, resolution: int = 64) -> float:
    """Occupancy IoU of two point sets on a shared axis-aligned grid."""
    both = np.vstack([pred_pts, gt_pts])
    lo, hi = both.min(axis=0), both.max(axis=0)
    scale = np.maximum(hi - lo, 1e-9)
    a = np.floor((pred_pts - lo) / scale * (resolution - 1)).astype(int)
    b = np.floor((gt_pts - lo) / scale * (resolution - 1)).astype(int)
    key = lambda arr: arr[:, 0] * resolution * resolution + arr[:, 1] * resolution + arr[:, 2]
    sa, sb = set(key(a).tolist()), set(key(b).tolist())
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter / union) if union else 0.0


def evaluate_geometry(pred_v: np.ndarray, pred_f: np.ndarray,
                      gt_v: np.ndarray, gt_f: np.ndarray,
                      samples: int = 20000, tau: float = 0.01,
                      seed: int = 0, align: bool = True) -> GeometryScores:
    pred_v = ict_normalize(pred_v) if align else normalize(pred_v)
    gt_v = ict_normalize(gt_v) if align else normalize(gt_v)
    p_pts, p_n = sample_surface(pred_v, pred_f, samples, seed)
    g_pts, g_n = sample_surface(gt_v, gt_f, samples, seed + 1)
    d_ab, _ = _nn_distance(p_pts, g_pts)
    d_ba, _ = _nn_distance(g_pts, p_pts)
    cd = float(0.5 * (d_ab.mean() + d_ba.mean()))
    prec = float((d_ab < tau).mean())
    rec = float((d_ba < tau).mean())
    f = 0.0 if prec + rec == 0 else float(2 * prec * rec / (prec + rec))
    f_005 = 0.0 if ((d_ab < 0.005).mean() + (d_ba < 0.005).mean()) == 0 else float(
        2 * (d_ab < 0.005).mean() * (d_ba < 0.005).mean() / ((d_ab < 0.005).mean() + (d_ba < 0.005).mean())
    )
    f_020 = 0.0 if ((d_ab < 0.02).mean() + (d_ba < 0.02).mean()) == 0 else float(
        2 * (d_ab < 0.02).mean() * (d_ba < 0.02).mean() / ((d_ab < 0.02).mean() + (d_ba < 0.02).mean())
    )
    pcts = chamfer_percentiles(d_ab, d_ba)
    nc = normal_consistency(p_pts, p_n, g_pts, g_n)
    iou = voxel_iou(p_pts, g_pts)
    return GeometryScores(chamfer=cd, chamfer_a_to_b=float(d_ab.mean()), chamfer_b_to_a=float(d_ba.mean()),
                          f_score=f, precision=prec, recall=rec,
                          normal_consistency=nc, iou=iou, tau=tau,
                          f_score_005=f_005, f_score_020=f_020,
                          chamfer_p50=pcts.get(50, 0.0), chamfer_p90=pcts.get(90, 0.0),
                          chamfer_p99=pcts.get(99, 0.0))


# ------------------------------------------------------------------ image scores
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64) / 255.0
    b = b.astype(np.float64) / 255.0
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    a = a.astype(np.float64) / 255.0
    b = b.astype(np.float64) / 255.0
    return float(structural_similarity(a, b, channel_axis=2, data_range=1.0))


def lpips_proxy(a: np.ndarray, b: np.ndarray) -> float:
    """Edge-similarity proxy used when the LPIPS network cannot be shipped offline.

    Computes 1 - cosine similarity of normalised gradient-magnitude features over
    three scales. It is reported under its own name, never as LPIPS.
    """
    import cv2

    feats = []
    for scale in (1, 2, 4):
        ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if scale > 1:
            ga = cv2.resize(ga, (ga.shape[1] // scale, ga.shape[0] // scale))
            gb = cv2.resize(gb, (gb.shape[1] // scale, gb.shape[0] // scale))
        fx_a, fy_a = np.gradient(ga)
        fx_b, fy_b = np.gradient(gb)
        fa = np.stack([np.hypot(fx_a, fy_a).ravel()])
        fb = np.stack([np.hypot(fx_b, fy_b).ravel()])
        denom = (np.linalg.norm(fa) * np.linalg.norm(fb))
        feats.append(0.0 if denom == 0 else float(fa @ fb.T / denom))
    return float(np.clip(1.0 - np.mean(feats), 0.0, 1.0))


def silhouette_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0.5
    b = mask_b > 0.5
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0
