"""Stage 3 - silhouette-aware volumetric inflation and implicit field building.

Mathematical model
------------------
For every pixel ``p`` inside the object silhouette we know

* the 2-D position ``(x, y)`` on the image plane, and
* a normalised closeness ``d(p) in [0, 1]`` produced by the depth network.

A pixel is inflated into a solid slab along the viewing axis ``z`` whose
thickness is

    t(p) = t_min + (t_max - t_min) * d(p) ** gamma ,
    t_max = kappa * W_object ,            (W_object = silhouette width)

occupying ``z in [-t(p)/2, +t(p)/2]`` and centred on the mid plane. The
silhouette itself is closed (``t >= t_min > 0`` on the rim), so the extracted
isosurface is watertight by construction. Because ``t`` is bounded by the
silhouette, the resulting mesh cannot leak outside the projected outline - the
classic failure mode of unconstrained monocular depth back-projection.

The occupancy is converted into a signed distance field (SDF) with a Euclidean
distance transform, lightly smoothed, and handed to marching cubes. This yields
smoother surfaces than meshing the binary occupancy directly, at the same cost.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from .config import PipelineConfig

log = logging.getLogger(__name__)


@dataclass
class VolumeResult:
    sdf: np.ndarray                 # (nz, ny, nx) float32, negative inside
    origin: tuple[float, float, float]   # world (x0, y0, z0) of voxel (0,0,0)
    spacing: float
    thickness_map: np.ndarray       # (S, S) world-space thickness
    stats: dict[str, Any]


def build_volume(mask: np.ndarray, closeness: np.ndarray, cfg: PipelineConfig) -> VolumeResult:
    S = mask.shape[0]
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise ValueError("Empty silhouette.")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    obj_w_px = x1 - x0
    obj_h_px = y1 - y0

    # World units: the full canvas spans [-1, 1] on x and y.
    px_to_world = 2.0 / S
    obj_w = obj_w_px * px_to_world
    obj_h = obj_h_px * px_to_world
    t_max = max(cfg.min_thickness, cfg.thickness_scale * obj_w)

    f = float(np.clip(cfg.thickness_floor_ratio, 0.0, 1.0))
    g = float(np.clip(cfg.thickness_depth_gain, 0.0, 1.0))
    d = np.clip(closeness, 0.0, 1.0) ** cfg.inflation_power
    shaped = np.clip(f + g * d, 0.0, 1.0)     # f keeps the rim closed, g carves the front
    thickness = cfg.min_thickness + (t_max - cfg.min_thickness) * shaped
    thickness = np.where(mask, thickness, cfg.min_thickness).astype(np.float32)

    margin = 0.04
    extent_x = obj_w + 2 * margin
    extent_y = obj_h + 2 * margin
    extent_z = t_max + 2 * margin
    h = max(extent_x, extent_y, extent_z) / max(cfg.voxel_res, 16)

    nx = max(8, int(np.ceil(extent_x / h)))
    ny = max(8, int(np.ceil(extent_y / h)))
    nz = max(8, int(np.ceil(extent_z / h)))

    xs_w = (np.arange(nx) + 0.5) * h - extent_x / 2
    ys_w = (np.arange(ny) + 0.5) * h - extent_y / 2
    zs_w = (np.arange(nz) + 0.5) * h - extent_z / 2

    # map world x,y -> canvas pixel indices (integer nearest neighbour)
    px = np.clip(((xs_w / px_to_world) + S / 2).astype(np.int32), 0, S - 1)
    py = np.clip((S / 2 - (ys_w / px_to_world)).astype(np.int32), 0, S - 1)

    m2 = mask[np.ix_(py, px)]                       # (ny, nx)
    t2 = thickness[np.ix_(py, px)]                  # (ny, nx)

    zz = zs_w[None, None, :]                        # (1,1,nz)
    half = (t2 * 0.5)[:, :, None]                   # (ny,nx,1)
    occupied = m2[:, :, None] & (np.abs(zz) <= half)
    occupied = occupied.transpose(2, 0, 1).astype(bool)   # (nz, ny, nx)

    # signed distance field (world units): + outside, - inside
    inside_dist = ndimage.distance_transform_edt(occupied) * h
    outside_dist = ndimage.distance_transform_edt(~occupied) * h
    sdf = (outside_dist - inside_dist).astype(np.float32)
    if cfg.field_smooth_sigma > 0:
        sdf = ndimage.gaussian_filter(sdf, cfg.field_smooth_sigma)


    stats = {
        "grid": [int(nx), int(ny), int(nz)],
        "voxel_size": float(h),
        "object_extent_world": [float(obj_w), float(obj_h)],
        "thickness_max": float(t_max),
        "occupancy_fraction": float(occupied.mean()),
        "bbox_px": [int(x0), int(y0), int(x1), int(y1)],
        "canvas_size": int(S),
    }
    # NOTE: ``sdf`` is indexed [z, y, x] (marching-cubes convention) while the
    # returned ``origin`` is expressed in world (x, y, z) order; the mesher
    # re-orders the vertices accordingly.
    return VolumeResult(
        sdf=sdf,
        origin=(-extent_x / 2.0, -extent_y / 2.0, -extent_z / 2.0),
        spacing=h,
        thickness_map=thickness,
        stats=stats,
    )


def calibrate_voxel_res(mask: np.ndarray, closeness: np.ndarray, cfg: PipelineConfig,
                        probe_res: int = 64) -> dict[str, Any]:
    """Pick the voxel resolution that lands closest to ``cfg.target_faces``.

    Marching-cubes output is watertight at *any* resolution, while quadric
    decimation of that output is not. We therefore control the triangle budget
    with the grid resolution instead of decimating, and calibrate the
    relationship ``F ~ c * res^2`` with one cheap probe run.
    """
    from .meshing import marching_cubes

    probe = cfg.with_overrides(voxel_res=probe_res)
    vol = build_volume(mask, closeness, probe)
    _, faces, _ = marching_cubes(vol.sdf, vol.spacing, vol.origin)
    c = max(len(faces), 1) / (probe_res ** 2)
    import math
    ideal = int(round(math.sqrt(max(cfg.target_faces, 500) / c)))
    res = int(np.clip(ideal, 32, 224))
    return {
        "probe_res": probe_res,
        "probe_faces": int(len(faces)),
        "faces_per_res2": float(c),
        "chosen_res": res,
        "predicted_faces": int(round(c * res ** 2)),
        "target_faces": int(cfg.target_faces),
    }
