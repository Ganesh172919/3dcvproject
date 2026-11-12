"""Stage 4 - isosurface extraction, mesh cleanup and UV unwrapping."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import PipelineConfig
from .volumetrics import VolumeResult

log = logging.getLogger(__name__)


@dataclass
class MeshResult:
    vertices: np.ndarray            # (V, 3) float32, world units
    faces: np.ndarray               # (F, 3) int32
    vertex_uv: np.ndarray           # (V, 2) float32 in [0, 1]
    face_uv: np.ndarray             # (F*3, 2) float32 (glTF layout)
    metrics: dict[str, Any] = field(default_factory=dict)
    face_normals_vertex: np.ndarray | None = None
    validation_vertices: np.ndarray | None = None   # pre-UV (manifold) topology
    validation_faces: np.ndarray | None = None
    vertex_colors: np.ndarray | None = None  # (V, 3) sRGB, in [0, 1]
    vertex_normals: np.ndarray | None = None


# --------------------------------------------------------------------------
# marching cubes
# --------------------------------------------------------------------------
def marching_cubes(sdf: np.ndarray, spacing: float, origin: tuple[float, float, float]):
    from skimage import measure

    verts, faces, normals, _ = measure.marching_cubes(sdf, level=0.0, spacing=(spacing,) * 3)
    # marching_cubes returns coordinates in array order (z, y, x); convert to
    # world order (x, y, z) so +X is right, +Y is up and +Z points at the camera.
    verts = verts.astype(np.float32)[:, [2, 1, 0]]
    normals = normals.astype(np.float32)[:, [2, 1, 0]]
    verts += np.array(origin, dtype=np.float32)[None, :]
    return verts, faces.astype(np.int32), normals


def cleanup_mesh(verts: np.ndarray, faces: np.ndarray, cfg: PipelineConfig):
    import open3d as o3d

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    if len(faces) == 0:
        raise ValueError("Isosurface extraction produced no triangles.")
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts.astype(np.float64)),
        o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.remove_non_manifold_edges()
    mesh.orient_triangles()
    mesh.compute_vertex_normals()

    # drop small floating shells
    tri = np.asarray(mesh.triangles)
    vts = np.asarray(mesh.vertices)
    clusters = o3d.geometry.TriangleMesh.cluster_connected_triangles(mesh)
    labels = np.asarray(clusters[0])
    counts = np.asarray(clusters[1])
    if labels.size and counts.size:
        keep_labels = {int(np.argmax(counts))}
        largest = counts.max()
        for i, c in enumerate(counts):
            if c >= cfg.remove_components_below * largest:
                keep_labels.add(i)
        keep = np.array([i for i, lb in enumerate(labels) if int(lb) in keep_labels])
        mesh = mesh.select_by_index(keep, cleanup=False)

    # smoothing (Taubin shrinks less than Laplacian)
    for _ in range(max(1, cfg.taubin_iters // 4)):
        mesh = mesh.filter_smooth_taubin(number_of_iterations=4,
                                         lambda_filter=cfg.taubin_lambda)
    mesh.compute_vertex_normals()

    v = np.asarray(mesh.vertices).astype(np.float32)
    f = np.asarray(mesh.triangles).astype(np.int32)
    if len(v) == 0 or len(f) == 0:
        raise ValueError("Mesh cleanup removed all geometry.")
    return v, f


def _fill_holes_o3d(v: np.ndarray, f: np.ndarray, hole_size: float):
    try:
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(v.astype(np.float64)),
            o3d.utility.Vector3iVector(f.astype(np.int32)))
        mesh = mesh.fill_holes(hole_size)
        return (np.asarray(mesh.vertices).astype(np.float32),
                np.asarray(mesh.triangles).astype(np.int32))
    except Exception:  # pragma: no cover
        return v, f


def _boundary_edge_count(f: np.ndarray) -> int:
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0), axis=1)
    _, c = np.unique(e, axis=0, return_counts=True)
    return int((c == 1).sum())


def decimate(v: np.ndarray, f: np.ndarray, target_faces: int,
             hole_size: float = 0.05) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fallback triangle-budget reduction.

    Only used when the requested budget is below what the coarsest usable
    voxel grid can produce. Decimation can open the surface, so the result is
    repaired and its watertightness is reported in the manifest.
    """
    import fast_simplification
    import open3d as o3d

    info: dict[str, Any] = {"method": "fast_simplification", "watertight_after": None}
    if len(f) <= target_faces:
        info["watertight_after"] = _boundary_edge_count(f) == 0
        return v, f, info
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    f = f[area > 1e-12]
    v2, f2 = fast_simplification.simplify(v.astype(np.float64), f.astype(np.int64),
                                          target_count=int(target_faces), agg=7)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(v2.astype(np.float64)),
        o3d.utility.Vector3iVector(f2.astype(np.int32)))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    try:
        mesh = mesh.fill_holes(hole_size)
    except Exception:
        pass
    mesh.remove_non_manifold_edges()
    mesh.orient_triangles()
    vv = np.asarray(mesh.vertices).astype(np.float32)
    ff = np.asarray(mesh.triangles).astype(np.int32)
    info["faces"] = int(len(ff))
    info["boundary_edges"] = _boundary_edge_count(ff)
    info["watertight_after"] = info["boundary_edges"] == 0
    return vv, ff, info


def normalize_transform(v: np.ndarray) -> tuple[np.ndarray, dict]:
    """Center at the bounding-box centre, scale so the largest axis is 1.0."""
    lo, hi = v.min(axis=0), v.max(axis=0)
    center = (lo + hi) / 2.0
    ext = float((hi - lo).max())
    scale = 1.0 / max(ext, 1e-9)
    out = ((v - center) * scale).astype(np.float32)
    return out, {
        "applied_center": [float(c) for c in center],
        "applied_scale": float(scale),
        "normalized_extent": [float(e * scale) for e in (hi - lo)],
    }


def planar_uv(v: np.ndarray, f: np.ndarray, canvas_size: int,
              normalization: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """View-projected (planar) UV parameterisation.

    The reconstruction is a shallow shell around the reference camera, so the
    most faithful parameterisation is simply the reference view itself:

    * front-facing triangles sample the reference image at their projected
      position, with atlas rows increasing down from the top;
    * back-facing triangles sample the mirrored image (``u -> 1 - u``), which is
      the standard symmetric-object assumption for single-view reconstruction;
    * a 1 % gutter separates the two halves so bilinear filtering never bleeds
      between the front and back sheets.

    Unlike an automatic atlas (xatlas), this produces zero seam fragmentation:
    every texel neighbour in UV space is a neighbour on the surface, so the
    baked texture stays sharp and continuous. It is also ~10x cheaper.
    """
    px_to_world = 2.0 / canvas_size
    scale = normalization["applied_scale"]
    center = np.asarray(normalization["applied_center"], dtype=np.float64)
    world = v.astype(np.float64) / scale + center[None, :]
    px = world[:, 0] / px_to_world + canvas_size / 2.0
    py = canvas_size / 2.0 - world[:, 1] / px_to_world
    # Normalise to the *object* bounding box (not the whole canvas) so the two
    # atlas sheets use the full texel budget instead of padding.
    px0, px1 = float(px.min()), float(px.max())
    py0, py1 = float(py.min()), float(py.max())
    u = np.clip((px - px0) / max(px1 - px0, 1e-6), 0.0, 1.0)
    w = np.clip((py - py0) / max(py1 - py0, 1e-6), 0.0, 1.0)

    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area_vec = n
    # Decide front/back with a *vertex* vote instead of a raw face test: edge-on
    # slivers have an ambiguous normal sign, and voting with the surrounding
    # surface keeps the atlas sheet assignment consistent (no speckles).
    vn = np.zeros_like(v, dtype=np.float64)
    np.add.at(vn, f[:, 0], area_vec)
    np.add.at(vn, f[:, 1], area_vec)
    np.add.at(vn, f[:, 2], area_vec)
    # area-weighted vote; the sign of the accumulated normal is scale invariant
    vfront = vn[:, 2] >= 0.0
    votes = vfront[f].sum(axis=1)
    front = votes >= 2

    gut = 0.01
    u_front = gut + u * (0.5 - 2 * gut)
    u_back = 0.5 + gut + (1.0 - u) * (0.5 - 2 * gut)
    corner_u = np.where(front[:, None], u_front[f], u_back[f])
    corner_uv = np.stack([corner_u.reshape(-1), w[f].reshape(-1)], axis=1).astype(np.float32)
    stats = {
        "parameterization": "planar-view-projection",
        "front_faces": int(front.sum()),
        "back_faces": int((~front).sum()),
        "uv_gutter": gut,
        "object_pixel_bbox": [px0, py0, px1, py1],
        "atlas_texels_per_sheet": 0.5 - 2 * gut,
    }
    return v, f.astype(np.int32), corner_uv, stats


def mesh_metrics(v: np.ndarray, f: np.ndarray, watertight_probe: bool = True) -> dict[str, Any]:
    tri = v[f]
    a = tri[:, 1] - tri[:, 0]
    b = tri[:, 2] - tri[:, 0]
    c = tri[:, 2] - tri[:, 1]
    la = np.linalg.norm(a, axis=1)
    lb = np.linalg.norm(b, axis=1)
    lc = np.linalg.norm(c, axis=1)
    cross = np.cross(a, b)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    longest = np.maximum(np.maximum(la, lb), lc)
    # standard triangle aspect ratio: longest edge / inscribed-circle diameter,
    # normalised so an equilateral triangle scores 1.0
    inradius = np.zeros_like(areas)
    valid = areas > 1e-14
    perimeter = la + lb + lc
    inradius[valid] = 2.0 * areas[valid] / np.maximum(perimeter[valid], 1e-12)
    aspect = np.zeros_like(areas)
    good = valid & (inradius > 1e-9)
    aspect[good] = (longest[good] / (2.0 * np.sqrt(3.0) * inradius[good]))
    degenerate = int((~good).sum())

    # edge manifoldness
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = int((counts == 1).sum())
    nonmanifold_edges = int((counts > 2).sum())

    euler = int(len(v) - len(uniq) + len(f))
    metrics = {
        "vertices": int(len(v)),
        "faces": int(len(f)),
        "triangles": int(len(f)),
        "surface_area": float(areas.sum()),
        "faces_under_1e-9_area": int((areas < 1e-9).sum()),
        "aspect_ratio_mean": float(aspect[good].mean()) if good.any() else 0.0,
        "aspect_ratio_p95": float(np.percentile(aspect[good], 95)) if good.any() else 0.0,
        "aspect_ratio_p99": float(np.percentile(aspect[good], 99)) if good.any() else 0.0,
        "degenerate_faces": degenerate,
        "edge_length_max": float(longest.max()),
        "edge_length_median": float(np.median(longest)),
        "boundary_edges": boundary_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "manifold": bool(boundary_edges == 0 and nonmanifold_edges == 0),
        "euler_characteristic": euler,
    }
    if watertight_probe:
        metrics["watertight"] = bool(boundary_edges == 0 and nonmanifold_edges == 0)
    lo, hi = v.min(axis=0), v.max(axis=0)
    metrics["bounding_box"] = [float(x) for x in (hi - lo)]
    metrics["centroid"] = [float(x) for x in v.mean(axis=0)]
    return metrics


def build_mesh(volume: VolumeResult, cfg: PipelineConfig) -> MeshResult:
    import trimesh

    verts, faces, _ = marching_cubes(volume.sdf, volume.spacing, volume.origin)
    raw_faces = len(faces)
    verts, faces = cleanup_mesh(verts, faces, cfg)
    verts, faces = _fill_holes_o3d(verts, faces, hole_size=0.02)
    # The grid resolution is calibrated to the face budget, so decimation is
    # only a fallback for very small budgets.
    dec_info = {"method": "none (resolution-calibrated)",
                "watertight_after": _boundary_edge_count(faces) == 0}
    if len(faces) > cfg.target_faces * 1.45:
        verts, faces, dec_info = decimate(verts, faces, cfg.target_faces)
    # Marching cubes and cleanup can produce consistently inward-facing shells.
    # Correct each closed component before the UV projection classifies front and back.
    surface = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    surface.fix_normals(multibody=True)
    faces = np.asarray(surface.faces, dtype=np.int32)
    verts, xform = normalize_transform(verts)
    # Topology metrics must be measured BEFORE UV unwrapping: xatlas splits
    # vertices along seams, which would otherwise look like boundary edges.
    base_metrics = mesh_metrics(verts, faces)
    new_v, new_f, corner_uv, uv_stats = planar_uv(verts, faces, cfg.canvas_size, xform)
    final_metrics = dict(base_metrics)
    final_metrics.update({
        "uv_parameterization": uv_stats,
        "pre_cleanup_faces": int(raw_faces),
        "post_cleanup_faces": int(len(faces)),
        "uv_corner_count": int(len(corner_uv)),
        "uv_bbox": [float(corner_uv[:, 0].min()), float(corner_uv[:, 1].min()),
                    float(corner_uv[:, 0].max()), float(corner_uv[:, 1].max())],
        "normalization": xform,
        "decimation": dec_info,
        "decimated_from_raw": bool(raw_faces != len(faces)),
    })
    return MeshResult(vertices=new_v, faces=new_f, vertex_uv=np.zeros((0, 2), np.float32),
                      face_uv=corner_uv, metrics=final_metrics,
                      validation_vertices=verts, validation_faces=faces)
