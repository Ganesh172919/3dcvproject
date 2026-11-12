"""Stage 5 - UV texture baking and PBR map synthesis.

Baking is a *visibility-aware front/back projection*:

* every texel of the atlas is mapped back to its 3-D position by barycentric
  interpolation over the unwrapped mesh;
* the position is projected onto the reference camera (orthographic, -Z view);
* front-facing texels sample the reference image directly;
* back-facing texels sample the horizontally mirrored image, which is the
  standard assumption for objects that are roughly symmetric about the view
  axis (a real limitation, documented in KNOWN_LIMITATIONS.md);
* a UV-space dilation pass removes seam bleed.

The PBR maps are *estimated*, not ground truth: roughness/metallic come from
image statistics and ambient occlusion is integrated from the reconstruction's
signed distance field. The UI labels them as estimated.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .config import PipelineConfig
from .meshing import MeshResult

log = logging.getLogger(__name__)

_CHUNK_ROWS = 192


@dataclass
class TextureResult:
    maps: dict[str, np.ndarray]        # name -> (H, W, 4) uint8 RGBA
    coverage: float
    stats: dict[str, Any]


def _barycentric_buffers(res: int, mesh: MeshResult, shrink_texels: float = 0.35):
    """Rasterise the UV atlas and return per-texel triangle ids.

    Each triangle is shrunk slightly toward its own centroid before filling.
    Without the shrink, adjacent islands compete for the same border texels and
    the last one drawn wins, which paints colours from a completely different
    part of the reference image into the seam ("confetti" artefacts).
    """
    uv = mesh.face_uv.reshape(-1, 3, 2).astype(np.float64)
    tri_id = np.full((res, res), -1, dtype=np.int32)
    scale = float(res)
    centroid = uv.mean(axis=1, keepdims=True)
    shrunk = centroid + (uv - centroid) * (1.0 - (2.0 * shrink_texels) / np.maximum(
        np.linalg.norm(uv - centroid, axis=2, keepdims=True) * scale, 1e-6))
    pts = np.rint(shrunk * scale).astype(np.int32)
    for i in range(len(pts)):
        cv2.fillConvexPoly(tri_id, pts[i], int(i), lineType=cv2.LINE_8, shift=0)
    return tri_id


def _texel_positions(res: int, mesh: MeshResult, tri_id: np.ndarray,
                     chunk: int = _CHUNK_ROWS):
    """Yield (rows, positions, bary, tri_ids, valid) for a chunk of texel rows."""
    pos = mesh.vertices
    faces = mesh.faces
    uv = mesh.face_uv.reshape(-1, 3, 2)
    centers_v = (np.arange(res) + 0.5) / res
    for r0 in range(0, res, chunk):
        r1 = min(res, r0 + chunk)
        ids = tri_id[r0:r1].reshape(-1)
        valid = ids >= 0
        if not valid.any():
            yield (r0, r1), None, None, None, valid
            continue
        vv = centers_v[r0:r1][:, None]
        uu = centers_v[None, :]
        P = np.stack(np.broadcast_arrays(uu, vv), axis=-1).reshape(-1, 2).astype(np.float32)
        ids_c = ids[valid]
        tri2 = uv[ids_c].astype(np.float32)    # (M,3,2)
        a = tri2[:, 0]
        v0 = tri2[:, 1] - a
        v1 = tri2[:, 2] - a
        v2 = P[valid] - a
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        den = np.where(np.abs(den) < np.float32(1e-12), np.float32(1e-12), den)
        b1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
        b2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
        b1 = np.clip(b1, -0.25, 1.25)
        b2 = np.clip(b2, -0.25, 1.25)
        b0 = 1.0 - b1 - b2
        bary = np.stack([b0, b1, b2], axis=1)
        corners = faces[ids_c]                 # (M,3)
        p = (pos[corners] * bary[:, :, None]).sum(axis=1)
        yield (r0, r1), p, bary, ids_c, valid


def _world_to_pixel(px: np.ndarray, S: int) -> np.ndarray:
    px_to_world = 2.0 / S
    x = px[:, 0] / px_to_world + S / 2.0
    y = S / 2.0 - px[:, 1] / px_to_world
    return np.stack([x, y], axis=1)


def _sample(img: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear-ish (nearest) sampling of an (H,W,C) float image."""
    h, w = img.shape[:2]
    xi = np.clip(np.round(xy[:, 0]).astype(np.int32), 0, w - 1)
    yi = np.clip(np.round(xy[:, 1]).astype(np.int32), 0, h - 1)
    return img[yi, xi]


def _dilate_uv(rgba: np.ndarray, iterations: int = 8) -> np.ndarray:
    """Bleed texture colour into empty UV gutters to avoid seam artefacts."""
    out = rgba.copy()
    alpha = out[..., 3]
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(iterations):
        empty = alpha == 0
        if not empty.any():
            break
        rgb = out[..., :3].astype(np.float32)
        wsum = cv2.filter2D((alpha > 0).astype(np.float32), -1, kernel, borderType=cv2.BORDER_REPLICATE)
        acc = np.zeros_like(rgb)
        for c in range(3):
            acc[..., c] = cv2.filter2D(rgb[..., c] * (alpha > 0), -1, kernel,
                                       borderType=cv2.BORDER_REPLICATE)
        fill = (wsum > 0) & empty
        for c in range(3):
            acc[..., c][fill] /= wsum[fill]
        out[..., :3][fill] = acc[..., :3][fill].astype(np.uint8)
        alpha[fill] = 1
    return out


def _extend_colors(img: np.ndarray, mask: np.ndarray, rounds: int = 32) -> np.ndarray:
    """Push object colour outward into the background, k-nearest ring at a time."""
    out = img.copy()
    known = mask.copy()
    h, w = known.shape
    ys, xs = np.where(~known)
    if ys.size == 0:
        return out
    dist_src = np.full((h, w), np.inf, dtype=np.float32)
    # iterative dilation of colour+mask
    color_acc = out.copy()
    k = np.ones((3, 3), np.float32)
    known_f = known.astype(np.float32)
    for _ in range(rounds):
        if known_f.all():
            break
        wsum = cv2.filter2D(known_f, -1, k, borderType=cv2.BORDER_REPLICATE)
        new_rgb = np.zeros_like(out)
        for c in range(3):
            new_rgb[..., c] = cv2.filter2D(color_acc[..., c] * known_f, -1, k,
                                           borderType=cv2.BORDER_REPLICATE)
        fill = (wsum > 0) & (known_f < 0.5)
        if not fill.any():
            break
        for c in range(3):
            new_rgb[..., c][fill] /= wsum[fill]
        color_acc[fill] = new_rgb[fill]
        known_f[fill] = 1.0
    out = color_acc
    _ = dist_src, ys, xs
    return out


def _sample_sdf(sdf: np.ndarray, origin, spacing: float, pts: np.ndarray) -> np.ndarray:
    idx = (pts - np.array(origin, dtype=np.float64)[None, :]) / spacing
    nz, ny, nx = sdf.shape
    coords = np.stack([idx[:, 2], idx[:, 1], idx[:, 0]], axis=1)  # z,y,x order
    inside = np.all((coords >= 0) & (coords < np.array([nz, ny, nx]) - 1), axis=1)
    out = np.full(len(pts), 0.05, dtype=np.float32)
    if inside.any():
        c = coords[inside]
        from scipy import ndimage
        out[inside] = ndimage.map_coordinates(sdf, c.T, order=1, mode="nearest")
    return out


def fit_texture_size(requested: int, budget_mb: int) -> int:
    """Shrink the atlas if the float working buffers would break the budget.

    Peak working set of the baker is roughly ``5 * N * 4`` bytes for the
    position/normal/albedo buffers plus ``~30 %`` headroom, with ``N = size**2``.
    """
    size = int(requested)
    while size > 256:
        n = size * size
        need_mb = (n * 4 * 9) / (1024 * 1024) * 1.35
        if need_mb <= budget_mb * 0.45:
            return size
        size = int(size / 1.3) // 8 * 8
    return 256


def bake(mesh: MeshResult, image_rgb: np.ndarray, image_alpha: np.ndarray,
         sdf: np.ndarray, origin, spacing: float, normal_matrix: dict,
         cfg: PipelineConfig) -> TextureResult:
    res = fit_texture_size(cfg.texture_size, cfg.memory_budget_mb)
    if res != cfg.texture_size:
        log.warning("Atlas reduced %d -> %d px to respect the memory budget",
                    cfg.texture_size, res)
    S = image_rgb.shape[0]
    img = image_rgb.astype(np.float32) / 255.0
    alpha_img = image_alpha.astype(np.float32)
    # Edge-extend the object colour across the background so that mirrored or
    # grazing-angle samples never pick up the (white / black) canvas.
    img = _extend_colors(img, alpha_img > 0.5, rounds=48)

    tri_id = _barycentric_buffers(res, mesh)
    rgb = np.zeros((res, res, 3), dtype=np.float32)
    t_alpha = np.zeros((res, res), dtype=np.float32)
    pos_map = np.zeros((res, res, 3), dtype=np.float32)
    normal_map_t = np.zeros((res, res, 3), dtype=np.float32)
    depth_sign = np.zeros((res, res), dtype=np.int8)

    faces = mesh.faces
    v = mesh.vertices
    tri = v[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn_norm = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)

    # recover original (pre-normalisation) world coordinates
    nscale = normal_matrix["applied_scale"]
    ncenter = np.array(normal_matrix["applied_center"], dtype=np.float32)

    covered = 0
    for (r0, r1), P, bary, ids, valid in _texel_positions(res, mesh, tri_id):
        if P is None:
            continue
        rows = r1 - r0
        flat_valid = valid.reshape(rows, res)
        orig = P / np.float32(nscale) + ncenter.astype(np.float32)
        xy = _world_to_pixel(orig, S)
        z = orig[:, 2]
        front = z >= 0.0
        sample_xy = xy.copy()
        if cfg.back_mirror_texture:
            mx = (S - 1) - xy[:, 0]
            sample_xy[:, 0] = np.where(front, xy[:, 0], mx)
        else:
            sample_xy[:, 0] = np.where(front, xy[:, 0], xy[:, 0])
        col = _sample(img, sample_xy)
        a = _sample(alpha_img[:, :, None], sample_xy)[:, 0]
        nrm = fn_norm[ids]
        nworld = nrm * np.where(front, 1.0, -1.0)[:, None]

        # write back into the atlas (row indices are chunk-local -> offset)
        local_r, local_c = np.where(flat_valid)
        rows_idx = local_r + r0
        rgb[rows_idx, local_c] = col
        t_alpha[rows_idx, local_c] = np.clip(a, 0.0, 1.0)
        pos_map[rows_idx, local_c] = orig
        normal_map_t[rows_idx, local_c] = nworld
        depth_sign[rows_idx, local_c] = np.where(front, 1, -1).astype(np.int8)
        covered += int(rows_idx.size)

    coverage = covered / float(res * res)
    front_fraction = float((depth_sign > 0).sum() / max(covered, 1))
    if coverage < 1e-4:
        raise ValueError("UV baking produced no covered texels.")

    # ---- de-lighting (optional) --------------------------------------
    if cfg.bake_texture:
        lum = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        cover = (t_alpha > 0).astype(np.float32)
        shaded = cv2.GaussianBlur(lum * cover, (0, 0), res * 0.02)
        weight = cv2.GaussianBlur(cover, (0, 0), res * 0.02)
        shade = np.maximum(shaded / np.maximum(weight, 1e-5), 1e-3)
        target = float((lum[cover > 0]).mean()) if (cover > 0).any() else 0.5
        gain = np.clip(target / shade, 0.85, 1.25)
        albedo = np.clip(rgb * gain[..., None], 0.0, 1.0)
        # the reference photo already contains baked-in studio lighting; a full
        # delighting pass over-saturates chroma, so only a mild correction is
        # applied and the result is blended back toward the measured radiance.
        albedo = 0.5 * albedo + 0.5 * rgb
        albedo = np.where((cover > 0)[..., None], albedo, rgb)
    else:
        albedo = rgb

    # ---- ambient occlusion from the SDF ------------------------------
    ao = np.ones((res, res), dtype=np.float32)
    if cfg.generate_pbr_maps:
        sel = t_alpha > 0
        pts = pos_map[sel]
        nrm = normal_map_t[sel]
        nlen = np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        nrm = nrm / nlen
        occ = np.zeros(len(pts), dtype=np.float32)
        for t, w in ((0.012, 0.45), (0.03, 0.3), (0.06, 0.25)):
            probe = pts + nrm * t
            d = _sample_sdf(sdf, origin, spacing, probe.astype(np.float64))
            occ += w * np.clip(1.0 - (d / (t * 1.6)), 0.0, 1.0)
        ao_vals = np.clip(1.0 - cfg.ao_strength * occ, 0.15, 1.0)
        ao[sel] = ao_vals

    # ---- roughness / metallic estimates ------------------------------
    gray = cv2.cvtColor(np.clip(albedo * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    local = cv2.GaussianBlur(gray.astype(np.float32) / 255.0, (0, 0), 3.0)
    detail = np.abs(gray.astype(np.float32) / 255.0 - local)
    rough = np.clip(0.42 + 1.6 * detail, 0.18, 0.95)
    sat = cv2.cvtColor(np.clip(albedo * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)[..., 1]
    metal = np.clip(((local > 0.72) & (sat < 40)).astype(np.float32) * 0.45, 0.0, 0.9)

    # normal/tangent estimation consumes the position buffer, so it runs before
    # the atlas buffers are released.
    # ---- tangent-space normal map from the baked surface -------------
    nmap = np.zeros((res, res, 3), dtype=np.float32)
    nmap[..., 2] = 1.0
    if cfg.generate_pbr_maps:
        px = pos_map.copy()
        px[t_alpha <= 0] = np.nan
        du = np.zeros_like(px)
        dv = np.zeros_like(px)
        du[:, 1:-1] = px[:, 2:] - px[:, :-2]
        dv[1:-1, :] = px[2:, :] - px[:-2, :]
        T = np.nan_to_num(du)
        B = np.nan_to_num(dv)
        valid = (t_alpha > 0) & (np.linalg.norm(T, axis=2) > 1e-9) & (np.linalg.norm(B, axis=2) > 1e-9)
        if valid.any():
            N = normal_map_t.copy()
            Nn = N / np.maximum(np.linalg.norm(N, axis=2, keepdims=True), 1e-9)
            Tn = T / np.maximum(np.linalg.norm(T, axis=2, keepdims=True), 1e-9)
            Bn = B / np.maximum(np.linalg.norm(B, axis=2, keepdims=True), 1e-9)
            tx = (Nn * Tn).sum(axis=2)
            ty = (Nn * Bn).sum(axis=2)
            tz = (Nn * np.cross(Tn, Bn)).sum(axis=2)
            tz = np.sign(np.where(np.abs(tz) < 1e-6, 1.0, tz)) * np.maximum(np.abs(tz), 0.15)
            enc = np.stack([tx * 0.5 + 0.5, ty * 0.5 + 0.5, np.clip(tz, 0, 1)], axis=2)
            nmap = np.where(valid[..., None], enc, nmap)
            nmap = cv2.GaussianBlur(nmap, (3, 3), 0.6)

    # release the largest transient buffers before packing
    del pos_map, normal_map_t, depth_sign
    # ---- pack ---------------------------------------------------------
    def rgba(rgb3: np.ndarray, a: np.ndarray | None = None) -> np.ndarray:
        out = np.zeros((res, res, 4), dtype=np.uint8)
        out[..., :3] = np.clip(rgb3 * 255.0, 0, 255).astype(np.uint8)
        out[..., 3] = (np.clip(a if a is not None else t_alpha, 0, 1) * 255).astype(np.uint8)
        return out

    base = rgba(albedo)
    base = _dilate_uv(base)
    nrm_rgba = rgba(nmap)
    orm = np.zeros((res, res, 4), dtype=np.uint8)
    orm[..., 0] = np.clip(ao * 255, 0, 255).astype(np.uint8)
    orm[..., 1] = np.clip(rough * 255, 0, 255).astype(np.uint8)
    orm[..., 2] = np.clip(metal * 255, 0, 255).astype(np.uint8)
    orm[..., 3] = base[..., 3]
    orm = _dilate_uv(orm)
    ao_rgba = np.zeros_like(orm)
    ao_rgba[..., 0] = orm[..., 0]
    ao_rgba[..., 1] = orm[..., 0]
    ao_rgba[..., 2] = orm[..., 0]
    ao_rgba[..., 3] = orm[..., 3]

    maps = {
        "basecolor": base,
        "normal": nrm_rgba,
        "orm": orm,
        "ao": ao_rgba,
        "roughness": np.stack([orm[..., 1]] * 3 + [orm[..., 3]], axis=2),
        "metallic": np.stack([orm[..., 2]] * 3 + [orm[..., 3]], axis=2),
    }
    stats = {
        "texture_size": res,
        "uv_coverage": float(coverage),
        "mean_ao": float(ao[t_alpha > 0].mean()) if (t_alpha > 0).any() else 1.0,
        "mean_roughness": float(rough[t_alpha > 0].mean()) if (t_alpha > 0).any() else 0.5,
        "mean_metallic": float(metal[t_alpha > 0].mean()) if (t_alpha > 0).any() else 0.0,
        "delit": bool(cfg.bake_texture),
        "back_face_handling": "mirrored-projection" if cfg.back_mirror_texture else "zero",
        "front_texel_fraction": front_fraction,
    }
    return TextureResult(maps=maps, coverage=coverage, stats=stats)
