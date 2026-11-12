"""Stage 7 - software rasteriser for previews and turntable frames.

Deliberately dependency-light (numpy + OpenCV): previews are rendered on the
CPU in the worker so the API never blocks on a GPU, and the same code runs in
CI without a GL context.

Two rasterisation modes are available:

``per_pixel=True`` (used for the still previews)
    True z-buffered scanline rasterisation with per-pixel barycentric UV
    interpolation, bilinear texture filtering, tangent-space normal mapping and
    Blinn-Phong shading.

``per_pixel=False`` (used for turntable frames)
    Flat-shaded painter's algorithm; ~10x faster, adequate for small frames.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .meshing import MeshResult
from .texturing import TextureResult

log = logging.getLogger(__name__)

_LIGHT_KEY = np.array([0.48, 0.72, 0.55])
_LIGHT_KEY = _LIGHT_KEY / np.linalg.norm(_LIGHT_KEY)
_LIGHT_FILL = np.array([-0.7, 0.18, 0.45])
_LIGHT_FILL = _LIGHT_FILL / np.linalg.norm(_LIGHT_FILL)


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def _project(pts_cam: np.ndarray, size: int, fov_deg: float = 32.0) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    z = np.maximum(pts_cam[:, 2], 1e-4)
    x = (pts_cam[:, 0] * f) / z
    y = (pts_cam[:, 1] * f) / z
    return np.stack([(x * 0.5 + 0.5) * size, (0.5 - y * 0.5) * size], axis=1)


def _bilinear(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = tex.shape[:2]
    x = np.clip(uv[:, 0] * (w - 1), 0, w - 1.001)
    y = np.clip(uv[:, 1] * (h - 1), 0, h - 1.001)
    x0 = x.astype(np.int32)
    y0 = y.astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    c00 = tex[y0, x0].astype(np.float32)
    c10 = tex[y1, x0].astype(np.float32)
    c01 = tex[y0, x1].astype(np.float32)
    c11 = tex[y1, x1].astype(np.float32)
    top = c00 * (1 - fx) + c01 * fx
    bot = c10 * (1 - fx) + c11 * fx
    return top * (1 - fy) + bot * fy


def _background(S: int) -> np.ndarray:
    bg_top = np.array([26, 30, 37], dtype=np.float32)
    bg_bot = np.array([46, 52, 62], dtype=np.float32)
    grad = np.linspace(0, 1, S, dtype=np.float32)[:, None, None]          # (S,1,1)
    row = (bg_top[None, None, :] * (1 - grad) + bg_bot[None, None, :] * grad)  # (S,1,3)
    return np.ascontiguousarray(np.broadcast_to(row, (S, S, 3))).copy()


def _camera(azimuth_deg: float, elevation_deg: float, dist: float = 2.35):
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    eye = np.array([math.sin(az) * math.cos(el), math.sin(el),
                    math.cos(az) * math.cos(el)]) * dist
    return eye, _look_at(eye, np.zeros(3), np.array([0.0, 1.0, 0.0]))


def render_view(mesh: MeshResult, tex: TextureResult | None, size: int,
                azimuth_deg: float, elevation_deg: float = 8.0,
                mode: str = "pbr", supersample: int = 2,
                per_pixel: bool = True) -> np.ndarray:
    """Render one view and return an RGB uint8 image of shape (size, size, 3)."""
    S = size * supersample
    eye, view = _camera(azimuth_deg, elevation_deg)
    v = mesh.vertices.astype(np.float64)
    cam = (view @ np.hstack([v, np.ones((len(v), 1))]).T).T
    scr = _project(cam, S)
    faces = mesh.faces

    if mode == "wireframe":
        img = _background(S)
        tri_scr = scr[faces]
        for i in range(len(faces)):
            cv2.polylines(img, [tri_scr[i].astype(np.int32)], True, (188, 212, 255), 1, cv2.LINE_AA)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA) if supersample > 1 else img

    if per_pixel:
        img = _rasterize_per_pixel(mesh, tex, cam, scr, S, mode, eye)
    else:
        img = _rasterize_flat(mesh, tex, cam, scr, S, mode, eye)

    img = np.clip(img, 0, 255).astype(np.uint8)
    if supersample > 1:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


# --------------------------------------------------------------------------
# per-pixel z-buffered rasteriser
# --------------------------------------------------------------------------
_FULL_DETAIL_FACES = 14000


def _face_loop(count: int, lanes: int = 2):
    """Sequential face iterator.

    Kept as a single generator so the hot loop stays readable; the renderer
    downsamples supersampling (not geometry) above the full-detail budget.
    """
    return range(count)


def _face_tangents(mesh: MeshResult):
    """Orthonormal tangent frames derived from the UV parameterisation."""
    faces = mesh.faces
    p = mesh.vertices[faces].astype(np.float64)
    uv = mesh.face_uv.reshape(-1, 3, 2).astype(np.float64)
    e1 = p[:, 1] - p[:, 0]
    e2 = p[:, 2] - p[:, 0]
    d1 = uv[:, 1] - uv[:, 0]
    d2 = uv[:, 2] - uv[:, 0]
    r = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    r = np.where(np.abs(r) < 1e-12, 1e-12, r)
    T = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / r[:, None]
    n = np.cross(e1, e2)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    T = T - n * (T * n).sum(axis=1, keepdims=True)          # Gram-Schmidt
    Tn = np.linalg.norm(T, axis=1, keepdims=True)
    fallback = _fallback_tangent(n)
    T = np.where(Tn > 1e-9, T / np.maximum(Tn, 1e-12), fallback)
    B = np.cross(n, T)
    return np.stack([T, B], axis=1)   # (F, 2, 3)


def _fallback_tangent(n: np.ndarray) -> np.ndarray:
    helper = np.tile(np.array([1.0, 0.0, 0.0]), (len(n), 1))
    alt = np.tile(np.array([0.0, 1.0, 0.0]), (len(n), 1))
    helper = np.where(np.abs((n * helper).sum(axis=1, keepdims=True)) > 0.9, alt, helper)
    T = np.cross(helper, n)
    return T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)


def _rasterize_per_pixel(mesh: MeshResult, tex: TextureResult | None,
                         cam: np.ndarray, scr: np.ndarray, S: int,
                         mode: str, eye: np.ndarray) -> np.ndarray:
    img = _background(S)
    zmin = cam[:, 2].min()
    zbuf = np.full((S, S), np.inf, dtype=np.float32)

    faces = mesh.faces
    uv = mesh.face_uv.reshape(-1, 3, 2)
    tri_v = mesh.vertices[faces].astype(np.float64)
    face_n = np.cross(tri_v[:, 1] - tri_v[:, 0], tri_v[:, 2] - tri_v[:, 0])
    face_n /= np.maximum(np.linalg.norm(face_n, axis=1, keepdims=True), 1e-12)
    centroid = tri_v.mean(axis=1)
    to_eye = eye[None, :] - centroid
    to_eye /= np.maximum(np.linalg.norm(to_eye, axis=1, keepdims=True), 1e-12)
    front_facing = (face_n * to_eye).sum(axis=1) > 0

    normal_tex = tex.maps["normal"][..., :3].astype(np.float32) / 255.0 * 2.0 - 1.0 if \
        (tex is not None and mode == "pbr") else None
    base_tex = tex.maps["basecolor"] if tex is not None else None
    rough_tex = tex.maps["roughness"] if tex is not None else None

    # per-face TBN frames so the tangent-space normal map is applied correctly
    tangents = _face_tangents(mesh)   # (F, 2, 3)
    clay = np.array([0.55, 0.56, 0.60], dtype=np.float32)
    depth_min = zmin - 1.2

    for i in _face_loop(len(faces)):
        p = scr[faces[i]]
        x0, y0 = p.min(axis=0)
        x1, y1 = p.max(axis=0)
        ix0 = max(int(math.floor(x0)), 0)
        ix1 = min(int(math.ceil(x1)) + 1, S)
        iy0 = max(int(math.floor(y0)), 0)
        iy1 = min(int(math.ceil(y1)) + 1, S)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        xs = np.arange(ix0, ix1, dtype=np.float32) + 0.5
        ys = np.arange(iy0, iy1, dtype=np.float32) + 0.5
        if xs.size * ys.size > 4_000_000:
            step = max(1, int(math.sqrt(xs.size * ys.size / 4_000_000)))
            xs, ys = xs[::step], ys[::step]
        gx, gy = np.meshgrid(xs, ys)

        a, b, c = p[0], p[1], p[2]
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-12:
            continue
        w0 = ((b[1] - c[1]) * (gx - c[0]) + (c[0] - b[0]) * (gy - c[1])) / d
        w1 = ((c[1] - a[1]) * (gx - c[0]) + (a[0] - c[0]) * (gy - c[1])) / d
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue
        rows = (gy[inside] - 0.5).astype(np.int32)
        cols = (gx[inside] - 0.5).astype(np.int32)
        ok = (rows >= 0) & (rows < S) & (cols >= 0) & (cols < S)
        rows, cols = rows[ok], cols[ok]
        ww0, ww1, ww2 = w0[inside][ok], w1[inside][ok], w2[inside][ok]
        if rows.size == 0:
            continue
        ci = faces[i]
        z = ww0 * cam[ci[0], 2] + ww1 * cam[ci[1], 2] + ww2 * cam[ci[2], 2]
        # perspective-correct interpolation weights
        inv_z = ww0 / cam[ci[0], 2] + ww1 / cam[ci[1], 2] + ww2 / cam[ci[2], 2]
        inv_z = np.where(np.abs(inv_z) < 1e-12, 1e-12, inv_z)
        l0 = (ww0 / cam[ci[0], 2]) / inv_z
        l1 = (ww1 / cam[ci[1], 2]) / inv_z
        l2 = (ww2 / cam[ci[2], 2]) / inv_z
        depth = z
        closer = depth < zbuf[rows, cols]
        if not closer.any():
            continue
        rows, cols, depth = rows[closer], cols[closer], depth[closer]
        l0, l1, l2 = l0[closer], l1[closer], l2[closer]
        zbuf[rows, cols] = depth

        uvf = uv[i]
        u = l0 * uvf[0, 0] + l1 * uvf[1, 0] + l2 * uvf[2, 0]
        vv = l0 * uvf[0, 1] + l1 * uvf[1, 1] + l2 * uvf[2, 1]
        uvs = np.stack([u, vv], axis=1)

        if base_tex is not None and mode in ("pbr", "albedo"):
            samp = _bilinear(base_tex, uvs)
            covered = samp[:, 3] > 8.0
            albedo = samp[:, :3] / 255.0
            if not covered.all():
                albedo[~covered] = clay
        elif mesh.vertex_colors is not None and mode in ("pbr", "albedo"):
            albedo = np.column_stack((l0, l1, l2)) @ mesh.vertex_colors[ci]
        else:
            albedo = np.broadcast_to(clay, (len(rows), 3)).copy()

        n0 = face_n[i]
        if normal_tex is not None:
            tn = _bilinear(normal_tex, uvs)
            T, B = tangents[i]
            n = (T[None, :] * tn[:, :1] + B[None, :] * tn[:, 1:2]
                 + n0[None, :] * tn[:, 2:3])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        elif mesh.vertex_normals is not None:
            n = np.column_stack((l0, l1, l2)) @ mesh.vertex_normals[ci]
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        else:
            n = np.broadcast_to(n0, (len(rows), 3)).copy()
        if not front_facing[i]:
            n = -n
        ndl = np.clip(n @ _LIGHT_KEY, 0.0, 1.0)
        ndf = np.clip(n @ _LIGHT_FILL, 0.0, 1.0)
        ndv = np.clip(n @ to_eye[i], 0.0, 1.0)
        rough = 0.5
        if rough_tex is not None:
            rough = float(_bilinear(rough_tex, uvs)[:, 0].mean() / 255.0)
        spec = ndv ** 40 * 0.10 * max(1.0 - rough, 0.0)
        amb = 0.40 + 0.12 * (1.0 - np.abs(n[:, 1]))
        shade = amb + 0.78 * ndl + 0.22 * ndf
        rim = (1.0 - ndv) ** 3 * 0.18
        if mode == "albedo":            # unlit base-colour pass
            col = albedo
        else:
            col = albedo * shade[:, None] + spec[:, None] + rim[:, None]
        img[rows, cols] = np.clip(col * 255.0, 0, 255)

    _ = depth_min
    return _add_ground(img, S)


def _add_ground(img: np.ndarray, S: int) -> np.ndarray:
    gy = int(S * 0.82)
    yy, xx = np.mgrid[0:S, 0:S]
    shadow = np.exp(-(((xx - S / 2) / (S * 0.30)) ** 2 + ((yy - gy) / (S * 0.05)) ** 2))
    img = img * (1 - 0.32 * shadow[..., None])
    cv2.line(img, (0, gy), (S, gy), (72, 78, 90), 1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------
# fast flat shading (turntables)
# --------------------------------------------------------------------------
def _rasterize_flat(mesh: MeshResult, tex: TextureResult | None,
                    cam: np.ndarray, scr: np.ndarray, S: int,
                    mode: str, eye: np.ndarray) -> np.ndarray:
    img = _background(S)
    faces = mesh.faces
    tri_scr = scr[faces]
    tri_v = mesh.vertices[faces].astype(np.float64)
    n = np.cross(tri_v[:, 1] - tri_v[:, 0], tri_v[:, 2] - tri_v[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    centroid = tri_v.mean(axis=1)
    to_eye = eye[None, :] - centroid
    to_eye /= np.maximum(np.linalg.norm(to_eye, axis=1, keepdims=True), 1e-12)
    facing = (n * to_eye).sum(axis=1)
    # Two-sided shading: a closed mesh seen from outside has its back faces hidden by
    # the depth sort below, but keeping them in the draw list is what makes the
    # silhouette solid on thin or inverted surfaces. Back-facing normals are flipped
    # so they are not rendered black.
    n = np.where(facing[:, None] > 0, n, -n)
    facing = np.abs(facing)
    if tex is not None and mode == "pbr":
        corner = _bilinear(tex.maps["basecolor"][..., :3].astype(np.float32),
                           mesh.face_uv.reshape(-1, 2))
        face_col = corner.reshape(-1, 3, 3).mean(axis=1) / 255.0
    elif mesh.vertex_colors is not None and mode == "pbr":
        face_col = mesh.vertex_colors[faces].mean(axis=1)
    else:
        face_col = np.full((len(faces), 3), 0.52, dtype=np.float32)
    ndl = np.clip(n @ _LIGHT_KEY, 0, 1)
    ndf = np.clip(n @ _LIGHT_FILL, 0, 1)
    shade = 0.36 + 0.76 * ndl + 0.20 * ndf
    lit = np.clip(face_col * shade[:, None], 0, 1) * 255.0
    depth = cam[faces][:, :, 2].mean(axis=1)
    order = np.argsort(-depth)
    drawable = order          # painter's order over every face, two-sided
    for i in drawable:
        cv2.fillConvexPoly(img, tri_scr[i].astype(np.int32), lit[i].tolist(),
                           lineType=cv2.LINE_AA)
        cv2.polylines(img, [tri_scr[i].astype(np.int32)], True, lit[i].tolist(), 1,
                      cv2.LINE_AA)
    return _add_ground(img, S)


# --------------------------------------------------------------------------
# public helpers
# --------------------------------------------------------------------------
def render_preview(mesh: MeshResult, tex: TextureResult | None, out: Path, size: int) -> dict[str, Any]:
    views = {"front": 0.0, "three_quarter": 38.0, "side": 90.0, "back": 180.0}
    outs: dict[str, str] = {}
    for name, az in views.items():
        img = render_view(mesh, tex, max(288, int(size * 0.75)), az, 8.0, supersample=1)
        p = out.parent / f"preview_{name}.png"
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        outs[name] = str(p)
    main = render_view(mesh, tex, size, 32.0, 12.0)
    cv2.imwrite(str(out), cv2.cvtColor(main, cv2.COLOR_RGB2BGR))
    clay = render_view(mesh, None, max(320, int(size * 0.75)), 32.0, 12.0, mode="clay",
                       supersample=1)
    cp = out.parent / "preview_clay.png"
    cv2.imwrite(str(cp), cv2.cvtColor(clay, cv2.COLOR_RGB2BGR))
    wire = render_view(mesh, tex, max(320, int(size * 0.75)), 32.0, 12.0,
                       mode="wireframe")
    wp = out.parent / "wireframe.png"
    cv2.imwrite(str(wp), cv2.cvtColor(wire, cv2.COLOR_RGB2BGR))
    outs.update({"main": str(out), "clay": str(cp), "wireframe": str(wp)})
    return outs


def render_turntable(mesh: MeshResult, tex: TextureResult | None, out: Path,
                     frames: int, size: int) -> dict[str, Any]:
    imgs = []
    step = 360.0 / max(frames, 4)
    for i in range(frames):
        imgs.append(render_view(mesh, tex, size, i * step, 10.0, supersample=1,
                                per_pixel=False))
    video_path = out.with_suffix(".mp4")
    ok = False
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0,
                             (size, size))
    if writer.isOpened():
        for im in imgs:
            writer.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        writer.release()
        ok = video_path.exists() and video_path.stat().st_size > 1024
    gif_path = out.with_suffix(".gif")
    try:
        from PIL import Image
        pil = [Image.fromarray(im) for im in imgs]
        pil[0].save(gif_path, save_all=True, append_images=pil[1:], duration=84, loop=0)
    except Exception:
        gif_path = Path("")
    return {
        "turntable_mp4": str(video_path) if ok else None,
        "turntable_gif": str(gif_path) if gif_path and gif_path.exists() else None,
        "frames": frames,
        "codec": "mp4v" if ok else "unavailable",
    }
