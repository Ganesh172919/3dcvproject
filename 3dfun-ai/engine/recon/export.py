"""Stage 6 - exporters (GLB, GLTF+bin, OBJ, PLY, STL) and GLB material wiring."""
from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import PipelineConfig
from .meshing import MeshResult
from .texturing import TextureResult

log = logging.getLogger(__name__)


@dataclass
class ExportResult:
    files: dict[str, Path]
    stats: dict[str, Any]


def _corner_expand(mesh: MeshResult):
    """Expand to one vertex per face-corner (required for per-corner UVs)."""
    corner_uv = mesh.face_uv.reshape(-1, 2).astype(np.float32)
    vidx = mesh.faces.reshape(-1).astype(np.int32)
    positions = mesh.vertices[vidx].astype(np.float32)
    normals = (_corner_normals(mesh, positions) if mesh.vertex_normals is None
               else mesh.vertex_normals[vidx].astype(np.float32))
    return positions, normals, corner_uv


def _corner_normals(mesh: MeshResult, positions: np.ndarray) -> np.ndarray:
    faces = mesh.faces
    tri = mesh.vertices[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).astype(np.float32)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    return np.repeat(fn, 3, axis=0)


def _png_bytes(arr_rgba: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(arr_rgba, cv2.COLOR_RGBA2BGRA))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


# --------------------------------------------------------------------------
# GLB
# --------------------------------------------------------------------------
def write_glb(path: Path, mesh: MeshResult, tex: TextureResult | None,
              meta: dict[str, Any], cfg: PipelineConfig) -> Path:
    positions, normals, uvs = _corner_expand(mesh)
    n_vert = len(positions)

    images: list[tuple[str, bytes]] = []
    if tex is not None:
        images = [
            ("basecolor.png", _png_bytes(tex.maps["basecolor"])),
            ("normal.png", _png_bytes(tex.maps["normal"])),
            ("orm.png", _png_bytes(tex.maps["orm"])),
        ]

    bin_parts: list[bytes] = []
    views: list[dict] = []
    accessors: list[dict] = []

    def add_accessor(arr: np.ndarray, comp_type: int, n_comp: int, target: int,
                     include_bounds: bool = False) -> int:
        data = np.ascontiguousarray(arr).tobytes()
        offset = sum(len(p) for p in bin_parts)
        pad = (-offset) % 4
        if pad:
            bin_parts.append(b"\x00" * pad)
            offset += pad
        bin_parts.append(data)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data),
                      "target": target})
        acc = {"bufferView": len(views) - 1, "componentType": comp_type,
               "count": int(arr.shape[0]), "type": {1: "SCALAR", 2: "VEC2", 3: "VEC3"}[n_comp]}
        if include_bounds:
            acc["min"] = [float(x) for x in np.asarray(arr).min(axis=0)]
            acc["max"] = [float(x) for x in np.asarray(arr).max(axis=0)]
        accessors.append(acc)
        return len(accessors) - 1

    acc_pos = add_accessor(positions, 5126, 3, 34962, include_bounds=True)
    acc_nrm = add_accessor(normals, 5126, 3, 34962)
    acc_uv = add_accessor(uvs, 5126, 2, 34962)
    attributes = {"POSITION": acc_pos, "NORMAL": acc_nrm, "TEXCOORD_0": acc_uv}
    if mesh.vertex_colors is not None:
        srgb = mesh.vertex_colors[mesh.faces.ravel()]
        linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
        attributes["COLOR_0"] = add_accessor(linear.astype("<f4"), 5126, 3, 34962)

    indices: np.ndarray | None = None
    acc_idx = None
    if n_vert < 65536:
        indices = np.arange(n_vert, dtype=np.uint16)
        acc_idx = add_accessor(indices, 5123, 1, 34963)
    else:
        indices = np.arange(n_vert, dtype=np.uint32)
        acc_idx = add_accessor(indices, 5125, 1, 34963)

    img_defs = []
    for i, (name, data) in enumerate(images):
        offset = sum(len(p) for p in bin_parts)
        bin_parts.append(data)
        pad = (-len(data)) % 4
        if pad:
            bin_parts.append(b"\x00" * pad)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data)})
        img_defs.append({"bufferView": len(views) - 1, "mimeType": "image/png", "name": name})

    material: dict[str, Any] = {
        "name": "3DFUN_PBR",
        "pbrMetallicRoughness": {
            "baseColorFactor": [1, 1, 1, 1],
            "metallicFactor": 1.0 if tex is not None else 0.0,
            "roughnessFactor": 1.0,
        },
        "doubleSided": False,
    }
    if tex is not None:
        material["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
        material["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": 1}
        material["normalTexture"] = {"index": 2, "scale": 1.0}
        material["occlusionTexture"] = {"index": 1, "strength": 0.9}

    gltf: dict[str, Any] = {
        "asset": {
            "version": "2.0",
            "generator": f"3DFUN AI reconstruction engine {cfg.pipeline_version}",
            "extras": {k: (v if isinstance(v, (str, int, float, bool, list, dict)) else str(v))
                       for k, v in meta.items()},
        },
        "scene": 0,
        "scenes": [{"nodes": [0], "name": "Scene"}],
        "nodes": [{"mesh": 0, "name": meta.get("asset_name", "model")}],
        "meshes": [{
            "name": "reconstruction",
            "primitives": [{
                "attributes": attributes,
                "indices": acc_idx,
                "material": 0,
                "mode": 4,
            }],
        }],
        "materials": [material],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": 0}],
    }
    if tex is not None:
        gltf["textures"] = [{"source": 0, "sampler": 0}, {"source": 2, "sampler": 0},
                            {"source": 1, "sampler": 0}]
        gltf["images"] = img_defs
        gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]

    bin_blob = b"".join(p for p in bin_parts)
    while len(bin_blob) % 4:
        bin_blob += b"\x00"
    gltf["buffers"][0]["byteLength"] = len(bin_blob)

    json_blob = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(json_blob) % 4:
        json_blob += b" "

    total = 12 + 8 + len(json_blob) + 8 + len(bin_blob)
    with path.open("wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(json_blob), 0x4E4F534A))
        fh.write(json_blob)
        fh.write(struct.pack("<II", len(bin_blob), 0x004E4942))
        fh.write(bin_blob)
    return path


def write_gltf_separate(outdir: Path, mesh: MeshResult, tex: TextureResult | None,
                        meta: dict[str, Any], cfg: PipelineConfig) -> dict[str, Path]:
    glb_path = write_glb(outdir / "_tmp.glb", mesh, tex, meta, cfg)
    data = glb_path.read_bytes()
    json_len = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + json_len].decode("utf-8"))
    bin_off = 20 + json_len + 8
    bin_blob = data[bin_off:]
    files: dict[str, Path] = {}

    bin_path = outdir / "model.bin"
    bin_path.write_bytes(bin_blob)
    buffer_views = gltf["bufferViews"]
    image_views = {img["bufferView"]: name for img, name in
                   zip(gltf.get("images", []), ["basecolor.png", "normal.png", "orm.png"])}
    for bv_idx, name in image_views.items():
        bv = buffer_views[bv_idx]
        (outdir / name).write_bytes(bin_blob[bv["byteOffset"]:bv["byteOffset"] + bv["byteLength"]])
        files[name] = outdir / name
        gltf["images"][list(image_views).index(bv_idx)]["uri"] = name
        gltf["images"][list(image_views).index(bv_idx)].pop("bufferView", None)
    gltf["buffers"][0]["uri"] = "model.bin"
    gltf_path = outdir / "model.gltf"
    gltf_path.write_text(json.dumps(gltf, indent=2), encoding="utf-8")
    glb_path.unlink(missing_ok=True)
    files["gltf"] = gltf_path
    files["bin"] = bin_path
    return files


# --------------------------------------------------------------------------
# OBJ / PLY / STL
# --------------------------------------------------------------------------
def write_obj(outdir: Path, mesh: MeshResult, tex: TextureResult | None,
              meta: dict[str, Any]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    obj = outdir / "model.obj"
    mtl = outdir / "model.mtl"
    positions, normals, uvs = _corner_expand(mesh)
    v = mesh.vertices
    lines = [f"# 3DFUN AI reconstruction - {meta.get('asset_name', 'model')}",
             f"# faces={len(mesh.faces)} vertices={len(v)}",
             "mtllib model.mtl", "o reconstruction"]
    vertices = v if mesh.vertex_colors is None else np.column_stack((v, mesh.vertex_colors))
    lines.extend("v " + " ".join(f"{value:.6f}" for value in row) for row in vertices)
    for t in uvs:
        # OBJ uses a bottom-left texture origin. The baked atlas and glTF use top-left.
        lines.append(f"vt {t[0]:.6f} {1.0 - t[1]:.6f}")
    vn = _corner_normals(mesh, positions)[::3] if mesh.vertex_normals is None else mesh.vertex_normals
    for x in vn:
        lines.append(f"vn {x[0]:.6f} {x[1]:.6f} {x[2]:.6f}")
    lines.append("usemtl 3DFUN_PBR")
    lines.append("s 1")
    for i, f in enumerate(mesh.faces):
        a, b, c = (int(x) + 1 for x in f)
        ti = i * 3
        na, nb, nc = (i + 1,) * 3 if mesh.vertex_normals is None else (a, b, c)
        lines.append(f"f {a}/{ti+1}/{na} {b}/{ti+2}/{nb} {c}/{ti+3}/{nc}")
    obj.write_text("\n".join(lines) + "\n", encoding="utf-8")
    files["obj"] = obj

    mtl_lines = ["# generated by 3DFUN AI", "newmtl 3DFUN_PBR",
                 "Ka 1.000 1.000 1.000", "Kd 1.000 1.000 1.000",
                 "Ks 0.200 0.200 0.200", "Ns 64.0", "d 1.0", "illum 2"]
    if tex is not None:
        cv2.imwrite(str(outdir / "basecolor.png"),
                    cv2.cvtColor(tex.maps["basecolor"], cv2.COLOR_RGBA2BGRA))
        cv2.imwrite(str(outdir / "normal.png"),
                    cv2.cvtColor(tex.maps["normal"], cv2.COLOR_RGBA2BGRA))
        cv2.imwrite(str(outdir / "orm.png"), cv2.cvtColor(tex.maps["orm"], cv2.COLOR_RGBA2BGRA))
        mtl_lines += ["map_Kd basecolor.png", "map_Bump -bm 1.0 normal.png",
                      "map_Ns orm.png"]
        files["basecolor.png"] = outdir / "basecolor.png"
        files["normal.png"] = outdir / "normal.png"
        files["orm.png"] = outdir / "orm.png"
    mtl.write_text("\n".join(mtl_lines) + "\n", encoding="utf-8")
    files["mtl"] = mtl
    return files


def write_ply(path: Path, mesh: MeshResult) -> Path:
    v = mesh.vertices
    f = mesh.faces
    header = [
        "ply", "format binary_little_endian 1.0",
        "comment generated by 3DFUN AI", f"element vertex {len(v)}",
        "property float x", "property float y", "property float z",
        "property float nx", "property float ny", "property float nz",
    ]
    colors = mesh.vertex_colors
    if colors is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header += [f"element face {len(f)}", "property list uchar int vertex_indices", "end_header"]
    vn = np.zeros_like(v)
    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    np.add.at(vn, f[:, 0], fn)
    np.add.at(vn, f[:, 1], fn)
    np.add.at(vn, f[:, 2], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    with path.open("wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        xyz_normals = np.hstack([v, vn]).astype("<f4")
        if colors is None:
            fh.write(xyz_normals.tobytes())
        else:
            vertices = np.empty(len(v), dtype=[("xyz_normals", "<f4", 6), ("rgb", "u1", 3)])
            vertices["xyz_normals"] = xyz_normals
            vertices["rgb"] = np.rint(colors * 255).astype(np.uint8)
            fh.write(vertices.tobytes())
        packed = np.zeros(len(f), dtype=[("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
        packed["n"] = 3
        packed["a"], packed["b"], packed["c"] = f[:, 0], f[:, 1], f[:, 2]
        fh.write(packed.tobytes())
    return path


def write_stl(path: Path, mesh: MeshResult) -> Path:
    import trimesh
    m = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    m.export(str(path), file_type="stl")
    return path


def export_all(outdir: Path, mesh: MeshResult, tex: TextureResult | None,
               meta: dict[str, Any], cfg: PipelineConfig) -> ExportResult:
    outdir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    formats = set(cfg.formats)
    for fmt in formats:
        if fmt == "glb":
            files["glb"] = write_glb(outdir / "model.glb", mesh, tex, meta, cfg)
        elif fmt == "gltf":
            files.update(write_gltf_separate(outdir, mesh, tex, meta, cfg))
        elif fmt == "obj":
            files.update(write_obj(outdir, mesh, tex, meta))
        elif fmt == "ply":
            files["ply"] = write_ply(outdir / "model.ply", mesh)
        elif fmt == "stl":
            files["stl"] = write_stl(outdir / "model.stl", mesh)
    if tex is not None:
        for name in ("basecolor", "normal", "orm", "ao", "roughness", "metallic"):
            p = outdir / f"{name}.png"
            cv2.imwrite(str(p), cv2.cvtColor(tex.maps[name], cv2.COLOR_RGBA2BGRA))
            files[f"tex_{name}"] = p
    sizes = {k: int(p.stat().st_size) for k, p in files.items() if p.exists()}
    return ExportResult(files=files, stats={"file_bytes": sizes, "formats": sorted(formats)})
