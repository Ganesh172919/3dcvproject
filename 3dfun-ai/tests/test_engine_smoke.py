"""Real-engine integration test (slow: ~20 s on the draft preset).

Marked ``slow`` — run with ``pytest -m slow``. It exercises the actual
reconstruction pipeline (matting, depth, volumetric inflation, marching cubes,
UV bake, export) instead of the mock provider, and asserts the watertightness
contract the product advertises.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

pytestmark = [pytest.mark.slow, pytest.mark.engine]

FAST = {
    "canvas_size": 384,
    "depth_input_size": 392,
    "voxel_res": 56,
    "target_faces": 6000,
    "texture_size": 512,
    "turntable_frames": 8,
    "render_size": 320,
    "formats": ("glb", "gltf", "obj", "ply", "stl"),
    "render_preview": True,
    "render_turntable": False,
    "bake_texture": True,
}
SOURCE = ROOT / "engine" / "testdata" / "sneaker.png"


@pytest.fixture(scope="module")
def draft_run(tmp_path_factory):
    from recon import reconstruct
    from recon.config import PipelineConfig

    out = tmp_path_factory.mktemp("recon")
    cfg = PipelineConfig(**FAST)
    result = reconstruct(SOURCE, out, cfg=cfg, asset_name="smoke")
    return out, result, cfg


def test_runs_and_writes_artefacts(draft_run):
    out, result, _ = draft_run
    assert (out / "model.glb").exists()
    assert (out / "model.obj").exists()
    assert (out / "manifest.json").exists()
    assert (out / "model.glb").stat().st_size > 50_000
    assert set(result.files) >= {"obj", "glb", "gltf", "bin", "ply", "stl", "mtl", "basecolor.png"}
    assert result.timings["total"] > 0


def test_mesh_is_watertight(draft_run):
    _, result, _ = draft_run
    mesh = result.manifest["mesh"]
    assert mesh["boundary_edges"] == 0
    assert mesh["nonmanifold_edges"] == 0
    assert mesh["watertight"] is True
    assert mesh["euler_characteristic"] == 2          # genus-0 shell
    assert mesh["faces"] >= 2000
    assert mesh["vertices"] > 0


def test_manifest_declares_its_limitations(draft_run):
    _, result, _ = draft_run
    manifest = result.manifest
    assert manifest["manifest_version"]
    assert manifest["engine"]["depth_model"]
    assert manifest["config_hash"]
    assert manifest["volume"]["voxel_size"] > 0
    # the view-projected atlas is a documented design choice, not an accident
    assert manifest["mesh"]["uv_parameterization"]["parameterization"] == "planar-view-projection"
    assert manifest["limitations"]


def test_exported_files_carry_the_full_geometry(draft_run):
    """Both exports must contain every triangle the engine reported.

    OBJ/glTF store one position per *corner* so that UV seams can be drawn
    without splitting triangles; the buffers are therefore larger than the
    source mesh but must not lose a single face.
    """
    out, result, _ = draft_run
    import trimesh

    reported = result.manifest["mesh"]["faces"]
    obj = trimesh.load(out / "model.obj", force="mesh", process=False)
    assert len(obj.faces) == reported
    scene = trimesh.load(out / "model.glb")
    glb_faces = sum(len(g.faces) for g in scene.geometry.values())
    assert glb_faces == reported
    assert (out / "model.ply").stat().st_size > 10_000


def test_ply_keeps_source_topology(draft_run):
    """The PLY is written from the raw mesh, so it stays closed and manifold."""
    out, result, _ = draft_run
    import trimesh

    ply = trimesh.load(out / "model.ply", force="mesh", process=False)
    assert ply.is_watertight
    assert len(ply.faces) == result.manifest["mesh"]["faces"]
    assert ply.volume > 0


def test_deterministic_for_identical_input(draft_run):
    """The recorded config hash must reproduce the *effective* configuration.

    The pipeline calibrates the voxel grid to the triangle budget, so the
    manifest stores the config that actually ran; rebuilding that config has to
    give the same hash bit for bit.
    """
    from recon.config import PipelineConfig

    _, result, _ = draft_run
    effective = dict(FAST)
    effective["voxel_res"] = result.manifest["resolution_calibration"]["chosen_res"]
    assert PipelineConfig(**effective).config_hash() == result.manifest["config_hash"]
    assert result.manifest["config"]["voxel_res"] == effective["voxel_res"]
