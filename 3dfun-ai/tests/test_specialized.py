import numpy as np
from PIL import Image
import pytest
import trimesh

from engine.specialized_3d.run_specialized import run_specialized_pipeline
from engine.specialized_3d.space_carving_construction import carve_visual_hull
from engine.specialized_3d.visualize_3d import generate_viewer_html


def test_all_specialized_modes_accept_grayscale(tmp_path):
    source = tmp_path / "gray.png"
    pixels = np.full((80, 64), 255, dtype=np.uint8)
    pixels[12:68, 14:50] = 25
    Image.fromarray(pixels).save(source)

    exports = run_specialized_pipeline(source, output_dir=tmp_path / "outputs", resolution=32)

    assert set(exports) == {"pointcloud", "revolution", "space_carving", "surface_mesh"}
    cloud = trimesh.load(exports["pointcloud"])
    assert len(cloud.vertices) > 0
    for mode in ("revolution", "space_carving", "surface_mesh"):
        mesh = trimesh.load(exports[mode], force="mesh")
        assert len(mesh.faces) > 0


def test_carving_rejects_mismatched_angles():
    mask = np.full((8, 8), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="one finite azimuth angle"):
        carve_visual_hull([mask, mask], [0])


def test_viewer_resolves_model_relative_to_output(tmp_path):
    model = tmp_path / "Sample mesh.obj"
    model.write_text("v 0 0 0\n", encoding="utf-8")
    viewer = generate_viewer_html(model, tmp_path / "viewer" / "view.html")
    assert 'loadModel("../Sample%20mesh.obj");' in viewer.read_text(encoding="utf-8")
