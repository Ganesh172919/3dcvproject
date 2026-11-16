# Specialized geometry exercises

These standalone CPU exercises use simple silhouette and depth cues. The Studio's main mesh pipeline also uses a pretrained relative-depth model.

Run from the project source directory after installing `requirements.txt`:

```sh
python -m engine.specialized_3d.run_specialized --image engine/testdata/sneaker.png --mode all --out out/specialized-demo
```

| Mode | Output | Assumptions |
| --- | --- | --- |
| `pointcloud` | PLY with color and normals | Uses a silhouette and intensity-based depth proxy by default. |
| `revolution` | Surface-of-revolution OBJ | Assumes the object is rotationally symmetric about a vertical axis. |
| `space_carving` | Visual hull OBJ and voxel NPZ | Orthographic silhouettes at known angles. The single-image demo reuses its silhouette from a second angle. |
| `surface_mesh` | Depth-guided SDF OBJ | Uses a bounded slab and heuristic depth to complete the volume. |

For actual orthographic images at known angles, pass them to the visual hull command in matching order:

```sh
python -m engine.specialized_3d.space_carving_construction --images front.png side.png --angles 0 90 --res 64
```

The PLY and OBJ outputs can be opened in CloudCompare or MeshLab. See the [main Studio tutorial](../tutorials/quickstart.md) for learned depth and textured export.
