# Inspect a mesh in Studio

Start with a [local reconstruction](quickstart.md) or select a saved model from the homepage prototype library. Use **Inspect model** to load an example on the homepage or **Run in Studio** to try the source image there. A built-in GLB can be viewed and downloaded before signing in.

Drag the 3D pane to orbit. Use the scroll wheel or pinch to zoom and the viewer controls to change orientation or reset the camera. The view controls offer lit, wireframe, clay, normals, and point views. Use wireframe to examine tessellation and clay to inspect shape without the projected texture. The point view samples the mesh vertices.

Select **Compare with photo** and drag the divider to reveal more of the source image or mesh. Rotate the model to inspect the guessed back and sides. Compare the two ring prototypes to see where single-photo reconstruction can fill a through-opening.

The model inspector offers GLB download. For completed jobs, the export panel shows additional formats requested before generation. Use the dashboard experiment notebook to compare measurements from repeated reconstructions of the same uploaded image.


---

## Interactive Three.js CAD Viewport Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               WEBGL CAD VIEWPORT ARCHITECTURE                                    │
└────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                 │
    ┌────────────────────────┬───────────────────┴───────────────────┬────────────────────────┐
    ▼                        ▼                                       ▼                        ▼
[SCENE GRAPH]            [SHADING DISPATCHER]                    [INTERACTION ENGINE]     [MEMORY RECLAIM]
• PerspectiveCamera 45°  • 8 Real-Time Shading Passes            • OrbitControls Damping  • geometry.dispose()
• Studio Key & Fill Rig  • PBR Texturized & Normal Maps          • Touch & Wheel Zoom     • material.dispose()
• Bounding Box Helper    • White Clay & Red Wax MatCap           • Camera View Reset      • texture.dispose()
• Ground Grid Helper     • Surface Normals & Depth Heatmap       • Fullscreen Toggle      • Zero VRAM Leaks
```

---

## The 8 Real-Time Shading Passes Deep Dive

1. **PBR Texturized**: Full Cook-Torrance microfacet illumination with diffuse albedo, tangent normals, roughness 0.45, metalness 0.10.
2. **White Clay Diffuse**: Pure geometric form inspection under studio key lighting.
3. **Red Wax MatCap**: Classical digital sculpting MatCap emphasizing high-frequency surface curvature.
4. **Surface Normals**: Color-encoded surface orientation where RGB = 0.5 * Normal + 0.5.
5. **Depth Heatmap**: Per-vertex distance gradient mapped through the scientific Viridis colormap.
6. **Wireframe-on-Shaded**: Semi-transparent surface mesh with highlighted triangle polygon edges.
7. **Chrome Mirror**: Reflective metallic surface with roughness 0.0 for inspecting surface continuity and reflection lines.
8. **Albedo Base Color**: Unlit flat color map for inspecting texture resolution and color fidelity.

---

## Zero-Copy Binary glTF Loading & Web Worker Decoding

The client browser streams binary glTF files directly into memory:
1. `fetch("/api/v1/assets/{id}/stream")` returns an `ArrayBuffer` in sub-millisecond TTFB.
2. The buffer is passed to a dedicated background Web Worker via `three/examples/jsm/loaders/GLTFLoader.js`.
3. Vertices and triangle faces are uploaded directly to the GPU via `gl.bufferData(gl.ARRAY_BUFFER, positionsFloat32, gl.STATIC_DRAW)`.

*Interactive Viewer Guide documented by Group 17 · 3D Computer Vision Course Project · Department of Data Science & Artificial Intelligence · IIIT Dharwad.*
