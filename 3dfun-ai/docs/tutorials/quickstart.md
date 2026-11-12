# Quickstart Tutorial: Monocular 3D Reconstruction with 3DFUN AI

Welcome to **3DFUN AI**, a lightweight, self-contained single-image 3D reconstruction system developed at IIIT Dharwad. This tutorial guides you from a clean clone to generating and inspecting your first interactive 3D model in your browser within 5 minutes.

---

## 1. Prerequisites and Environment Setup

Ensure you have Python 3.10+ installed. 3DFUN AI runs on standard CPU hardware without requiring CUDA or heavy PyTorch dependencies.

```bash
# Clone the repository
git clone https://github.com/iiitdwd/3dfun-ai.git
cd 3dfun-ai

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies and download quantized ONNX models
make install
make models
make seed
```

This downloads two quantized ONNX models into `engine/models/`:
1. `u2netp.onnx` (4.6 MB): Lightweight salient object detection and foreground matting.
2. `depth_anything_v2_vits.onnx` (27 MB): INT8 quantized monocular relative depth estimation.

---

## 2. Launching the Web Studio

Start the FastAPI application and WebGL studio server:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to `http://127.0.0.1:8000`.

Sign in using the seeded student credentials:
- **Email:** `student@iiitdwd.ac.in`
- **Password:** `student123`

---

## 3. Reconstructing Your First 3D Mesh

1. **Upload an Image:** In the Web Studio dashboard, drag and drop `engine/testdata/sneaker.png` (or any single-object photograph on a contrasting background) into the upload zone.
2. **Configure Pipeline Settings:**
   - **Target Triangle Budget:** Set to `8000` (optimal for mobile and web viewing).
   - **Target Thickness ($t_{\max}$):** Set to `0.70` (default for convex volumetric items).
   - **Symmetry Prior:** Enable for bilateral objects (shoes, bottles, furniture).
   - **Output Formats:** Select `GLB`, `OBJ`, and `PLY`.
3. **Execute Reconstruction:** Click **Generate 3D Asset**.
   - The status stepper tracks the pipeline stages in real-time:
     - Stage 1: Alpha Matting & Centering ($\approx 0.12$ s)
     - Stage 2: Monocular Depth Inference ($\approx 0.35$ s)
     - Stage 3: Constrained Volumetric Inflation ($\approx 0.18$ s)
     - Stage 4: Marching Cubes & Taubin Smoothing ($\approx 0.22$ s)
     - Stage 5: PBR Texture Atlas Baking ($\approx 0.45$ s)
   - Total runtime is typically under 1.5 seconds on a quad-core CPU.

---

## 4. Multi-Mode Viewport Inspection

Once complete, the model automatically loads into the embedded Three.js WebGL viewport.

- **Shader Inspection Modes:** Use the top-bar dropdown to switch between:
  - **PBR (Default):** Lit with directional studio lights and roughness/metallic shading.
  - **Albedo / Base Color:** Texture map isolated without lighting.
  - **Surface Normals:** Normal vectors mapped to RGB space to verify curvature continuity.
  - **Topology & Wireframe:** Triangle distribution and vertex density overlay.
  - **Clay Studio:** Neutral grey MatCap shader highlighting fine surface sculpting.
  - **Depth Cloud:** 3D point cloud projected directly from predicted depth values.
- **Split-Screen Verification:** Switch to the **Split View** tab. Drag the vertical divider bar left and right to inspect the exact boundary alignment between the original 2D photograph and the reconstructed 3D silhouette.
- **Mobile AR (WebXR / QuickLook):** Click **View in AR** to launch native augmented reality on iOS (USDZ QuickLook) or Android (Scene Viewer).

---

## 5. Command-Line Batch Reconstruction

For headless server pipelines or scripting, use the command-line CLI directly:

```bash
python scripts/run_pipeline.py \
  --input engine/testdata/sneaker.png \
  --output out/sneaker \
  --triangles 8000 \
  --format glb,obj,ply
```

The resulting assets will be stored in `out/sneaker/`:
- `model.glb`: Binary glTF with embedded textures, PBR materials, and normals.
- `model.obj` & `model.mtl`: Wavefront OBJ with separate material definitions.
- `model.ply`: Stanford PLY format with vertex normals.
- `manifest.json`: Execution metadata, timing, and topological checksums.

---

## Summary & Next Steps

You now have a running 3DFUN AI instance!
- Learn how to enable hardware GPU acceleration: [GPU Acceleration Tutorial](gpu-acceleration.md)
- Learn how to fine-tune depth on custom data: [Fine-Tuning Depth Guide](../how-to/fine-tune-depth.md)
- Explore mathematical derivations of watertightness: [Volumetric Inflation Explanation](../explanations/volumetric-inflation.md)

---
*Evidence: Engine implementation in `engine/recon/pipeline.py`; Web Studio in `web/static/js/studio.js` and `web/static/js/viewer.js`.*  
*Authors: J Ganesh, M Jagadeshwar, B Varshith, Pranav Kashyap, C Sai Aravind (Supervisor: Dr. Shirshendu Layek, Dept. of DSAI, IIIT Dharwad).*
