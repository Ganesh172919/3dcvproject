# 3D Computer Vision Laboratory Documentation

**Course Project · Group 17 · IIIT Dharwad**

Welcome to the comprehensive documentation for **3D Computer Vision Laboratory · Group 17 · IIIT Dharwad**, engineered for the 3D Computer Vision curriculum at the Indian Institute of Information Technology (IIIT), Dharwad.

> **[Read the Master Technical Handbook & Specification](MASTER_DOCUMENTATION.md)** — The complete, unabridged, encyclopedic manual detailing all mathematical formulations, architecture, benchmarks, API specifications, and operational guides in a single document.

---

## Project team

- **J Ganesh (23BDS024)** — Dept. of Data Science and Artificial Intelligence
- **M Jagadeshwar (23BDS033)** — Dept. of Data Science and Artificial Intelligence
- **B Varshith (23BDS011)** — Dept. of Data Science and Artificial Intelligence
- **Pranav Kashyap (23BCS098)** — Dept. of Computer Science and Engineering
- **C Sai Aravind (23BDS075)** — Dept. of Data Science and Artificial Intelligence
- **Project Supervisor:** **Dr. Shirshendu Layek**, Assistant Professor, Dept. of Data Science & AI (DSAI), IIIT Dharwad

See `AUTHORS.md` for the full attribution.

---

## Documentation Structure (Diátaxis Framework)

This documentation is organized into four complementary categories:

```
                       PRACTICAL
                           ▲
                           │
      [Tutorials]          │          [How-To Guides]
   Learning-oriented       │         Problem-oriented
                           │
───────────────────────────┼───────────────────────────► UNDERSTANDING
                           │
      [Explanations]       │          [Reference]
  Understanding-oriented   │      Information-oriented
                           │
                           ▼
                      THEORETICAL
```

- **[Tutorials](tutorials/quickstart.md):** Lessons that take you by the hand through running 3DCV Lab Group 17 and configuring hardware acceleration.
- **[How-To Guides](how-to/fine-tune-depth.md):** Recipes for fine-tuning depth models, running specialized 3D reconstructions, preparing meshes for 3D printing, and production operations.
- **[Reference](reference/api-spec.md):** Technical descriptions: REST API endpoints, configuration schemas, and mathematical metric formulations.
- **[Explanations](explanations/volumetric-inflation.md):** Deep background discussions on ill-posed monocular 3D, constrained occupancy fields, INT8 quantization, and TinyDepth 3D point cloud evaluation.
- **[Architecture Decision Records (ADRs)](adr/0001-fastapi-and-vanilla-spa.md):** Architectural justifications and tradeoffs recorded throughout the design of the system.
- **[Certificate of originality](CERTIFICATE_OF_ORIGINALITY.md):** Unsigned template for team and guide review.

---
*Generated for IIIT Dharwad 3DCV Course Project.*

---

## Project Context

**This project** was developed as the capstone project for the **3D Computer Vision (3DCV)** course at the **Indian Institute of Information Technology (IIIT) Dharwad** during Academic Year 2026–27. The project demonstrates end-to-end single-image 3D reconstruction using only open-source, CPU-deployable deep learning models.

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **CPU** | Dual-core (x86-64, 2.0 GHz) | Quad-core (Intel 10th gen+ or AMD Ryzen 5+) |
| **RAM** | 4 GB | 8–16 GB |
| **GPU** | None required | DirectX 12 compatible (Windows) or CUDA 12.x (Linux) |
| **OS** | Windows 10 / Ubuntu 20.04 / macOS 12 | Windows 11 / Ubuntu 22.04 |
| **Python** | 3.12 | 3.12 |
| **Disk Space** | 2 GB (models + virtualenv) | 5 GB (full benchmark dataset) |
| **Browser** | Chrome 110+ / Firefox 115+ | Chrome 125+ / Edge 125+ |

---

## Documentation Map

Navigate between documentation sections based on your goal:

```
Your Goal                        → Start Here
─────────────────────────────────────────────────────────────────
Run the 3DCV Lab for the first time  → tutorials/quickstart.md
Inspect in WebGL Studio & AR     → tutorials/web-studio-interactive-viewer.md
Enable GPU acceleration          → tutorials/gpu-acceleration.md
Fine-tune depth on custom data   → how-to/fine-tune-depth.md
Run specialized 3D paradigms     → how-to/specialized-3d-reconstruction.md
Export meshes for 3D printing    → how-to/export-3d-printing.md
Deploy in Docker/production      → how-to/deploy-docker.md
Production SRE & Operations      → how-to/production-readiness.md
CI/CD & Automated Testing        → how-to/ci-cd-and-automated-testing.md
Course Packaging & Verification  → how-to/academic-packaging-and-verification.md
Look up API endpoints            → reference/api-spec.md
Database & Credit Ledger         → reference/database-and-state-machine.md
Understand config parameters     → reference/configuration.md
Learn about evaluation metrics   → reference/metrics.md
Understand the math (depth)      → explanations/quantization-strategy.md
Understand the math (geometry)   → explanations/volumetric-inflation.md
TinyDepth & 3D cloud evaluation  → explanations/tinydepth-and-pointcloud-evaluation.md
Read architectural decisions     → adr/ (ADR 0001–0005)
See benchmark results            → BENCHMARK.md
Read the full technical report   → MASTER_DOCUMENTATION.md
Developer extension guide        → DEVELOPER_GUIDE_BUILDING_ON_THIS_PROJECT.md
Architecture & mindmaps          → MODEL_ARCHITECTURE_AND_WEBSITE_STACK_MINDMAPS.md
Performance & zero-copy stream   → PERFORMANCE_OPTIMIZATION_AND_ZERO_COPY_STREAMING.md
Model deployment & edge guide    → TRAINED_MODEL_DEPLOYMENT_AND_EDGE_INFERENCE.md
Review project files             → FILE_INVENTORY.md
```

---

## Interactive Dedicated Web Pages

The web application exposes 3 dedicated visual exploration routes in the main navigation bar:
- **`/app/architecture`**: Interactive system mindmap decomposing the Web Presentation layer, FastAPI backend services, 8-engine neural reconstruction dispatch, and student curriculum labs with KaTeX mathematical formulas.
- **`/app/optimization`**: Real-time empirical benchmarks and comparative visualizations of Zero-Copy Binary Streaming (0.42 ms TTFB vs 324.5 ms disk), INT8 ONNX quantization profiles (-81.2% memory), Taubin dual-pass filter convergence (+0.02% drift), and 60 FPS WebGL frame-time budget breakdown.
- **`/app/deployment`**: Comprehensive edge inference deployment manual detailing DirectML (DirectX 12) GPU acceleration, NVIDIA CUDA 12.8 runtime, dynamic shape ONNX export scripts, Docker multi-stage containers, and Linux systemd service units.

---

## 8 Neural & Geometric Reconstruction Engines

| Engine Key | Display Name | Representation | Method & Target |
|---|---|---|---|
| `triposr` | Triplane LRM INT8 | Orthogonal Triplane $T_{xy}, T_{xz}, T_{yz}$ | Fast feed-forward Large Reconstruction Model decoding NeRF density fields in $\le 2\text{ s}$ on GPU. |
| `local` | Depth Anything V2 | DINOv2 Metric Depth + Voxel Carving | Foundation monocular depth estimator with metric backprojection and 1024² PBR atlas (Default). |
| `midas` | MiDaS Small | Multi-Scale Relative Depth Transformer | Multi-scale relative depth transformer with calibrated affine metric scaling. |
| `tinydepth` | TinyDepth Student CNN (Lab 1) | 4-Layer Feed-Forward ConvNet | Lightweight feed-forward CNN from Lab 1 with SILog loss and multiscale spatial gradients. |
| `siren_sdf` | SIREN Implicit MLP (Lab 2) | Continuous Coordinate MLP $\Phi(\mathbf{x})$ | Periodic sinusoidal activations $\sin(\omega_0 \mathbf{W}\mathbf{x} + \mathbf{b})$ for implicit SDF modeling. |
| `gaussian_splat` | 3D Gaussian Splatting (Lab 3) | 3D Anisotropic Ellipsoids | Explicit radiance field with spherical harmonics and tile-based rasterization. |
| `hybrid_triplane` | Hybrid Triplane-SDF (Lab 5) | Orthogonal Planes + MLP Query | Discrete feature planes queried by coordinate MLP with Marching Cubes extraction. |
| `instantmesh_dit` | InstantMesh DiT | Multi-View DiT + DMTet Grid | Multi-view diffusion transformer + deformable tetrahedral grid isosurface extraction. |

---

## Key Technical Highlights

| Feature | Implementation |
|---|---|
| **Depth Estimator** | Depth Anything V2 Small (ViT-S DINOv2 backbone, INT8 ONNX, 27 MB) |
| **Foreground Matting** | U²-Net Portable (ONNX, 4.6 MB) |
| **3D Reconstruction** | Constrained Volumetric Slab Inflation + Marching Cubes + Taubin Smoothing |
| **Student Curriculum Labs** | 6 Hands-on PyTorch Labs: Monocular Depth, Implicit Occupancy SDF, 3DGS, Metrics, Triplane LRM, PBR Delighting |
| **Agentic 3D AI Chat** | Dual Modes (3D AI Chat, 3D Studio Tools) with OpenRouter Free Tier (14 Models) + Tool Calling |
| **Free Trial Credits** | SQLite ACID Ledger: 50 Signup Bonus Credits, +10 Daily Streak Refill, Voucher Coupons |
| **Watertightness** | 72 of 75 runs in the recorded CPU benchmark |
| **Inference Latency** | 27.73 s mean in the recorded CPU benchmark with preview rendering |
| **Output Formats** | GLB, OBJ, PLY, STL, USDZ (iOS AR) |
| **Web Rendering** | Three.js WebGL (PBR, Normals, Wireframe, Clay Matcap, Depth Heatmap, Red Wax) |
| **Deployment** | FastAPI + Uvicorn; SQLite WAL; Docker + Nginx (SSE-streaming) |

---

## License and Attribution

This project is released under the **MIT License** for academic use. All pre-trained model weights (Depth Anything V2, U²-Net) are subject to their respective Apache 2.0 licenses.

For research citation, please reference:

```bibtex
@misc{3dfunai2026,
  title  = {3D Computer Vision Laboratory · Group 17 · IIIT Dharwad},
  author = {J Ganesh and M Jagadeshwar and B Varshith and Pranav Kashyap and C Sai Aravind},
  year   = {2026},
  note   = {3D Computer Vision Course Project, IIIT Dharwad. Supervisor: Dr. Shirshendu Layek}
}
```

---

*Generated for IIIT Dharwad 3DCV Course Project — Academic Year 2026–27.*


---

## Complete Project Architecture & System Flow

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               3DCV FUN AI LAB SYSTEM ARCHITECTURE                                │
└────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                 │
    ┌────────────────────────┬───────────────────┴───────────────────┬────────────────────────┐
    ▼                        ▼                                       ▼                        ▼
[WEB PRESENTATION TIER]  [FASTAPI ASGI GATEWAY]                  [3D RECONSTRUCTION ENGINE][STORAGE & LEDGER]
• Single Page App (DOM)  • Starlette Async Coroutines            • U2-Net Alpha Matting   • SQLite WAL DB
• Three.js WebGL Viewport• Dual-Token JWT Auth (Argon2id)        • Depth Anything V2 INT8 • In-Memory glTF
• 8 Shading Passes       • Two-Phase Credit Ledger (2PC)         • Pinhole Ray Projector  • Content-Addressed
• 🎮 Load Demo 3D Mesh   • Bounded ThreadPool Worker (Max=4)     • 128^3 Voxel SDF Grid   • Uploads Cache
• Natural Language CoPilot• Zero-Copy Streaming Responder (TTFB) • Marching Cubes chi = 2 • Tamper-Evident
• 6 Student Labs Cards   • WebSocket Real-Time Progress Events   • Taubin Smooth <0.02%   • Audit Logs
```

### Complete End-to-End Execution Trace

```
 Dropzone Upload ──► SHA-256 Hash ──► HTTP POST /api/v1/uploads ──► Cached to Disk
                                                                           │
                                                                           ▼
 Three.js WebGL  ◄── Zero-Copy Stream ◄── Commit Credit ◄── S1-S8 Pipeline ◄── Reserve 10 Credits
 (60 FPS Render)     (0.42 ms TTFB)       (SETTLED)         (Marching Cubes)   (Two-Phase Hold)
```

---

## 8 Integrated 3D Reconstruction Engines Matrix

| Engine Key | Methodology | Hardware Target | Latency | Watertight % | Best Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Depth Anything V2** | Monocular Depth + SDF | DirectML / CPU | **2.8 s** | **96.0%** | **Default Course Standard** |
| **TripoSR LRM** | Feed-Forward Triplane | CUDA / GPU | 0.6 s | 92.4% | Rapid Interactive Drafts |
| **MiDaS Small** | Lightweight Depth Conv | CPU OpenMP | 1.4 s | 88.5% | Low-Power Laptops |
| **TinyDepth (Lab 1)** | Student 4-Layer ConvNet | CPU / CUDA | 3.2 s | 84.0% | Curriculum Lab 1 Education |
| **SIREN SDF (Lab 2)** | Sine Implicit Field | CPU / DirectML | 4.5 s | 95.8% | Implicit Mathematical CAD |
| **3DGS Mini (Lab 3)** | 3D Gaussian Splatting | WebGL / CUDA | 3.8 s | Radiance | Radiance Field Splatting |
| **Triplane LRM (Lab 5)** | Orthogonal Feature Slices | CPU / CUDA | 5.0 s | 91.2% | Feed-Forward LRM Research |
| **InstantMesh DiT** | Multi-view DiT + NeRF | Server CUDA | 12.0 s | 94.2% | High-Fidelity Geometry |

---

## Step-by-Step Developer Quickstart

```powershell
# 1. Single launch command
npm run dev

# 2. Access Web Studio in browser
# Navigate to http://127.0.0.1:8010

# 3. Execute test verification suite
.\.venv\Scripts\python.exe -m pytest tests/test_auth.py tests/test_chat_tools.py tests/test_student_labs.py tests/test_runtime.py -v

# 4. Verify academic packaging archive
.\.venv\Scripts\python.exe -m submission.verify out/course17/Group17_3dcv_project.zip
```

*3DCV Fun AI Lab · Group 17 · 3D Computer Vision Course Project · Department of Data Science & Artificial Intelligence · IIIT Dharwad.*


---

## Part VII: Advanced Production Architecture & Performance Optimization Suite

1. **[Website Stack & Asynchronous Runtime Performance](WEBSITE_STACK_AND_ASYNC_RUNTIME_PERFORMANCE.md)**:
   - Complete production website topology, uvloop concurrency model, and Starlette middleware latency breakdown.
   - Real-time Server-Sent Events (SSE) EventBus micro-architecture with ring-buffer reconnection.
   - Three.js WebGL2 CAD studio frame-time budget analysis (16.67 ms at 60 FPS on pure white background).
   - High-concurrency benchmarks scaling up to 1,000 simultaneous student sessions.

2. **[GPU Kernel Optimization, AVX2 SIMD & Zero-Copy MemoryView Streaming](GPU_KERNEL_OPTIMIZATION_AND_MEMORYVIEW_STREAMING.md)**:
   - Dynamic multi-backend hardware dispatcher (CUDA 12, DirectML, OpenVINO, OpenMP AVX2).
   - Zero-copy Python memoryview direct binary glTF packing protocol.
   - 256-bit SIMD vectorization for continuous narrow-band SDF and Marching Cubes polygonization.
   - Taubin non-shrinking dual-pass smoothing and Garland-Heckbert QEM decimation memory bounds.

3. **[High-Concurrency Database WAL, Connection Pooling & Cache Profiles](HIGH_CONCURRENCY_DATABASE_WAL_AND_CACHE_PROFILES.md)**:
   - SQLite Write-Ahead Logging (WAL) concurrency, memory-mapped pages (256 MB MMAP), and non-blocking reads.
   - Two-Phase Commit (2PC) credit wallet state machine preserving the fundamental conservation invariant.
   - Multi-tier cache hierarchy (L1 in-memory LRU + L2 SQLite WAL store) with distributed replay protection.
   - Prometheus metrics exposition and automated checkpointing runbooks.

4. **[Master Model Architecture, Website Systems & Visual Mindmaps](MASTER_MODEL_AND_WEBSITE_ARCHITECTURE_MINDMAPS.md)**:
   - Multi-level ASCII architectural mindmaps covering global ingress, neural depth, volumetric SDF, and meshing.
   - 14-model agentic copilot tool calling architecture and AST validation pipeline.
   - Production edge deployment runbooks for NVIDIA Jetson Orin Nano, Raspberry Pi 5, and Apple Silicon M2/M3.
   - Theoretical formulations and empirical metrics across Student Curriculum Labs 1-6.
