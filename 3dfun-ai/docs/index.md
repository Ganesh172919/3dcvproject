# 3DFUN AI Documentation

Welcome to the comprehensive documentation for **3DFUN AI: An End-to-End Image-to-3D Generation Framework**, engineered for the 3D Computer Vision curriculum at the Indian Institute of Information Technology (IIIT), Dharwad.

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

- **[Tutorials](tutorials/quickstart.md):** Lessons that take you by the hand through running 3DFUN AI and configuring hardware acceleration.
- **[How-To Guides](how-to/fine-tune-depth.md):** Recipes for fine-tuning depth models, running specialized 3D reconstructions, preparing meshes for 3D printing, and production operations.
- **[Reference](reference/api-spec.md):** Technical descriptions: REST API endpoints, configuration schemas, and mathematical metric formulations.
- **[Explanations](explanations/volumetric-inflation.md):** Deep background discussions on ill-posed monocular 3D, constrained occupancy fields, INT8 quantization, and TinyDepth 3D point cloud evaluation.
- **[Architecture Decision Records (ADRs)](adr/0001-fastapi-and-vanilla-spa.md):** Architectural justifications and tradeoffs recorded throughout the design of the system.
- **[Certificate of originality](CERTIFICATE_OF_ORIGINALITY.md):** Unsigned template for team and guide review.

---
*Generated for IIIT Dharwad 3DCV Course Project.*

---

## Project Context

**3DFUN AI** was developed as the capstone project for the **3D Computer Vision (3DCV)** course at the **Indian Institute of Information Technology (IIIT) Dharwad** during Academic Year 2026–27. The project demonstrates end-to-end single-image 3D reconstruction using only open-source, CPU-deployable deep learning models.

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
Run 3DFUN AI for the first time  → tutorials/quickstart.md
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
Review project files             → FILE_INVENTORY.md
```

---

## Key Technical Highlights

| Feature | Implementation |
|---|---|
| **Depth Estimator** | Depth Anything V2 Small (ViT-S DINOv2 backbone, INT8 ONNX, 27 MB) |
| **Foreground Matting** | U²-Net Portable (ONNX, 4.6 MB) |
| **3D Reconstruction** | Constrained Volumetric Slab Inflation + Marching Cubes + Taubin Smoothing |
| **Student Curriculum Labs** | 4 Hands-on PyTorch Labs: Monocular Depth, Implicit Occupancy SDF, 3DGS, Metrics |
| **Agentic AI Copilot** | Dual Personas (Professor 3D, Studio Director) with real backend tool execution |
| **Free Trial Credits** | SQLite ACID Ledger: 60 Signup Credits, +10 Daily Streak Refill, Voucher Coupons |
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
  title  = {3DFUN AI: An End-to-End Image-to-3D Generation Framework},
  author = {J Ganesh and M Jagadeshwar and B Varshith and Pranav Kashyap and C Sai Aravind},
  year   = {2026},
  note   = {3D Computer Vision Course Project, IIIT Dharwad. Supervisor: Dr. Shirshendu Layek}
}
```

---

*Generated for IIIT Dharwad 3DCV Course Project — Academic Year 2026–27.*
