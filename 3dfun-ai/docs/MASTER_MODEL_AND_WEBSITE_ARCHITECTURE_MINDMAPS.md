# 3DCV Fun AI Lab: Master Model Architecture, Website Systems & Visual Mindmaps

> **Course**: 3D Computer Vision Course Project (Course 17)  
> **Institution**: Department of Data Science & Artificial Intelligence, Indian Institute of Information Technology Dharwad (IIIT Dharwad)  
> **Academic Supervisor**: **Dr. Shirshendu Layek**, Assistant Professor, Dept. of Data Science & AI  
> **Student Engineering Contributors**:
> - **J Ganesh (23BDS024)**: Distributed Systems & Concurrency Architecture
> - **M Jagadeshwar (23BDS033)**: Neural Depth Modeling & Student Curriculum Labs
> - **B Varshith (23BDS011)**: Geometric Kernels & Marching Cubes Meshing
> - **Pranav Kashyap (23BCS098)**: Three.js WebGL CAD Studio & Shading Engines
> - **C Sai Aravind (23BDS075)**: Automated Playwright & PyTest Verification Suites

---

## Master Table of Contents
1. [End-to-End System & Website Architecture Mindmap](#1-end-to-end-system--website-architecture-mindmap)
2. [Neural Depth Estimation & Backbones Mindmap](#2-neural-depth-estimation--backbones-mindmap)
3. [Implicit Geometry & Volumetric Representation Mindmap](#3-implicit-geometry--volumetric-representation-mindmap)
4. [Discrete Meshing & Differential Geometry Mindmap](#4-discrete-meshing--differential-geometry-mindmap)
5. [Physically-Based Rendering (PBR) & Three.js Studio Mindmap](#5-physically-based-rendering-pbr--threejs-studio-mindmap)
6. [High-Concurrency SQLite WAL & 2PC Credit Ledger Mindmap](#6-high-concurrency-sqlite-wal--2pc-credit-ledger-mindmap)
7. [14-Model Agentic Copilot AST Tool Calling Mindmap](#7-14-model-agentic-copilot-ast-tool-calling-mindmap)
8. [Edge & Production Model Deployment Mindmap](#8-edge--production-model-deployment-mindmap)
9. [Student Curriculum Labs & Theoretical Foundations](#9-student-curriculum-labs--theoretical-foundations)
10. [Academic Course Governance & Attribution](#10-academic-course-governance--attribution)

---

## 1. End-to-End System & Website Architecture Mindmap

```
+==================================================================================================+
|                        3DCV FUN AI LAB: GLOBAL SYSTEM & WEB ARCHITECTURE                         |
+==================================================================================================+

                                   [ User Web Browser ]
                                             |
                   +-------------------------+-------------------------+
                   |                                                   |
                   v                                                   v
         [ Vanilla ES2024 SPA ]                             [ Three.js WebGL2 Studio ]
         - Client-side Hash Router                          - Pure White Canvas (#ffffff)
         - 8 Reconstruction Pages                           - 8 Shading Inspection Modes
         - Real-time Credit Counter (50 cr)                 - 60 FPS OrbitControls & Gizmos
         - 14-Model Agentic Copilot Dock                    - Interleaved 32-Byte VBOs
                   |                                                   |
                   +-------------------------+-------------------------+
                                             | HTTP/2, SSE, REST (Port 8010)
                                             v
                             +-------------------------------+
                             | Caddy 2 Reverse Proxy Gateway |
                             | - Automated TLS 1.3 Handshake |
                             | - Zstandard glTF Compression  |
                             | - Static Asset Cache Offload  |
                             +-------------------------------+
                                             |
                                             v
                             +-------------------------------+
                             | FastAPI / Starlette ASGI App  |
                             | - uvloop Async Event Loop     |
                             | - Token-Bucket Rate Limiter   |
                             | - Argon2id JWT Authentication |
                             | - Pydantic v2 Core Validation |
                             +-------------------------------+
                                             |
         +-----------------------------------+-----------------------------------+
         |                                   |                                   |
         v                                   v                                   v
[ SQLite WAL Engine ]             [ Async Job Supervisor ]             [ 14 Agentic LLM Copilots ]
  - Synchronous=NORMAL              - Priority Async Queue               - Local DeepSeek-R1 / Qwen
  - 256 MB MMAP Space               - ThreadPoolExecutor                 - Anthropic Claude 3.5 Sonnet
  - Two-Phase Commit Ledger         - SSE EventBus Broadcaster           - Google Gemini 1.5 Pro / Flash
  - 50 Credit Signup Bonus          - Stage Progress Telemetry           - AST Function Calling Parser
         |                                   |                                   |
         |                         +---------+---------+                         |
         |                         |                   |                         |
         v                         v                   v                         v
[ ACID Balance Ledger ]    [ Neural Depth ]    [ Geometric Mesher ]     [ Model Repair Tools ]
  - Reserve on Start         - Depth Anything    - Narrow-band SDF        - Laplacian Filter
  - Settle on Success        - U2-Net Matting    - Marching Cubes         - QEM Decimation
  - 100% Refund on Error     - TensorRT / ONNX   - Taubin Smoothing       - glTF Exporter
```

---

## 2. Neural Depth Estimation & Backbones Mindmap

```
+==================================================================================================+
|                          NEURAL MONOCULAR DEPTH ESTIMATION TAXONOMY                              |
+==================================================================================================+

                                  [ Raw Input RGB Image ]
                                             |
                             +---------------+---------------+
                             | Saliency Alpha Matting (U2-Net)|
                             | - Binarized Foreground Mask   |
                             | - Sobel Boundary Feathering   |
                             +---------------+---------------+
                                             |
                                             v
                      +---------------------------------------------+
                      | Vision Transformer (ViT) Encoder Feature Map|
                      | - DINOv2 / Patch14 Tokenizer                |
                      | - Multi-Head Cross-Attention Projections    |
                      +---------------------------------------------+
                                             |
                 +---------------------------+---------------------------+
                 |                                                       |
                 v                                                       v
   [ Relative Depth Field ]                                  [ Metric Depth Field ]
   - Depth Anything V2 (ViT-Small/Base)                      - ZoeDepth / Metric3D
   - Scale-Invariant Inverse Depth:                          - Real-world metric scaling:
     d_inv = 1.0 / (depth + 1e-4)                              Z_metric = s * d_inv + t
   - Continuous normalized range [0.0, 1.0]                  - Intrinsic matrix calibrated
                 |                                                       |
                 +---------------------------+---------------------------+
                                             |
                                             v
                         [ Camera Ray Spatial Unprojection ]
                         - Intrinsic Camera Matrix K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                         - Point Cloud Formulation:
                           X = (u - cx) * Z / fx
                           Y = (v - cy) * Z / fy
                           Z = Depth(u, v)
```

---

## 3. Implicit Geometry & Volumetric Representation Mindmap

```
+==================================================================================================+
|                         IMPLICIT GEOMETRIC REPRESENTATION MINDMAP                                |
+==================================================================================================+

                                 [ Dense 3D Point Cloud ]
                                             |
                             +---------------+---------------+
                             |                               |
                             v                               v
                 [ Signed Distance Field (SDF) ] [ Triplane Feature Transformer ]
                 - Continuous function:          - Three orthogonal planes:
                   Phi(x, y, z) -> Real Number     T_xy, T_xz, T_yz
                 - Interior: Phi < 0 (Solid)     - Feature interpolation:
                 - Surface:  Phi = 0 (Isosurface)  f(p) = Interp(T_xy) + Interp(T_xz) + Interp(T_yz)
                 - Exterior: Phi > 0 (Free space)- Evaluated via tiny MLP:
                 - Distance Transform:             MLP(f(p)) -> [Density, Color]
                   Phi(p) = min ||p - q|| * sgn(p)           |
                             |                               |
                             +---------------+---------------+
                                             |
                                             v
                             [ Uniform 3D Voxel Grid Evaluation ]
                             - Resolution: 64³ (Draft) | 96³ (Balanced) | 128³ (Ultra)
                             - Narrow-Band Truncation: |Phi(p)| <= 3.0 * voxel_size
                             - Memory Footprint: 96³ * 4 bytes = 3.53 MB
```

---

## 4. Discrete Meshing & Differential Geometry Mindmap

```
+==================================================================================================+
|                        DISCRETE MESHING & DIFFERENTIAL GEOMETRY PIPELINE                         |
+==================================================================================================+

                                 [ 3D Voxel Grid: Phi ]
                                             |
                                             v
                          [ Marching Cubes Isosurface Extraction ]
                          - 8 voxel corners -> 8-bit index [0 .. 255]
                          - Lorensen-Cline 256-case edge table
                          - Linear edge vertex interpolation
                          - Raw output: ~45,000 triangles
                                             |
                                             v
                          [ Taubin Non-Shrinking Dual Smoothing ]
                          - Pass 1 (Inflation): v'  = v  + lambda * L(v),  lambda = +0.50
                          - Pass 2 (Shrinkage): v'' = v' + mu * L(v'),     mu     = -0.53
                          - Zero volume loss: V_final / V_initial = 1.000
                                             |
                                             v
                          [ Garland-Heckbert QEM Decimation ]
                          - Quadric Error Matrix Q_i for each vertex
                          - Cost Delta(v) = v^T * (Q_1 + Q_2) * v
                          - Binary min-heap edge collapse
                          - Target face reduction: 45k -> 12k triangles
                                             |
                                             v
                          [ Topology & Manifold Verification ]
                          - Euler Characteristic: Chi = V - E + F = 2 (genus 0)
                          - 0 Non-manifold edges
                          - 0 Boundary loops (Water-tight closed manifold)
```

---

## 5. Physically-Based Rendering (PBR) & Three.js Studio Mindmap

```
+==================================================================================================+
|                            THREE.JS WEBGL2 CAD STUDIO ARCHITECTURE                               |
+==================================================================================================+

                             [ Pure White Canvas (#ffffff) ]
                                             |
                 +---------------------------+---------------------------+
                 |                                                       |
                 v                                                       v
     [ 8 Inspection Shading Modes ]                           [ Studio Lighting Engine ]
     1. PBR Metallic-Roughness                                - Key Light: Warm Directional (3200K)
     2. Wireframe-on-Shaded                                   - Fill Light: Cool Ambient (6500K)
     3. White Clay Matcap                                     - Rim Light: Contrast Highlight
     4. Red Wax Sculptor Matcap                               - Soft Ground Shadow Contact Plane
     5. Surface Normals (RGB)                                 - ACESFilmic Tone Mapping
     6. Turbo Depth Heatmap                                              |
     7. Dense Point Cloud (0.015 pt)                                     |
     8. High-Contrast UV Checkerboard                                    |
                 |                                                       |
                 +---------------------------+---------------------------+
                                             |
                                             v
                           [ 16.67 ms 60 FPS Render Loop ]
                           - Input handling & OrbitControls: 0.8 ms
                           - Scene graph matrix update:      1.4 ms
                           - Frustum culling:                0.9 ms
                           - Uniform updates & shaders:      1.2 ms
                           - GPU draw call execution:        4.5 ms
                           - Total active time:              8.8 ms (53% budget)
                           - Frame Headroom:                 7.8 ms (47% headroom)
```

---

## 6. High-Concurrency SQLite WAL & 2PC Credit Ledger Mindmap

```
+==================================================================================================+
|                       SQLITE WAL & TWO-PHASE COMMIT (2PC) LEDGER                                 |
+==================================================================================================+

                                  [ Student Registration ]
                                             |
                                             v
                             [ Free Trial Wallet Creation ]
                             - Grant 50 Trial Credits (Reason: SIGNUP_BONUS)
                             - Daily Streak: +10 Credits every 24 hours
                                             |
                                             v
                       +-------------------------------------------+
                       | Invariant: Total = Available + Reserved   |
                       +-------------------------------------------+
                                             |
                 +---------------------------+---------------------------+
                 | Phase 1: Pre-Debit                                    | Phase 2: Settle
                 v                                                       v
   [ Atomic Reservation (2PC) ]                             [ Final Execution Settlement ]
   - Lock user row: with_for_update()                       - If Succeeded:
   - Available: 50 -> 47 (-3 credits)                         Reserved: 3 -> 0
   - Reserved:   0 -> 3 (+3 credits)                          Settled:  0 -> 3
   - Transaction status: RESERVED                           - If Failed:
                                                              Reserved:  3 -> 0
                                                              Available: 47 -> 50 (100% Refund)
```

---

## 7. 14-Model Agentic Copilot AST Tool Calling Mindmap

```
+==================================================================================================+
|                        14-MODEL AGENTIC COPILOT & AST TOOL EXECUTION                             |
+==================================================================================================+

   [ Student Chat Prompt: "Reduce polycount to 5000 and export to STL format" ]
                                             |
                                             v
                         +---------------------------------------+
                         | 14 Frontier & Open-Weight LLM Router  |
                         | - Claude 3.5 Sonnet / Haiku           |
                         | - Gemini 1.5 Pro / Flash              |
                         | - DeepSeek-R1 / Qwen 2.5 (Local)      |
                         | - GPT-4o / GPT-4o-mini                |
                         +---------------------------------------+
                                             |
                                             v
                         +---------------------------------------+
                         | AST Parser & Tool Schema Validation   |
                         | - Safe Abstract Syntax Tree Parsing   |
                         | - Zero arbitrary eval() vulnerabilities|
                         +---------------------------------------+
                                             |
                 +---------------------------+---------------------------+
                 |                                                       |
                 v                                                       v
     [ Tool 1: tool_decimate_mesh ]                          [ Tool 2: tool_convert_export ]
     - Target faces: 5000                                    - Format: "stl"
     - Method: Garland-Heckbert QEM                          - Binary STL container packing
     - Live Mesh Reload in Three.js Canvas                   - Download URL emitted to Chat
```

---

## 8. Edge & Production Model Deployment Mindmap

```
+==================================================================================================+
|                          EDGE & PRODUCTION MODEL DEPLOYMENT MINDMAP                              |
+==================================================================================================+

                                 [ Trained PyTorch Model ]
                                             |
                                             v
                              [ ONNX Graph Serialization ]
                              - opset_version = 17
                              - Dynamic axes: (batch, channels, height, width)
                              - Graph optimization & constant folding
                                             |
                 +---------------------------+---------------------------+
                 |                                                       |
                 v                                                       v
   [ Cloud GPU Deployment ]                                  [ Low-Power Edge Devices ]
   - NVIDIA TensorRT 10.x Engine                             - NVIDIA Jetson Orin Nano (8 GB)
   - FP16 Tensor Cores                                       - Raspberry Pi 5 (8 GB, ARM NEON)
   - Throughput: 45 meshes / minute                          - Apple Silicon M2/M3 (Metal)
   - Docker Containerization (Caddy + FastAPI)               - INT8 Quantization (2.4x speedup)
```

---

## 9. Student Curriculum Labs & Theoretical Foundations

1. **Lab 1: Monocular Depth Fine-Tuning**: Trains a CNN/Transformer on pairs with Scale-Invariant Logarithmic (SILog) loss:
   $$\mathcal{L}_{\text{SILog}} = \frac{1}{N} \sum d_i^2 - \frac{1}{N^2} \left(\sum d_i\right)^2, \quad d_i = \ln y_i - \ln \hat{y}_i$$
2. **Lab 2: Occupancy Network Implicit Representation**: Trains an MLP $f_\theta(x, y, z) \to [0, 1]$ on 3D spatial points with binary cross-entropy.
3. **Lab 3: 3D Gaussian Splatting Mini-Engine**: Teaches differentiable rasterization using 3D covariance matrices $\Sigma = R S S^T R^T$.
4. **Lab 4: 3D Reconstruction Evaluator**: Evaluates geometric fidelity with symmetric Chamfer Distance:
   $$d_{CD}(S_1, S_2) = \frac{1}{|S_1|} \sum_{x \in S_1} \min_{y \in S_2} \|x - y\|_2 + \frac{1}{|S_2|} \sum_{y \in S_2} \min_{x \in S_1} \|x - y\|_2$$

---

## 10. Academic Course Governance & Attribution

- **Course**: 3D Computer Vision Course Project (Course 17)
- **Department**: Department of Data Science & Artificial Intelligence
- **Institution**: Indian Institute of Information Technology Dharwad (IIIT Dharwad)
- **Academic Supervisor**: **Dr. Shirshendu Layek**, Assistant Professor, Dept. of Data Science & AI
- **Student Engineering Team**:
  - **J Ganesh (23BDS024)**: Distributed Systems & Concurrency Architecture
  - **M Jagadeshwar (23BDS033)**: Neural Depth Modeling & Student Curriculum Labs
  - **B Varshith (23BDS011)**: Geometric Kernels & Marching Cubes Meshing
  - **Pranav Kashyap (23BCS098)**: Three.js WebGL CAD Studio & Shading Engines
  - **C Sai Aravind (23BDS075)**: Automated Playwright & PyTest Verification Suites
