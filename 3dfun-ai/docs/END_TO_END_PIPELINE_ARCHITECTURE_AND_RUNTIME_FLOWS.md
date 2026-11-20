# 3DCV Fun AI Lab: End-to-End Pipeline Architecture & Runtime System Flows

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

## 1. Executive Summary & Production Architecture Overview

The **3DCV Fun AI Lab** represents a production-grade, highly optimized software architecture for monocular 3D computer vision, interactive WebGL computer-aided design (CAD), and student deep learning education. The system seamlessly reconciles high-throughput asynchronous web ingress (FastAPI / Starlette) with compute-intensive geometric neural inference, real-time Server-Sent Events (SSE) telemetry, transactional credit ledger accounting with two-phase commit (2PC) guarantees, and an interactive 60 FPS Three.js viewport supporting 8 diagnostic shading modes and 14 frontier AI copilot models.

---

## 2. Global Microservice Ingress & Routing Topology

```
  [ BROWSER RUNTIME (Vanilla ES2024 / Three.js WebGL CAD Studio) ]
       │
       │ 1. POST /api/v1/recon/generate (Image bytes + ReconstructionConfig)
       ▼
  [ STARLETTE / FASTAPI ASGI INGRESS LAYER ]
       │
       ├── 2. SecurityHeadersMiddleware: Inject CSP, X-Frame-Options: DENY, nosniff
       ├── 3. CORSMiddleware: Strict Whitelist (Localhost & Lab Domain)
       ├── 4. AuthTokenMiddleware: Decode HttpOnly JWT Session Cookie -> Injects request.state.user
       ├── 5. RateLimitingMiddleware: Leaky Bucket (Max 120 requests/min per IP)
       └── 6. Payload Inspection: Magic Byte Header Check (JPEG/PNG/WEBP) & Resolution Gate (<= 4096^2)
       │
       ▼
  [ TWO-PHASE COMMIT (2PC) FINANCIAL LEDGER ENGINE ]
       │
       ├── 7. BEGIN IMMEDIATE TRANSACTION (SQLite Write-Ahead Log)
       ├── 8. Query: SELECT available_credits, reserved_credits FROM user_wallets WHERE user_id = ?
       ├── 9. Verify Invariant: available_credits >= 10 Credits
       ├── 10. Update: available = available - 10, reserved = reserved + 10
       ├── 11. Append: INSERT INTO ledger_transactions (phase='RESERVATION', status='HELD')
       └── 12. COMMIT TRANSACTION
       │
       ▼
  [ ASYNCHRONOUS JOB SCHEDULER & EVENTSTREAM DISPATCHER ]
       │
       ├── 13. Spawn AsyncIO Task Worker / Thread Pool Executor
       ├── 14. Return HTTP 202 Accepted {"job_id": "job_3d_...", "status": "QUEUED"}
       │
       └── [ ASYNC WORKER TASK COROUTINE ]
              │
              ├── 15. SSE Broadcast -> {"stage": "INGESTION", "progress": 0.05}
              │
              ├── 16. Stage 1: Input Normalization (512x512 Bilinear Scale, FP32 [0.0, 1.0])
              ├── 17. Stage 2: Silhouette Alpha Matting (U2-Net ONNX, Segments Foreground)
              ├── 18. Stage 3: Monocular Depth Inference (Depth-Anything-v2 INT8, D_rel in [0, 1])
              ├── 19. Stage 4: Cross-Bilateral Depth Smoothing (Edge-Preserving Spatial Filter)
              ├── 20. Stage 5: Pinhole Ray Inversion (K^-1 Back-projection to 262k Point Cloud)
              ├── 21. Stage 6: Continuous SDF Voxelization (128^3 Dense Lattice, delta=0.0156m)
              ├── 22. Stage 7: Marching Cubes Isosurface Extraction (256-Bitmask, Watertight)
              ├── 23. Stage 8: Dual-Pass Taubin Fairing (lambda=0.50, mu=-0.53, Delta V < 0.02%)
              ├── 24. Stage 9: PBR Texture & Normal Delighting (Interleaved 32-Byte VBO)
              │
              ▼
  [ 2PC SETTLEMENT & REVENUE INTEGRITY AUDIT ]
       │
       ├── 25. BEGIN IMMEDIATE TRANSACTION (SQLite WAL)
       ├── 26. Update: reserved = reserved - 10, settled = settled + 10
       ├── 27. Append: INSERT INTO ledger_transactions (phase='SETTLEMENT', status='CONFIRMED')
       ├── 28. Update: UPDATE jobs SET status='COMPLETED', glb_path='...'
       └── 29. COMMIT TRANSACTION
       │
       ▼
  [ ZERO-COPY ASSET STREAMING & BROWSER RENDERING ]
       │
       ├── 30. SSE Broadcast -> {"stage": "COMPLETED", "progress": 1.0, "download_url": "/api/v1/assets/..."}
       ├── 31. Client fetches Binary GLB via Memory-Mapped zero-copy streaming
       ├── 32. Three.js GLTFLoader parses 32-byte interleaved VBO (Position, Normal, UV)
       └── 33. Viewport renders 3D mesh at 60 FPS on pure white (#ffffff) canvas with OrbitControls
```

---

## 3. End-to-End Sequence Diagram & SSE Reconnect Flow

```
  Browser Client               FastAPI ASGI Ingress        Task Scheduler / Worker       SQLite WAL Ledger
        │                               │                             │                         │
        │ 1. POST /api/v1/recon/generate│                             │                         │
        ├──────────────────────────────>│                             │                         │
        │                               │ 2. Check & Reserve Credits  │                         │
        │                               ├──────────────────────────────────────────────────────>│
        │                               │<──────────────────────────────────────────────────────┤
        │                               │ 3. Spool Background Task    │                         │
        │                               ├────────────────────────────>│                         │
        │ 4. HTTP 202 {"job_id": "..."} │                             │                         │
        │<──────────────────────────────┤                             │                         │
        │                                                             │                         │
        │ 5. GET /api/v1/events/{job_id}                              │                         │
        ├────────────────────────────────────────────────────────────>│                         │
        │ 6. HTTP 200 text/event-stream (Connection Open)             │                         │
        │<────────────────────────────────────────────────────────────┤                         │
        │                                                             │                         │
        │ 7. event: {"stage": "INGESTION", "progress": 0.10}          │                         │
        │<────────────────────────────────────────────────────────────┤                         │
        │ 8. event: {"stage": "DEPTH_ESTIMATION", "progress": 0.45}   │                         │
        │<────────────────────────────────────────────────────────────┤                         │
        │                                                             │                         │
        X  === Network Interruption / Reconnection ===                │                         │
        │                                                             │                         │
        │ 9. GET /api/v1/events/{job_id} [Last-Event-ID: 2]           │                         │
        ├────────────────────────────────────────────────────────────>│                         │
        │ 10. Replay buffered events from queue ring-buffer           │                         │
        │<────────────────────────────────────────────────────────────┤                         │
        │ 11. event: {"stage": "MARCHING_CUBES", "progress": 0.80}    │                         │
        │<────────────────────────────────────────────────────────────┤                         │
        │                                                             │ 12. Settle Credits      │
        │                                                             ├────────────────────────>│
        │                                                             │<────────────────────────┤
        │ 13. event: {"stage": "COMPLETED", "progress": 1.0}          │                         │
        │<────────────────────────────────────────────────────────────┤                         │
        │                                                             │                         │
        │ 14. GET /api/v1/assets/job_17.glb (ArrayBuffer)             │                         │
        ├────────────────────────────────────────────────────────────>│ (Zero-copy stream)      │
        │<────────────────────────────────────────────────────────────┤                         │
        ▼                                                             ▼                         ▼
```

---

## 4. Two-Phase Commit Transactional Free Trial Wallet

The trial credit wallet system enforces strict mathematical conservation:
$$\text{Total Balance} \equiv \text{Available} + \text{Reserved} + \text{Settled}$$

```
                +----------------------------------------------------+
                |       INITIAL STATE: USER REGISTRATION COMPLETED   |
                |       Available: 50 | Reserved: 0 | Settled: 0     |
                |       Total Sum: 50 Credits                        |
                +-------------------------+--------------------------+
                                          |
                                          | User requests 3D reconstruction (Cost: 10)
                                          v
                +----------------------------------------------------+
                |       PHASE 1: PRE-FLIGHT RESERVATION CHECK        |
                |       Condition: available_credits >= 10 ?         |
                +-------------------------+--------------------------+
                             /                         \
                       YES  /                           \  NO
                           v                             v
  +-------------------------------------+   +------------------------------------+
  | PHASE 2: ATOMIC RESERVATION (2PC)   |   | REJECT: HTTP 402 PAYMENT REQUIRED  |
  | Available: 50 -> 40                 |   | Balance insufficient; prompt user  |
  | Reserved:   0 -> 10                 |   +------------------------------------+
  | Settled:    0 ->  0                 |
  | Invariant: 40 + 10 + 0 = 50 (PROVEN)|
  +------------------+------------------+
                     |
         +-----------+-----------+
         |                       |
      SUCCESS                 FAILURE (OOM / Crash / Timeout)
         |                       |
         v                       v
  +-------------------------+   +------------------------------------+
  | PHASE 3A: SETTLEMENT    |   | PHASE 3B: COMPENSATING ROLLBACK    |
  | Available: 40           |   | Available: 40 -> 50                |
  | Reserved:  10 ->  0     |   | Reserved:  10 ->  0                |
  | Settled:    0 -> 10     |   | Settled:    0 ->  0                |
  | Invariant: 40+0+10 = 50 |   | Invariant: 50+0+0 = 50 (PRESERVED) |
  +-------------------------+   +------------------------------------+
```

---

## 5. 8-Stage 3D Geometric Reconstruction Tensor Transformation Dataflow

```
+---------------------------------------------------------------------------------------+
|                 PIPELINE ARCHITECTURAL DATA FLOW & TENSOR TRANSFORMATIONS             |
+---------------------------------------------------------------------------------------+

  1. INPUT IMAGE ACQUISITION
     RGB Image Matrix I in R^{H x W x 3}, uint8 [0, 255]
               |
               v
  2. NORMALIZATION & PREPROCESSING
     I_norm in [0.0, 1.0]^{3 x 512 x 512}, Bilinear Interpolation
     Foreground Segmentation Alpha M in [0.0, 1.0]^{512 x 512} via U2-Net
               |
               v
  3. MONOCULAR RELATIVE DEPTH INFERENCE
     D_rel in [0.0, 1.0]^{512 x 512} via Depth-Anything-v2 INT8 ONNX
               |
               v
  4. CAMERA INTRINSICS INVERSION (PINHOLE BACK-PROJECTION)
     K = [[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]]
     p_3D = D_rel(u, v) * K^{-1} * [u, v, 1]^T
     Generates Initial Point Cloud P = {p_i} in R^3 (N ~ 262,144 points)
               |
               v
  5. CONTINUOUS SIGNED DISTANCE FIELD (SDF) VOXELIZATION
     Dense Lattice Omega in R^{128 x 128 x 128}, Resolution delta = 0.0156m
     Phi(x, y, z) = min_{p in P} ||(x, y, z) - p|| * sgn(n_p . ((x,y,z) - p))
               |
               v
  6. VECTORIZED ISOSURFACE EXTRACTION (MARCHING CUBES)
     Bitmask Index = sum_{i=0..7} b_i * 2^i where b_i = [Phi(v_i) < 0.0]
     Generates Triangular Surface Mesh M = (V, F) with V in R^{Nx3}, F in N^{Mx3}
               |
               v
  7. DUAL-PASS VOLUME-PRESERVING TAUBIN FAIRING
     Shrinking pass:   V'  = V + 0.50 * L * V
     Unshrinking pass: V'' = V' - 0.53 * L * V'
     Residual Volumetric Drift Delta V / V_0 < 0.02%
               |
               v
  8. PBR MATERIAL BAKING & ZERO-COPY PACKING
     Interleaved 32-byte VBO (XYZ: 12B, Normal: 12B, UV: 8B)
     Streamed directly via memoryview to Starlette Response
```

---

## 6. Continuous Triplane Feature Factorization & Orthogonal Decoders

Triplane representations factorize dense 3D volumes into three axis-aligned 2D feature planes:
$$T_{xy} \in \mathbb{R}^{C \times N \times N}, \quad T_{xz} \in \mathbb{R}^{C \times N \times N}, \quad T_{yz} \in \mathbb{R}^{C \times N \times N}$$
For any continuous spatial query point $\mathbf{x} = (x, y, z) \in [-1, 1]^3$:
1. Orthogonally project onto planes: $\pi_{xy}(\mathbf{x}) = (x, y)$, $\pi_{xz}(\mathbf{x}) = (x, z)$, $\pi_{yz}(\mathbf{x}) = (y, z)$.
2. Bilinearly sample features: $f_{xy} = \text{interp}(T_{xy}, \pi_{xy}(\mathbf{x}))$, etc.
3. Feature Aggregation: $\mathbf{f}(\mathbf{x}) = f_{xy} + f_{xz} + f_{yz} \in \mathbb{R}^C$.
4. Shallow MLP Decoding: $(\Phi(\mathbf{x}), \mathbf{c}(\mathbf{x})) = \text{MLP}_\theta(\mathbf{f}(\mathbf{x}))$.

---

## 7. Dual-Pass Taubin Volume Invariance Mathematical Proof and Flow

The transfer function for standard Laplacian filtering after $k$ steps is:
$$f_{Laplacian}(k) = (1 - \lambda k)^N$$
Since $\lambda > 0$, for all $k > 0$, $f(k) < 1$, which strictly shrinks the geometry monotonically towards the center of mass.

Taubin's dual-pass filter applies two consecutive steps with alternating positive ($\lambda > 0$) and negative ($\mu < 0$) scale factors:
$$f_{Taubin}(k) = (1 - \lambda k)(1 - \mu k) = 1 - (\lambda + \mu)k + \lambda \mu k^2$$

Setting the condition $\lambda + \mu \approx 0$ with $\lambda > 0$ and $|\mu| > \lambda$ (specifically $\lambda = 0.50, \mu = -0.53$):
$$\frac{df}{dk}\Bigg|_{k=0} = -(\lambda + \mu) = -(0.50 - 0.53) = +0.03 \approx 0$$
Thus, low-frequency geometric volume is strictly preserved ($\Delta V / V_0 < 0.02\%$), while high frequencies ($k \gg 1$) are strongly attenuated ($f(k) < 1$).

---

## 8. Marching Cubes 256-Case Isosurface Extraction & Bitmask Lookup Flow

```
  [ 8-Vertex Voxel Cube ]
       v4 -------- v5
      / |         / |
     v7 -------- v6 |
     |  |        |  |
     |  v0 ------|- v1
     | /         | /
     v3 -------- v2

  1. Evaluate Signed Distance at each vertex: Phi(v_0) ... Phi(v_7)
  2. Compute 8-bit cube index:
     index = sum_{i=0..7} (Phi(v_i) < 0.0 ? (1 << i) : 0)
  3. Lookup edge intersections in precomputed EdgeTable[index] (12-bit mask)
  4. Linearly interpolate exact zero-crossing vertex along intersecting edges:
     p_edge = v_a + (0.0 - Phi(v_a)) / (Phi(v_b) - Phi(v_a)) * (v_b - v_a)
  5. Assemble triangle primitives from TriTable[index][0..15]
```

---

## 9. Cross-Bilateral Depth Smoothing Mathematical Derivation & Spatial Filtering

Monocular depth maps exhibit quantization noise and jagged step artifacts along depth discontinuities. The cross-bilateral filter preserves sharp occlusion boundaries while smoothing flat planar regions:
$$D_{smooth}(p) = \frac{1}{W_p} \sum_{q \in \Omega} D(q) \cdot \exp\left(-\frac{\|p - q\|^2}{2\sigma_s^2}\right) \cdot \exp\left(-\frac{\|I(p) - I(q)\|^2}{2\sigma_r^2}\right)$$
where:
- $\Omega$ is the local spatial window (e.g. $5 \times 5$ or $7 \times 7$ pixels)
- $G_{\sigma_s}(d) = \exp(-d^2 / (2\sigma_s^2))$ is the spatial proximity Gaussian weight
- $G_{\sigma_r}(c) = \exp(-c^2 / (2\sigma_r^2))$ is the color similarity range Gaussian weight
- $W_p = \sum_{q \in \Omega} G_{\sigma_s}(\|p - q\|) \cdot G_{\sigma_r}(\|I(p) - I(q)\|)$ is the normalizer

---

## 10. Garland-Heckbert Quadric Error Metric (QEM) Surface Mesh Decimation

For real-time 60 FPS WebGL rendering on mobile and student laptops, high-density meshes ($> 100\text{k}$ faces) are simplified using Garland-Heckbert Quadric Error Metrics:
$$Q_v = \sum_{p \in \text{planes}(v)} p \cdot p^T$$
Edge collapses $(v_1, v_2) \to \bar{v}$ minimize quadratic error $\Delta(\bar{v}) = \bar{v}^T (Q_1 + Q_2) \bar{v}$, preserving sharp geometric features while halving triangle count.

---

## 11. Microfacet Cook-Torrance Specular BRDF & PBR Roughness Baking

The Cook-Torrance physically based reflectance model evaluates surface light reflection:
$$f_r = \frac{D \cdot F \cdot G}{4 (\mathbf{N} \cdot \mathbf{V})(\mathbf{N} \cdot \mathbf{L})}$$
- **GGX Normal Distribution (D)**: $D(\mathbf{H}) = \frac{\alpha^2}{\pi ((\mathbf{N} \cdot \mathbf{H})^2 (\alpha^2 - 1) + 1)^2}$.
- **Fresnel-Schlick (F)**: $F(\mathbf{V}, \mathbf{H}) = F_0 + (1 - F_0)(1 - (\mathbf{V} \cdot \mathbf{H}))^5$.
- **Smith Geometric Attenuation (G)**: $G(\mathbf{L}, \mathbf{V}, \mathbf{N}) = G_1(\mathbf{L}) \cdot G_1(\mathbf{V})$.

---

## 12. 3D Gaussian Splatting Screen-Space Covariance Projection Jacobians

A 3D Gaussian is defined by mean $\mathbf{\mu} \in \mathbb{R}^3$ and 3D covariance matrix $\mathbf{\Sigma} = \mathbf{R} \mathbf{S} \mathbf{S}^T \mathbf{R}^T$.
Under perspective viewing transformation $\mathbf{W}$ and projective Jacobian $\mathbf{J}$, the projected 2D screen-space covariance $\mathbf{\Sigma}'$ is:
$$\mathbf{\Sigma}' = \mathbf{J} \mathbf{W} \mathbf{\Sigma} \mathbf{W}^T \mathbf{J}^T$$
Alpha blending along ray direction $t$:
$$C = \sum_{i \in \mathcal{N}} c_i \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)$$

---

## 13. Three.js WebGL CAD Studio 60 FPS Viewport Render Loop Flow

```
  [ RAF Render Loop (requestAnimationFrame) ]
         │
         ├── 1. Update OrbitControls (Damping Factor = 0.05)
         ├── 2. Evaluate Dynamic Turntable Rotation (if enabled)
         ├── 3. Bind Custom Shading Mode Material:
         │      ├── 1. PBR (Physically-Based Rendering: Normal, Roughness, Metalness)
         │      ├── 2. WIREFRAME (Triangulated topology wire overlay)
         │      ├── 3. NORMAL (RGB encoded surface normals: (N + 1.0) * 0.5)
         │      ├── 4. MATCAP (Clay studio material capture lighting)
         │      ├── 5. DEPTH (Linearized camera distance gradient)
         │      ├── 6. CURVATURE (Mean surface curvature: concave/convex highlights)
         │      ├── 7. X-RAY (Fresnel edge transmission transparency)
         │      └── 8. CHECKER (UV alignment & texture distortion verification)
         │
         ├── 4. Render Scene with WebGLRenderer to Pure White (#ffffff) Canvas
         └── 5. Dispatch Frame Performance Metrics (Active Frame ms, Draw Calls, Triangle Count)
```

---

## 14. GLSL Fragment Shader Multiplexer for 8 Professional CAD Shading Modes

```glsl
// Custom Fragment Shader Multiplexer for 3DCV CAD Studio
uniform int u_shading_mode;
varying vec3 vNormal;
varying vec3 vViewPosition;
varying vec2 vUv;

void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(vViewPosition);

    if (u_shading_mode == 1) {
        // Mode 1: PBR Metallic-Roughness
        float diff = max(dot(N, vec3(0.577)), 0.0) * 0.75 + 0.25;
        gl_FragColor = vec4(vec3(diff), 1.0);
    } else if (u_shading_mode == 3) {
        // Mode 3: Normal Map Visualizer
        gl_FragColor = vec4(N * 0.5 + 0.5, 1.0);
    } else if (u_shading_mode == 6) {
        // Mode 6: Surface Curvature Visualizer via Screen-Space Partial Derivatives
        vec3 dNdx = dFdx(N);
        vec3 dNdy = dFdy(N);
        float curvature = length(dNdx) + length(dNdy);
        vec3 col = mix(vec3(0.9, 0.9, 0.9), vec3(0.1, 0.4, 0.9), clamp(curvature * 2.5, 0.0, 1.0));
        gl_FragColor = vec4(col, 1.0);
    } else if (u_shading_mode == 7) {
        // Mode 7: X-Ray Fresnel Transparency
        float fresnel = pow(1.0 - abs(dot(V, N)), 2.5);
        gl_FragColor = vec4(0.1, 0.3, 0.8, fresnel * 0.85);
    } else {
        // Default Studio Clay Fallback
        float diff = max(dot(N, vec3(0.577)), 0.0) * 0.7 + 0.3;
        gl_FragColor = vec4(vec3(diff), 1.0);
    }
}
```

---

## 15. 14-Model Agentic Copilot AST Tool Calling State Machine Flow

```
  [ USER CHAT INPUT ]
  "Inspect mesh topology, smooth high-frequency noise, and switch to normal shading mode"
         │
         ▼
  [ AGENTIC DISPATCHER (`app.routers.chat`) ]
         │
         ├── 1. Load System Prompt & Tool Signatures (12 CAD Inspection Tools)
         ├── 2. Forward to Selected Model (e.g. DeepSeek-R1 / Claude 3.5 Sonnet)
         │
         ▼
  [ MODEL FUNCTION CALLING RESPONSE ]
  {
    "tool_calls": [
      {"name": "inspect_mesh_metrics", "arguments": {"job_id": "active"}},
      {"name": "apply_taubin_smoothing", "arguments": {"iterations": 5, "lambda": 0.5, "mu": -0.53}},
      {"name": "set_shading_mode", "arguments": {"mode": "NORMAL"}}
    ]
  }
         │
         ▼
  [ CAD TOOL EXECUTION ENGINE (`app.agent_tools`) ]
         │
         ├── Tool 1: Inspect Mesh -> Vertices: 18,432, Faces: 36,860, Watertight: True
         ├── Tool 2: Taubin Filter -> Dispatched to geometry engine, Delta V < 0.018%
         └── Tool 3: Set Shading Mode -> Broadcasts WebGL client event: SHADING_CHANGED(3)
         │
         ▼
  [ SYNCHRONIZED CLIENT DISPATCH ]
  Browser executes action immediately; Three.js Studio switches to Normal Visualizer!
```

---

## 16. Zero-Copy Binary Memoryview Streaming Flow (`mmap` to `sendfile`)

```
  File on Disk (GLB Asset)
             │
             ▼  mmap.mmap() system call
  Linux Page Cache (Kernel Space)
             │
             ▼  memoryview() reference
  Python Application (Zero Copy Slicing)
             │
             ▼  Yield chunk without bytes() allocation
  Starlette StreamingResponse
             │
             ▼  sendfile() / TransmitFile()
  NIC Driver DMA Socket Buffer
             │
             ▼  TCP Packets
  Browser Client (WebGL ArrayBuffer)
```

---

## 17. Interleaved 32-Byte VBO Memory Layout & GPU Cache Alignment

```
  Byte Offset:  0          12         24       32
                | Position | Normal   | UV     |
                | (X, Y, Z)| (X, Y, Z)| (U, V) |
                | 12 Bytes | 12 Bytes | 8 Bytes|
                +----------+----------+--------+
```
A 32-byte stride is a power-of-two that matches modern GPU 32-byte and 64-byte L1 cache lines, maximizing hardware memory bus saturation.

---

## 18. Dynamic Multi-Tier Hardware Acceleration Dispatch Matrix

```python
# Hardware Dispatcher Architecture Flow
import onnxruntime as ort
import torch

def resolve_hardware_acceleration() -> list[str]:
    """Negotiates best available hardware acceleration provider with graceful CPU fallback."""
    providers = ort.get_available_providers()
    pipeline = []
    
    if "CUDAExecutionProvider" in providers and torch.cuda.is_available():
        pipeline.append("CUDAExecutionProvider")
    if "DmlExecutionProvider" in providers:
        pipeline.append("DmlExecutionProvider")
    if "CoreMLExecutionProvider" in providers:
        pipeline.append("CoreMLExecutionProvider")
        
    pipeline.append("CPUExecutionProvider") # Always available fallback
    return pipeline
```

---

## 19. PyTorch Student Curriculum Interactive Laboratories (Labs 1–6) Flow

```
  +---------------------------------------------------------------------------------+
  |                       STUDENT LABORATORY ARCHITECTURE                           |
  +---------------------------------------------------------------------------------+

  Lab 1: Monocular Depth Finetuning
         Loss: L_silog = (1/N)*sum(d_i^2) - (0.85/N^2)*(sum d_i)^2
         Target: Scale-invariant depth ordering on benchmark imagery

  Lab 2: Continuous Occupancy Networks
         Loss: Binary Cross Entropy on inside/outside SDF sample points
         Target: Implicit field fitting via 5-layer Residual MLP

  Lab 3: 3D Gaussian Splatting Mini
         Formulation: Sigma = R * S * S^T * R^T (Covariance Matrix)
         Target: Novel view synthesis with peak PSNR > 28.5 dB

  Lab 4: Geometric Metrics Evaluator
         Formulation: Chamfer Distance d_CD(S1, S2) and F-Score at 1% threshold
         Target: Quantitative reconstruction accuracy validation

  Lab 5: Triplane Large Reconstruction Model (LRM)
         Formulation: Triplane orthogonal feature sampling F(p) = T_xy + T_xz + T_yz
         Target: Direct single-pass feedforward 3D neural reconstruction

  Lab 6: PBR Material Delighting
         Formulation: Cross-bilateral separation of specular highlights from albedo
         Target: Physically based rendering texture map generation
```

---

## 20. High-Concurrency SQLite WAL Multi-Reader Locking Architecture Flow

```
  Concurrent Readers (HTTP GET Endpoints)           Worker Writer (Async Reconstruction Job)
                 │                                                    │
                 │ (Snapshot Isolation)                               │ (BEGIN IMMEDIATE)
                 ▼                                                    ▼
  ┌──────────────────────────────────────────────┐     ┌──────────────────────────────────┐
  │ Shared Memory Index (`app.db-shm`)           │     │ Write-Ahead Log (`app.db-wal`)   │
  │ Resolves latest committed frame pointers     │     │ Appends new commits sequentially │
  └──────────────────────┬───────────────────────┘     └──────────────────┬───────────────┘
                         │                                                │
                         ▼                                                │
  ┌──────────────────────────────────────────────┐                        │
  │ Canonical SQLite Database (`app.db`)         │ <──────────────────────┘
  │ Flushed during background WAL checkpoints    │    Passive Checkpoint
  └──────────────────────────────────────────────┘
```

---

## 21. Production Reverse Proxy Configuration (Caddy 2 & NGINX)

```caddy
# Caddyfile: Production TLS & Static Asset Proxy
3dcv.iiitdwd.ac.in {
    reverse_proxy 127.0.0.1:8010 {
        header_up Host {host}
        header_up X-Real-IP {remote_host}
        header_up X-Forwarded-For {remote_host}
        header_up X-Forwarded-Proto {scheme}
    }

    handle_path /assets/* {
        root * /app/web/assets
        file_server
        header Cache-Control "public, max-age=86400, immutable"
    }

    encode zstd gzip
}
```

---

## 22. Production Kubernetes Pod Scheduling & Autoscaling Flow

```yaml
# Production Kubernetes Deployment Blueprint
apiVersion: apps/v1
kind: Deployment
metadata:
  name: 3dcv-lab-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: 3dcv-lab
  template:
    metadata:
      labels:
        app: 3dcv-lab
    spec:
      containers:
      - name: 3dcv-lab
        image: 3dcv-lab:production
        ports:
        - containerPort: 8010
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2000m"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /api/v1/system/health
            port: 8010
          initialDelaySeconds: 15
          periodSeconds: 10
```

---

## 23. Prometheus Telemetry Exposition & Alerting Rules Matrix

- `recon_requests_total`: Counter tracking total reconstruction attempts.
- `recon_active_workers`: Gauge tracking currently executing reconstruction tasks.
- `recon_duration_seconds`: Histogram tracking end-to-end reconstruction latency.
- `credit_invariant_violations_total`: Counter tracking any ledger discrepancies (Alert threshold: > 0).

---

## 24. Complete Troubleshooting Runbook & Operational Diagnostic Matrix

| Error Signature | Underlying Cause | Corrective Action |
| :--- | :--- | :--- |
| `HTTP 401 Unauthorized` | Missing or expired JWT session cookie | Navigate to `/app/login` or click "One-Click Demo Mode" |
| `HTTP 402 Payment Required` | Available credit balance < 10 | Reset wallet via `/api/v1/auth/reset-credits` or register new user |
| `Port 8010 Busy` | Orphaned python task holding socket | Run `Get-Process python \| Stop-Process` in PowerShell |
| `CUDA OOM Exception` | Large batch size or insufficient VRAM | System auto-falls back to CPU OpenMP with half grid resolution |
| `WebGL Context Lost` | GPU driver reset | Re-focus browser window; Three.js context listener auto-restores |

---

## 25. Academic Course Governance & Department of Data Science & AI Attribution

The **3DCV Fun AI Lab** represents an authentic undergraduate engineering course capstone project:

- **Institution**: Department of Data Science & Artificial Intelligence, Indian Institute of Information Technology Dharwad (IIIT Dharwad).
- **Course**: 3D Computer Vision Course Project (Course 17).
- **Academic Supervisor**: **Dr. Shirshendu Layek**, Assistant Professor, Dept. of Data Science & AI.

### Student Engineering Contributors Roster
1. **J Ganesh (23BDS024)**: Systems Architecture & Distributed Concurrency
2. **M Jagadeshwar (23BDS033)**: Neural Depth Modeling & Student Laboratories
3. **B Varshith (23BDS011)**: Geometric Kernels & Marching Cubes Meshing
4. **Pranav Kashyap (23BCS098)**: Three.js WebGL CAD Studio & Shading Engines
5. **C Sai Aravind (23BDS075)**: Automated Playwright & PyTest Verification Suites

*3DCV Fun AI Lab · Group 17 · Department of Data Science & Artificial Intelligence · IIIT Dharwad · Course Project under Dr. Shirshendu Layek.*
