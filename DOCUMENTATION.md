# 3DFUN AI — Master Documentation & Engineering Portal

Welcome to the definitive documentation portal for **3DFUN AI: An End-to-End Image-to-3D Reconstruction Framework with Constrained Volumetric Inflation**.

Developed for the 3D Computer Vision (3DCV) course at the **Indian Institute of Information Technology, Dharwad (IIIT Dharwad)** under the faculty guidance of **Dr. Shirshendu Layek** (Dept. of DSAI).

---

## Master Documentation Suite

| Document | Description | Direct Link |
|---|---|---|
| **Master Technical Handbook** | The exhaustive 11-section master manual covering all mathematical derivations, algorithms, systems engineering, state machines, and operational runbooks. | [MASTER_DOCUMENTATION.md](file:///C:/Users/RAVIPRAKASH/Downloads/3dcvproject/3dfun-ai/docs/MASTER_DOCUMENTATION.md) |
| **System Architecture Specification** | Complete architectural breakdown of the 3-tier system (Engine, FastAPI Backend, WebGL Studio). | [ARCHITECTURE.md](file:///C:/Users/RAVIPRAKASH/Downloads/3dcvproject/3dfun-ai/docs/ARCHITECTURE.md) |
| **Scientific Benchmark Protocol** | Description of the 25-scene evaluation dataset (13 synthetic analytic + 12 CC0 real photos), metrics ($d_{\text{CD}}$, $F@1\%$, IoU, NC, SSIM), and baselines. | [BENCHMARK.md](file:///C:/Users/RAVIPRAKASH/Downloads/3dcvproject/3dfun-ai/docs/BENCHMARK.md) |
| **Volumetric Inflation Theory** | Detailed mathematical exposition of constrained occupancy fields, slab thickness equations, and level-set theory. | [volumetric-inflation.md](file:///C:/Users/RAVIPRAKASH/Downloads/3dcvproject/3dfun-ai/docs/explanations/volumetric-inflation.md) |
| **INT8 Quantization Strategy** | Theoretical analysis of Vision Transformer post-training quantization, symmetric/asymmetric calibration, and CPU SIMD acceleration. | [quantization-strategy.md](file:///C:/Users/RAVIPRAKASH/Downloads/3dcvproject/3dfun-ai/docs/explanations/quantization-strategy.md) |
| **14-Week Student Lab Notebook** | Chronological engineering and research sprints detailing algorithm breakthroughs, guide meetings, and debug sessions. | [STUDENT_LAB_NOTEBOOK.md](file:///C:/Users/RAVIPRAKASH/Downloads/3dcvproject/3dfun-ai/docs/STUDENT_LAB_NOTEBOOK.md) |
| **Viva-Voce Oral Defense Guide** | 12-minute synchronized presentation outline, second-by-second cue cards, and 20+ challenging theoretical examiner FAQs with complete answers. | [VIVA_VOCE_DEFENSE_GUIDE.md](file:///C:/Users/RAVIPRAKASH/Downloads/3dcvproject/3dfun-ai/docs/VIVA_VOCE_DEFENSE_GUIDE.md) |
| **Multi-Hardware Benchmark Log** | Empirical performance comparisons across 5 student laptops (Intel i7, AMD Ryzen 7, Apple M2) across CPU, DirectML, and CUDA runtimes. | [STUDENT_HARDWARE_BENCHMARK_LOG.md](file:///C:/Users/RAVIPRAKASH/Downloads/3dcvproject/3dfun-ai/docs/STUDENT_HARDWARE_BENCHMARK_LOG.md) |
| **Production Docker Deployment** | Multi-stage Dockerfile, Docker Compose topology, and Nginx reverse proxy configuration with SSE buffer disabling. | [deploy-docker.md](file:///C:/Users/RAVIPRAKASH/Downloads/3dcvproject/3dfun-ai/docs/how-to/deploy-docker.md) |
| **REST API Specification** | Endpoints, JWT authentication, SSE event protocol, and Pydantic schemas. | [api-spec.md](file:///C:/Users/RAVIPRAKASH/Downloads/3dcvproject/3dfun-ai/docs/reference/api-spec.md) |
| **Project Technical Report** | Reproducible academic report generated directly from raw benchmark runs. | [report.md](file:///C:/Users/RAVIPRAKASH/Downloads/3dcvproject/3dfun-ai/docs/report.md) |
| **IEEE Conference Paper** | Full 8-page IEEEtran camera-ready paper source and compiled PDF. | [main.pdf](file:///C:/Users/RAVIPRAKASH/Downloads/3dcvproject/3dfun-ai/paper/main.pdf) · [main.tex](file:///C:/Users/RAVIPRAKASH/Downloads/3dcvproject/3dfun-ai/paper/main.tex) |

---

## Repository Architecture

```
3dcvproject/
├── DOCUMENTATION.md                 # Root portal to the complete documentation suite
├── IEEEtran.cls                     # IEEE conference document class
└── 3dfun-ai/                        # Primary project package
    ├── engine/                      # 3D Reconstruction Engine (NumPy, SciPy, ONNX Runtime)
    │   ├── recon/                   # Math, depth, volume, meshing, texturing, export
    │   └── run_recon.py             # Standalone CLI entrypoint
    ├── app/                         # FastAPI Backend (auth, jobs, SSE, credits, storage)
    ├── web/                         # Vanilla ES6+ WebGL Studio (Three.js r160, zero npm)
    ├── benchmarks/                  # 25-scene benchmark suite & baseline implementations
    ├── training/                    # Synthetic data generator, SILog fine-tuner, Optuna optimizer
    ├── paper/                       # IEEEtran 8-page paper and LaTeX sources
    ├── presentations/               # 16:9 presentation deck and defense speaker notes
    ├── results/                     # Raw CSV logs, summaries, environment telemetry
    └── docs/                        # Complete MkDocs Material documentation source
```
