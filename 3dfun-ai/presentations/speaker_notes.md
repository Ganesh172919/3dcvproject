# 3DFUN AI — Oral Presentation Defense Notes
**Course:** 3D Computer Vision Course Project (Autumn 2026)
**Institution:** Indian Institute of Information Technology Dharwad
**Guide:** Dr. Shirshendu Layek, Assistant Professor, Dept. of Data Science and AI, IIIT Dharwad
**Defense Duration:** 12 Minutes Total (15 Slides · 5 Presenters · 3 Slides Each)

---

## Defense Schedule & Speaker Roster
| Presenter | Roll No | Presentation Slot | Slides | Focus Topic |
| :--- | :--- | :--- | :--- | :--- |
| **J Ganesh** | 23BDS024 | 0:00 – 2:15 (2:15) | Slides 1–3 | Problem Ill-Posedness, Bas-Relief Ambiguity, System Constraints |
| **M Jagadeshwar** | 23BDS033 | 2:15 – 4:45 (2:30) | Slides 4–6 | Related Work Taxonomy, Modular Architecture, 6-Stage Pipeline |
| **B Varshith** | 23BDS011 | 4:45 – 7:30 (2:45) | Slides 7–9 | Benchmark Protocol, Multi-Engine Results, Latency Profiling |
| **Pranav Kashyap** | 23BCS098 | 7:30 – 10:15 (2:45) | Slides 10–12 | Component Ablations, Qualitative Heatmaps, Web Studio & AR |
| **C Sai Aravind** | 23BDS075 | 10:15 – 12:00 (1:45) | Slides 13–15 | Limitations & Edge Cases, Conclusion & Roadmap, Defense FAQ |

---

## Part 1: J Ganesh (23BDS024)
*Slides 1 to 3 [0:00 – 2:15] · Problem & Theoretical Constraints*

### Slide 1
[Speaker: J Ganesh (23BDS024) | 0:00 - 0:45] Good morning professors and examiners. We are team 3DFUN AI from IIIT Dharwad, working under the guidance of Dr. Shirshendu Layek. Our team of five has divided this 12-minute defense into five structured sections: I will cover the problem formulation and constraints in Slides 1–3; Jagadeshwar will present our architecture and 6-stage pipeline in Slides 4–6; Varshith will cover benchmark protocol and empirical results in Slides 7–9; Pranav will detail component ablations and our full-stack studio in Slides 10–12; and Sai Aravind will address failure modes, future directions, and conclude in Slides 13–15. Every number in this presentation is read dynamically from results/.

---

### Slide 2
[Speaker: J Ganesh (23BDS024) | 0:45 - 1:30] Monocular 3D reconstruction is fundamentally ill-posed: any single view captures only the visible front surface, leaving the back surface, interior, and occluded concavities unobserved. Deep generative diffusion models require massive VRAM and hallucinate unpredictable backs. Our approach introduces two mathematically verifiable priors: a silhouette prism constraint and depth-dependent thickness. The pipeline then measures mesh closure after cleanup.

---

### Slide 3
[Speaker: J Ganesh (23BDS024) | 1:30 - 2:15] We designed 3DFUN AI around strict accessibility constraints: zero GPU dependency in the target runtime. Using Depth Anything V2 Small quantized to INT8 (27 MB) and U²-Net portable (4.6 MB), the evaluated pipeline runs on the reference dual-core CPU. The right side shows the measured runtime including rendering, mean peak RSS, triangle count and closure rate.

---

## Part 2: M Jagadeshwar (23BDS033)
*Slides 4 to 6 [2:15 – 4:45] · Architecture & Volumetric Pipeline*

### Slide 4
[Speaker: M Jagadeshwar (23BDS033) | 2:15 - 3:00] Thank you Ganesh. In related work, we honestly categorize existing families. GPU-only large reconstruction and diffusion models could not run locally, so we transparently label them 'not evaluated' rather than citing mismatched numbers. Instead, we re-implemented two rigorous classical baselines—TSDF fusion and Visual Hull—evaluated on identical inputs, triangle budgets, and voxel resolutions.

---

### Slide 5
[Speaker: M Jagadeshwar (23BDS033) | 3:00 - 3:45] System architecture follows a clean three-tier separation: an interactive vanilla JavaScript + Three.js WebGL Studio, a robust FastAPI backend with credit reservations, JWT authentication, and Server-Sent Events, and a decoupled NumPy reconstruction engine. Because the engine is an adapter, the benchmark measures the exact code path the product runs.

---

### Slide 6
[Speaker: M Jagadeshwar (23BDS033) | 3:45 - 4:45] The reconstruction pipeline consists of 6 stages: 1) U²-Net foreground matting, 2) Depth Anything V2 monocular depth, 3) bilateral symmetry enforcement, 4) monotonic volumetric inflation via Euclidean Distance Transform, 5) Marching Cubes and Taubin smoothing, and 6) PBR texture baking. The code attempts to fill small holes before validating the mesh. Handing over to Varshith for Part 3.

---

## Part 3: B Varshith (23BDS011)
*Slides 7 to 9 [4:45 – 7:30] · Benchmarking, Metrics & Profiling*

### Slide 7
[Speaker: B Varshith (23BDS011) | 4:45 - 5:30] Thank you Jagadeshwar. Our benchmark suite evaluates 25 scenes: 13 procedural analytic models with closed-form ground truth geometry, and 12 CC0 real-world photographs. Geometric accuracy is evaluated with Chamfer Distance, F-Score@1%, Volumetric IoU, and Normal Consistency, while appearance is measured using SSIM and silhouette IoU.

---

### Slide 8
[Speaker: B Varshith (23BDS011) | 5:30 - 6:30] Read the table row by row, then present the F-score bar chart on the right. Highlight where 3DFUN AI leads: F-score and Volumetric IoU due to closed-slab inflation. Explain the higher end-to-end runtime, which includes preview rendering. Stress that all metrics were recorded across 3 repeated runs per scene on identical hardware.

---

### Slide 9
[Speaker: B Varshith (23BDS011) | 6:30 - 7:30] Examining runtime profiling: core reconstruction accounts for only a minor fraction of total execution time. Software turntable and preview rendering on CPU account for over 65% of latency. Disabling turntable generation reduces total runtime by nearly half, as demonstrated in our ablation table. Handing over to Pranav for Part 4.

---

## Part 4: Pranav Kashyap (23BCS098)
*Slides 10 to 12 [7:30 – 10:15] · Ablation, Quality & Web Studio*

### Slide 10
[Speaker: Pranav Kashyap (23BCS098) | 7:30 - 8:30] Thank you Varshith. Our ablation isolates every pipeline stage: removing the slab core causes the steepest Chamfer penalty and destroys watertightness; removing the depth term creates a flat silhouette plaque; and Taubin smoothing reduces voxel staircase artifacts while limiting shrinkage. The thickness sweep rigorously confirms κ = 0.70 as our optimal empirical setting.

---

### Slide 11
[Speaker: Pranav Kashyap (23BCS098) | 8:30 - 9:30] In qualitative evaluation under identical canonical poses: Visual Hull produces extruded prismatic blocks with zero curvature; TSDF suffers from open edges and torn boundaries; 3DFUN AI preserves front surface relief in these examples. Occluded hollow interiors can be filled by the slab prior.

---

### Slide 12
[Speaker: Pranav Kashyap (23BCS098) | 9:30 - 10:15] Use the sample photo to explain the upload, engine selection, and progress stages. Orbit a completed mesh, then show the export actions and the Experiment notebook.

---

## Part 5: C Sai Aravind (23BDS075)
*Slides 13 to 15 [10:15 – 12:00] · Edge Cases, Roadmap & Defense*

### Slide 13
[Speaker: C Sai Aravind (23BDS075) | 10:15 - 11:00] Thank you Pranav. We openly document all limitations: strongly asymmetric objects receive symmetric backs; semi-transparent fur or glass degrades foreground matting; and planar UV back-projection stretches at grazing angles. Every failure mode is automatically flagged and saved to failures.csv with full transparency.

---

### Slide 14
[Speaker: C Sai Aravind (23BDS075) | 11:00 - 11:30] The prototype generates meshes and records their topology. Future directions include learned occupancy regressors for complex backs and conformal UV unwrapping. The README gives the commands for reproducing the benchmark and rebuilding the deliverables.

---

### Slide 15
[Speaker: C Sai Aravind (23BDS075) | 11:30 - 12:00] Team contributions: Ganesh led the core volumetric engine; Jagadeshwar owned meshing and texture baking; Varshith developed the benchmark harness and metrics; Pranav engineered the FastAPI backend and credit ledger; and Sai Aravind built the Web Studio and viewer. We thank Dr. Shirshendu Layek for his mentorship and welcome questions from the examination committee.

---
