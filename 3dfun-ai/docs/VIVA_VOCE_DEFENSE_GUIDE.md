# Viva preparation

Use the [presentation deck](../presentations/3DFUN-AI-Project.pptx) and its speaker notes to rehearse. The times below are a suggested 12-minute split.

| Time | Slides | Topic |
|---|---:|---|
| 0:00–2:15 | 1–3 | The single-view problem, what the photo reveals, and the project's scope |
| 2:15–4:45 | 4–6 | Related work, system architecture, and depth-based reconstruction |
| 4:45–7:30 | 7–9 | Dataset splits, metrics, baseline results, and runtime |
| 7:30–10:15 | 10–12 | Ablations, examples, and the Studio demo |
| 10:15–12:00 | 13–15 | Failure cases, conclusions, and team contributions |

## Questions to prepare

- **Why is reconstruction ambiguous?** A single photo shows the visible surface. It leaves the back, cavities, and absolute scale underdetermined.
- **How do the depth-based models differ from TripoSR?** Depth Anything V2 and MiDaS estimate relative depth and pass it to the course geometry pipeline. TripoSR predicts a 3D field and uses vertex colour.
- **Which models were evaluated in the paper?** The course depth-based method, a TSDF baseline, and a visual-hull baseline. The other Studio engines are available for local comparison.
- **How was success measured?** Explain the synthetic reference metrics separately from the runtime, topology, and appearance measurements available for photographs. See the [benchmark protocol](BENCHMARK.md).
- **What does watertightness tell us?** It measures mesh closure. The course method passed the closure check in 72 of its 75 recorded runs. Even a closed mesh can misrepresent hidden geometry.
- **How can a reviewer trace a result?** Locate the matching configuration in [the summary table](../results/summary/main_table.csv), the [per-run CSV](../results/raw/runs.csv), and [the recorded environment](../results/environment.json).

## Live demo

Open Studio, select the teapot photo and Draft quality, and generate a model. Rotate the mesh and open the export actions. The homepage includes completed examples for inspecting and downloading a GLB during a timed presentation.
