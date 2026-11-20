# First reconstruction

Use Python 3.12 from the project source directory. In the course archive this is `code/`. Create a virtual environment with `python -m venv .venv`. Activate it with `.\.venv\Scripts\Activate.ps1` on Windows or `source .venv/bin/activate` on macOS and Linux.

```sh
python -m pip install -r requirements.txt
python scripts/fetch_models.py
python -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

Open [3DCV Lab Group 17](http://127.0.0.1:8010/), open Studio, and create a local account. Choose **Depth Anything V2** and **Draft**, then upload `engine/testdata/sneaker.png` or choose **Sneaker** from the built-in samples. Select **Generate model**. When it finishes, rotate the mesh and download the GLB. See the [viewer guide](web-studio-interactive-viewer.md) for inspection controls.

Run the same method from a terminal:

```sh
python engine/run_recon.py engine/testdata/sneaker.png --quality draft --out out/sneaker
```

Open `model.glb` from the output directory. Read `manifest.json` for per-stage timings and geometry diagnostics. Use the [troubleshooting guide](../TROUBLESHOOTING.md) if generation fails.


---

## Complete System Architecture Flow

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                QUICKSTART SYSTEM ARCHITECTURE                                    │
└────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                 │
    ┌────────────────────────┬───────────────────┴───────────────────┬────────────────────────┐
    ▼                        ▼                                       ▼                        ▼
[SINGLE COMMAND LAUNCH]  [STUDIO CAD WORKSPACE]                  [8 RECON ENGINES]        [STUDENT LABS PORTAL]
• npm run dev            • Three.js WebGL CAD Viewport           • Depth Anything V2 INT8 • 6 PyTorch Labs
• Binds to 127.0.0.1:8010• 8 Real-Time Shading Passes            • TripoSR LRM            • Real-time Loss Curves
• Fast Uvicorn Server    • 🎮 Load Demo 3D Mesh Quickstart       • MiDaS Small            • Metric Benchmark PLY
• SQLite WAL Database    • Natural Language Copilot Chat Bar     • SIREN Implicit SDF     • Watertight Verification
```

---

## 4-Step Developer Onboarding Guide

1. **Clone & Bootstrap**:
   ```powershell
   cd 3dfun-ai
   npm run dev
   ```
2. **Access Web Studio**:
   Open `http://127.0.0.1:8010/app/studio` in your browser.
3. **Execute Verification Suites**:
   ```powershell
   .\.venv\Scripts\python.exe -m pytest tests/test_auth.py tests/test_chat_tools.py tests/test_student_labs.py tests/test_runtime.py -v
   ```
4. **Audit Course Packaging**:
   ```powershell
   .\.venv\Scripts\python.exe -m submission.verify out/course17/Group17_3dcv_project.zip
   ```

*Quickstart Guide documented by Group 17 · 3D Computer Vision Course Project · Department of Data Science & Artificial Intelligence · IIIT Dharwad.*
