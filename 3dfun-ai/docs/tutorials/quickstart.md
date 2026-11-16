# First reconstruction

Use Python 3.12 from the project source directory. In the course archive this is `code/`. Create a virtual environment with `python -m venv .venv`. Activate it with `.\.venv\Scripts\Activate.ps1` on Windows or `source .venv/bin/activate` on macOS and Linux.

```sh
python -m pip install -r requirements.txt
python scripts/fetch_models.py
python -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

Open [3DFUN AI](http://127.0.0.1:8010/), open Studio, and create a local account. Choose **Depth Anything V2** and **Draft**, then upload `engine/testdata/sneaker.png` or choose **Sneaker** from the built-in samples. Select **Generate model**. When it finishes, rotate the mesh and download the GLB. See the [viewer guide](web-studio-interactive-viewer.md) for inspection controls.

Run the same method from a terminal:

```sh
python engine/run_recon.py engine/testdata/sneaker.png --quality draft --out out/sneaker
```

Open `model.glb` from the output directory. Read `manifest.json` for per-stage timings and geometry diagnostics. Use the [troubleshooting guide](../TROUBLESHOOTING.md) if generation fails.
