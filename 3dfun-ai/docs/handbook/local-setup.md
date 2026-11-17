# Local setup and model use

Follow the `README.md` to install the pinned CPU dependencies with Python 3.12. Run commands from the source folder, called `code/` when you extract the archive.

## Download and check weights

```sh
python scripts/fetch_models.py
python scripts/fetch_models.py --check
```

The base download installs U²-NetP and Depth Anything V2 Small. Use `--midas` or `--triposr` for an optional model and `--all` for both. Each downloaded file is checked against the digest expected by the project. Public weight sources and project changes are listed in the archive README. Windows GPU environments use the provided `requirements-directml.txt` or `requirements-cuda.txt` in a separate virtual environment.

## Run Studio locally

```sh
./run.ps1
```

Run the command above in Windows PowerShell. On macOS or Linux, use `bash run.sh` with the Python environment activated. Open [Studio](http://127.0.0.1:8010/app/studio), sign in with the demo account or register locally, and upload the sneaker or box from `sample_data/`. Select Depth Anything V2 and Draft. When the job completes, orbit the mesh and export GLB. Use a simple background and keep the whole object in view for your own photos.

The default model runs on CPU. With a compatible GPU build, set `RECON_DEVICE` to `cuda` or `directml` in `.env` and restart. Check the provider actually used in the generation details. This submission's end-to-end checks use CPU.

The chat can answer a few course questions locally. To enable the optional Gemini provider, set `GEMINI_API_KEY` and a model available to your account in a local `.env` created from `.env.example`. Cloud chat sends messages to Gemini. Mesh generation uses the local reconstruction weights.

**Evidence** `scripts/fetch_models.py` · `engine/recon/runtime.py` · `app/config.py`
