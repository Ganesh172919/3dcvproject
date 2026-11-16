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
python -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

Open [Studio](http://127.0.0.1:8010/app/studio), register a local account, and upload the sneaker or box from `sample_data/`. Select Depth Anything V2 and Draft. When the job completes, orbit the mesh and export GLB. Use a simple background and keep the whole object in view for your own photos.

The default model runs on CPU. With a compatible GPU build, set `RECON_DEVICE` to `cuda` or `directml` in `.env` and restart. Check the provider actually used in the generation details. This submission's end-to-end checks use CPU.

Optional chat requires your Gemini API key. Set `GEMINI_API_KEY` and an available `GEMINI_MODEL` in a local `.env` created from the `.env.example`. The server sends chat messages to Gemini. Local mesh generation uses the downloaded reconstruction weights.

**Evidence** `scripts/fetch_models.py` · `engine/recon/runtime.py` · `app/config.py`
