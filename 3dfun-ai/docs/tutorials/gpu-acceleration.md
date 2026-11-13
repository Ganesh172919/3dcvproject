# GPU inference in 3DFUN AI

Use the same local photos and ONNX weights across execution providers. The Studio records the provider used at each inference stage in the model manifest. The dashboard shows those providers alongside the run measurements.

## Windows

Run these commands from the project directory. Stop the current server first if it already uses port 8010.

```powershell
# DirectX 12 graphics on Windows
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Device directml

# NVIDIA with CUDA 12.8 and cuDNN 9.x available
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Device cuda
```

The launcher creates a separate virtual environment for each ONNX Runtime package. Use `-InstallModels` to fetch every supported checkpoint. Use `-Device cpu` to start the default CPU build.

## Linux with NVIDIA

Create a separate Python 3.12 environment and activate it. Install the CUDA 12.8 and cuDNN 9.x runtime libraries for the selected ONNX Runtime wheel. Then run:

```bash
python -m pip install -r requirements-cuda.txt
python scripts/fetch_models.py --all
RECON_DEVICE=cuda ./run.sh
```

## Compare a run

1. Open the Studio and select a photo.
2. Run Draft with Depth Anything V2 or MiDaS Small.
3. Open the model details and inspect the inference provider.
4. Run the same photo and quality on the CPU, then compare the results in the dashboard's experiment notebook.

DirectML can run background matting and the two depth estimators. The TripoSR encoder and decoder use the CPU in the DirectML build. The CUDA build can request GPU inference for all three engines. Unsupported operations may execute on the CPU, and an ONNX GPU session that fails is retried on the CPU. Meshing, texture baking, preview rendering, and export run on the CPU.

For CUDA, `RECON_CUDA_MEMORY_MB` limits the ONNX CUDA allocator arena to 2,048 MiB by default. The CUDA driver and libraries can allocate memory outside that arena.

Use the prototype's measured times to compare your machine's providers. An integrated GPU can take longer than the CPU when session setup and transfer time dominate.

*Authors: J Ganesh, M Jagadeshwar, B Varshith, Pranav Kashyap, C Sai Aravind (Supervisor: Dr. Shirshendu Layek, Dept. of DSAI, IIIT Dharwad).*
