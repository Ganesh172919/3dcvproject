# Configuration reference

The app reads `.env` in the project source directory. Process environment values take precedence. See `.env.example` and `app/config.py` for the full application settings.

| Setting | Default or behavior |
| --- | --- |
| `APP_URL` | `http://localhost:8000`. Set it to the origin you use for the app. |
| `NODE_ENV` | `development`. |
| `DATABASE_URL` | SQLite in the project's `data/` directory. |
| `STORAGE_DRIVER` | `local`. |
| `STORAGE_ROOT` | `data/storage` under the project directory. |
| `GENERATION_PROVIDER` | `local`. |
| `RECON_DEVICE` | CPU by default. Use a matching DirectML or CUDA wheel for GPU inference. |
| `RECON_WARMUP` | `false`. |
| `MAX_UPLOAD_MB` | `20`. |
| `JOB_CONCURRENCY` | `1`. |
| `JOB_TIMEOUT_SECONDS` | `480`. |
| `SIGNUP_CREDITS` | `60`. |
| `ALLOW_REGISTRATION` | `true`. |
| `GEMINI_API_KEY` | Optional server-side chat key. |
| `GEMINI_MODEL` | Optional chat model override. |

The Windows launcher calls Uvicorn on port 8010 by default. The POSIX launcher defaults to port 8000. The server reports limits and installed model features in `GET /api/v1/meta`.

## Surface extraction and smoothing settings

Draft, Standard, and High map to presets in `engine/recon/config.py`. The CLI accepts `--quality`, `--faces`, and `--voxels`. Other slab and meshing settings live in `PipelineConfig` and the calibration program. For measured parameter changes, see the [benchmark](../BENCHMARK.md) and [theory notes](../explanations/volumetric-inflation.md).
