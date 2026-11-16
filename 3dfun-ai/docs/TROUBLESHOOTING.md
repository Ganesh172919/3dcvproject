# Troubleshooting

| Symptom | Check |
| --- | --- |
| Studio loads but generation cannot start | Review `data.checks.engine` in `GET /api/v1/health` and run `python scripts/fetch_models.py --check`. Use `--all --check` to verify optional provider files too. |
| A model download failed | Run the same downloader command again. It verifies the expected SHA-256 digest before accepting a file. |
| A different engine is unavailable | Run `python scripts/fetch_models.py --all`. Check installed execution providers in `GET /api/v1/meta`. |
| Studio returns 401 | Sign in again. The default refresh cookies are tied to the local browser session. Use the same hostname and port throughout the demo. |
| The page has no WebGL viewer | Open the Studio through FastAPI. A static file opened directly in a browser cannot resolve the app's module URLs. Try a browser with WebGL enabled. |
| The mesh closes a hole or loses a narrow part | Compare the input and foreground mask with the mesh. Single-view silhouette cleanup and the thickness prior can erase thin parts or fill openings. |
| A job exceeds available memory | Start with Draft and run one job at a time. Disable turntable export when it is unnecessary. |
| Gemini reports missing configuration | Put `GEMINI_API_KEY` in the project-root `.env` and use **Check setup**. Reconstruction uses the local ONNX models independently of chat. |
| Benchmark results differ from the report | Compare your dataset and environment with the recorded metadata. A regenerated dataset uses only available cached photos. |
| Report or demo link returns 404 in a source checkout | Build the course artifacts into `out/course17/` before opening either link. The extracted course archive has both files at its root. |

See the [operations guide](handbook/operations.md) for local checks and the [packaging guide](how-to/academic-packaging-and-verification.md) to rebuild the course deliverables.
