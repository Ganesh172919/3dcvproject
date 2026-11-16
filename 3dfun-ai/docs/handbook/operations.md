# Operations and troubleshooting

Start from the [local setup guide](local-setup.md). The package was exercised as a single-machine course build.

| Check | What to inspect |
| --- | --- |
| [Health](http://127.0.0.1:8010/api/v1/health) | Read `data.status` and `checks.engine`. An HTTP 200 response can include a degraded engine when weights are missing. |
| [Readiness](http://127.0.0.1:8010/api/v1/health/ready) | Checks that the app can query the database. It does not perform a reconstruction. |
| [Model availability](http://127.0.0.1:8010/api/v1/meta) | Inspect provider selection and the MiDaS and TripoSR flags. See the generation details for the provider used on a particular job. |
| [Metrics](http://127.0.0.1:8010/api/v1/metrics) | App counters and timings in Prometheus text format. |

- **Studio opens but a model is unavailable.** Run `python scripts/fetch_models.py --check` from the source folder. Fetch the missing weights and restart Studio. Add `--all --check` when checking the optional models too.
- **An upload fails.** Check the size shown by the upload control. Use PNG, JPEG, WebP, or BMP. Keep the object and its outline in frame.
- **Reconstruction fails or stalls.** Note the generation ID, stage, and error from Studio. Look for the corresponding server log entries. Try Draft with the shipped sample to check whether the issue is specific to the uploaded image or provider.
- **A GPU provider falls back to CPU.** Inspect the requested and available providers before retrying. Mesh cleanup and some model stages can still use CPU. Keep the working CPU demo available for review.
- **A 3D preview is unavailable.** Download the GLB and open it in a separate viewer. Compare the mesh from several angles because hidden surfaces are inferred.

## Back up and restore the local instance

Wait for active jobs to finish, then stop the server. Back up the database and asset storage together while the server is stopped. In the default setup they are under `data/`. If you set `DATABASE_URL` or `STORAGE_ROOT`, use those locations instead. Restore both from the same backup before restarting. Open Studio and check one previously completed asset and one new generation. Keep credentials, local uploads and backups outside the course archive.

For deployment beyond loopback, review `NODE_ENV`, `APP_URL`, separate access and refresh secrets, TLS, and `COOKIE_SECURE`. The app warns about insecure production settings. Configure ingress, access, backup retention and capacity for the target environment, and validate them there. Cloud deployment and multi-worker operation are outside the verified course setup.

**Evidence** `app/routers/system.py` · `app/config.py` · `app/storage.py`
