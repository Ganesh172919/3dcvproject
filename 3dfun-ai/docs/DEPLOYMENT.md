# 3DFUN AI — deployment and operations

The benchmark reference environment is recorded in `results/environment.json`. The app also has
a Windows PowerShell launcher for local use.

---

## 1. What you need

| Requirement | Value on the reference machine |
|---|---|
| OS / arch | Linux x86-64 |
| CPU | Intel Xeon @ 2.60 GHz, **2 cores**, no GPU |
| RAM | 2,030,416 kB (~2 GB) — the constraint that shapes every design decision |
| Python | 3.13.16 |
| Disk | ~1.1 GB for code + models + results (179 MB of that is `results/raw/meshes`) |

Runtime dependencies are pinned in `requirements.txt`; the exact resolved versions used for the
results are in `results/environment.json`. The reference benchmark uses these two model files:

* `engine/models/depth_anything_v2_small_quantized.onnx` — 27.26 MB,
  SHA-256 `fcf51f1b…547f20d`
* `engine/models/u2netp.onnx` — 4.57 MB, SHA-256 `309c8469…f4ddd8`

`make models` fetches the benchmark and Studio model files and **refuses to install a file whose SHA-256 does not match** the
pinned digest, so a corrupted or substituted network cannot silently change a reported number.

## 2. Local run (the supported path)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
make models                  # downloads and checksums the Studio models
bash run.sh --seed           # http://localhost:8000, dev data seeded
```

Open `http://localhost:8000` — the studio, dashboard, benchmark page and API docs are all served
from the same process. `PORT=9000 ./run.sh` changes the port; `HOST` defaults to `0.0.0.0` so the
server is reachable from another container or a preview proxy.

Install the development requirements to verify the saved artifacts. Run the benchmark first if you need a new set of measurements.

```bash
make install-dev
make benchmark   # overwrites the saved results with a new benchmark
make all         # rebuild figures, paper, slides and report
make verify
```

The 225-run benchmark is the longest step. See §6 for how to resume it.

## 3. Configuration

The application reads the project-root `.env` and gives the process environment precedence.
See `.env.example` and `app/config.py` for defaults. You can also supply values from your process manager:

```bash
export DATABASE_URL="sqlite:///$(pwd)/data/3dfun.db"
export STORAGE_DRIVER=local STORAGE_ROOT="$(pwd)/data/storage"
export JWT_ACCESS_SECRET="$(python3 -c 'import secrets;print(secrets.token_urlsafe(48))')"
export JWT_REFRESH_SECRET="$(python3 -c 'import secrets;print(secrets.token_urlsafe(48))')"
export ALLOW_REGISTRATION=false COOKIE_SECURE=true
```

The keys that actually exist, with their defaults and meaning, are the `Settings` fields:

| Variable | Default | Notes |
|---|---|---|
| `APP_URL` | `http://localhost:8000` | used in emails/links |
| `NODE_ENV` | `development` | `production` enables `validate_production()` |
| `DATABASE_URL` | `sqlite:///data/3dfun.db` | PostgreSQL works unchanged via SQLAlchemy |
| `REDIS_URL` | empty | optional; the queue falls back to the in-process worker |
| `JWT_ACCESS_SECRET` / `JWT_REFRESH_SECRET` | dev-only strings | production refuses to start with the dev defaults |
| `ACCESS_TOKEN_TTL` / `REFRESH_TOKEN_TTL` | 900 / 2592000 s | |
| `COOKIE_SECURE`, `ALLOW_REGISTRATION` | false / true | |
| `STORAGE_DRIVER`, `STORAGE_ROOT`, `STORAGE_ENDPOINT`, `STORAGE_BUCKET`, `STORAGE_ACCESS_KEY`, `STORAGE_SECRET_KEY`, `SIGNED_URL_TTL` | local, `data/storage`, … | S3-compatible stores use `STORAGE_DRIVER=s3` |
| `MAX_UPLOAD_MB` | 20 | enforced before decode |
| `RATE_LIMIT_MAX` / `RATE_LIMIT_WINDOW` / `LOGIN_MAX_ATTEMPTS` | 120 / 60 s / 8 | |
| `JOB_TIMEOUT_SECONDS` / `JOB_CONCURRENCY` | 480 / 1 | per-job watchdog; keep concurrency at 1 on 2 GB |
| `SIGNUP_CREDITS`, `CREDITS_DRAFT`, `CREDITS_STANDARD`, `CREDITS_HIGH` | 60, 4, 8, 16 | ledger units |
| `GENERATION_PROVIDER` (`local`\|`http`\|`mock`), `GENERATION_API_URL`, `GENERATION_API_KEY`, `GENERATION_MODEL` | local | |
| `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `EMAIL_PROVIDER`, `RESEND_API_KEY` | empty / console | optional integrations; unused paths are inert |
| `LOG_LEVEL`, `SENTRY_DSN` | INFO, empty | |

`app/config.py::Settings.validate_production()` returns a list of blocking problems (dev JWT
secrets, `COOKIE_SECURE=false`, local storage in production). Run it in your entrypoint:

```python
from app.config import settings
problems = settings.validate_production()
if problems:
    raise SystemExit("refusing to start: " + "; ".join(problems))
```

> `.env.example` is kept in sync with this table: every variable it lists is read by
> `app/config.py::Settings`, and every `Settings` field is listed there. If a key is not in
> `app/config.py`, setting it does nothing — check the table, not the file, when unsure.

## 4. Running behind a process manager

`run.sh` uses `uvicorn --reload` (development). For a deployment, drop `--reload`, run one worker
(the engine is memory-bound and `JOB_CONCURRENCY` should stay at 1 on 2 GB), and terminate TLS at
the proxy:

```
# /etc/systemd/system/3dfun.service
[Service]
WorkingDirectory=/srv/3dfun-ai
Environment=APP_URL=https://3dfun.example.org
Environment=NODE_ENV=production
Environment=COOKIE_SECURE=true
Environment=STORAGE_DRIVER=s3
EnvironmentFile=/etc/3dfun/secrets.env
ExecStart=/srv/3dfun-ai/.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
```

Health and readiness are built in and are the right probe targets:

```bash
curl -fsS http://127.0.0.1:8000/api/v1/health        # liveness
curl -fsS http://127.0.0.1:8000/api/v1/health/ready  # DB + engine + models present
curl -fsS http://127.0.0.1:8000/api/v1/metrics       # Prometheus text
```

## 5. Static assets and the SPA

`web/app/index.html` imports three.js from `/static/vendor/three.module.js` through an import map,
so the app must be served by the FastAPI process (or by a proxy that serves `web/` at the same
origin). Absolute `/static/...` paths break if you mount the studio under a sub-path — deploy it at
the origin root, or set the base in the import map.

## 6. Operations that matter

* **Crash recovery.** State lives in SQLite (`data/3dfun.db`) and every artefact on disk, so
  nothing is lost, but the in-process queue is **not** durable: a row that was `processing` when
  the process died stays in that state. Recovery is explicit — call
  `POST /api/v1/generations/{id}/retry`, which is allowed for `FAILED` *and* `CANCELLED` rows and
  re-reserves credits. Per-job progress is timestamped on the row, so `SELECT id, stage,
  updated_at FROM generations WHERE status='PROCESSING'` finds the stale ones. If you deploy with
  more than one worker, put a real queue behind `REDIS_URL` instead of relying on the in-process
  runner.
* **Interrupting a benchmark.** `results/raw/runs.jsonl` is appended per finished cell; rerunning
  `make benchmark` reuses cached meshes *with their original timings*. `--force` starts clean.
* **Bounded resources.** `PipelineConfig.memory_budget_mb` (1400) is passed to the engine and used
  to size the texture atlas; the measured peaks are 926–933 MB for all three methods. The benchmark has been
  OOM-killed at ~735 MB in a smaller cgroup — do not run it next to a dev server on 2 GB.
* **Backups.** Two things must be backed up together: the database (`data/3dfun.db`) and
  `data/storage/` (content-addressed by SHA-256, so a file is either present or regenerable from
  the manifest). A DB restore without storage yields dangling asset rows.
* **Secret rotation.** `python3 -c "from app.config import generate_secret; print(generate_secret())"`
  emits a suitable value; rotating `JWT_*` invalidates sessions (by design).

## 7. Resource envelope (measured)

| Operation | Time | Peak RSS |
|---|---|---|
| draft reconstruction, 8 000 tris | 27.7 s ± 5.3 (render included) | ~926–933 MB |
| same, preview/turntable off | ~3.2 s (stage sum minus render) | ~930 MB |
| TSDF baseline | 1.43 s ± 0.24 | ~930 MB |
| visual-hull baseline | 2.13 s ± 0.33 | ~930 MB |
| full 225-cell benchmark | see README reproduce table | ≤ 933 MB |

The renderer is the single most expensive component (24.56 s of the 27.70 s mean) because it
rasterises 10 turntable frames plus previews in NumPy on 2 cores. That is a product decision, not a
reconstruction cost, and both numbers are reported.
