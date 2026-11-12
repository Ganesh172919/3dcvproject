# 3DFUN AI — troubleshooting

Every entry below is a failure that actually occurred while building this project, or an error the
code itself can emit. Error codes are quoted exactly as the API returns them
(`{"error": {"code": ..., "message": ...}}`). If your problem is not here, the fastest path is
`make verify` — it prints which check failed and why.

---

## 0. First three commands

```bash
python3 -m pytest tests -q                 # 39 tests, ~15 s — logic and API
python3 scripts/verify_submission.py --fast  # 30 checks, ~1 min — artefacts and numbers
python3 scripts/audit_renders.py            # page PNGs + text-outside-page audit
```

If those three pass, the repository is intact and the problem is environmental.

---

## 1. Setup

**`RuntimeError: Missing required environment variable: ...`**
`app/config.py::_env(name)` only raises when it is called with `default=None`; every current
`Settings` field has a default, so this means a modified checkout. Unset the variable you added, or
give it a default.

**The app ignores my `.env` file.**
By design — there is no dotenv loader. Export the variables (`export JWT_ACCESS_SECRET=...`) or use
a systemd `EnvironmentFile=`. See `docs/DEPLOYMENT.md` §3 for the authoritative key list.

**`make models` refuses the download.**
It verifies SHA-256 against the pinned digests (`u2netp.onnx` `309c8469…f4ddd8`,
`depth_anything_v2_small_quantized.onnx` `fcf51f1b…547f20d`). A mismatch means a partial download or
a different upstream revision. Delete `engine/models/*.onnx` and retry; do not "fix" the hash.

**`ModuleNotFoundError: No module named 'onnxruntime'` (or `cv2`, `trimesh`).**
`pip install -r requirements.txt`. The engine is imported lazily, so the API can boot without them —
reconstructions will fail with `ENGINE_UNAVAILABLE` until they are installed.

**`libGL.so.1: cannot open shared object file`.**
Only `open3d` needs it. `open3d` is **optional** and unused by the shipped path; remove it or
`apt-get install libgl1 libgomp1`. Nothing in `results/` depends on it.

---

## 2. Reconstruction

**`ENGINE_ERROR` / job fails at the `geometry` stage.**
Read `results/logs/<entry>__<method>__r<k>.log`, then that run's `manifest.json`. The usual causes
are an image with an empty matte (uniform background, object touching the border) and an object so
thin that it vanishes below `min_thickness = 0.02`. Credits are refunded automatically; the API
returns a reference code, never a stack trace.

**`OUT_OF_MEMORY` or the process is OOM-killed around 735 MB.**
The box is the limit, not the code: the benchmark has been killed at ~735 MB in a small cgroup while
measured peaks are 926–933 MB. Stop the dev server while benchmarking, keep
`JOB_CONCURRENCY=1`, and do not run `make benchmark` and `./run.sh` at once.

**`ENGINE_UNAVAILABLE`.**
`/api/v1/health/ready` tells you which dependency is missing (models, ONNX runtime, writable storage
root). It is the probe to use, not `/health`.

**The preview looks holed / fissured but the mesh is watertight.**
Known renderer artefact, fixed in `engine/recon/render.py`: the per-pixel rasteriser is *single-sided*
by design and drops back faces, which makes a correct closed mesh look perforated. Use the
two-sided flat rasteriser (`mode="clay"`, `per_pixel=False`) for comparisons. Do **not** patch the
mesh to satisfy the renderer — that is how the first qualitative figure was wrong.

**Texture is stretched at grazing angles.**
Expected: the atlas is a planar view projection, exact on the camera-facing half and stretched on
silhouette-tangent surfaces. Documented in the paper's limitations, not a bug.

---

## 3. Benchmark

**`seconds` is NaN in a summary row.**
A cached mesh was reused without its timing. The harness reads the original wall clock back from
the progress log or `manifest.json#timings_seconds.total`; if a run directory predates that fix,
delete it and rerun that cell (or `make benchmark --force`). Never let a NaN reach a table —
`num()` in the generators prints `not evaluated` instead of a number, which the verifier flags.

**`max deviation 0.36 s` from `verify_submission.py`.**
`results/raw/runs.csv` disagrees with `results/raw/runs.jsonl`. `runs.csv` is derived; regenerate it
from `runs.jsonl` rather than editing either by hand.

**A re-run is much faster than the first one.**
It is reusing meshes and their original timings (that is the resumability contract). Use `--force`
when you want fresh measurements, and say so if you quote the numbers.

**Thickness sweep values do not match the shipped preset.**
Sweep run directories zero-pad floats (`ts0.70`). Format the value the same way or the cache misses
and you silently get a fresh run.

**A baseline looks artificially bad (holes, missing back faces).**
Check whether it was rendered per-pixel. The per-pixel path is single-sided; benchmark comparisons
must use the two-sided flat path, and all three methods must share the same renderer.

**`operands could not be broadcast together` from `silhouette_iou`.**
The masks have different sizes. Resize both to a common size (the harness renders at 120 px against
a 180 px reference and resizes before comparing).

---

## 4. Documents

**`File ended while scanning use of \@xdblarg`.**
Unbalanced braces inside an f-string that emits `\caption{...}`. Grep the generator around the line
number in the error, count braces in the emitted string, then `python3 scripts/generate_paper.py`
and rebuild.

**Overfull `\hbox` in the paper.**
The fix is content, not `\sloppy`: split the equation, narrow a table to `p{17mm}ccc`, shorten the
column header, wrap the verbatim block in `\footnotesize`, or reflow the caption.

**`\%` printed literally, or `\` swallowed.**
Inside f-strings use `chr(92) + "%"` rather than `"\\%"`; Python collapses escapes differently
across versions and the LaTeX output is what matters. The same applies to `_vt()`,
`ssim_sentence()` and `model_size()` in `scripts/generate_paper.py`.

**pandoc PDF fails on `docs/report.md`.**
No xelatex on this machine — use `--pdf-engine=pdflatex`, and keep `report.md` ASCII. Any Unicode
character (κ, ±, →) breaks the build; the generator writes ASCII on purpose.

**Paper/deck/report numbers disagree with a CSV.**
`python3 scripts/verify_submission.py | grep cross-check`. The generators read
`results/summary/*.csv`; if a number is stale, one of the generators was not re-run after the last
benchmark. Re-run `make figures paper slides report`, in that order.

**A slide overflows the 16:9 frame.**
`scripts/generate_slides.py` rescales the picture on slide 11 when `pic.top + pic.height` exceeds
7.2 in. Text overflow is caught by `scripts/audit_renders.py --deck` (python-pptx shape geometry;
LibreOffice is not installed here).

---

## 5. Web UI

**Blank page, console shows a failed `/static/vendor/three.module.js`.**
The studio must be served by the FastAPI process at the origin root; the import map uses absolute
`/static/...` paths. `./run.sh` does this. Opening `web/app/index.html` from the filesystem will not
work.

**Benchmark page shows no numbers.**
It reads `GET /api/v1/results/summary`, which serves `results/summary/*.csv`. If `results/` is
missing, seed or run the benchmark first.

**Everything is 401 after a restart.**
Expected: rotating `JWT_*` invalidates sessions. Log in again.

**`RATE_LIMITED` during manual testing.**
`RATE_LIMIT_MAX` (default 120/min) and `LOGIN_MAX_ATTEMPTS` (8). Raise them in the environment
rather than disabling the limiter.

---

## 6. API codes you may see, and what they mean

| Code | Meaning / action |
|---|---|
| `UNAUTHENTICATED`, `TOKEN_EXPIRED`, `NO_REFRESH_TOKEN` | log in again; rotate secrets deliberately |
| `INVALID_CREDENTIALS`, `ACCOUNT_LOCKED`, `ACCOUNT_DISABLED` | 8 failed logins locks the account |
| `CSRF_FAILED`, `SESSION_REVOKED` | stale tab or rotated session |
| `RATE_LIMITED` | back off; raise the limit only in dev |
| `FILE_TOO_LARGE`, `INVALID_IMAGE`, `INVALID_FORMAT` | `MAX_UPLOAD_MB=20`; PNG/JPEG/WEBP only |
| `INSUFFICIENT_CREDITS` | ledger is append-only; check `GET /api/v1/billing/ledger` |
| `NOT_RETRYABLE`, `NOT_CANCELLABLE` | state machine: only `FAILED`/`CANCELLED` retry, `QUEUED`/`PROCESSING` cancel |
| `PROVIDER_MISCONFIGURED`, `PROVIDER_UNAVAILABLE`, `PROVIDER_NOT_FOUND` | `GENERATION_PROVIDER` / `GENERATION_API_URL` |
| `ENGINE_ERROR`, `ENGINE_UNAVAILABLE`, `OUT_OF_MEMORY`, `TIMEOUT` | see §2; credits refunded |
| `UNKNOWN_JOB` | the row was deleted; the SSE stream cannot resume |
| `INTERNAL` / `INTERNAL_ERROR` | a reference code is returned — grep it in the server log |
