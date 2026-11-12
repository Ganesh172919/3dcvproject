# 3DFUN AI — verification record

This is the evidence document: what is checked, how, and the state of every check at the time of
submission. It is produced from `results/verification.json`, which
`python3 scripts/verify_submission.py` rewrites on every run — so the numbers below can be
re-confirmed rather than believed.

Team and guide: `AUTHORS.md`. Measurements: `docs/report.md`, `paper/main.pdf`. Protocol:
`docs/BENCHMARK.md`.

---

## 1. The gate

```bash
make verify          # or: python3 scripts/verify_submission.py
```

```
3DFUN AI — submission verification
============================================================
30 passed, 0 warnings, 0 failed
```

`--fast` skips the pytest suite (29 checks). Everything else is cheap; the full run takes about a
minute because it invokes pytest and greps every number in three artefacts.

## 2. The 30 checks

| # | Check | What it actually proves | Result |
|---|---|---|---|
| 1 | required files present | the deliverable tree exists: paper, deck, report, results, engine, app, docs | PASS — all present |
| 2 | no API keys or private keys in the tree | no secrets committed | PASS — clean |
| 3 | large weights are git-ignored | the 31.8 MB of ONNX is out of the repo and pinned by hash | PASS — 2 weight files, pattern present |
| 4 | `main_table.csv` exists | results landed | PASS — 3 method rows |
| 5 | both baselines present | the comparison is not single-method | PASS — `proposed, tsdf, visual_hull` |
| 6 | environment record complete | hardware, library versions, seed, dataset, commit all recorded | PASS |
| 7 | ≥ 3 repetitions per configuration | variance is measurable, not assumed | PASS — repetition counts `[3]` |
| 8 | raw per-run rows recorded | 225 rows for 25 × 3 × 3 | PASS — 225 rows |
| 9 | per-run logs written | 225 logs for 75 configurations: every run is debuggable | PASS |
| 10 | summary means match the mean of the raw runs | no hand-edited aggregate | PASS — max deviation 0.0000 s |
| 11 | figures referenced by the paper exist | no dangling figure | PASS — 11 figures, 4 referenced |
| 12 | figures are non-trivial | not blank plots | PASS |
| 13 | paper has no undefined references | LaTeX resolved | PASS |
| 14 | no undefined citations | all 24 BibTeX entries resolve | PASS |
| 15 | no overfull hboxes | the PDF is not silently damaged | PASS |
| 16 | paper length | 6–8 pages | PASS — 8 pages |
| 17 | author block: five students, in order | team order is exactly J Ganesh 23BDS024, M Jagadeshwar 23BDS033, B Varshith 23BDS011, Pranav Kashyap 23BCS098, C Sai Aravind 23BDS075 | PASS — 5/5, order True |
| 18 | guide named | Dr. Shirshendu Layek (DSAI, IIIT Dharwad) | PASS |
| 19 | deck length | 12–15 slides | PASS — 15 |
| 20 | deck is 16:9 | 13.333 × 7.500 in, ratio 1.7777 | PASS |
| 21 | all slide shapes inside the canvas | no shape off-frame | PASS |
| 22 | no text box appears clipped | estimated text height vs box height | PASS |
| 23 | speaker notes on every slide | the deck is presentable | PASS — 15/15 |
| 24 | numbers in the paper trace to `results/` | 119 numeric tokens | PASS — 119/119 |
| 25 | numbers in the report trace to `results/` | 280 numeric tokens | PASS — 280/280 |
| 26 | numbers in the deck trace to `results/` | 70 numeric tokens | PASS — 70/70 |
| 27 | report documents what was not evaluated | explicit "not evaluated" section | PASS |
| 28 | all team members and the guide appear in the report | attribution | PASS — 6/6 |
| 29 | README documents reproduce and verify commands | a reviewer can run it | PASS |
| 30 | test suite passes | 39 tests | PASS — 39 passed |

Checks 24–26 are the important ones for honesty: they parse every number out of the artefacts and
require each one to match a value derived from `results/` (library versions and pinned digests are
whitelisted). A number that cannot be traced fails the run.

## 3. The test suite

```bash
python3 -m pytest tests -q      # 39 passed
```

| File | Tests | Covers |
|---|---|---|
| `tests/test_auth.py` | 17 | Argon2id hashing and salting, JWT round-trip, TOTP window, registration + signup grant, duplicate email, weak password, login/logout/login, non-leaking bad password, 401s, session listing and revocation, role escalation, admin-only routes, CSRF not required for bearer tokens, enumeration-safe password reset, rate limiter |
| `tests/test_generation.py` | 16 | estimate = charged cost, reserve-then-spend, insufficient credits blocked, append-only ledger, full upload → generation → asset lifecycle on the `mock` provider, ordered and replayable SSE events, attempts recorded, retry rejected for a completed job *and* accepted for a failed one, cancel refunds, cross-user isolation, private asset hidden from the gallery, upload rejection (non-image, oversized, deduplicated), generation rate limit |
| `tests/test_engine_smoke.py` | 6 | the real engine on `engine/testdata/sneaker.png`: artefacts written, mesh watertight, manifest declares its limitations, exported files carry the full geometry, PLY keeps source topology, deterministic for identical input |

The slow engine test runs the actual pipeline on `engine/testdata/sneaker.png` with a reduced grid
(`voxel_res=56`, `target_faces=6000`); `tests/browser/smoke.py` is a manual UI smoke script.

## 4. Render audit

```bash
python3 scripts/audit_renders.py
```

Renders every page of `paper/main.pdf` and `docs/report.pdf` to PNG and checks for text outside the
page box and blank pages; `--deck` checks PowerPoint shape geometry (LibreOffice is not installed on
this machine, so the deck audit uses python-pptx intersection tests instead of a render). Result:
paper and report clean, 8 pages each; deck 15/15 shapes inside the canvas.

## 5. Reproducing the numbers from scratch

```bash
make models && make benchmark      # 225 cells -> results/raw, results/summary
make figures paper slides report   # 11 figures, 8-page PDF, 15 slides, 8-page report
make verify                        # this document's table, re-derived
```

Runtimes are listed in the README reproduce table. Interruption is safe (§6 of
`docs/BENCHMARK.md`); `--force` is the only way to get fresh timings.

## 6. Current status and honest caveats

* All 30 checks pass, 0 warnings, 0 failed; acceptance checklist 17/17
  (`docs/ACCEPTANCE_CHECKLIST.md`).
* Inventory: 3,871 files, 83,096 code and documentation lines (`docs/FILE_INVENTORY.md`).
* **Preset scope.** Every reported number is the `draft` preset. `standard` and `high` exist and are
  selectable in the product but were **not** benchmarked, so no number is quoted for them.
* **Baseline scope.** Only the TSDF and visual-hull baselines are evaluated. Learned single-image
  regressors, multi-view photogrammetry and diffusion priors are **not evaluated** because they
  cannot run on the 2-core, 2 GB, GPU-less evaluation machine; the report says so in place of a
  number.
* **Runtime.** `proposed` is ~19× slower than TSDF in wall clock, almost entirely the
  preview/turntable renderer (24.56 s of 27.70 s). Both numbers are reported; the comparison is not
  hidden behind one.
* **Real images have no geometric reference.** They are scored on silhouette IoU and image metrics
  only. Chamfer/F/IoU on them does not exist and is not reported.
* **Texture at grazing angles** is stretched by construction (planar view projection). Documented in
  the limitations, not measured as a separate metric.
