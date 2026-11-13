# Verifying the submission

The packaged artifacts include the IEEE-format paper, project report, 15-slide deck, saved evaluation data, and `results/verification.json`. The course benchmark covers the CPU environment in `results/environment.json`. The web prototype can also run other Studio engines.

## Windows

From the extracted project folder, start the app and install the models.

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Install -Seed
```

Open [the lab](http://127.0.0.1:8010/), sign in with the demo account, and try the teapot example in Draft quality. Orbit the result, download a GLB, and check the run in the Experiment notebook.

From another PowerShell window in the same project folder:

```powershell
uv pip install --python .\.venv\Scripts\python.exe -r requirements-dev.txt
.\.venv\Scripts\python.exe scripts/fetch_models.py --all --check
.\.venv\Scripts\python.exe scripts/verify_submission.py
```

Use `--fast` for artifact and evidence checks alone. The verifier writes a new JSON record on each run. The full check runs pytest.

## Rebuild the paper, report and deck

```text
python scripts/generate_paper.py --pdf
python scripts/generate_report.py
python scripts/generate_slides.py
```

The paper build uses Tectonic or pdfLaTeX and BibTeX. The report PDF uses Pandoc and XeLaTeX or Tectonic. Both builds read the saved benchmark summaries. Use the benchmark instructions in the [README](../README.md) to rerun the experiments and regenerate the figures.

After rebuilding the PowerPoint, export slide PNGs into `presentations/slides_png/` using filenames `Slide1.PNG` through `Slide15.PNG` to refresh the website's presentation preview.

## Package a fresh submission

From the source checkout with Git available, run the full verifier and then `python scripts/package_submission.py`. The script writes the ZIP and its SHA-256 checksum to `out/submission/`. The archive includes the referenced evaluation images and analytic meshes. Models are fetched by the launcher.
