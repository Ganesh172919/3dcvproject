# Build and verify the course submission

Run these commands from the project source directory. In an extracted course archive this is the `code/` directory. Use Python 3.12 and install the development requirements.

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check app engine benchmarks scripts submission tests tools training
python -m mkdocs build --strict
```

## Build the report and demo

Compile `submission/report.tex` with a local Tectonic installation. The figures already exist in `results/course17/`.

```sh
python -c "from pathlib import Path; Path('out/course17').mkdir(parents=True, exist_ok=True)"
cd submission
tectonic -o ../out/course17 report.tex
cd ..
python -m submission.demo
```

The result is a four-page report and an 84-second captioned MP4. To rerender the depth figure from a new CPU experiment, run the `training/README.md` with `--run --ablation`, then use `python -m submission.figures --experiment PATH`. Review newly generated figures and measurements together before replacing the saved evidence.

## Build and verify the ZIP

```sh
python -m submission.package
python -m submission.verify out/course17/Group17_3dcv_project.zip
```

The archive includes the report, demo, README, requirements, source code, recorded measurements, and two sample photos or renders. The packager builds the analytic box and its reference mesh. It excludes model weights, checkpoints, virtual environments, databases, uploads, and local environment files.

Extract a copy to a fresh directory and follow the archive README. Run the sneaker CLI demo and compare its manifest and GLB with the Studio workflow. Check the course roster with the instructor before uploading.
