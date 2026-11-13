# Acceptance checklist

This checks which files are present. See results/verification.json for execution results.

- [x] Team and guide named in README, AUTHORS.md, paper and deck
- [x] Engine source and export pipeline included
- [x] Web app, API and demo account seed command
- [x] Test suite and submission verifier included
- [x] Evaluation set manifest and image attribution
- [x] Two baseline implementations and per-run evidence
- [x] Raw results, logs and environment record in results/
- [x] Summary tables, ablation and thickness sweep
- [x] Figures generated from results/
- [x] Paper PDF, LaTeX source and bibliography
- [x] Presentation deck
- [x] Project report in Markdown and PDF
- [x] Reproduce with one command per artefact (`make benchmark|figures|paper|slides|report`)
- [x] Large ONNX weights git-ignored and fetched by `make models`
- [x] LICENSE, .gitignore, sample input, demo commands

## Reproduce

```bash
make install && make models && make seed
make benchmark && make figures && make paper && make slides && make report
make verify
```
