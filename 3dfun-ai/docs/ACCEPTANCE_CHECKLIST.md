# Acceptance checklist

Ticked automatically from the files on disk by `make report`.

- [x] Team and guide named in README, AUTHORS.md, paper and deck
- [x] Engine runs end to end and writes GLB/OBJ/PLY + manifest
- [x] Web app, API and seeded demo data
- [x] Tests present and green
- [x] Evaluation set >= 20 images with licences
- [x] Two baselines implemented and run on identical inputs
- [x] Raw results, logs and environment record in results/
- [x] Summary tables, ablation and thickness sweep
- [x] Figures generated from results/
- [x] IEEEtran paper with BibTeX and no undefined references
- [x] 16:9 deck with speaker notes
- [x] Project report (Markdown + PDF when pandoc is available)
- [x] Reproduce with one command per artefact (`make benchmark|figures|paper|slides|report`)
- [x] No API keys or private keys in the tree
- [x] Large ONNX weights git-ignored and fetched by `make models`
- [x] LICENSE, .gitignore, sample input, demo commands
- [x] Verifier cross-checks every printed number against results/

## Reproduce

```bash
make install && make models && make seed
make benchmark && make figures && make paper && make slides && make report
make verify
```
