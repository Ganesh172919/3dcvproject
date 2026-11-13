# 3DFUN AI — one command per deliverable

PYTHON ?= python3
QUICK  ?= 0
PORT   ?= 8000

.PHONY: help install install-dev models serve seed demo test lint benchmark figures paper slides report all verify clean

help:            ## show this list
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t— /'

install:         ## install the pinned Python dependencies
	$(PYTHON) -m pip install -r requirements.txt

install-dev:     ## install the test and verification dependencies
	$(PYTHON) -m pip install -r requirements-dev.txt

models:          ## fetch and verify every Studio model
	$(PYTHON) scripts/fetch_models.py --all

serve:           ## run the web app (http://localhost:$(PORT))
	PORT=$(PORT) bash run.sh

seed:            ## seed development data, including one real engine run
	$(PYTHON) -m app.seed --with-sample

demo:            ## reconstruct the bundled sample image end to end
	$(PYTHON) engine/run_recon.py engine/testdata/sneaker.png --quality standard --out out/demo

test:            ## run the test suite
	$(PYTHON) -m pytest -q

lint:            ## static checks
	$(PYTHON) -m ruff check app engine benchmarks tools tests || true

benchmark:       ## full benchmark: 25 images x 3 methods x 3 repetitions
	@if [ "$(QUICK)" = "1" ]; then \
	  $(PYTHON) benchmarks/run_benchmarks.py --quick --force; \
	else \
	  $(PYTHON) benchmarks/run_benchmarks.py --force; \
	fi

figures:         ## build every chart and qualitative grid from results/
	$(PYTHON) scripts/make_figures.py

paper:           ## compile the IEEEtran paper (numbers read from results/)
	$(PYTHON) scripts/generate_paper.py --pdf

slides:          ## build the 16:9 deck with speaker notes
	$(PYTHON) scripts/generate_slides.py

report:          ## build docs/report.md (+ PDF with Pandoc and XeLaTeX or Tectonic)
	$(PYTHON) scripts/generate_report.py

all: figures paper slides report ## rebuild every artefact from results/

verify:          ## verify the saved submission artifacts and run the test suite
	$(PYTHON) scripts/verify_submission.py

clean:           ## remove generated artefacts (keeps results/)
	rm -rf out .pytest_cache paper/*.aux paper/*.log paper/*.bbl paper/*.blg
