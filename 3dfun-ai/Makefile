# 3DFUN AI — one command per deliverable
# Every target below was executed for this submission; see PROOFS.md.

PYTHON ?= python3
QUICK  ?= 0
PORT   ?= 8000

.PHONY: help install models serve seed demo test lint benchmark figures paper slides report all verify clean

help:            ## show this list
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t— /'

install:         ## install the pinned Python dependencies
	$(PYTHON) -m pip install -r requirements.txt

models:          ## fetch the two ONNX graphs into engine/models/
	$(PYTHON) scripts/fetch_models.py

serve:           ## run the web app (http://localhost:$(PORT))
	PORT=$(PORT) ./run.sh

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
	$(PYTHON) scripts/generate_paper.py
	cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null && \
	  bibtex main >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null && \
	  pdflatex -interaction=nonstopmode main.tex | tail -3

slides:          ## build the 16:9 deck with speaker notes
	$(PYTHON) scripts/generate_slides.py

report:          ## build docs/report.md (+ PDF when pandoc is available)
	$(PYTHON) scripts/generate_report.py

all: figures paper slides report ## rebuild every artefact from results/

verify:          ## full submission check (tests, build, number cross-check)
	$(PYTHON) scripts/verify_submission.py

clean:           ## remove generated artefacts (keeps results/)
	rm -rf out .pytest_cache paper/*.aux paper/*.log paper/*.bbl paper/*.blg
