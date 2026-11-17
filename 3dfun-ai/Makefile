# 3DFUN AI — one command per deliverable

PYTHON ?= python3
QUICK  ?= 0
PORT   ?= 8010
TECTONIC ?= tectonic

.PHONY: help install install-dev models serve seed demo test lint benchmark figures report package verify clean

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
	$(PYTHON) -m ruff check app engine benchmarks scripts submission tests tools training

benchmark:       ## full benchmark: 25 images x 3 methods x 3 repetitions
	@if [ "$(QUICK)" = "1" ]; then \
	  $(PYTHON) benchmarks/run_benchmarks.py --quick --force; \
	else \
	  $(PYTHON) benchmarks/run_benchmarks.py --force; \
	fi

figures:         ## build legacy benchmark plots from locally recorded results
	$(PYTHON) scripts/main.py

report:          ## rebuild the IEEE report and browser preview
	mkdir -p out/course17
	cd submission && $(TECTONIC) -o ../out/course17 report.tex
	$(PYTHON) -m submission.report_preview

package: report  ## rebuild the demo and assemble the course ZIP
	$(PYTHON) -m submission.demo
	$(PYTHON) -m submission.package

verify: test lint ## check the course ZIP, report and video
	$(PYTHON) -m submission.verify out/course17/Group17_3dcv_project.zip

clean:           ## remove generated artefacts (keeps results/)
	rm -rf out .pytest_cache paper/*.aux paper/*.log paper/*.bbl paper/*.blg
