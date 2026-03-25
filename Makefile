.RECIPEPREFIX := >

PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
RUN_PYTHON := $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),$(PYTHON))

.PHONY: create_environment install requirements test paper-artifacts figures paper verify-paper clean eval-sequence

create_environment:
>$(PYTHON) -m venv $(VENV)

install: create_environment
>$(VENV_PIP) install --upgrade pip
>$(VENV_PIP) install -e .

requirements: install
>$(VENV_PIP) install -r requirements.txt

test:
>PYTHONPATH=src $(RUN_PYTHON) -m unittest discover -s tests

paper-artifacts:
>PYTHONPATH=src $(RUN_PYTHON) -m route_rangers.cli.generate_paper_artifacts

figures: paper-artifacts
>PYTHONPATH=src $(RUN_PYTHON) -m route_rangers.visualization.plot_results

paper: paper-artifacts figures
>cd docs && latexmk -pdf -interaction=nonstopmode paper.tex

verify-paper: test paper

clean:
>find . -type d -name __pycache__ -prune -exec rm -rf {} +
>rm -rf cache
>rm -f slurm-*.out run.log results.tsv docs/*.aux docs/*.bbl docs/*.blg docs/*.fdb_latexmk docs/*.fls docs/*.log docs/*.out docs/*.stdout

eval-sequence:
>chmod +x scripts/run_full_eval_sequence.sh
>VENV_PYTHON=$(VENV_PYTHON) scripts/run_full_eval_sequence.sh
