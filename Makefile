.RECIPEPREFIX := >

PYTHON ?= python
VENV ?= .venv
ifeq ($(OS),Windows_NT)
VENV_PYTHON := $(VENV)/Scripts/python.exe
VENV_PIP := $(VENV)/Scripts/pip.exe
else
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
endif
RUN_PYTHON := $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),$(PYTHON))

.PHONY: create_environment install requirements lint test paper-artifacts figures paper verify-paper clean eval-sequence

create_environment:
>$(PYTHON) -m venv $(VENV)

install: create_environment
>$(VENV_PIP) install --upgrade pip
>$(VENV_PIP) install -e .

requirements: install
>$(VENV_PIP) install -r requirements.txt

lint:
>PYTHONPATH=src $(RUN_PYTHON) -m ruff check src tests

test:
>PYTHONPATH=src $(RUN_PYTHON) -m unittest discover -s tests

paper-artifacts:
>PYTHONPATH=src $(RUN_PYTHON) -m route_rangers.cli.generate_paper_artifacts

figures: paper-artifacts
>PYTHONPATH=src $(RUN_PYTHON) -m route_rangers.visualization.plot_results

paper: paper-artifacts figures
>$(RUN_PYTHON) -c "print('No LaTeX paper source is tracked in this snapshot. Generated tables are in reports/paper and figures are in docs/figures.')"

verify-paper: test paper-artifacts figures

clean:
>$(RUN_PYTHON) -c "import pathlib, shutil; root = pathlib.Path('.'); [shutil.rmtree(path, ignore_errors=True) for path in root.rglob('__pycache__')]; shutil.rmtree('cache', ignore_errors=True); [path.unlink() for pattern in ('slurm-*.out', 'run.log', 'results.tsv') for path in root.glob(pattern)]"

eval-sequence:
>chmod +x scripts/run_full_eval_sequence.sh
>VENV_PYTHON=$(VENV_PYTHON) scripts/run_full_eval_sequence.sh
