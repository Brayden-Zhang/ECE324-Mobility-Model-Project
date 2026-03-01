.RECIPEPREFIX := >

PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip

.PHONY: create_environment install requirements test

create_environment:
>$(PYTHON) -m venv $(VENV)

install: create_environment
>$(VENV_PIP) install --upgrade pip
>$(VENV_PIP) install -e .

requirements: install
>$(VENV_PIP) install -r requirements.txt

test:
>PYTHONPATH=src $(PYTHON) -m unittest discover -s tests
