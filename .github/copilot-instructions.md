# Copilot instructions for Route Rangers

## Project context
- This repository follows a Cookiecutter Data Science (CCDS) layout.
- Reusable Python code belongs in `src/route_rangers/`.
- `notebooks/` is for exploration; move stable logic from notebooks into `src/`.

## Tech stack
- Python 3.11+
- Packaging/build config in `pyproject.toml`
- Unit tests use `unittest` and are located in `tests/`

## Development expectations
- Prefer small, focused changes that keep existing structure and naming.
- Keep data files organized by CCDS conventions (`data/raw`, `data/interim`, `data/processed`).
- Avoid introducing new dependencies unless they are necessary.

## Validation before finishing
- Run tests: `make test`
- If Ruff is available in the environment, run lint checks:
  - `python -m ruff check src tests`

## Documentation and clarity
- Update README or docs when behavior or workflow changes.
- Keep functions and modules readable, with clear names and minimal side effects.
