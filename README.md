# TrajectoryFM-HMT Reproducibility Snapshot

This repository is organized as a lightweight, reproducible project checkout for
the ECE324 mobility-model project. It keeps the packaged Python code under
`src/route_rangers/`, small sample datasets under `data/`, checked-in result
artifacts under `reports/`, and checkpoint download instructions under
`checkpoints/`.

## What is reproducible here

- install the package and run the unit test suite;
- regenerate the checked-in paper-style tables in `reports/paper/`;
- regenerate the figures in `docs/figures/`;
- run local evaluation commands on the bundled sample datasets;
- train the bundled UniTraj-compatible baseline through the maintained package
  CLI;
- use the preserved HMT pretraining entrypoint in `train_hmt.py`.

The repository still includes the HMT pretraining script. Running the full
training stack depends on the training-time data loaders and optional research
dependencies expected by that workflow.

## Quick start

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
python -m pip install -r requirements.txt
```

Run the repository checks:

```bash
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
python -m route_rangers.cli.generate_paper_artifacts
python -m route_rangers.visualization.plot_results
```

## Minimal reproduction workflow

1. Install dependencies with the commands above.
2. Run `python -m unittest discover -s tests`.
3. Regenerate report tables with
   `python -m route_rangers.cli.generate_paper_artifacts`.
4. Regenerate figures with `python -m route_rangers.visualization.plot_results`.
5. Inspect outputs in `reports/paper/`, `reports/foundation/`, and
   `docs/figures/`.

## Repository layout

- `src/route_rangers/`: maintained Python package.
- `tests/`: unit tests covering config and report-generation helpers.
- `data/`: sample datasets plus the data catalog.
- `reports/`: checked-in evaluation summaries and paper-ready exports.
- `checkpoints/`: release/checkpoint download notes only; large weights are not
  committed.
- `scripts/`: cluster and orchestration scripts kept for experiment traceability.
- `main.py`: compatibility wrapper for the maintained UniTraj baseline trainer.
- `train_hmt.py`: main HMT pretraining entrypoint.

## Data

Tracked lightweight assets are documented in `data/DATA_CATALOG.md`. The main
sample files used for local reproduction are:

- `data/samples/worldtrace_sample.pkl`
- `data/samples/worldtrace_sample_nyc.pkl`
- `data/samples/poi_mobility_sample.pkl`
- `data/processed/context/osm_context.json`

Large raw datasets are intentionally not versioned and should stay under
`data/raw/` if prepared locally.

## Example commands

Generate the maintained paper-style tables:

```bash
$env:PYTHONPATH = "src"
python -m route_rangers.cli.generate_paper_artifacts
```

Generate the figures:

```bash
$env:PYTHONPATH = "src"
python -m route_rangers.visualization.plot_results
```

Run the benchmark wrapper on the local sample data with a published checkpoint:

```bash
$env:PYTHONPATH = "src"
python -m route_rangers.cli.run_benchmarks `
  --checkpoint checkpoints\hmt_worldtrace_stage1_step_25000.pt `
  --local_data data\samples\worldtrace_sample.pkl `
  --split_mode both `
  --output cache\benchmark_results.json
```

Train the bundled UniTraj-compatible baseline:

```bash
$env:PYTHONPATH = "src"
python main.py `
  --data_path data\samples\worldtrace_sample.pkl `
  --output_dir checkpoints\unitraj_smoke `
  --epochs 1 `
  --batch_size 32
```

## Checkpoints

Large model checkpoints are distributed as GitHub Release assets instead of
being committed to the repository.

- Release page: `v1.0-checkpoint`
- Direct asset: `hmt_worldtrace_stage1_step_25000.pt`
