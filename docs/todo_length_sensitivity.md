# TODO: Length Sensitivity Analysis — UniTraj / WorldTraj

This document tracks the plan for building and running a systematic
**length-sensitivity** study on the UniTraj WorldTraj trajectory-foundation model.

"Length sensitivity" means understanding how prediction quality (ADE / FDE / Miss-Rate)
changes as we vary:

- **Observation horizon** (`T_obs`): how many historical time-steps the model sees.
- **Prediction horizon** (`T_pred`): how many future time-steps the model must forecast.

---

## Milestones

### 1 — Framework scaffold ✅
- [x] Abstract `TrajectoryModel` base class (`src/route_rangers/models/worldtraj/model.py`)
- [x] `LinearExtrapolationModel` stub (drop-in baseline for integration testing)
- [x] ADE / FDE / Miss-Rate metrics (`src/route_rangers/models/worldtraj/metrics.py`)
- [x] `LengthSensitivityExperiment` sweep utility (`src/route_rangers/models/worldtraj/length_sensitivity.py`)
- [x] Unit tests (`tests/test_length_sensitivity.py`)

### 2 — Data preparation
- [ ] Identify the dataset to use (nuScenes / Argoverse 2 / Waymo Open Motion).
- [ ] Write a preprocessing script in `src/route_rangers/data/` that converts raw
      tracks into fixed-size NumPy arrays with shape `(N, T, 2)` (x, y coordinates).
- [ ] Store processed arrays in `data/processed/trajectories.npy` (or per-split files).
- [ ] Add a data-loading helper to `worldtraj/model.py` (or a new `dataset.py`).

### 3 — UniTraj / WorldTraj model integration
- [ ] Install / vendor UniTraj (pip-installable or git-submodule under `src/`).
- [ ] Subclass `TrajectoryModel` with a `UniTrajModel` wrapper that:
      - Loads the pretrained WorldTraj checkpoint from `models/worldtraj_checkpoint/`.
      - Accepts `observed` array of shape `(N, T_obs, 2)`.
      - Returns `predicted` array of shape `(N, T_pred, 2)`.
- [ ] Verify the wrapper passes existing unit tests.

### 4 — Length-sensitivity sweep
- [ ] Define sweep grids in a config (or directly in the experiment script):
      ```
      T_obs_values  = [10, 20, 30, 50]   # observation lengths (timesteps)
      T_pred_values = [10, 20, 30, 50]   # prediction horizons (timesteps)
      ```
- [ ] Run `LengthSensitivityExperiment` with the real `UniTrajModel`.
- [ ] Save the result matrix (CSV + pickle) to `reports/length_sensitivity_results.csv`.

### 5 — Analysis and visualisation
- [ ] Plot ADE / FDE heat-maps (`T_obs` × `T_pred`) using `plot_results.py`.
- [ ] Plot line-charts: fix `T_pred`, vary `T_obs` (and vice-versa).
- [ ] Interpret results: at what observation length does the model saturate?
      Is there a prediction horizon beyond which accuracy degrades sharply?
- [ ] Document findings in `reports/length_sensitivity_report.md`.

### 6 — (Stretch) Statistical robustness
- [ ] Run multiple random seeds and report mean ± std.
- [ ] Compare against `LinearExtrapolationModel` baseline at every grid point.
- [ ] Significance test: paired t-test / Wilcoxon across lengths.

---

## Key files

| Path | Purpose |
|------|---------|
| `src/route_rangers/models/worldtraj/model.py` | Model interface + linear stub |
| `src/route_rangers/models/worldtraj/metrics.py` | ADE / FDE / Miss-Rate |
| `src/route_rangers/models/worldtraj/length_sensitivity.py` | Sweep experiment |
| `tests/test_length_sensitivity.py` | Unit tests for the framework |
| `data/processed/` | Preprocessed trajectory arrays |
| `models/worldtraj_checkpoint/` | Downloaded model weights (gitignored) |
| `reports/length_sensitivity_results.csv` | Experiment output |

---

## References

- UniTraj paper: <https://arxiv.org/abs/2403.15098>
- WorldTraj / foundation model checkpoint: (link TBD once model access is arranged)
- nuScenes dataset: <https://www.nuscenes.org/>
- Argoverse 2 dataset: <https://www.argoverse.org/av2.html>
