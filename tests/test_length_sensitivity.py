"""Unit tests for the WorldTraj length-sensitivity framework.

Covers:
* TrajectoryModel subclass contract (LinearExtrapolationModel)
* Individual metric functions (ADE, FDE, Miss-Rate)
* LengthSensitivityExperiment sweep logic

All tests use purely synthetic NumPy arrays so no external data or model
weights are required.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from route_rangers.models.worldtraj.length_sensitivity import (
    LengthSensitivityExperiment,
    LengthSensitivityResult,
)
from route_rangers.models.worldtraj.metrics import (
    compute_ade,
    compute_fde,
    compute_miss_rate,
)
from route_rangers.models.worldtraj.model import (
    LinearExtrapolationModel,
    TrajectoryModel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _straight_line_trajectories(
    n_agents: int,
    t_total: int,
    velocity: tuple[float, float] = (1.0, 0.0),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return perfectly straight-line trajectories with configurable velocity.

    Each agent starts at a random position and moves at constant ``velocity``
    every step.  Because the motion is perfectly linear, a
    :class:`LinearExtrapolationModel` should predict *exactly* zero error.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    starts = rng.uniform(-10, 10, size=(n_agents, 2))  # (N, 2)
    steps = np.arange(t_total, dtype=np.float64)       # (T,)
    vx, vy = velocity
    # (N, 1, 2) + (1, T, 1) * (1, 1, 2) → (N, T, 2)
    dx = np.stack([vx * steps, vy * steps], axis=-1)   # (T, 2)
    return starts[:, np.newaxis, :] + dx[np.newaxis, :, :]


# ---------------------------------------------------------------------------
# TrajectoryModel contract
# ---------------------------------------------------------------------------

class TestTrajectoryModelContract(unittest.TestCase):
    """Verify the abstract base-class contract."""

    def test_linear_model_is_subclass(self) -> None:
        self.assertTrue(issubclass(LinearExtrapolationModel, TrajectoryModel))

    def test_constructor_rejects_invalid_t_obs(self) -> None:
        with self.assertRaises(ValueError):
            LinearExtrapolationModel(t_obs=1, t_pred=10)  # needs >= 2

    def test_constructor_rejects_zero_t_pred(self) -> None:
        with self.assertRaises(ValueError):
            LinearExtrapolationModel(t_obs=5, t_pred=0)

    def test_repr_contains_class_name(self) -> None:
        model = LinearExtrapolationModel(t_obs=10, t_pred=20)
        self.assertIn("LinearExtrapolationModel", repr(model))
        self.assertIn("10", repr(model))
        self.assertIn("20", repr(model))


# ---------------------------------------------------------------------------
# LinearExtrapolationModel predictions
# ---------------------------------------------------------------------------

class TestLinearExtrapolationModel(unittest.TestCase):
    """Functional tests for the constant-velocity baseline model."""

    def setUp(self) -> None:
        self.t_obs = 10
        self.t_pred = 20
        self.model = LinearExtrapolationModel(t_obs=self.t_obs, t_pred=self.t_pred)

    def test_output_shape(self) -> None:
        n = 50
        observed = np.zeros((n, self.t_obs, 2))
        predicted = self.model.predict(observed)
        self.assertEqual(predicted.shape, (n, self.t_pred, 2))

    def test_zero_ade_for_straight_line_motion(self) -> None:
        """On perfectly linear trajectories the model should have zero ADE."""
        trajs = _straight_line_trajectories(n_agents=30, t_total=self.t_obs + self.t_pred)
        observed = trajs[:, : self.t_obs, :]
        ground_truth = trajs[:, self.t_obs :, :]
        predicted = self.model.predict(observed)
        ade = compute_ade(predicted, ground_truth)
        self.assertAlmostEqual(ade, 0.0, places=10)

    def test_stationary_agents_predict_stationary(self) -> None:
        """Agents that do not move should be predicted to stay put."""
        n = 5
        observed = np.ones((n, self.t_obs, 2)) * 3.0  # constant position
        predicted = self.model.predict(observed)
        np.testing.assert_allclose(predicted, 3.0)

    def test_wrong_input_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.model.predict(np.zeros((10, self.t_obs + 1, 2)))

    def test_output_dtype_is_float64(self) -> None:
        observed = np.zeros((5, self.t_obs, 2), dtype=np.float32)
        predicted = self.model.predict(observed)
        self.assertEqual(predicted.dtype, np.float64)


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

class TestMetrics(unittest.TestCase):
    """Tests for compute_ade, compute_fde, compute_miss_rate."""

    def _perfect_prediction(self) -> tuple[np.ndarray, np.ndarray]:
        gt = np.random.default_rng(0).standard_normal((20, 15, 2))
        return gt.copy(), gt

    def test_ade_zero_for_perfect_prediction(self) -> None:
        pred, gt = self._perfect_prediction()
        self.assertAlmostEqual(compute_ade(pred, gt), 0.0, places=12)

    def test_fde_zero_for_perfect_prediction(self) -> None:
        pred, gt = self._perfect_prediction()
        self.assertAlmostEqual(compute_fde(pred, gt), 0.0, places=12)

    def test_miss_rate_zero_for_perfect_prediction(self) -> None:
        pred, gt = self._perfect_prediction()
        self.assertAlmostEqual(compute_miss_rate(pred, gt), 0.0, places=12)

    def test_miss_rate_one_for_large_error(self) -> None:
        """All agents miss if prediction is 1000 m off."""
        n, t = 10, 5
        pred = np.zeros((n, t, 2))
        gt = np.full((n, t, 2), 1000.0)
        self.assertAlmostEqual(compute_miss_rate(pred, gt, threshold=2.0), 1.0)

    def test_ade_known_value(self) -> None:
        """One agent, one step, displacement of 3-4-5 triangle → ADE = 5."""
        pred = np.array([[[3.0, 4.0]]])
        gt = np.zeros((1, 1, 2))
        self.assertAlmostEqual(compute_ade(pred, gt), 5.0, places=10)

    def test_fde_equals_ade_for_single_step(self) -> None:
        rng = np.random.default_rng(7)
        pred = rng.standard_normal((10, 1, 2))
        gt = rng.standard_normal((10, 1, 2))
        self.assertAlmostEqual(compute_ade(pred, gt), compute_fde(pred, gt), places=12)

    def test_mismatched_shapes_raise(self) -> None:
        with self.assertRaises(ValueError):
            compute_ade(np.zeros((5, 10, 2)), np.zeros((5, 11, 2)))

    def test_negative_threshold_raises(self) -> None:
        pred = gt = np.zeros((5, 10, 2))
        with self.assertRaises(ValueError):
            compute_miss_rate(pred, gt, threshold=-1.0)

    def test_ade_non_negative(self) -> None:
        rng = np.random.default_rng(1)
        pred = rng.standard_normal((20, 30, 2))
        gt = rng.standard_normal((20, 30, 2))
        self.assertGreaterEqual(compute_ade(pred, gt), 0.0)

    def test_fde_uses_last_step_only(self) -> None:
        """FDE must equal the displacement at the final step, not the average."""
        n, t = 4, 5
        pred = np.zeros((n, t, 2))
        gt = np.zeros((n, t, 2))
        # Only final step has error of 3-4-5 = 5 m
        gt[:, -1, 0] = 3.0
        gt[:, -1, 1] = 4.0
        self.assertAlmostEqual(compute_fde(pred, gt), 5.0, places=10)
        # ADE is diluted across all steps; only 1 of t steps has error
        self.assertAlmostEqual(compute_ade(pred, gt), 5.0 / t, places=10)


# ---------------------------------------------------------------------------
# LengthSensitivityExperiment
# ---------------------------------------------------------------------------

class TestLengthSensitivityExperiment(unittest.TestCase):
    """Tests for the sweep experiment class."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(99)
        self.n_agents = 40
        self.t_total = 60
        self.t_obs_values = [10, 20]
        self.t_pred_values = [10, 20]

    def _straight_trajs(self) -> np.ndarray:
        return _straight_line_trajectories(
            n_agents=self.n_agents,
            t_total=self.t_total,
            rng=self.rng,
        )

    def test_result_count_matches_grid(self) -> None:
        experiment = LengthSensitivityExperiment(
            model_cls=LinearExtrapolationModel,
            t_obs_values=self.t_obs_values,
            t_pred_values=self.t_pred_values,
        )
        results = experiment.run(self._straight_trajs())
        self.assertEqual(
            len(results),
            len(self.t_obs_values) * len(self.t_pred_values),
        )

    def test_result_type(self) -> None:
        experiment = LengthSensitivityExperiment(
            model_cls=LinearExtrapolationModel,
            t_obs_values=[10],
            t_pred_values=[10],
        )
        results = experiment.run(self._straight_trajs())
        self.assertIsInstance(results[0], LengthSensitivityResult)

    def test_grid_covers_all_combinations(self) -> None:
        experiment = LengthSensitivityExperiment(
            model_cls=LinearExtrapolationModel,
            t_obs_values=self.t_obs_values,
            t_pred_values=self.t_pred_values,
        )
        results = experiment.run(self._straight_trajs())
        pairs = {(r.t_obs, r.t_pred) for r in results}
        expected = {
            (t_obs, t_pred)
            for t_obs in self.t_obs_values
            for t_pred in self.t_pred_values
        }
        self.assertEqual(pairs, expected)

    def test_zero_ade_for_linear_model_on_straight_lines(self) -> None:
        """LinearExtrapolationModel must yield near-zero ADE on straight data."""
        experiment = LengthSensitivityExperiment(
            model_cls=LinearExtrapolationModel,
            t_obs_values=self.t_obs_values,
            t_pred_values=self.t_pred_values,
        )
        results = experiment.run(self._straight_trajs())
        for r in results:
            self.assertAlmostEqual(r.ade, 0.0, places=8,
                                   msg=f"ADE non-zero at t_obs={r.t_obs}, t_pred={r.t_pred}")

    def test_n_agents_matches_input(self) -> None:
        experiment = LengthSensitivityExperiment(
            model_cls=LinearExtrapolationModel,
            t_obs_values=[10],
            t_pred_values=[10],
        )
        results = experiment.run(self._straight_trajs())
        self.assertEqual(results[0].n_agents, self.n_agents)

    def test_agents_skipped_when_trajectories_too_short(self) -> None:
        """Agents with fewer than t_obs+t_pred steps must be excluded."""
        # total length = 25, so t_obs=20 + t_pred=20 = 40 requires skipping
        short_trajs = _straight_line_trajectories(
            n_agents=10,
            t_total=25,
            rng=self.rng,
        )
        experiment = LengthSensitivityExperiment(
            model_cls=LinearExtrapolationModel,
            t_obs_values=[20],
            t_pred_values=[20],
        )
        results = experiment.run(short_trajs)
        self.assertEqual(results[0].n_agents, 0)
        self.assertTrue(math.isnan(results[0].ade))

    def test_invalid_input_shape_raises(self) -> None:
        experiment = LengthSensitivityExperiment(
            model_cls=LinearExtrapolationModel,
            t_obs_values=[10],
            t_pred_values=[10],
        )
        with self.assertRaises(ValueError):
            experiment.run(np.zeros((10, 20)))  # missing last dim

    def test_longer_pred_horizon_increases_ade_for_noisy_model(self) -> None:
        """For a naïve model on curved paths longer horizons should yield higher ADE."""
        rng = np.random.default_rng(5)
        # Curved (random walk) trajectories — linear model accumulates error.
        noise = rng.standard_normal((self.n_agents, self.t_total, 2)) * 0.5
        curved_trajs = noise.cumsum(axis=1)

        experiment_short = LengthSensitivityExperiment(
            model_cls=LinearExtrapolationModel,
            t_obs_values=[10],
            t_pred_values=[5],
        )
        experiment_long = LengthSensitivityExperiment(
            model_cls=LinearExtrapolationModel,
            t_obs_values=[10],
            t_pred_values=[20],
        )
        r_short = experiment_short.run(curved_trajs)[0]
        r_long = experiment_long.run(curved_trajs)[0]
        self.assertLess(r_short.ade, r_long.ade,
                        "Expected ADE to increase with longer prediction horizon")


if __name__ == "__main__":
    unittest.main()
