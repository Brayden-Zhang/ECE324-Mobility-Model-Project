"""Length-sensitivity sweep for trajectory foundation models.

``LengthSensitivityExperiment`` systematically evaluates how a
:class:`~route_rangers.models.worldtraj.model.TrajectoryModel` performs as the
**observation horizon** (``t_obs``) and **prediction horizon** (``t_pred``) are
varied over configurable grids.

Typical usage
-------------
::

    import numpy as np
    from route_rangers.models.worldtraj import (
        LinearExtrapolationModel,
        LengthSensitivityExperiment,
    )

    # Synthetic ground-truth trajectories: 100 agents, 60 steps, (x, y)
    rng = np.random.default_rng(0)
    full_trajectories = rng.standard_normal((100, 60, 2)).cumsum(axis=1)

    experiment = LengthSensitivityExperiment(
        model_cls=LinearExtrapolationModel,
        t_obs_values=[10, 20, 30],
        t_pred_values=[10, 20, 30],
    )
    results = experiment.run(full_trajectories)
    # results is a list[dict] — one entry per (t_obs, t_pred) combination.

Output schema
-------------
Each element of the returned list is a ``dict`` with keys:

``t_obs``
    The observation horizon used in this trial.
``t_pred``
    The prediction horizon used in this trial.
``ade``
    Average Displacement Error (metres).
``fde``
    Final Displacement Error (metres).
``miss_rate``
    Miss Rate at the configured ``miss_threshold`` (fraction in [0, 1]).
``n_agents``
    Number of agents included in this trial (trajectories too short to
    provide at least ``t_obs + t_pred`` steps are silently skipped).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence, Type

import numpy as np
from numpy.typing import NDArray

from route_rangers.models.worldtraj.metrics import (
    compute_ade,
    compute_fde,
    compute_miss_rate,
)
from route_rangers.models.worldtraj.model import TrajectoryModel

if TYPE_CHECKING:
    pass


@dataclass
class LengthSensitivityResult:
    """Result for a single (t_obs, t_pred) trial."""

    t_obs: int
    t_pred: int
    ade: float
    fde: float
    miss_rate: float
    n_agents: int


@dataclass
class LengthSensitivityExperiment:
    """Sweep ``t_obs`` × ``t_pred`` and record ADE / FDE / Miss-Rate.

    Parameters
    ----------
    model_cls:
        A class (not an instance) that is a sub-class of
        :class:`~route_rangers.models.worldtraj.model.TrajectoryModel`.
        For every ``(t_obs, t_pred)`` combination a fresh instance is
        constructed via ``model_cls(t_obs=..., t_pred=..., **model_kwargs)``.
    t_obs_values:
        Sequence of observation-horizon lengths to sweep.
    t_pred_values:
        Sequence of prediction-horizon lengths to sweep.
    model_kwargs:
        Additional keyword arguments forwarded to the model constructor
        (e.g. ``dt=0.1`` for :class:`LinearExtrapolationModel`).
    miss_threshold:
        Distance threshold in metres used for the Miss-Rate metric
        (default 2.0 m).
    """

    model_cls: Type[TrajectoryModel]
    t_obs_values: Sequence[int]
    t_pred_values: Sequence[int]
    model_kwargs: dict = field(default_factory=dict)
    miss_threshold: float = 2.0

    def run(
        self,
        full_trajectories: NDArray[np.float64],
    ) -> list[LengthSensitivityResult]:
        """Execute the sweep over all (t_obs, t_pred) combinations.

        Parameters
        ----------
        full_trajectories:
            Array of shape ``(N, T_total, 2)`` containing complete agent
            trajectories.  Each trial slices the first ``t_obs`` steps as
            observed input and the next ``t_pred`` steps as ground truth.
            Agents whose total length is less than ``t_obs + t_pred`` are
            **skipped** for that trial; if *all* agents are skipped the
            trial is still recorded but with ``n_agents=0`` and NaN metrics.

        Returns
        -------
        list[LengthSensitivityResult]
            One result per ``(t_obs, t_pred)`` pair, in the order defined
            by iterating ``t_obs_values`` (outer) × ``t_pred_values`` (inner).
        """
        full_trajectories = np.asarray(full_trajectories, dtype=np.float64)
        if full_trajectories.ndim != 3 or full_trajectories.shape[2] != 2:
            raise ValueError(
                f"full_trajectories must have shape (N, T, 2), "
                f"got {full_trajectories.shape}"
            )

        results: list[LengthSensitivityResult] = []

        for t_obs in self.t_obs_values:
            for t_pred in self.t_pred_values:
                result = self._run_single(full_trajectories, t_obs, t_pred)
                results.append(result)

        return results

    def _run_single(
        self,
        full_trajectories: NDArray[np.float64],
        t_obs: int,
        t_pred: int,
    ) -> LengthSensitivityResult:
        """Run a single (t_obs, t_pred) trial and return its metrics."""
        required_len = t_obs + t_pred
        # All agents in a dense array share the same time-axis length,
        # so a single scalar check is sufficient.
        if full_trajectories.shape[1] < required_len:
            return LengthSensitivityResult(
                t_obs=t_obs,
                t_pred=t_pred,
                ade=float("nan"),
                fde=float("nan"),
                miss_rate=float("nan"),
                n_agents=0,
            )

        trajectories = full_trajectories  # (N, T_total, 2)
        n_agents = trajectories.shape[0]

        observed = trajectories[:, :t_obs, :]              # (N', t_obs, 2)
        ground_truth = trajectories[:, t_obs : t_obs + t_pred, :]  # (N', t_pred, 2)

        model = self.model_cls(t_obs=t_obs, t_pred=t_pred, **self.model_kwargs)
        predicted = model.predict(observed)                # (N', t_pred, 2)

        return LengthSensitivityResult(
            t_obs=t_obs,
            t_pred=t_pred,
            ade=compute_ade(predicted, ground_truth),
            fde=compute_fde(predicted, ground_truth),
            miss_rate=compute_miss_rate(predicted, ground_truth, threshold=self.miss_threshold),
            n_agents=n_agents,
        )
