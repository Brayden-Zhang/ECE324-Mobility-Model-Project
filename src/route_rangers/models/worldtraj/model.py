"""Trajectory model interface and built-in stubs.

This module defines:

* ``TrajectoryModel`` — abstract base class that every model wrapper must
  implement.  A real UniTraj / WorldTraj integration would subclass this and
  load the pretrained checkpoint inside ``__init__``.

* ``LinearExtrapolationModel`` — a simple constant-velocity baseline that
  requires no learned weights.  It is used in unit-tests and as a sanity-check
  reference when running the length-sensitivity sweep.

Coordinate convention
---------------------
All arrays use the (x, y) Cartesian plane with units of *metres* and a
time-step of 0.1 s (10 Hz), matching the nuScenes / Argoverse 2 convention.
Adapt the ``dt`` class attribute if your dataset uses a different frequency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class TrajectoryModel(ABC):
    """Abstract interface for a trajectory-prediction model.

    Sub-classes must implement :meth:`predict`.  The constructor may load
    model weights from disk or initialise any other state.

    Parameters
    ----------
    t_obs:
        Number of historical time-steps provided as input to the model.
    t_pred:
        Number of future time-steps the model is asked to forecast.
    """

    def __init__(self, t_obs: int, t_pred: int) -> None:
        if t_obs < 1:
            raise ValueError(f"t_obs must be >= 1, got {t_obs}")
        if t_pred < 1:
            raise ValueError(f"t_pred must be >= 1, got {t_pred}")
        self.t_obs = t_obs
        self.t_pred = t_pred

    @abstractmethod
    def predict(self, observed: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict future trajectory positions.

        Parameters
        ----------
        observed:
            Array of shape ``(N, t_obs, 2)`` containing the observed (x, y)
            positions of *N* agents over the last ``t_obs`` time-steps.

        Returns
        -------
        NDArray[np.float64]
            Array of shape ``(N, t_pred, 2)`` with the predicted future
            positions for each agent.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(t_obs={self.t_obs}, t_pred={self.t_pred})"


class LinearExtrapolationModel(TrajectoryModel):
    """Constant-velocity (linear extrapolation) baseline model.

    For each agent the model estimates a velocity vector from the last two
    observed positions and extrapolates it forward for ``t_pred`` steps.
    This requires no training data or model weights and therefore serves as
    an ideal drop-in stub while the real UniTraj / WorldTraj integration is
    being developed.

    Parameters
    ----------
    t_obs:
        Number of observed time-steps (must be >= 2 for velocity estimation).
    t_pred:
        Number of future time-steps to predict.
    dt:
        Time between consecutive steps in seconds (default 0.1 s / 10 Hz).
    """

    #: Time resolution in seconds — matches nuScenes / Argoverse 2 default.
    dt: float = 0.1

    def __init__(self, t_obs: int, t_pred: int, dt: float = 0.1) -> None:
        if t_obs < 2:
            raise ValueError(
                f"LinearExtrapolationModel requires t_obs >= 2 to estimate "
                f"velocity, got t_obs={t_obs}"
            )
        super().__init__(t_obs, t_pred)
        self.dt = dt

    def predict(self, observed: NDArray[np.float64]) -> NDArray[np.float64]:
        """Extrapolate each agent's trajectory at constant velocity.

        Parameters
        ----------
        observed:
            Shape ``(N, t_obs, 2)``.

        Returns
        -------
        NDArray[np.float64]
            Shape ``(N, t_pred, 2)``.

        Raises
        ------
        ValueError
            If ``observed`` has an unexpected shape.
        """
        observed = np.asarray(observed, dtype=np.float64)
        if observed.ndim != 3 or observed.shape[1] != self.t_obs or observed.shape[2] != 2:
            raise ValueError(
                f"Expected observed shape (N, {self.t_obs}, 2), "
                f"got {observed.shape}"
            )

        last_pos = observed[:, -1, :]          # (N, 2)
        velocity = observed[:, -1, :] - observed[:, -2, :]  # (N, 2)  per step

        steps = np.arange(1, self.t_pred + 1, dtype=np.float64)  # (t_pred,)
        # (N, 1, 2) + (1, t_pred, 1) * (N, 1, 2) → broadcast → (N, t_pred, 2)
        predicted = last_pos[:, np.newaxis, :] + steps[np.newaxis, :, np.newaxis] * velocity[:, np.newaxis, :]
        return predicted
