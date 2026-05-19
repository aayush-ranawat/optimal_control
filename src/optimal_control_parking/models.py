from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProblemSpec:
    alpha: float
    mu: float
    samples: int = 400

    def __post_init__(self) -> None:
        if not (0.0 < self.mu < 1.0):
            raise ValueError("mu must lie strictly between 0 and 1.")
        if not (0.0 <= self.alpha <= np.pi / 2.0):
            raise ValueError("alpha must lie in [0, pi/2] radians.")
        if self.samples < 32:
            raise ValueError("samples must be at least 32.")


@dataclass(frozen=True)
class AnalyticalSolution:
    tau: np.ndarray
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    v: np.ndarray
    omega: np.ndarray
    z: float
    tf: float
    m: float
    phi: float
    eta: float
    cost: float
    control_radius_error: float
    boundary_error: np.ndarray


@dataclass(frozen=True)
class NumericalSolution:
    tau: np.ndarray
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    v: np.ndarray
    omega: np.ndarray
    lambda1: np.ndarray
    lambda2: np.ndarray
    lambda3: np.ndarray
    tf: float
    success: bool
    max_state_gap: float
    boundary_error: np.ndarray
    hamiltonian_final: float
