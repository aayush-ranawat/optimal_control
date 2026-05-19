from __future__ import annotations

import numpy as np
from scipy import integrate

from .models import AnalyticalSolution, NumericalSolution, ProblemSpec


def solve_numerical(spec: ProblemSpec, analytical: AnalyticalSolution) -> NumericalSolution:
    mu = spec.mu
    target_x = np.cos(spec.alpha)
    target_y = np.sin(spec.alpha)

    tau = analytical.tau.copy()
    regression = np.column_stack([np.cos(analytical.theta), np.sin(analytical.theta)])
    lam12, *_ = np.linalg.lstsq(regression, -mu * analytical.v, rcond=None)
    y_guess = np.vstack(
        [
            analytical.x,
            analytical.y,
            analytical.theta,
            lam12[0] * np.ones_like(tau),
            lam12[1] * np.ones_like(tau),
            -mu * analytical.omega,
        ]
    )

    def dynamics(tau_grid: np.ndarray, state: np.ndarray, p: np.ndarray) -> np.ndarray:
        tf = p[0]
        x, y, theta, lam1, lam2, lam3 = state
        v = -(lam1 * np.cos(theta) + lam2 * np.sin(theta)) / mu
        omega = -lam3 / mu
        lam3_dot = tf * v * (lam1 * np.sin(theta) - lam2 * np.cos(theta))
        zeros = np.zeros_like(theta)
        return np.vstack(
            [
                tf * v * np.cos(theta),
                tf * v * np.sin(theta),
                tf * omega,
                zeros,
                zeros,
                lam3_dot,
            ]
        )

    def boundary_conditions(ya: np.ndarray, yb: np.ndarray, p: np.ndarray) -> np.ndarray:
        tf = p[0]
        theta_f = yb[2]
        lam1_f = yb[3]
        lam2_f = yb[4]
        lam3_f = yb[5]
        v_f = -(lam1_f * np.cos(theta_f) + lam2_f * np.sin(theta_f)) / mu
        omega_f = -lam3_f / mu
        h_final = tf * ((1.0 - mu) + 0.5 * mu * (v_f * v_f + omega_f * omega_f) + lam1_f * v_f * np.cos(theta_f) + lam2_f * v_f * np.sin(theta_f) + lam3_f * omega_f)
        return np.array(
            [
                ya[0],
                ya[1],
                ya[2],
                yb[0] - target_x,
                yb[1] - target_y,
                yb[2],
                h_final,
            ]
        )

    result = integrate.solve_bvp(
        dynamics,
        boundary_conditions,
        tau,
        y_guess,
        p=np.array([analytical.tf]),
        tol=1e-6,
        max_nodes=20000,
    )

    dense_tau = np.linspace(0.0, 1.0, spec.samples)
    dense_state = result.sol(dense_tau)
    theta = dense_state[2]
    lam1 = dense_state[3]
    lam2 = dense_state[4]
    lam3 = dense_state[5]
    v = -(lam1 * np.cos(theta) + lam2 * np.sin(theta)) / mu
    omega = -lam3 / mu

    boundary_error = np.array(
        [dense_state[0, -1] - target_x, dense_state[1, -1] - target_y, theta[-1]]
    )
    state_gap = float(
        max(
            np.max(np.abs(dense_state[0] - analytical.x)),
            np.max(np.abs(dense_state[1] - analytical.y)),
            np.max(np.abs(theta - analytical.theta)),
        )
    )

    theta_f = theta[-1]
    hamiltonian_final = float(
        result.p[0]
        * (
            (1.0 - mu)
            + 0.5 * mu * (v[-1] * v[-1] + omega[-1] * omega[-1])
            + lam1[-1] * v[-1] * np.cos(theta_f)
            + lam2[-1] * v[-1] * np.sin(theta_f)
            + lam3[-1] * omega[-1]
        )
    )

    return NumericalSolution(
        tau=dense_tau,
        x=dense_state[0],
        y=dense_state[1],
        theta=theta,
        v=v,
        omega=omega,
        lambda1=lam1,
        lambda2=lam2,
        lambda3=lam3,
        tf=float(result.p[0]),
        success=bool(result.success),
        max_state_gap=state_gap,
        boundary_error=boundary_error,
        hamiltonian_final=hamiltonian_final,
    )
