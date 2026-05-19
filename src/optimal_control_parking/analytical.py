from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate, optimize, special

from .models import AnalyticalSolution, ProblemSpec


@dataclass(frozen=True)
class _ClosedFormParameters:
    z: float
    tf: float
    m: float
    phi: float
    eta: float


def _elliptic_bundle(u: np.ndarray, m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sn, cn, dn, am = special.ellipj(u, m)
    e_inc = special.ellipeinc(am, m)
    return sn, cn, dn, am, e_inc


def _parameterize_candidates(spec: ProblemSpec, z: float, tf: float) -> list[_ClosedFormParameters]:
    mu = spec.mu
    if z <= 0.0 or tf <= 0.0:
        raise ValueError("z and tf must be positive.")

    m = 2.0 * mu * (1.0 - mu) / (z * z)
    if not (0.0 < m < 1.0):
        raise ValueError("Derived elliptic modulus m must lie in (0, 1).")

    k_m = special.ellipk(m)
    eta = k_m - (tf * z) / (2.0 * mu)
    sn_eta, cn_eta, _, _, e_eta = _elliptic_bundle(np.asarray([eta]), m)
    sin_phi = float(np.sqrt(m) * sn_eta[0])
    sin_phi = float(np.clip(sin_phi, -1.0, 1.0))
    base_phi = float(np.arcsin(sin_phi))

    candidates = []
    for phi in (base_phi, np.pi - base_phi, -np.pi - base_phi):
        candidates.append(_ClosedFormParameters(z=z, tf=tf, m=m, phi=float(phi), eta=eta))
    return candidates


def _parameterize(spec: ProblemSpec, z: float, tf: float) -> _ClosedFormParameters:
    candidates = _parameterize_candidates(spec, z, tf)
    trajectories = [_trajectory_from_parameters(spec, params) for params in candidates]
    best_index = int(np.argmin([np.linalg.norm(sol.boundary_error[:2]) for sol in trajectories]))
    return candidates[best_index]


def _trajectory_from_parameters(spec: ProblemSpec, params: _ClosedFormParameters) -> AnalyticalSolution:
    tau = np.linspace(0.0, 1.0, spec.samples)
    u = (params.tf * params.z / spec.mu) * tau + params.eta

    sn, cn, _, _, _ = _elliptic_bundle(u, params.m)

    gain = np.sqrt(2.0 * spec.mu * (1.0 - spec.mu)) / spec.mu
    v = gain * sn
    omega = gain * cn
    theta = integrate.cumulative_trapezoid(params.tf * omega, tau, initial=0.0)
    x = integrate.cumulative_trapezoid(params.tf * v * np.cos(theta), tau, initial=0.0)
    y = integrate.cumulative_trapezoid(params.tf * v * np.sin(theta), tau, initial=0.0)

    integrand = (1.0 - spec.mu) + 0.5 * spec.mu * (v * v + omega * omega)
    cost = float(params.tf * np.trapezoid(integrand, tau))

    target = np.array([np.cos(spec.alpha), np.sin(spec.alpha), 0.0])
    boundary_error = np.array([x[-1], y[-1], theta[-1]]) - target
    control_radius = v * v + omega * omega
    control_radius_target = 2.0 * (1.0 - spec.mu) / spec.mu
    control_radius_error = float(np.max(np.abs(control_radius - control_radius_target)))

    return AnalyticalSolution(
        tau=tau,
        x=x,
        y=y,
        theta=theta,
        v=v,
        omega=omega,
        z=params.z,
        tf=params.tf,
        m=params.m,
        phi=params.phi,
        eta=params.eta,
        cost=cost,
        control_radius_error=control_radius_error,
        boundary_error=boundary_error,
    )


def _residuals(q: np.ndarray, spec: ProblemSpec) -> np.ndarray:
    z, tf = q
    if z <= 0.0 or tf <= 0.0:
        return np.array([1e3 + abs(z), 1e3 + abs(tf)])

    try:
        params = _parameterize(spec, z, tf)
        sol = _trajectory_from_parameters(spec, params)
    except ValueError:
        return np.array([1e3, 1e3])

    return sol.boundary_error[:2]


def _initial_guess(spec: ProblemSpec) -> np.ndarray:
    # The controls lie on V^2 + omega^2 = 2(1-mu)/mu, so z must exceed sqrt(2mu(1-mu)).
    z_min = np.sqrt(2.0 * spec.mu * (1.0 - spec.mu)) * 1.02
    tf_seed = max(2.0, np.pi * spec.mu / max(z_min, 1e-6))
    seeds = []
    for z_scale in (1.05, 1.25, 1.6, 2.0, 2.8):
        for tf_scale in (0.75, 1.0, 1.5, 2.0, 3.0):
            seeds.append(np.array([z_min * z_scale, tf_seed * tf_scale]))
    best = seeds[0]
    best_norm = float("inf")
    for guess in seeds:
        norm = float(np.linalg.norm(_residuals(guess, spec)))
        if np.isfinite(norm) and norm < best_norm:
            best = guess
            best_norm = norm
    return best


def _solve_local(spec: ProblemSpec, guess: np.ndarray) -> AnalyticalSolution:
    lower_bounds = [np.sqrt(2.0 * spec.mu * (1.0 - spec.mu)) * 1.001, 1e-6]
    upper_bounds = [np.inf, np.inf]
    result = optimize.least_squares(
        lambda q: _residuals(q, spec),
        guess,
        bounds=(lower_bounds, upper_bounds),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=4000,
    )
    params = _parameterize(spec, float(result.x[0]), float(result.x[1]))
    return _trajectory_from_parameters(spec, params)


def _search_global(spec: ProblemSpec) -> AnalyticalSolution:
    z_floor = np.sqrt(2.0 * spec.mu * (1.0 - spec.mu)) * 1.001
    # Small mu can require a different basin than moderate/high mu.
    # Try several search boxes and refine each candidate locally.
    search_boxes = [
        ((z_floor, 8.0), (0.05, 20.0)),
        ((z_floor, 20.0), (0.05, 50.0)),
        ((z_floor, 40.0), (0.05, 80.0)),
    ]

    best_sol: AnalyticalSolution | None = None
    best_norm = float("inf")
    objective = lambda q: float(np.dot(_residuals(q, spec), _residuals(q, spec)))

    for z_bounds, tf_bounds in search_boxes:
        result = optimize.differential_evolution(
            objective,
            bounds=[z_bounds, tf_bounds],
            seed=0,
            polish=True,
            tol=1e-10,
            maxiter=150,
        )
        sol = _solve_local(spec, result.x)
        norm = float(np.linalg.norm(sol.boundary_error[:2], ord=np.inf))
        if norm < best_norm:
            best_sol = sol
            best_norm = norm
        if norm <= 1e-6:
            return sol

    if best_sol is None:
        raise RuntimeError("Global analytical search failed to produce a candidate solution.")
    return best_sol


def solve_analytical(spec: ProblemSpec) -> AnalyticalSolution:
    guesses = [_initial_guess(spec)]
    base = guesses[0]
    guesses.extend(
        [
            np.array([base[0] * 0.8, base[1] * 0.8]),
            np.array([base[0] * 0.8, base[1] * 1.5]),
            np.array([base[0] * 1.5, base[1] * 0.8]),
            np.array([base[0] * 1.5, base[1] * 1.5]),
            np.array([base[0] * 2.0, base[1] * 2.0]),
        ]
    )
    best_sol: AnalyticalSolution | None = None
    best_norm = float("inf")
    for guess in guesses:
        sol = _solve_local(spec, guess)
        norm = float(np.linalg.norm(sol.boundary_error[:2], ord=np.inf))
        if norm < best_norm:
            best_sol = sol
            best_norm = norm
        if norm <= 1e-6:
            return sol

    sol = _search_global(spec)
    if np.linalg.norm(sol.boundary_error[:2], ord=np.inf) <= 1e-6:
        return sol

    if np.linalg.norm(sol.boundary_error[:2], ord=np.inf) > 1e-6:
        raise RuntimeError(
            "Failed to satisfy the analytical terminal-position equations. "
            f"Residual={sol.boundary_error[:2]}"
        )
    return sol
