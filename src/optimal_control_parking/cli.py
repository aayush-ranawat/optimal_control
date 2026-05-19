from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .analytical import solve_analytical
from .models import ProblemSpec
from .numerical import solve_numerical


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the minimum energy-time optimal parking controller from Kim and Singh (ACC 2022)."
    )
    parser.add_argument("--alpha-deg", type=float, default=45.0, help="Terminal polar angle in degrees on the unit circle.")
    parser.add_argument("--mu", type=float, default=0.5, help="Energy-time weighting factor, strictly between 0 and 1.")
    parser.add_argument("--samples", type=int, default=500, help="Number of trajectory samples.")
    parser.add_argument("--no-bvp", action="store_true", help="Skip numerical TPBVP validation.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for plots and summary JSON.")
    return parser


def _plot_results(output_dir: Path, analytical, numerical) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(analytical.x, analytical.y, label="Analytical", linewidth=2.0)
    if numerical is not None:
        axes[0].plot(numerical.x, numerical.y, "--", label="Numerical BVP", linewidth=1.5)
    axes[0].scatter([0.0, analytical.x[-1]], [0.0, analytical.y[-1]], c=["black", "tab:red"], s=35)
    axes[0].set_title("Trajectory")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(analytical.tau, analytical.v, label="V", linewidth=2.0)
    axes[1].plot(analytical.tau, analytical.omega, label="omega", linewidth=2.0)
    axes[1].set_title("Optimal Controls")
    axes[1].set_xlabel("tau")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(analytical.tau, analytical.theta, label="theta", linewidth=2.0)
    if numerical is not None:
        axes[2].plot(numerical.tau, numerical.theta, "--", label="theta (BVP)", linewidth=1.5)
    axes[2].set_title("Heading")
    axes[2].set_xlabel("tau")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_controls.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    control_radius = analytical.v * analytical.v + analytical.omega * analytical.omega
    ax.plot(analytical.v, analytical.omega, linewidth=2.0)
    ax.set_title("Control Locus")
    ax.set_xlabel("V")
    ax.set_ylabel("omega")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_dir / "control_locus.png", dpi=180)
    plt.close(fig)

    np.savetxt(
        output_dir / "trajectory.csv",
        np.column_stack([analytical.tau, analytical.x, analytical.y, analytical.theta, analytical.v, analytical.omega, control_radius]),
        delimiter=",",
        header="tau,x,y,theta,v,omega,v2_plus_omega2",
        comments="",
    )


def main() -> None:
    args = _build_parser().parse_args()
    spec = ProblemSpec(alpha=np.deg2rad(args.alpha_deg), mu=args.mu, samples=args.samples)

    analytical = solve_analytical(spec)
    numerical = None if args.no_bvp else solve_numerical(spec, analytical)

    output_dir = Path(args.output_dir)
    _plot_results(output_dir, analytical, numerical)

    summary = {
        "paper": "Kim and Singh, Minimum Energy-Time Optimal Control of Wheeled Mobile Robots: Application to Parallel Parking, ACC 2022",
        "alpha_deg": args.alpha_deg,
        "mu": args.mu,
        "z": analytical.z,
        "tf": analytical.tf,
        "m": analytical.m,
        "phi_deg": float(np.rad2deg(analytical.phi)),
        "eta": analytical.eta,
        "cost": analytical.cost,
        "analytical_boundary_error": analytical.boundary_error.tolist(),
        "control_circle_max_error": analytical.control_radius_error,
        "numerical": None
        if numerical is None
        else {
            "success": numerical.success,
            "tf": numerical.tf,
            "boundary_error": numerical.boundary_error.tolist(),
            "max_state_gap_vs_analytical": numerical.max_state_gap,
            "hamiltonian_final": numerical.hamiltonian_final,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
