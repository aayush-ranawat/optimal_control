# Optimal Control Parking

This project implements the minimum energy-time optimal control algorithm from:

Youngjin Kim and Tarunraj Singh, "Minimum Energy-Time Optimal Control of Wheeled Mobile Robots: Application to Parallel Parking," ACC 2022.

The code follows the paper's analytical reduction:

- solve the two nonlinear terminal equations in `z` and `T_f`
- evaluate the closed-form states and controls using Jacobi elliptic functions
- validate the result against the Hamiltonian two-point boundary value problem with `scipy.integrate.solve_bvp`

## Setup

```bash
uv venv .venv
uv sync
```

## Run

```bash
uv run optimal-control-parking --alpha-deg 45 --mu 0.5
```

Outputs are written to `outputs/`:

- `summary.json`
- `trajectory.csv`
- `trajectory_controls.png`
- `control_locus.png`

## Notes

- The terminal state is constrained to `(cos(alpha), sin(alpha), 0)` on the unit circle, matching the paper.
- `mu` must lie in `(0, 1)`.
- The analytical solver enforces the paper's control-circle identity `V^2 + omega^2 = 2(1-mu)/mu`.
