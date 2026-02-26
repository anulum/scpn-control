#!/usr/bin/env python3
"""Digital twin closed-loop demo.

Runs the tokamak digital twin with H-infinity and PID controllers,
comparing final displacement errors and demonstrating chaos-monkey
sensor fault injection.

Usage:
    python examples/digital_twin_demo.py
"""
from __future__ import annotations

import json
import sys

import numpy as np

from scpn_control.control.tokamak_digital_twin import run_digital_twin
from scpn_control.control.h_infinity_controller import get_radial_robust_controller


def main() -> None:
    seed = 42
    steps = 200

    # --- Baseline run (PID) ---
    summary_pid = run_digital_twin(
        time_steps=steps,
        save_plot=False,
        verbose=False,
        seed=seed,
        sensor_dropout_prob=0.0,
        sensor_noise_std=0.0,
    )

    # --- Chaos monkey run (PID, 10% dropout + noise) ---
    summary_chaos = run_digital_twin(
        time_steps=steps,
        save_plot=False,
        verbose=False,
        seed=seed,
        chaos_monkey=True,
        sensor_dropout_prob=0.10,
        sensor_noise_std=0.02,
    )

    # --- H-infinity closed-loop comparison ---
    ctrl = get_radial_robust_controller(gamma_growth=100.0, damping=10.0)
    ctrl.reset()
    dt = 0.001
    from scipy.linalg import expm

    A, B2, C2 = ctrl.A, ctrl.B2, ctrl.C2
    n = A.shape[0]
    M = np.zeros((n + B2.shape[1], n + B2.shape[1]))
    M[:n, :n] = A * dt
    M[:n, n:] = B2 * dt
    eM = expm(M)
    Ad, Bd = eM[:n, :n], eM[:n, n:]

    x = np.array([0.1, 0.0])
    errors = []
    for _ in range(steps):
        y = (C2 @ x).item()
        u = ctrl.step(y, dt)
        x = Ad @ x + Bd.ravel() * u
        errors.append(abs(y))

    results = {
        "pid_baseline": {
            "final_avg_temp": float(summary_pid["final_avg_temp"]),
            "final_islands": int(summary_pid["final_islands_px"]),
            "final_reward": float(summary_pid["final_reward"]),
        },
        "pid_chaos_monkey": {
            "dropout_prob": 0.10,
            "noise_std": 0.02,
            "final_avg_temp": float(summary_chaos["final_avg_temp"]),
            "final_islands": int(summary_chaos["final_islands_px"]),
            "final_reward": float(summary_chaos["final_reward"]),
            "sensor_dropouts": int(summary_chaos["sensor_dropouts_total"]),
        },
        "h_infinity": {
            "gamma": float(ctrl.gamma),
            "gain_margin_db": float(ctrl.gain_margin_db),
            "final_error": float(errors[-1]),
            "converged_step": next(
                (i for i, e in enumerate(errors) if e < 0.01 * errors[0]), steps
            ),
        },
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
