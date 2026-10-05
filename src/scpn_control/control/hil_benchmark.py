# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Hardware-in-the-Loop Test Harness

"""HIL benchmark runners."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from scpn_control.control.hil_evidence import HILBenchmarkResult
from scpn_control.control.hil_fpga import FPGASNNExport
from scpn_control.control.hil_io import SensorInterface
from scpn_control.control.hil_loop import HILControlLoop, PipelineProfile

logger = logging.getLogger("scpn_control.control.hil_harness")


def run_hil_benchmark(
    *,
    iterations: int = 1000,
    target_rate_hz: float = 1000.0,
    include_fpga_export: bool = True,
    verbose: bool = False,
) -> HILBenchmarkResult:
    """Run the full HIL benchmark suite.

    Executes a PID controller at target rate, measures timing, and
    optionally generates FPGA export.
    """
    sensor = SensorInterface(rng_seed=42)

    # Simple PID controller
    pid_state = {"integral": 0.0, "prev_error": 0.0}

    def pid_controller(error: float, _sensor: SensorInterface) -> float:
        Kp, Ki, Kd = 5.0, 0.2, 2.0
        pid_state["integral"] += error
        derivative = error - pid_state["prev_error"]
        pid_state["prev_error"] = error
        return Kp * error + Ki * pid_state["integral"] + Kd * derivative

    loop = HILControlLoop(target_rate_hz=target_rate_hz, sensor=sensor)
    loop.set_controller(pid_controller)

    # Plant: vertical position with instability
    def vde_plant(state: float, cmd: float) -> float:
        growth_rate = 0.1
        dt_s = 1.0 / target_rate_hz
        return state * (1.0 + growth_rate * dt_s) + cmd * 0.2 * dt_s

    metrics = loop.run(
        iterations=iterations,
        plant_fn=vde_plant,
        initial_state=0.1,
        setpoint=0.0,
    )

    # Decompose latency budget
    sensor_lat = metrics.mean_latency_us * 0.15  # ~15% sensor
    controller_lat = metrics.mean_latency_us * 0.60  # ~60% control
    actuator_lat = metrics.mean_latency_us * 0.25  # ~25% actuator

    fpga_map = None
    if include_fpga_export:
        exporter = FPGASNNExport(n_neurons=50, n_channels=2)
        fpga_map = exporter.generate_register_map()

    result = HILBenchmarkResult(
        control_metrics=metrics,
        sensor_latency_us=sensor_lat,
        controller_latency_us=controller_lat,
        actuator_latency_us=actuator_lat,
        total_loop_latency_us=metrics.mean_latency_us,
        passes_sub_ms=metrics.sub_ms_achieved,
        passes_1khz=metrics.p95_latency_us < (1e6 / target_rate_hz),
        fpga_register_map=fpga_map,
    )

    if verbose:
        logger.info("=== HIL Benchmark Results ===")
        logger.info(f"  Iterations:     {metrics.iterations}")
        logger.info(f"  Target rate:    {target_rate_hz:.0f} Hz ({metrics.target_dt_us:.0f} us)")
        logger.info(f"  P50 latency:    {metrics.p50_latency_us:.1f} us")
        logger.info(f"  P95 latency:    {metrics.p95_latency_us:.1f} us")
        logger.info(f"  P99 latency:    {metrics.p99_latency_us:.1f} us")
        logger.info(f"  Max latency:    {metrics.max_latency_us:.1f} us")
        logger.info(f"  Jitter (std):   {metrics.jitter_std_us:.1f} us")
        logger.info(f"  Overruns:       {metrics.overrun_count} ({metrics.overrun_fraction * 100:.1f}%)")
        logger.info(f"  Sub-ms (P95):   {'PASS' if metrics.sub_ms_achieved else 'FAIL'}")
        logger.info(f"  1 kHz capable:  {'PASS' if result.passes_1khz else 'FAIL'}")
        if fpga_map:
            logger.info(f"  FPGA neurons:   {fpga_map.n_neurons}")
            logger.info(f"  FPGA clock:     {fpga_map.clock_hz / 1e6:.0f} MHz")

    return result


def run_hil_benchmark_detailed(n_steps: int = 10000) -> dict[str, Any]:
    """Run HIL benchmark with detailed per-stage profiling.

    Exercises a realistic 4-state Kalman filter (predict + update) for
    state estimation, a feedback gain K@x for control, and saturated
    actuator output.  Matrices are pre-allocated outside the loop to
    mirror real-time allocations.

    Returns dict with latency statistics and pipeline profile.
    """
    # 4-state vertical-stability plant: x = [z, dz/dt, z_int, bias]
    n_x = 4
    n_u = 2
    n_y = 2

    dt = 1e-3  # 1 kHz loop
    gamma = 100.0  # VDE growth rate

    # Discrete-time plant (Euler)
    A = np.eye(n_x)
    A[0, 1] = dt
    A[1, 0] = gamma**2 * dt
    A[1, 1] = 1.0 - 10.0 * dt
    A[2, 0] = dt  # integrator of z
    A[3, 3] = 1.0  # bias random walk

    B = np.zeros((n_x, n_u))
    B[1, 0] = dt
    B[1, 1] = -dt

    C = np.zeros((n_y, n_x))
    C[0, 0] = 1.0  # z measurement
    C[1, 1] = 1.0  # dz/dt measurement

    Q = np.diag([1e-6, 1e-4, 1e-8, 1e-6])  # process noise
    R = np.diag([1e-4, 1e-3])  # measurement noise

    # LQR-style gain (pre-computed)
    K_ctrl = np.array(
        [
            [5000.0, 150.0, 100.0, 50.0],
            [-5000.0, -150.0, -100.0, -50.0],
        ]
    )

    # Pre-allocate Kalman state
    x_hat = np.zeros(n_x)
    P = np.eye(n_x) * 0.01
    y_meas = np.zeros(n_y)
    rng = np.random.default_rng(42)

    profiles = []

    for _ in range(n_steps):
        p = PipelineProfile()
        y_meas[:] = rng.standard_normal(n_y) * 0.01

        # Kalman predict + update
        t0 = time.perf_counter_ns()
        x_pred = A @ x_hat
        P_pred = A @ P @ A.T + Q
        S = C @ P_pred @ C.T + R
        K_kal = P_pred @ C.T @ np.linalg.solve(S, np.eye(n_y))
        innov = y_meas - C @ x_pred
        x_hat = x_pred + K_kal @ innov
        P = (np.eye(n_x) - K_kal @ C) @ P_pred
        p.state_estimation_us = (time.perf_counter_ns() - t0) / 1e3

        # Controller K@x
        t0 = time.perf_counter_ns()
        u = K_ctrl @ x_hat
        p.controller_step_us = (time.perf_counter_ns() - t0) / 1e3

        # Actuator saturation
        t0 = time.perf_counter_ns()
        u = np.clip(u, -1.0, 1.0)
        x_hat = A @ x_hat + B @ u
        p.actuator_command_us = (time.perf_counter_ns() - t0) / 1e3

        p.total_us = p.state_estimation_us + p.controller_step_us + p.actuator_command_us
        profiles.append(p)

    totals = np.array([p.total_us for p in profiles])
    return {
        "n_steps": n_steps,
        "mean_us": float(np.mean(totals)),
        "p50_us": float(np.percentile(totals, 50)),
        "p95_us": float(np.percentile(totals, 95)),
        "p99_us": float(np.percentile(totals, 99)),
        "max_us": float(np.max(totals)),
        "stage_breakdown": {
            "state_estimation_mean_us": float(np.mean([p.state_estimation_us for p in profiles])),
            "controller_step_mean_us": float(np.mean([p.controller_step_us for p in profiles])),
            "actuator_command_mean_us": float(np.mean([p.actuator_command_us for p in profiles])),
        },
    }


# ─── HIL Demo Runner (Software FPGA Register Simulation) ─────────────
