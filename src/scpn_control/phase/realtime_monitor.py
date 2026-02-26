# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Real-Time Phase Sync Monitor
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Real-time dashboard hook: UPDE phase sync + LyapunovGuard + DIRECTOR_AI export.

Provides a tick-by-tick interface for live control dashboards.  Each call to
``tick()`` advances the UPDE by one step, checks Lyapunov stability, and
returns a snapshot suitable for streaming to a frontend or logging via
DIRECTOR_AI's AuditLogger.

Usage::

    monitor = RealtimeMonitor.from_paper27(psi_driver=0.0)
    for sample in sensor_stream:
        snap = monitor.tick()
        if not snap["guard_approved"]:
            trigger_safety_halt()
        dashboard.push(snap)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from scpn_control.phase.knm import KnmSpec, build_knm_paper27, OMEGA_N_16
from scpn_control.phase.kuramoto import lyapunov_v, order_parameter
from scpn_control.phase.lyapunov_guard import LyapunovGuard
from scpn_control.phase.upde import UPDESystem

FloatArray = NDArray[np.float64]


@dataclass
class RealtimeMonitor:
    """Tick-by-tick UPDE monitor with LyapunovGuard."""

    upde: UPDESystem
    guard: LyapunovGuard
    theta_layers: list[FloatArray]
    omega_layers: list[FloatArray]
    psi_driver: float = 0.0
    pac_gamma: float = 0.0
    _tick_count: int = field(default=0, init=False)

    @classmethod
    def from_paper27(
        cls,
        L: int = 16,
        N_per: int = 50,
        *,
        dt: float = 1e-3,
        zeta_uniform: float = 0.5,
        psi_driver: float = 0.0,
        pac_gamma: float = 0.0,
        guard_window: int = 50,
        guard_max_violations: int = 3,
        seed: int = 42,
    ) -> RealtimeMonitor:
        """Build from Paper 27 defaults."""
        spec = build_knm_paper27(L=L, zeta_uniform=zeta_uniform)
        upde = UPDESystem(spec=spec, dt=dt, psi_mode="external")
        guard = LyapunovGuard(window=guard_window, dt=dt, max_violations=guard_max_violations)

        rng = np.random.default_rng(seed)
        theta = [rng.uniform(-np.pi, np.pi, N_per) for _ in range(L)]
        omega = [OMEGA_N_16[m % 16] + rng.normal(0, 0.2, N_per) for m in range(L)]

        return cls(
            upde=upde, guard=guard,
            theta_layers=theta, omega_layers=omega,
            psi_driver=psi_driver, pac_gamma=pac_gamma,
        )

    def tick(self) -> dict:
        """Advance one UPDE step and return dashboard snapshot."""
        t0 = time.perf_counter_ns()

        out = self.upde.step(
            self.theta_layers, self.omega_layers,
            psi_driver=self.psi_driver, pac_gamma=self.pac_gamma,
        )
        self.theta_layers = out["theta1"]
        self._tick_count += 1

        all_theta = np.concatenate([t.ravel() for t in self.theta_layers])
        verdict = self.guard.check(all_theta, out["Psi_global"])

        elapsed_us = (time.perf_counter_ns() - t0) / 1000.0

        return {
            "tick": self._tick_count,
            "R_global": float(out["R_global"]),
            "R_layer": out["R_layer"].tolist(),
            "Psi_global": float(out["Psi_global"]),
            "V_global": float(out["V_global"]),
            "V_layer": out["V_layer"].tolist(),
            "lambda_exp": verdict.lambda_exp,
            "guard_approved": verdict.approved,
            "guard_score": verdict.score,
            "guard_violations": verdict.consecutive_violations,
            "latency_us": elapsed_us,
            "director_ai": self.guard.to_director_ai_dict(verdict),
        }

    def reset(self, seed: int = 42) -> None:
        """Reset oscillator phases and guard state."""
        rng = np.random.default_rng(seed)
        L = len(self.theta_layers)
        N = self.theta_layers[0].shape[0]
        self.theta_layers = [rng.uniform(-np.pi, np.pi, N) for _ in range(L)]
        self.guard.reset()
        self._tick_count = 0
