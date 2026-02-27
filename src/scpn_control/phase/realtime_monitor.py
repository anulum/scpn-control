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
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray

from scpn_control.phase.knm import OMEGA_N_16, build_knm_paper27
from scpn_control.phase.lyapunov_guard import LyapunovGuard
from scpn_control.phase.upde import UPDESystem

FloatArray = NDArray[np.float64]


@dataclass
class TrajectoryRecorder:
    """Accumulates tick snapshots for batch export."""

    R_global: list[float] = field(default_factory=list)
    R_layer: list[list[float]] = field(default_factory=list)
    V_global: list[float] = field(default_factory=list)
    V_layer: list[list[float]] = field(default_factory=list)
    lambda_exp: list[float] = field(default_factory=list)
    guard_approved: list[bool] = field(default_factory=list)
    latency_us: list[float] = field(default_factory=list)
    Psi_global: list[float] = field(default_factory=list)

    def record(self, snap: dict) -> None:
        self.R_global.append(snap["R_global"])
        self.R_layer.append(snap["R_layer"])
        self.V_global.append(snap["V_global"])
        self.V_layer.append(snap["V_layer"])
        self.lambda_exp.append(snap["lambda_exp"])
        self.guard_approved.append(snap["guard_approved"])
        self.latency_us.append(snap["latency_us"])
        self.Psi_global.append(snap["Psi_global"])

    def clear(self) -> None:
        for lst in (self.R_global, self.R_layer, self.V_global, self.V_layer,
                    self.lambda_exp, self.guard_approved, self.latency_us, self.Psi_global):
            lst.clear()

    @property
    def n_ticks(self) -> int:
        return len(self.R_global)


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
    _recorder: TrajectoryRecorder = field(default_factory=TrajectoryRecorder, init=False)

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

    def tick(self, *, record: bool = True) -> dict:
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

        snap = {
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
        if record:
            self._recorder.record(snap)
        return snap

    @property
    def recorder(self) -> TrajectoryRecorder:
        return self._recorder

    def reset(self, seed: int = 42) -> None:
        """Reset oscillator phases, guard state, and recorder."""
        rng = np.random.default_rng(seed)
        L = len(self.theta_layers)
        N = self.theta_layers[0].shape[0]
        self.theta_layers = [np.asarray(rng.uniform(-np.pi, np.pi, N)) for _ in range(L)]
        self.guard.reset()
        self._recorder.clear()
        self._tick_count = 0

    def save_hdf5(self, path: Union[str, Path]) -> Path:
        """Export recorded trajectory to HDF5.

        Datasets: R_global, R_layer, V_global, V_layer, lambda_exp,
        guard_approved, latency_us, Psi_global.  Attributes: L, N_per,
        psi_driver, pac_gamma, n_ticks.

        Requires h5py (``pip install h5py``).
        """
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("pip install h5py") from exc

        path = Path(path)
        rec = self._recorder
        with h5py.File(path, "w") as f:
            f.create_dataset("R_global", data=np.array(rec.R_global, dtype=np.float64))
            f.create_dataset("R_layer", data=np.array(rec.R_layer, dtype=np.float64))
            f.create_dataset("V_global", data=np.array(rec.V_global, dtype=np.float64))
            f.create_dataset("V_layer", data=np.array(rec.V_layer, dtype=np.float64))
            f.create_dataset("lambda_exp", data=np.array(rec.lambda_exp, dtype=np.float64))
            f.create_dataset("guard_approved", data=np.array(rec.guard_approved, dtype=bool))
            f.create_dataset("latency_us", data=np.array(rec.latency_us, dtype=np.float64))
            f.create_dataset("Psi_global", data=np.array(rec.Psi_global, dtype=np.float64))
            f.attrs["L"] = len(self.theta_layers)
            f.attrs["N_per"] = self.theta_layers[0].shape[0]
            f.attrs["psi_driver"] = self.psi_driver
            f.attrs["pac_gamma"] = self.pac_gamma
            f.attrs["n_ticks"] = rec.n_ticks
        return path

    def save_npz(self, path: Union[str, Path]) -> Path:
        """Export recorded trajectory to compressed NPZ (no h5py needed)."""
        path = Path(path)
        rec = self._recorder
        np.savez_compressed(
            path,
            R_global=np.array(rec.R_global, dtype=np.float64),
            R_layer=np.array(rec.R_layer, dtype=np.float64),
            V_global=np.array(rec.V_global, dtype=np.float64),
            V_layer=np.array(rec.V_layer, dtype=np.float64),
            lambda_exp=np.array(rec.lambda_exp, dtype=np.float64),
            guard_approved=np.array(rec.guard_approved, dtype=bool),
            latency_us=np.array(rec.latency_us, dtype=np.float64),
            Psi_global=np.array(rec.Psi_global, dtype=np.float64),
        )
        return path
