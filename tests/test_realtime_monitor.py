# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — RealtimeMonitor plasma builder and adaptive-tick tests.
from __future__ import annotations

from typing import Any

import numpy as np

from scpn_control.phase.adaptive_knm import AdaptiveKnmEngine
from scpn_control.phase.plasma_knm import build_knm_plasma
from scpn_control.phase.realtime_monitor import RealtimeMonitor


def test_from_plasma_builds_and_ticks() -> None:
    monitor = RealtimeMonitor.from_plasma(L=4, N_per=10, mode="baseline", seed=7)
    assert len(monitor.theta_layers) == 4
    assert monitor.theta_layers[0].shape == (10,)
    assert monitor.adaptive_engine is None

    snap = monitor.tick()
    assert snap["tick"] == 1
    assert "adaptive" not in snap
    assert np.isfinite(snap["R_global"])
    assert np.isfinite(snap["lambda_exp"])


def test_tick_rejects_nonfinite_pac_gamma() -> None:
    monitor = RealtimeMonitor.from_paper27(L=3, N_per=10, seed=42)
    # The constructor validates pac_gamma, so mutate it after construction to
    # exercise the per-tick fail-closed path in tick().
    monitor.pac_gamma = float("nan")
    snap = monitor.tick()
    assert snap["guard_approved"] is False
    assert snap["error_type"] == "ValueError"
    assert snap["error"] == "pac_gamma must be finite"
    assert monitor.recorder.n_ticks == 1


def test_tick_rejects_nonfinite_psi_driver() -> None:
    monitor = RealtimeMonitor.from_paper27(L=3, N_per=10, seed=42)
    monitor.psi_driver = float("inf")
    snap = monitor.tick(record=False)
    assert snap["tick"] == 1
    assert snap["guard_approved"] is False
    assert snap["error_type"] == "ValueError"
    assert snap["error"] == "psi_driver must be finite"
    assert monitor.recorder.n_ticks == 0


def test_tick_fails_closed_when_upde_step_raises(monkeypatch: Any) -> None:
    monitor = RealtimeMonitor.from_paper27(L=3, N_per=10, seed=42)

    def raise_step(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("native solver unavailable")

    monkeypatch.setattr(monitor.upde, "step", raise_step)
    snap = monitor.tick()

    assert snap["tick"] == 1
    assert snap["R_global"] == 0.0
    assert snap["R_layer"] == [0.0, 0.0, 0.0]
    assert snap["V_global"] == 0.0
    assert snap["V_layer"] == [0.0, 0.0, 0.0]
    assert snap["guard_approved"] is False
    assert snap["guard_score"] == 0.0
    assert snap["director_ai"]["approved"] is False
    assert snap["director_ai"]["halt_reason"] == "RealtimeMonitor tick failed closed"
    assert snap["error_type"] == "RuntimeError"
    assert snap["error"] == "native solver unavailable"
    assert monitor.recorder.guard_approved == [False]


def test_adaptive_engine_branch_overrides_and_summarizes() -> None:
    spec = build_knm_plasma(mode="baseline", L=4, zeta_uniform=0.0)
    engine = AdaptiveKnmEngine(spec)
    monitor = RealtimeMonitor.from_plasma(
        L=4,
        N_per=10,
        mode="baseline",
        seed=7,
        adaptive_engine=engine,
    )
    assert monitor.adaptive_engine is engine

    # First tick exercises the zero-initialised cached-state branch
    # (_last_R_layer is None); subsequent ticks exercise the cached path.
    first = monitor.tick(beta_n=1.2, q95=3.4, disruption_risk=0.1, mirnov_rms=0.05)
    assert "adaptive" in first, first
    assert first["adaptive"]["L"] == 4

    second = monitor.tick(beta_n=1.5, q95=3.2, disruption_risk=0.2, mirnov_rms=0.07)
    assert "adaptive" in second
    assert monitor.recorder.n_ticks == 2
    assert np.isfinite(second["R_global"])
