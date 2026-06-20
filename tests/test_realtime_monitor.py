# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — RealtimeMonitor plasma builder and adaptive-tick tests.
from __future__ import annotations

import numpy as np
import pytest

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
    # exercise the per-tick guard in tick().
    monitor.pac_gamma = float("nan")
    with pytest.raises(ValueError, match="pac_gamma must be finite"):
        monitor.tick()


def test_tick_rejects_nonfinite_psi_driver() -> None:
    monitor = RealtimeMonitor.from_paper27(L=3, N_per=10, seed=42)
    monitor.psi_driver = float("inf")
    with pytest.raises(ValueError, match="psi_driver must be finite"):
        monitor.tick()


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
    assert "adaptive" in first
    assert first["adaptive"]["L"] == 4

    second = monitor.tick(beta_n=1.5, q95=3.2, disruption_risk=0.2, mirnov_rms=0.07)
    assert "adaptive" in second
    assert monitor.recorder.n_ticks == 2
    assert np.isfinite(second["R_global"])
