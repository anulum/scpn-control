# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Adaptive Knm diagnostic wiring tests.
"""Focused tests for adaptive Knm diagnostic-channel wiring."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.phase.adaptive_knm import AdaptiveKnmConfig, AdaptiveKnmEngine, DiagnosticSnapshot
from scpn_control.phase.plasma_knm import build_knm_plasma


def _snapshot(
    *,
    lambda_exp: float = -0.01,
    q95: float = 3.0,
    mirnov_rms: float = 0.0,
    v_value: float = 0.3,
    r_value: float = 0.5,
    disruption_risk: float = 0.0,
) -> DiagnosticSnapshot:
    """Build a diagnostic snapshot for the eight-layer plasma matrix."""
    layers = 8
    return DiagnosticSnapshot(
        R_layer=np.full(layers, r_value, dtype=np.float64),
        V_layer=np.full(layers, v_value, dtype=np.float64),
        lambda_exp=lambda_exp,
        beta_n=0.0,
        q95=q95,
        disruption_risk=disruption_risk,
        mirnov_rms=mirnov_rms,
        guard_approved=True,
    )


def _risk_isolation_config(
    *,
    lambda_risk_gain_s: float = 0.0,
    q95_low_risk_gain: float = 0.0,
    mirnov_risk_gain: float = 0.0,
    mirnov_rms_reference: float = 1.0,
) -> AdaptiveKnmConfig:
    """Return a config that isolates the risk-pair diagnostic channel."""
    return AdaptiveKnmConfig(
        beta_scale=0.0,
        risk_gain=0.5,
        lambda_risk_gain_s=lambda_risk_gain_s,
        q95_low_risk_gain=q95_low_risk_gain,
        mirnov_risk_gain=mirnov_risk_gain,
        mirnov_rms_reference=mirnov_rms_reference,
        coherence_Kp=0.0,
        coherence_Ki=0.0,
        max_delta_per_tick=1.0,
    )


def test_positive_lambda_exp_drives_risk_pair_coupling() -> None:
    """Positive Lyapunov exponent contributes to the MHD-pair risk drive."""
    spec = build_knm_plasma(mode="baseline", L=8)
    cfg = _risk_isolation_config(lambda_risk_gain_s=1.0)
    engine = AdaptiveKnmEngine(spec, cfg)
    baseline = engine.K_current.copy()

    adapted = engine.update(_snapshot(lambda_exp=0.4))

    assert adapted[2, 5] > baseline[2, 5]
    assert adapted[5, 2] == pytest.approx(adapted[2, 5])


def test_low_q95_and_mirnov_rms_drive_risk_pair_coupling() -> None:
    """Low q95 and Mirnov RMS contribute to the bounded scalar risk drive."""
    spec = build_knm_plasma(mode="baseline", L=8)
    cfg = _risk_isolation_config(q95_low_risk_gain=1.0, mirnov_risk_gain=1.0, mirnov_rms_reference=0.5)
    engine = AdaptiveKnmEngine(spec, cfg)
    baseline = engine.K_current.copy()

    adapted = engine.update(_snapshot(q95=2.0, mirnov_rms=0.5))

    assert adapted[2, 5] > baseline[2, 5]
    assert adapted[3, 5] > baseline[3, 5]


def test_v_layer_scales_low_coherence_diagonal_boost() -> None:
    """Layer Lyapunov V scales the diagonal PI boost when R is below target."""
    spec = build_knm_plasma(mode="baseline", L=8)
    cfg = AdaptiveKnmConfig(
        beta_scale=0.0,
        risk_gain=0.0,
        coherence_Kp=0.2,
        coherence_Ki=0.0,
        lyapunov_v_gain=1.0,
        max_delta_per_tick=1.0,
    )
    low_v_engine = AdaptiveKnmEngine(spec, cfg)
    high_v_engine = AdaptiveKnmEngine(spec, cfg)

    low_v = low_v_engine.update(_snapshot(v_value=0.0, r_value=0.2))
    high_v = high_v_engine.update(_snapshot(v_value=2.0, r_value=0.2))

    assert float(np.diag(high_v).mean()) > float(np.diag(low_v).mean())


@pytest.mark.parametrize(
    "bad_config",
    [
        AdaptiveKnmConfig(lambda_risk_gain_s=-0.1),
        AdaptiveKnmConfig(q95_reference=0.0),
        AdaptiveKnmConfig(mirnov_rms_reference=-1.0),
        AdaptiveKnmConfig(lyapunov_v_gain=float("nan")),
        AdaptiveKnmConfig(coherence_R_target=1.1),
        AdaptiveKnmConfig(risk_pairs=((-1, 2),)),
    ],
)
def test_config_validation_rejects_invalid_diagnostic_scales(bad_config: AdaptiveKnmConfig) -> None:
    """Invalid gain or reference constants fail before state mutation."""
    spec = build_knm_plasma(mode="baseline", L=8)

    with pytest.raises(ValueError):
        AdaptiveKnmEngine(spec, bad_config)
