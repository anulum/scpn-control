# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disruption Contracts.
"""Replay one labelled shot through the bounded mitigation pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np

from scpn_control.control.advanced_soc_fusion_learning import FusionAIAgent
from scpn_control.control.disruption_roc import score_risk_series
from scpn_control.control.spi_mitigation import ShatteredPelletInjection
from scpn_control.core._validators import (
    require_1d_array,
    require_fraction,
    require_int,
    require_positive_float,
)


def run_real_shot_replay(
    *,
    shot_data: dict[str, Any],
    rl_agent: FusionAIAgent,
    base_tbr: float = 1.15,
    risk_threshold: float = 0.65,
    spi_trigger_risk: float = 0.80,
    window_size: int = 128,
) -> dict[str, Any]:
    """Replay a real tokamak shot through the disruption mitigation pipeline.

    Parameters
    ----------
    shot_data : dict
        NPZ-loaded shot data with keys: time_s, Ip_MA, BT_T, beta_N, q95,
        ne_1e19, n1_amp, n2_amp, locked_mode_amp, dBdt_gauss_per_s,
        vertical_position_m, is_disruption, disruption_time_idx, disruption_type.
    rl_agent : FusionAIAgent
        Reserved for the episode API; this deterministic replay does not
        choose an RL action.
    base_tbr : float
        Baseline tritium breeding ratio.
    risk_threshold : float
        Risk level above which alarm is raised.
    spi_trigger_risk : float
        Risk level above which SPI mitigation is triggered.
    window_size : int
        Sliding window size for disruption predictor, smaller than the shot.

    Returns
    -------
    dict with replay results including risk time-series and mitigation outcomes.
    """
    _ = require_positive_float("base_tbr", base_tbr)
    risk_threshold = require_fraction("risk_threshold", risk_threshold)
    spi_trigger_risk = require_fraction("spi_trigger_risk", spi_trigger_risk)
    if spi_trigger_risk < risk_threshold:
        raise ValueError("spi_trigger_risk must be >= risk_threshold.")
    window_size = require_int("window_size", window_size, 8)

    time_s = require_1d_array(
        "shot_data.time_s",
        shot_data.get("time_s", []),
        minimum_size=16,
    )
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError("shot_data.time_s must be strictly increasing.")

    n_steps = int(time_s.size)
    if window_size >= n_steps:
        raise ValueError(f"window_size must be < number of samples ({n_steps}), got {window_size}.")
    n1_amp = require_1d_array(
        "shot_data.n1_amp",
        shot_data.get("n1_amp", np.zeros(n_steps, dtype=np.float64)),
        expected_size=n_steps,
    )
    n2_amp = require_1d_array(
        "shot_data.n2_amp",
        shot_data.get("n2_amp", np.zeros(n_steps, dtype=np.float64)),
        expected_size=n_steps,
    )
    dBdt = require_1d_array(
        "shot_data.dBdt_gauss_per_s",
        shot_data.get("dBdt_gauss_per_s", np.zeros(n_steps, dtype=np.float64)),
        expected_size=n_steps,
    )
    beta_N = require_1d_array(
        "shot_data.beta_N",
        shot_data.get("beta_N", np.ones(n_steps, dtype=np.float64) * 2.0),
        expected_size=n_steps,
    )
    Ip_MA = require_1d_array(
        "shot_data.Ip_MA",
        shot_data.get("Ip_MA", np.ones(n_steps, dtype=np.float64)),
        expected_size=n_steps,
    )
    label_value = np.asarray(shot_data.get("is_disruption", False))
    if label_value.ndim != 0 or label_value.dtype.kind != "b":
        raise ValueError("is_disruption must be a scalar Boolean.")
    is_disruption = bool(label_value.item())
    index_value = np.asarray(shot_data.get("disruption_time_idx", -1))
    if index_value.ndim != 0 or index_value.dtype.kind not in "iu":
        raise ValueError("disruption_time_idx must be a scalar integer.")
    disruption_time_idx = int(index_value.item())
    if disruption_time_idx >= n_steps:
        raise ValueError(f"disruption_time_idx must be < number of samples ({n_steps}), got {disruption_time_idx}.")
    if disruption_time_idx < -1:
        raise ValueError("disruption_time_idx must be >= -1.")
    if is_disruption and disruption_time_idx < 0:
        raise ValueError("a disruptive shot needs a nonnegative disruption_time_idx.")
    if not is_disruption and disruption_time_idx != -1:
        raise ValueError("a safe shot needs disruption_time_idx=-1.")

    # Canonical per-window scoring lives in disruption_roc.score_risk_series (the
    # single source of truth shared with the FAIR-MAST evaluation harness). The
    # validation above guarantees 8 <= window_size < n_steps and finite inputs,
    # so every predictor window has enough finite samples.
    risk_series = score_risk_series(dBdt, n1_amp, n2_amp, window_size=window_size)
    alarm_series = np.zeros(n_steps, dtype=bool)
    first_alarm_idx = -1
    spi_triggered = False
    spi_trigger_idx = -1

    for t in range(window_size, n_steps):
        risk = float(risk_series[t])

        if risk > risk_threshold:
            alarm_series[t] = True
            if first_alarm_idx < 0:
                first_alarm_idx = t

        # SPI mitigation trigger
        if risk > spi_trigger_risk and not spi_triggered:
            spi_triggered = True
            spi_trigger_idx = t

    # Compute mitigation outcome
    if spi_triggered:
        pre_energy_mj = float(np.clip(300 + 50 * np.mean(beta_N), 200, 500))
        pre_current_ma = float(np.clip(np.mean(Ip_MA), 0.5, 20))
        risk_now = float(np.clip(risk_series[spi_trigger_idx], 0.0, 1.0))
        disturbance_now = float(np.clip(risk_now + 0.10 * np.mean(n1_amp), 0.0, 1.0))
        cocktail = ShatteredPelletInjection.estimate_mitigation_cocktail(
            risk_score=risk_now,
            disturbance=disturbance_now,
            action_bias=0.0,
        )
        neon_mol = float(cocktail["neon_quantity_mol"])
        argon_mol = float(cocktail["argon_quantity_mol"])
        xenon_mol = float(cocktail["xenon_quantity_mol"])
        total_impurity_mol = float(cocktail["total_quantity_mol"])

        spi = ShatteredPelletInjection(
            Plasma_Energy_MJ=pre_energy_mj,
            Plasma_Current_MA=pre_current_ma,
        )
        _, _, _, spi_diag = spi.trigger_mitigation(
            neon_quantity_mol=neon_mol,
            argon_quantity_mol=argon_mol,
            xenon_quantity_mol=xenon_mol,
            return_diagnostics=True,
            duration_s=0.03,
            dt_s=5e-5,
            verbose=False,
        )
        tau_cq_ms = float(spi_diag["tau_cq_ms_mean"])
        final_current = float(spi_diag["final_current_MA"])
        z_eff = float(spi_diag["z_eff"])
    else:
        tau_cq_ms = 0.0
        final_current = float(np.mean(Ip_MA))
        z_eff = 1.0
        neon_mol = 0.0
        argon_mol = 0.0
        xenon_mol = 0.0
        total_impurity_mol = 0.0

    # Detection timing
    detection_lead_ms = -1.0
    if is_disruption and disruption_time_idx > 0 and first_alarm_idx > 0:
        dt_arr = time_s if time_s.size > 0 else np.arange(n_steps) * 0.001
        detection_lead_ms = float((dt_arr[disruption_time_idx] - dt_arr[first_alarm_idx]) * 1000)

    # Prevention determination
    prevented = False
    if is_disruption:
        if spi_triggered and spi_trigger_idx < disruption_time_idx:
            # SPI triggered before disruption — check if it mitigated
            post_risk = np.mean(risk_series[spi_trigger_idx : min(spi_trigger_idx + 50, n_steps)])
            prevented = bool(post_risk < 0.88 and tau_cq_ms > 0)
    else:
        prevented = not spi_triggered  # For safe shots, not triggering SPI = correct

    return {
        "n_steps": n_steps,
        "is_disruption": is_disruption,
        "disruption_time_idx": disruption_time_idx,
        "first_alarm_idx": first_alarm_idx,
        "spi_triggered": spi_triggered,
        "spi_trigger_idx": spi_trigger_idx,
        "detection_lead_ms": round(detection_lead_ms, 1),
        "prevented": prevented,
        "neon_mol": round(neon_mol, 4),
        "argon_mol": round(argon_mol, 4),
        "xenon_mol": round(xenon_mol, 4),
        "total_impurity_mol": round(total_impurity_mol, 4),
        "tau_cq_ms": round(tau_cq_ms, 2),
        "final_current_MA": round(final_current, 3),
        "z_eff": round(z_eff, 2),
        "peak_risk": round(float(np.max(risk_series)), 4),
        "mean_risk": round(float(np.mean(risk_series)), 4),
        "risk_series": risk_series.tolist(),
    }
