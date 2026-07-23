# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — FusionKernel phase-sync step helpers

"""Reduced-order phase-sync steps used by the CONTROL fusion kernel.

This leaf owns the Paper-27 / ζ·sin(Ψ−θ) phase-reduction step and multi-step
Lyapunov tracking helpers previously living on
:class:`~scpn_control.core.fusion_kernel.FusionKernel`. The CONTROL product
surface remains first-class under dual-home C and keeps thin wrappers that
supply ``phase_sync`` config defaults. The core Kuramoto numerics stay in
:mod:`scpn_control.phase.kuramoto` (not duplicated here).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from scpn_control._typing import FloatArray
from scpn_control.phase.kuramoto import (
    kuramoto_sakaguchi_step,
    lyapunov_exponent,
    lyapunov_v,
)


def phase_sync_step(
    theta: FloatArray,
    omega: FloatArray,
    *,
    dt: float = 1e-3,
    K: float | None = None,
    alpha: float | None = None,
    zeta: float | None = None,
    psi_driver: float | None = None,
    psi_mode: str | None = None,
    actuation_gain: float | None = None,
    phase_sync_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reduced-order plasma sync kernel (phase reduction).

    dθ_i/dt = ω_i + K·R·sin(ψ_r − θ_i − α) + ζ·sin(Ψ − θ_i)

    Ψ is exogenous when psi_mode="external" (no dotΨ equation).
    This is the reviewer's ζ sin(Ψ−θ) injection for plasma sync stability.

    Parameters
    ----------
    theta, omega :
        Oscillator phases and natural frequencies.
    dt :
        Integration step.
    K, alpha, zeta, psi_driver, psi_mode, actuation_gain :
        Optional overrides; when ``None``, values are taken from
        ``phase_sync_cfg`` with documented defaults.
    phase_sync_cfg :
        Optional mapping of defaults (typically
        ``FusionKernel.cfg["phase_sync"]``).
    """
    cfg = dict(phase_sync_cfg or {})
    k_eff = float(cfg.get("K", 1.0) if K is None else K)
    alpha_eff = float(cfg.get("alpha", 0.0) if alpha is None else alpha)
    zeta_eff = float(cfg.get("zeta", 0.0) if zeta is None else zeta)
    psi_mode_eff = str(cfg.get("psi_mode", "external") if psi_mode is None else psi_mode)
    gain = float(cfg.get("actuation_gain", 1.0) if actuation_gain is None else actuation_gain)

    return kuramoto_sakaguchi_step(
        theta=np.asarray(theta, dtype=np.float64),
        omega=np.asarray(omega, dtype=np.float64),
        dt=dt,
        K=k_eff * gain,
        alpha=alpha_eff,
        zeta=zeta_eff * gain,
        psi_driver=psi_driver,
        psi_mode=psi_mode_eff,
        wrap=True,
    )


def phase_sync_step_lyapunov(
    theta: FloatArray,
    omega: FloatArray,
    *,
    n_steps: int = 100,
    dt: float = 1e-3,
    K: float | None = None,
    zeta: float | None = None,
    psi_driver: float | None = None,
    psi_mode: str | None = None,
    phase_sync_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Multi-step phase sync with Lyapunov stability tracking.

    Returns final state, R trajectory, V trajectory, and λ exponent.
    λ < 0 ⟹ stable convergence toward Ψ.

    Parameters
    ----------
    theta, omega :
        Oscillator phases and natural frequencies.
    n_steps :
        Number of integration steps.
    dt :
        Integration step size.
    K, zeta, psi_driver, psi_mode :
        Optional overrides; when ``None``, values are taken from
        ``phase_sync_cfg`` with documented defaults.
    phase_sync_cfg :
        Optional mapping of defaults (typically
        ``FusionKernel.cfg["phase_sync"]``).
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and > 0")

    cfg = dict(phase_sync_cfg or {})
    k_eff = float(cfg.get("K", 1.0) if K is None else K)
    zeta_eff = float(cfg.get("zeta", 0.0) if zeta is None else zeta)
    psi_mode_eff = str(cfg.get("psi_mode", "external") if psi_mode is None else psi_mode)

    th = np.asarray(theta, dtype=np.float64)
    om = np.asarray(omega, dtype=np.float64)
    r_hist: list[float] = []
    v_hist: list[float] = []

    for _ in range(n_steps):
        out = kuramoto_sakaguchi_step(
            th,
            om,
            dt=dt,
            K=k_eff,
            zeta=zeta_eff,
            psi_driver=psi_driver,
            psi_mode=psi_mode_eff,
        )
        th = out["theta1"]
        r_hist.append(out["R"])
        v_hist.append(lyapunov_v(th, out["Psi"]))

    lam = lyapunov_exponent(v_hist, dt)
    return {
        "theta_final": th,
        "R_hist": np.array(r_hist),
        "V_hist": np.array(v_hist),
        "lambda": lam,
        "stable": lam < 0.0,
    }
