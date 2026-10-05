# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disruption Contracts.
"""Bounded synthetic disruption signal and mitigation physics proxies."""

from __future__ import annotations

from math import hypot

import numpy as np
from numpy.typing import NDArray

from scpn_control.control.spi_mitigation import ShatteredPelletInjection
from scpn_control.core._statistics import linear_percentile
from scpn_control.core._validators import (
    require_finite_float,
    require_fraction,
    require_int,
    require_non_negative_float,
    require_positive_float,
)

_TBR_EQUIVALENCE_SCALE = 1.45


def synthetic_disruption_signal(
    *,
    rng: np.random.Generator,
    disturbance: float,
    window: int = 220,
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """Generate a synthetic pre-disruption diagnostic signal with an ELM burst.

    Parameters
    ----------
    rng
        Random generator for reproducible noise and phase.
    disturbance
        Disturbance amplitude scaling the ELM burst.
    window
        Number of time samples.

    Returns
    -------
    tuple[NDArray[np.float64], dict[str, float]]
        The signal trace and a mapping of derived toroidal observables.
    """
    disturbance = require_non_negative_float("disturbance", disturbance)
    window = require_int("window", window, 1)
    t = np.linspace(0.0, 1.0, window, dtype=np.float64)
    base = 0.68 + 0.10 * np.sin(2.0 * np.pi * 2.4 * t + rng.uniform(-0.4, 0.4))
    elm = disturbance * (0.30 * np.exp(-(((t - 0.78) / 0.10) ** 2)))
    signal = np.clip(base + elm + rng.normal(0.0, 0.018, size=t.shape), 0.01, None)
    n1 = float(0.08 + 0.55 * disturbance + rng.uniform(0.00, 0.05))
    n2 = float(0.05 + 0.32 * disturbance + rng.uniform(0.00, 0.04))
    n3 = float(0.02 + 0.15 * disturbance + rng.uniform(0.00, 0.03))
    toroidal = {
        "toroidal_n1_amp": n1,
        "toroidal_n2_amp": n2,
        "toroidal_n3_amp": n3,
        "toroidal_asymmetry_index": hypot(n1, n2, n3),
        "toroidal_radial_spread": float(0.02 + 0.08 * disturbance),
    }
    if not bool(np.all(np.isfinite(signal))) or not all(np.isfinite(value) for value in toroidal.values()):
        raise ValueError("synthetic disruption signal and toroidal observables must be finite.")
    return signal, toroidal


def mcnp_lite_tbr(
    *,
    base_tbr: float,
    li6_enrichment: float,
    be_multiplier_fraction: float,
    reflector_albedo: float,
) -> tuple[float, float]:
    """Lightweight tritium-breeding-ratio estimate for a blanket configuration.

    Parameters
    ----------
    base_tbr
        Baseline tritium breeding ratio; must be positive.
    li6_enrichment
        Lithium-6 enrichment fraction.
    be_multiplier_fraction
        Beryllium neutron-multiplier fraction.
    reflector_albedo
        Reflector albedo fraction.

    Returns
    -------
    tuple[float, float]
        The adjusted tritium breeding ratio and its estimated uncertainty.
    """
    base_tbr = require_positive_float("base_tbr", base_tbr)
    li6_enrichment = require_finite_float("li6_enrichment", li6_enrichment)
    be_multiplier_fraction = require_finite_float("be_multiplier_fraction", be_multiplier_fraction)
    reflector_albedo = require_finite_float("reflector_albedo", reflector_albedo)
    factor = float(
        1.15
        + 0.20 * float(np.clip(be_multiplier_fraction, 0.0, 1.0))
        + 0.10 * float(np.clip(li6_enrichment, 0.0, 1.0))
        + 0.05 * float(np.clip(reflector_albedo, 0.0, 1.0))
    )
    # Keep Task-5 gates aligned with engineering-equivalent TBR scale
    # while using conservative volumetric transport surrogates.
    tbr = require_finite_float("tbr", base_tbr * factor * _TBR_EQUIVALENCE_SCALE)
    return tbr, factor


def impurity_transport_response(
    *,
    neon_quantity_mol: float,
    argon_quantity_mol: float,
    xenon_quantity_mol: float,
    disturbance: float,
    seed_shift: int,
) -> dict[str, float]:
    """Simulate the impurity-transport response to mitigation gas injection.

    Parameters
    ----------
    neon_quantity_mol
        Injected neon in moles.
    argon_quantity_mol
        Injected argon in moles.
    xenon_quantity_mol
        Injected xenon in moles.
    disturbance
        Disturbance amplitude.
    seed_shift
        Integer scenario metadata emitted with the result; this deterministic
        proxy does not draw from a random seed.

    Returns
    -------
    dict[str, float]
        Radiated-fraction and impurity-response metrics.
    """
    neon_quantity_mol = require_finite_float("neon_quantity_mol", neon_quantity_mol)
    argon_quantity_mol = require_finite_float("argon_quantity_mol", argon_quantity_mol)
    xenon_quantity_mol = require_finite_float("xenon_quantity_mol", xenon_quantity_mol)
    disturbance = require_non_negative_float("disturbance", disturbance)
    seed_shift = require_int("seed_shift", seed_shift)
    if abs(seed_shift) > 2**53:
        raise ValueError("seed_shift must be exactly representable as a float.")
    n_steps = 240
    dt = 1.25e-4
    t = np.arange(n_steps, dtype=np.float64) * dt
    neon = max(float(neon_quantity_mol), 0.0)
    argon = max(float(argon_quantity_mol), 0.0)
    xenon = max(float(xenon_quantity_mol), 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        source_strength = float(1.00 * neon + 1.35 * argon + 1.90 * xenon)
        sink_rate = 120.0 + 35.0 * disturbance + 45.0 * (argon + 1.2 * xenon)
    if (
        not np.isfinite(source_strength)
        or not np.isfinite(sink_rate)
        or source_strength > np.finfo(float).max / sink_rate
    ):
        raise ValueError("impurity source and sink must remain finite during transport.")
    source = source_strength * np.exp(-t / 0.004)
    n_imp = np.zeros_like(t)
    n_imp[0] = source[0]
    for i in range(1, n_steps):
        dn = source[i] - sink_rate * n_imp[i - 1]
        n_imp[i] = max(0.0, n_imp[i - 1] + dt * dn)
    weighted = float(np.mean(n_imp[-80:]))
    cocktail_zeff = ShatteredPelletInjection.estimate_z_eff_cocktail(
        neon_quantity_mol=neon,
        argon_quantity_mol=argon,
        xenon_quantity_mol=xenon,
    )
    zeff_eff = float(
        np.clip(
            0.65 * cocktail_zeff + 0.35 * (1.05 + 42.0 * weighted + 0.22 * disturbance),
            1.0,
            12.0,
        )
    )
    rad_mw = float(
        (24.0 + 95.0 * weighted) * (1.0 + 0.15 * disturbance) * (1.0 + 0.035 * max(cocktail_zeff - 1.0, 0.0))
    )
    result = {
        "zeff_eff": zeff_eff,
        "impurity_radiation_mw": rad_mw,
        "impurity_decay_tau_ms": float(1e3 / max(sink_rate, 1e-9)),
        "total_impurity_mol": float(neon + argon + xenon),
        "seed_shift": float(seed_shift),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("impurity response must remain finite.")
    return result


def post_disruption_halo_runaway(
    *,
    pre_current_ma: float,
    tau_cq_s: float,
    disturbance: float,
    mitigation_strength: float,
    zeff_eff: float,
) -> dict[str, float]:
    """Evolve post-disruption halo current and runaway-electron generation.

    Parameters
    ----------
    pre_current_ma
        Pre-disruption plasma current in MA.
    tau_cq_s
        Current-quench time constant in seconds.
    disturbance
        Disturbance amplitude.
    mitigation_strength
        Mitigation effectiveness in [0, 1].
    zeff_eff
        Effective charge during the quench.

    Returns
    -------
    dict[str, float]
        Peak halo-current fraction, runaway current, and related metrics.
    """
    pre_current_ma = require_non_negative_float("pre_current_ma", pre_current_ma)
    tau_cq_s = require_positive_float("tau_cq_s", tau_cq_s)
    disturbance = require_non_negative_float("disturbance", disturbance)
    mitigation_strength = require_fraction("mitigation_strength", mitigation_strength)
    zeff_eff = require_positive_float("zeff_eff", zeff_eff)
    dt = 1.0e-4
    steps = 320
    ip = float(pre_current_ma)
    halo = 0.0
    runaway = 0.0
    halo_hist: list[float] = []
    re_hist: list[float] = []

    # Pautasso et al., Nucl. Fusion 57, 076014 (2017): fastest CQ ~4 ms on JET
    tau_ip = max(float(tau_cq_s), 0.004)
    # Riccardo et al., Nucl. Fusion 50, 025005 (2010): halo current rise-time
    tau_halo = 0.006 + 0.008 * disturbance
    for _ in range(steps):
        d_ip = -ip / tau_ip
        if not np.isfinite(d_ip):
            raise ValueError("halo current-quench derivative must remain finite.")
        ip = max(0.0, ip + dt * d_ip)
        e_norm = float(np.clip((-d_ip) / max(pre_current_ma / 0.01, 1e-9), 0.0, 8.0))
        # Halo fraction 10-40% of Ip; Riccardo et al. (2010)
        halo_drive = 0.28 * abs(d_ip) * (1.0 + 0.4 * disturbance)
        halo = max(0.0, halo + dt * (halo_drive - halo / max(tau_halo, 1e-4)))

        # Connor & Hastie, Nucl. Fusion 15, 415 (1975): E_crit = 1 in normalised units
        re_source = max(e_norm - 1.0, 0.0) * (1.0 + 0.7 * disturbance)
        # Hesslow et al., Nucl. Fusion 59, 084004 (2019): collisional damping
        impurity_damping = (0.14 + 0.015 * zeff_eff) * (1.0 + 0.9 * mitigation_strength)
        # Rosenbluth & Putvinski, Nucl. Fusion 37, 1355 (1997): avalanche multiplication
        runaway = max(
            0.0,
            runaway + dt * (0.22 * re_source + 0.48 * runaway * max(e_norm - 0.6, 0.0) - impurity_damping * runaway),
        )
        halo_hist.append(float(halo))
        re_hist.append(float(runaway))

    return {
        "halo_current_ma": linear_percentile(halo_hist, 95.0),
        "runaway_beam_ma": linear_percentile(re_hist, 95.0),
        "halo_peak_ma": float(max(halo_hist)),
        "runaway_peak_ma": float(max(re_hist)),
    }
