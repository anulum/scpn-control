# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Halo Runaway Physics
"""Bounded halo and runaway ensemble simulation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from scpn_control.control._halo_current_model import HaloCurrentModel
from scpn_control.control._runaway_electron_model import RunawayElectronModel
from scpn_control.core._statistics import linear_percentile
from scpn_control.core._validators import require_range

logger = logging.getLogger("scpn_control.control.halo_re_physics")


@dataclass
class DisruptionMitigationReport:
    """Combined disruption mitigation ensemble report."""

    ensemble_runs: int
    prevention_rate: float
    mean_halo_peak_ma: float
    p95_halo_peak_ma: float
    mean_re_peak_ma: float
    p95_re_peak_ma: float
    mean_tpf_product: float
    passes_iter_limits: bool
    per_run_details: list[dict[str, Any]]


def run_disruption_ensemble(
    *,
    ensemble_runs: int = 50,
    seed: int = 42,
    plasma_current_range: tuple[float, float] = (11.0, 16.5),
    plasma_energy_range: tuple[float, float] = (240.0, 420.0),
    neon_range: tuple[float, float] = (0.03, 0.24),
    verbose: bool = False,
) -> DisruptionMitigationReport:
    """Run a full disruption mitigation ensemble with physics-based halo/RE models.

    Evaluates ``ensemble_runs`` independent disruption scenarios with randomised
    initial conditions. Reports the prevention rate (fraction of runs where
    halo ≤ limits AND RE ≤ limits AND wall damage acceptable).

    ITER limits (per ITER DDD):
        - TPF × I_halo / I_p ≤ 0.75
        - I_RE_peak ≤ 1.0 MA
        - Halo peak ≤ 3.0 MA
    """
    if ensemble_runs <= 0:
        raise ValueError(f"ensemble_runs must be > 0, got {ensemble_runs!r}")
    plasma_current_range = require_range("plasma_current_range", plasma_current_range, min_allowed=0.0)
    plasma_energy_range = require_range("plasma_energy_range", plasma_energy_range, min_allowed=0.0)
    neon_range = require_range("neon_range", neon_range, min_allowed=0.0)

    rng = np.random.default_rng(seed)

    per_run: list[dict[str, Any]] = []
    prevented_count = 0

    for run_idx in range(ensemble_runs):
        # Randomise initial conditions
        Ip_ma = rng.uniform(*plasma_current_range)
        W_mj = rng.uniform(*plasma_energy_range)
        disturbance = rng.uniform(0.0, 1.0)

        # --- AEGIS CONTROL LOOP (Simulated) ---
        # 1. Sense: Predict risk based on current state and disturbance
        # 15MA machines are inherently high risk.
        risk_score = 0.4 * (Ip_ma / 15.0) + 0.6 * disturbance

        # 2. Act: Decide on mitigation strategy
        if risk_score > 0.5:  # More sensitive trigger
            mitigation_triggered = True
            seed_re_fraction = 1e-15  # Near-total seed suppression
            tpf_suppression = 0.4
            tau_cq_base = 1.2  # Very soft quench
        else:
            mitigation_triggered = False
            seed_re_fraction = 1e-10
            tpf_suppression = 0.8
            tau_cq_base = 0.6

        # SPI Z_eff from impurity cocktail (Ne/Ar/Xe)
        from scpn_control.control.spi_mitigation import ShatteredPelletInjection

        cocktail = ShatteredPelletInjection.estimate_mitigation_cocktail(
            risk_score=risk_score,
            disturbance=disturbance,
            action_bias=1.0 if mitigation_triggered else -0.5,
        )
        impurity_total_mol = float(np.clip(cocktail["total_quantity_mol"], *neon_range))
        scale = impurity_total_mol / max(float(cocktail["total_quantity_mol"]), 1e-12)
        neon_mol = float(cocktail["neon_quantity_mol"] * scale)
        argon_mol = float(cocktail["argon_quantity_mol"] * scale)
        xenon_mol = float(cocktail["xenon_quantity_mol"] * scale)

        z_eff = ShatteredPelletInjection.estimate_z_eff_cocktail(
            neon_quantity_mol=neon_mol,
            argon_quantity_mol=argon_mol,
            xenon_quantity_mol=xenon_mol,
        )
        tau_cq = ShatteredPelletInjection.estimate_tau_cq(tau_cq_base, z_eff)

        # TPF varies with disturbance (1.5-2.5)
        tpf = (1.5 + 1.0 * disturbance) * tpf_suppression

        # Halo current model
        halo_model = HaloCurrentModel(
            plasma_current_ma=Ip_ma,
            tpf=tpf,
            contact_fraction=0.2 + 0.2 * disturbance,
        )
        halo_result = halo_model.simulate(tau_cq_s=tau_cq, duration_s=0.05)

        # Runaway electron model
        re_model = RunawayElectronModel(
            n_e=1e20,
            T_e_keV=20.0,
            z_eff=z_eff,
            neon_mol=impurity_total_mol,
        )
        # Thermal quench temperature scaling
        T_e_post = max(0.02, 1.0 * (1.0 - 0.98 * min(impurity_total_mol / 0.8, 1.0)))
        re_result = re_model.simulate(
            plasma_current_ma=Ip_ma,
            tau_cq_s=tau_cq,
            T_e_quench_keV=T_e_post,
            neon_z_eff=z_eff,
            neon_mol=impurity_total_mol,
            seed_re_fraction=seed_re_fraction,
        )

        # Prevention criteria (ITER DDD limits)
        halo_ok = halo_result.peak_halo_ma <= 3.0
        tpf_ok = halo_result.peak_tpf_product <= 0.75
        re_ok = re_result.peak_re_current_ma <= 1.0
        prevented = halo_ok and tpf_ok and re_ok

        if prevented:  # pragma: no cover — depends on physics parameter regime
            prevented_count += 1

        run_detail = {
            "run": run_idx,
            "Ip_ma": Ip_ma,
            "W_mj": W_mj,
            "neon_mol": neon_mol,
            "argon_mol": argon_mol,
            "xenon_mol": xenon_mol,
            "total_impurity_mol": impurity_total_mol,
            "disturbance": disturbance,
            "mitigation_triggered": mitigation_triggered,
            "z_eff": z_eff,
            "tau_cq_s": tau_cq,
            "tpf": tpf,
            "halo_peak_ma": halo_result.peak_halo_ma,
            "tpf_product": halo_result.peak_tpf_product,
            "wall_force_mn_m": halo_result.wall_force_mn_m,
            "re_peak_ma": re_result.peak_re_current_ma,
            "re_final_ma": re_result.final_re_current_ma,
            "avalanche_gain": re_result.avalanche_gain,
            "prevented": prevented,
        }
        per_run.append(run_detail)

        if verbose:
            status = "PREVENTED" if prevented else "FAILED"
            logger.info(
                "  Run %3d: Ip=%.1fMA impurities=%.3fmol halo=%.2fMA RE=%.3fMA -> %s",
                run_idx,
                Ip_ma,
                impurity_total_mol,
                halo_result.peak_halo_ma,
                re_result.peak_re_current_ma,
                status,
            )

    prevention_rate = prevented_count / max(ensemble_runs, 1)
    halo_peaks = [r["halo_peak_ma"] for r in per_run]
    re_peaks = [r["re_peak_ma"] for r in per_run]
    tpf_products = [r["tpf_product"] for r in per_run]
    p95_halo_peak_ma = linear_percentile(halo_peaks, 95.0)
    p95_re_peak_ma = linear_percentile(re_peaks, 95.0)

    passes_iter = prevention_rate >= 0.90 and p95_halo_peak_ma <= 3.4 and p95_re_peak_ma <= 1.0

    return DisruptionMitigationReport(
        ensemble_runs=ensemble_runs,
        prevention_rate=prevention_rate,
        mean_halo_peak_ma=float(np.mean(halo_peaks)),
        p95_halo_peak_ma=p95_halo_peak_ma,
        mean_re_peak_ma=float(np.mean(re_peaks)),
        p95_re_peak_ma=p95_re_peak_ma,
        mean_tpf_product=float(np.mean(tpf_products)),
        passes_iter_limits=passes_iter,
        per_run_details=per_run,
    )
