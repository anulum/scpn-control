# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disruption Contracts.
"""Synthetic disruption episode orchestration and learning."""

from __future__ import annotations

import numpy as np

from scpn_control.control._disruption_episode_physics import (
    impurity_transport_response,
    mcnp_lite_tbr,
    post_disruption_halo_runaway,
    synthetic_disruption_signal,
)
from scpn_control.control.advanced_soc_fusion_learning import FusionAIAgent
from scpn_control.control.disruption_predictor import predict_disruption_risk
from scpn_control.control.spi_mitigation import ShatteredPelletInjection
from scpn_control.core._validators import require_finite_float, require_positive_float
from scpn_control.core.global_design_scanner import GlobalDesignExplorer


def run_disruption_episode(
    *,
    rng: np.random.Generator,
    rl_agent: FusionAIAgent,
    base_tbr: float,
    explorer: GlobalDesignExplorer,
) -> dict[str, float | bool]:
    """Run one synthetic disruption episode through the RL agent and explorer.

    Parameters
    ----------
    rng
        Random generator for the episode draw.
    rl_agent
        The fusion RL agent scoring disruption risk and actions.
    base_tbr
        Baseline tritium breeding ratio; must be positive.
    explorer
        The global design explorer evaluated for the episode.

    Returns
    -------
    dict[str, float | bool]
        Episode metrics including the risk before and after mitigation and the
        mitigation outcome.
    """
    base_tbr = require_positive_float("base_tbr", base_tbr)
    disturbance = float(rng.uniform(0.0, 1.0))
    pre_energy_mj = float(rng.uniform(240.0, 420.0))
    pre_current_ma = float(rng.uniform(11.0, 16.5))
    signal, toroidal = synthetic_disruption_signal(rng=rng, disturbance=disturbance)
    risk_before = float(
        np.clip(
            require_finite_float("risk_before", predict_disruption_risk(signal, toroidal)),
            0.0,
            1.0,
        )
    )

    rl_state = rl_agent.discretize_state(12.0 * risk_before, 4.0 * disturbance)
    rl_action = int(rl_agent.choose_action(rl_state, rng))
    rl_action_bias = {-1: -1.0, 0: 0.0, 1: 1.0}[rl_action - 1]

    cocktail = ShatteredPelletInjection.estimate_mitigation_cocktail(
        risk_score=risk_before,
        disturbance=disturbance,
        action_bias=rl_action_bias,
    )
    neon_quantity_mol = float(cocktail["neon_quantity_mol"])
    argon_quantity_mol = float(cocktail["argon_quantity_mol"])
    xenon_quantity_mol = float(cocktail["xenon_quantity_mol"])
    total_impurity_mol = float(cocktail["total_quantity_mol"])
    spi = ShatteredPelletInjection(
        Plasma_Energy_MJ=pre_energy_mj,
        Plasma_Current_MA=pre_current_ma,
    )
    _, _, _, spi_diag = spi.trigger_mitigation(
        neon_quantity_mol=neon_quantity_mol,
        argon_quantity_mol=argon_quantity_mol,
        xenon_quantity_mol=xenon_quantity_mol,
        return_diagnostics=True,
        duration_s=0.03,
        dt_s=5e-5,
        verbose=False,
    )
    tau_cq_s = float(spi_diag["tau_cq_ms_mean"]) * 1e-3
    final_current_ma = float(spi_diag["final_current_MA"])
    quench_fraction = float(np.clip((pre_current_ma - final_current_ma) / pre_current_ma, 0.0, 1.0))
    mitigation_strength = float(
        np.clip(
            1.60 * total_impurity_mol + 0.03 * float(spi_diag["z_eff"]) + 0.10 * (rl_action_bias + 1.0),
            0.08,
            0.95,
        )
    )
    impurity = impurity_transport_response(
        neon_quantity_mol=neon_quantity_mol,
        argon_quantity_mol=argon_quantity_mol,
        xenon_quantity_mol=xenon_quantity_mol,
        disturbance=disturbance,
        seed_shift=rl_action,
    )
    zeff = float(0.6 * float(spi_diag["z_eff"]) + 0.4 * impurity["zeff_eff"])
    impurity_radiation_mw = float(impurity["impurity_radiation_mw"] * (0.72 + 0.28 * quench_fraction))
    post_dyn = post_disruption_halo_runaway(
        pre_current_ma=pre_current_ma,
        tau_cq_s=tau_cq_s,
        disturbance=disturbance,
        mitigation_strength=mitigation_strength,
        zeff_eff=zeff,
    )
    halo_current_ma = float(post_dyn["halo_current_ma"])
    runaway_beam_ma = float(post_dyn["runaway_beam_ma"])
    post_toroidal = {
        "toroidal_n1_amp": float(max(0.0, toroidal["toroidal_n1_amp"] * (1.0 - 0.75 * mitigation_strength))),
        "toroidal_n2_amp": float(max(0.0, toroidal["toroidal_n2_amp"] * (1.0 - 0.70 * mitigation_strength))),
        "toroidal_n3_amp": float(max(0.0, toroidal["toroidal_n3_amp"] * (1.0 - 0.65 * mitigation_strength))),
        "toroidal_asymmetry_index": float(
            max(
                0.0,
                toroidal["toroidal_asymmetry_index"] * (1.0 - 0.72 * mitigation_strength),
            )
        ),
        "toroidal_radial_spread": float(
            max(
                0.0,
                toroidal["toroidal_radial_spread"] * (1.0 - 0.60 * mitigation_strength),
            )
        ),
    }
    post_signal = np.clip(signal * (1.0 - 0.60 * mitigation_strength), 0.01, None)
    risk_after_model = float(
        np.clip(
            require_finite_float("risk_after_model", predict_disruption_risk(post_signal, post_toroidal)),
            0.0,
            1.0,
        )
    )
    risk_after = float(
        np.clip(
            0.45 * risk_after_model + 0.55 * (risk_before * (1.0 - 0.80 * mitigation_strength) + 0.03 * disturbance),
            0.0,
            1.0,
        )
    )

    wall_damage_index = float(
        np.clip(
            0.18 * halo_current_ma + 0.55 * runaway_beam_ma + 5.0e-4 * impurity_radiation_mw + 0.10 * disturbance,
            0.0,
            3.0,
        )
    )

    r_maj = float(rng.uniform(1.2, 1.6))
    b_t = float(rng.uniform(9.0, 12.0))
    ip = float(rng.uniform(3.5, 8.0))
    design = explorer.evaluate_design(r_maj, b_t, ip)
    q_proxy = float(7.5 + 0.10 * np.sqrt(max(float(design["Q"]), 0.0)) * (1.0 - 0.25 * disturbance))
    li6_enrichment = float(rng.uniform(0.85, 1.0))
    be_multiplier_fraction = float(rng.uniform(0.35, 0.95))
    reflector_albedo = float(rng.uniform(0.30, 0.90))
    tbr_proxy, _ = mcnp_lite_tbr(
        base_tbr=base_tbr,
        li6_enrichment=li6_enrichment,
        be_multiplier_fraction=be_multiplier_fraction,
        reflector_albedo=reflector_albedo,
    )

    no_wall_damage = bool(wall_damage_index < 1.10)
    objective_success = bool(q_proxy >= 10.0 and tbr_proxy >= 1.0 and no_wall_damage)
    prevented = bool(risk_after < 0.88 and no_wall_damage and runaway_beam_ma < 1.00)

    reward = (
        2.0 * float(q_proxy >= 10.0)
        + 2.0 * float(tbr_proxy >= 1.0)
        + 1.5 * float(no_wall_damage)
        + 1.2 * float(prevented)
        - 1.4 * wall_damage_index
        - 1.1 * risk_after
    )
    next_state = rl_agent.discretize_state(12.0 * risk_after, 4.0 * max(0.0, disturbance - mitigation_strength))
    rl_agent.learn(rl_state, rl_action, next_state, reward)

    return {
        "disturbance": disturbance,
        "risk_before": risk_before,
        "risk_after": risk_after,
        "neon_quantity_mol": neon_quantity_mol,
        "argon_quantity_mol": argon_quantity_mol,
        "xenon_quantity_mol": xenon_quantity_mol,
        "total_impurity_mol": total_impurity_mol,
        "zeff": zeff,
        "impurity_decay_tau_ms": float(impurity["impurity_decay_tau_ms"]),
        "halo_current_ma": halo_current_ma,
        "halo_peak_ma": float(post_dyn["halo_peak_ma"]),
        "runaway_beam_ma": runaway_beam_ma,
        "runaway_peak_ma": float(post_dyn["runaway_peak_ma"]),
        "impurity_radiation_mw": impurity_radiation_mw,
        "wall_damage_index": wall_damage_index,
        "q_proxy": q_proxy,
        "tbr_proxy": tbr_proxy,
        "no_wall_damage": no_wall_damage,
        "objective_success": objective_success,
        "prevented": prevented,
    }
