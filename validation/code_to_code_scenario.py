# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Code-to-code declared inputs and profile initialisation.

"""Define and validate the bounded high-level transport scenario.

R0/a use metres, B0 tesla, I_p amperes, temperatures keV, density 10^19 m^-3,
P_aux MW and time seconds. Shared inputs do not imply identical equilibria,
sources, composition, boundary evolution or transport closures.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from validation.code_to_code_comparison import _canonical_json, _finite_number

ITER_SCENARIO: dict[str, Any] = {
    "name": "ITER_15MA_baseline",
    "R0": 6.2,  # m
    "a": 2.0,  # m
    "B0": 5.3,  # T
    "I_p": 15.0e6,  # A
    "kappa": 1.7,
    "delta": 0.33,
    "n_e0": 10.0,  # 10^19 m^-3
    "T_e0": 10.0,  # keV
    "T_i0": 10.0,  # keV
    "P_aux": 50.0,  # MW
    "n_rho": 50,
    "dt": 0.01,  # s
    "n_steps": 100,
    "t_final": 1.0,  # s
}


def validate_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    """Normalize required scalar inputs before any solver or filesystem action.

    Parameters
    ----------
    scenario : dict
        ITER_SCENARIO keys with finite real scalars, excluding booleans.
        n_rho is an integer >=3; n_steps is an integer >=0. Zero steps with
        zero t_final observes initialisation without evolution. Other scalar
        domains are positive except nonnegative power/time and |delta|<1.

    Returns
    -------
    dict
        Independent copy with normalised numeric fields; finite JSON-compatible
        extra metadata keys are retained.

    Raises
    ------
    ValueError
        A required key is missing, a value/domain/metadata is invalid, R0<=a, or
        n_steps*dt differs from t_final beyond 1e-12 relative/absolute tolerance.
    """
    if not isinstance(scenario.get("name"), str) or not scenario["name"].strip():
        raise ValueError("scenario name must be a nonempty string")
    out = dict(scenario)
    for key in ("R0", "a", "B0", "I_p", "kappa", "delta", "n_e0", "T_e0", "T_i0", "P_aux", "dt", "t_final"):
        if not _finite_number(scenario.get(key)):
            raise ValueError("scenario scalar must be a finite real number: " + key)
        out[key] = float(scenario[key])
    for key, minimum in (("n_rho", 3), ("n_steps", 0)):
        value = scenario.get(key)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
            or not _finite_number(value)
        ):
            raise ValueError("scenario count has an invalid integer domain: " + key)
        out[key] = int(value)
    if any(out[key] <= 0 for key in ("R0", "a", "B0", "I_p", "kappa", "n_e0", "T_e0", "T_i0", "dt")):
        raise ValueError("scenario positive scalars must be greater than zero")
    if out["P_aux"] < 0 or out["t_final"] < 0 or abs(out["delta"]) >= 1:
        raise ValueError("scenario power, time or triangularity is outside its domain")
    if out["R0"] <= out["a"]:
        raise ValueError("scenario requires R0 greater than a")
    if not math.isclose(out["n_steps"] * out["dt"], out["t_final"], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("scenario t_final must equal n_steps times dt")
    try:
        _canonical_json(out)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("scenario metadata must be finite JSON-compatible values") from None
    return out


def initial_profiles(scenario: dict[str, Any], rho: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    """Sample common linear axis-to-edge profiles on a supplied radial grid.

    Parameters
    ----------
    scenario : dict
        Validated required inputs, also checked here.
    rho : numpy.ndarray
        Finite ordered normalised coordinates in [0,1], shape (N,), N>=2.

    Returns
    -------
    dict
        Te_initial/Ti_initial in keV and ne_initial in 10^19 m^-3, shape (N,).
        Each thermal edge is max(0.1,min(0.05*axis,1.0)) keV. Density edge
        is 0.1*axis. These positive edges initialise both adapters' profiles.

    Raises
    ------
    ValueError
        Scenario or radial grid is invalid.
    """
    from validation.code_to_code_comparison import _profile_coordinates_are_valid

    s = validate_scenario(scenario)
    if not _profile_coordinates_are_valid({"rho": rho}, ()):
        raise ValueError("initial profile rho must be finite, ordered and within [0,1]")
    result = {}
    for key, axis in (("Te_initial", s["T_e0"]), ("Ti_initial", s["T_i0"]), ("ne_initial", s["n_e0"])):
        edge = 0.1 * axis if key == "ne_initial" else max(0.1, min(0.05 * axis, 1.0))
        result[key] = np.asarray(axis + (edge - axis) * rho, dtype=np.float64)
    return result


def _torax_config_dict(scenario: dict[str, Any]) -> dict[str, Any]:
    """Map declared high-level inputs to the existing TORAX configuration schema.

    Temperatures are keV, densities m^-3, current A and heat power W.
    The fixed dt now equals the local declared dt. Profile edges use the same
    policy as initial_profiles. Circular geometry, Ne/Zeff1.6, constant
    transport and TORAX source/current evolution still differ from CONTROL.
    API compatibility and a successful provider run must be verified against
    the installed TORAX; constructing this mapping establishes neither.
    Unit reference: https://torax.readthedocs.io/en/stable/configuration.html.
    """
    scenario = validate_scenario(scenario)
    ion_edge_temp = max(0.1, min(float(scenario["T_i0"]) * 0.05, 1.0))
    edge_temp = max(0.1, min(float(scenario["T_e0"]) * 0.05, 1.0))
    edge_density = float(scenario["n_e0"]) * 1.0e18
    fixed_dt = float(scenario["dt"])
    return {
        "plasma_composition": {
            "main_ion": {"D": 0.5, "T": 0.5},
            "impurity": "Ne",
            "Z_eff": 1.6,
        },
        "profile_conditions": {
            "Ip": float(scenario["I_p"]),
            "T_i": {0.0: {0.0: float(scenario["T_i0"]), 1.0: ion_edge_temp}},
            "T_i_right_bc": ion_edge_temp,
            "T_e": {0.0: {0.0: float(scenario["T_e0"]), 1.0: edge_temp}},
            "T_e_right_bc": edge_temp,
            "n_e": {0.0: {0.0: float(scenario["n_e0"]) * 1.0e19, 1.0: edge_density}},
            "n_e_right_bc": edge_density,
            "nbar": float(scenario["n_e0"]) * 0.8e19,
            "n_e_nbar_is_fGW": False,
            "normalize_n_e_to_nbar": False,
            "initial_psi_mode": "j",
        },
        "numerics": {
            "t_final": float(scenario["t_final"]),
            "fixed_dt": fixed_dt,
            "evolve_ion_heat": True,
            "evolve_electron_heat": True,
            "evolve_current": True,
            "evolve_density": True,
            "resistivity_multiplier": 50.0,
            "max_dt": fixed_dt,
        },
        "geometry": {
            "geometry_type": "circular",
            "n_rho": int(scenario["n_rho"]),
            "R_major": float(scenario["R0"]),
            "a_minor": float(scenario["a"]),
            "B_0": float(scenario["B0"]),
            "elongation_LCFS": float(scenario["kappa"]),
        },
        "neoclassical": {
            "bootstrap_current": {},
        },
        "sources": {
            "generic_current": {
                "fraction_of_total_current": 0.15,
                "gaussian_width": 0.075,
                "gaussian_location": 0.36,
            },
            "generic_particle": {
                "S_total": 0.0,
                "deposition_location": 0.3,
                "particle_width": 0.25,
            },
            "gas_puff": {
                "S_total": 0.0,
                "puff_decay_length": 0.3,
            },
            "pellet": {
                "S_total": 0.0,
                "pellet_width": 0.1,
                "pellet_deposition_location": 0.85,
            },
            "generic_heat": {
                "P_total": float(scenario["P_aux"]) * 1.0e6,
                "electron_heat_fraction": 0.5,
                "gaussian_location": 0.2,
                "gaussian_width": 0.25,
            },
            "fusion": {},
            "ei_exchange": {},
            "ohmic": {},
        },
        "pedestal": {},
        "transport": {
            "model_name": "constant",
        },
        "solver": {
            "solver_type": "linear",
        },
        "time_step_calculator": {
            "calculator_type": "fixed",
        },
    }
