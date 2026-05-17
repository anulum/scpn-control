# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — code-to-code benchmark tests
# © 1998–2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────
"""Unit coverage for the TORAX code-to-code benchmark adapter."""

from __future__ import annotations

import numpy as np
import xarray as xr
from xarray import DataTree

from validation import code_to_code_benchmark as c2c


def test_torax_config_maps_scenario_fields() -> None:
    cfg = c2c._torax_config_dict(c2c.ITER_SCENARIO)

    assert cfg["profile_conditions"]["Ip"] == c2c.ITER_SCENARIO["I_p"]
    assert cfg["geometry"]["R_major"] == c2c.ITER_SCENARIO["R0"]
    assert cfg["geometry"]["a_minor"] == c2c.ITER_SCENARIO["a"]
    assert cfg["geometry"]["B_0"] == c2c.ITER_SCENARIO["B0"]
    assert cfg["geometry"]["n_rho"] == c2c.ITER_SCENARIO["n_rho"]
    assert cfg["sources"]["generic_heat"]["P_total"] == c2c.ITER_SCENARIO["P_aux"] * 1.0e6


def test_extract_torax_result_reads_profiles_and_scalars() -> None:
    time = np.array([0.0, 1.0])
    rho = np.array([0.0, 0.5, 1.0])
    profiles = xr.Dataset(
        {
            "T_e": (("time", "rho_norm"), np.array([[1.0, 0.8, 0.2], [8.0, 5.0, 1.0]])),
            "T_i": (("time", "rho_norm"), np.array([[1.1, 0.9, 0.2], [7.0, 4.0, 0.8]])),
            "n_e": (("time", "rho_norm"), np.array([[1e20, 8e19, 3e19], [9e19, 7e19, 2e19]])),
        },
        coords={"time": time, "rho_norm": rho},
    )
    scalars = xr.Dataset(
        {
            "W_thermal_total": ("time", np.array([1.0e8, 2.5e8])),
            "tau_E": ("time", np.array([0.7, 1.2])),
        },
        coords={"time": time},
    )
    tree = DataTree.from_dict({"/profiles": profiles, "/scalars": scalars})

    result = c2c._extract_torax_result(tree, c2c.ITER_SCENARIO, wall_time=0.25)

    assert result["status"] == "done"
    assert result["Te_avg"] == np.mean([8.0, 5.0, 1.0])
    assert result["Ti_avg"] == np.mean([7.0, 4.0, 0.8])
    assert result["tau_E_s"] == 1.2
    assert result["W_thermal_total_J"] == 2.5e8


def test_compare_results_includes_ion_temperature_metrics() -> None:
    scpn = {
        "rho": [0.0, 0.5, 1.0],
        "Te_final": [8.0, 5.0, 1.0],
        "Ti_final": [7.0, 4.0, 1.0],
    }
    torax = {
        "rho": [0.0, 1.0],
        "Te_final": [7.0, 1.0],
        "Ti_final": [6.0, 1.0],
        "tau_E_s": 1.1,
    }

    result = c2c._compare_results(scpn, torax)

    assert result["comparison"]["Te_rmse_keV"] > 0.0
    assert result["comparison"]["Ti_rmse_keV"] > 0.0
    assert result["comparison"]["torax_tau_E_s"] == 1.1


def test_compare_results_keeps_electron_metrics_when_torax_omits_ion_profile() -> None:
    scpn = {
        "rho": [0.0, 0.5, 1.0],
        "Te_final": [8.0, 5.0, 1.0],
        "Ti_final": [7.0, 4.0, 1.0],
    }
    torax = {
        "rho": [0.0, 1.0],
        "Te_final": [7.0, 1.0],
        "tau_E_s": 1.1,
    }

    result = c2c._compare_results(scpn, torax)

    assert result["comparison"]["Te_rmse_keV"] > 0.0
    assert "Ti_rmse_keV" not in result["comparison"]
    assert result["comparison"]["torax_tau_E_s"] == 1.1
