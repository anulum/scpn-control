# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Optional TORAX execution and retained output extraction.

"""Retain the optional TORAX entry path with an explicit runtime evidence gap.

Provider configuration/output compatibility depends on the installed TORAX.
This optional adapter is not exercised by a fake backend when TORAX is absent.
"""

from __future__ import annotations

import pprint
import time
from pathlib import Path
from typing import Any

import numpy as np

from validation.code_to_code_scenario import _torax_config_dict

TORAX_TMP_CONFIG = Path("validation/reports/_tmp_torax_c2c_config.py")


def write_torax_config(path: Path, scenario: dict[str, Any]) -> None:
    """Write a new Python configuration module for independent TORAX preparation.

    Parameters
    ----------
    path : pathlib.Path
        New caller-relative configuration filename; parent directories are
        created. Existing files, symlinks and hard links are never overwritten.
    scenario : dict
        High-level scalar controls validated by the shared scenario adapter.

    Raises
    ------
    ValueError
        A scenario scalar or count is invalid.
    OSError
        Exclusive UTF-8 creation fails, including an existing destination.

    Notes
    -----
    The module contains only CONFIG assigned to Python literals rendered by
    pprint. Current units/profile/fixed-dt mappings follow the TORAX config
    documentation; runtime/API compatibility requires the actual provider.
    Export is neither execution evidence nor physical-reference admission.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = pprint.pformat(_torax_config_dict(scenario), sort_dicts=True, width=100)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(f"CONFIG = {payload}\n")


_write_torax_config = write_torax_config


def _extract_torax_result(data_tree: Any, scenario: dict[str, Any], wall_time: float) -> dict[str, Any]:
    """Read the final time slice from the legacy TORAX DataTree shape.

    profiles.dataset must provide rho_norm coordinates and time-indexed
    T_e/T_i [keV] and n_e [m^-3]. scalars.dataset may provide W_thermal_total
    [J] and tau_E [s]. Shape/schema access and float conversion errors propagate.
    Arrays become lists and means are unweighted arithmetic sample means.
    wall_time [s] is caller-supplied; no runtime/provider authenticity,
    coordinates/shape validation or physical admission is established here.
    """
    profiles = data_tree["profiles"].dataset
    scalars = data_tree["scalars"].dataset

    rho = np.asarray(profiles.coords["rho_norm"].values, dtype=np.float64)
    te = np.asarray(profiles["T_e"].isel(time=-1).values, dtype=np.float64)
    ti = np.asarray(profiles["T_i"].isel(time=-1).values, dtype=np.float64)
    ne = np.asarray(profiles["n_e"].isel(time=-1).values, dtype=np.float64)

    result = {
        "code": "torax",
        "scenario": scenario["name"],
        "status": "done",
        "rho": rho.tolist(),
        "Te_final": te.tolist(),
        "Ti_final": ti.tolist(),
        "ne_final": ne.tolist(),
        "Te_avg": float(np.mean(te)),
        "Ti_avg": float(np.mean(ti)),
        "ne_avg_m3": float(np.mean(ne)),
        "wall_time_s": wall_time,
    }
    if "W_thermal_total" in scalars:
        result["W_thermal_total_J"] = float(scalars["W_thermal_total"].isel(time=-1).values)
    if "tau_E" in scalars:
        result["tau_E_s"] = float(scalars["tau_E"].isel(time=-1).values)
    return result


def _run_torax(scenario: dict[str, Any]) -> dict[str, Any] | None:
    """Invoke the installed provider with the configured diagnostic scenario.

    ImportError prints the unavailable-provider notice and returns None.
    Otherwise exclusively create TORAX_TMP_CONFIG, call the existing
    build_torax_config_from_file/run_simulation API and extract its last state.
    A pre-existing config is refused without removal. After successful creation,
    cleanup runs even if provider build/run/extraction fails; errors propagate.
    This legacy fixed filename is sequential, without a concurrent writer
    lock. Installed-provider/API compatibility and actual runtime must be
    independently exercised before its output is treated as provider evidence.
    """
    try:
        import torax
    except ImportError:
        print("TORAX not installed — skipping TORAX comparison.")
        print("Install with: pip install torax")
        return None

    _write_torax_config(TORAX_TMP_CONFIG, scenario)
    try:
        cfg = torax.build_torax_config_from_file(TORAX_TMP_CONFIG.resolve())
        t0 = time.perf_counter()
        data_tree, _ = torax.run_simulation(cfg, progress_bar=False)
        wall_time = time.perf_counter() - t0
        return _extract_torax_result(data_tree, scenario, wall_time)
    finally:
        TORAX_TMP_CONFIG.unlink(missing_ok=True)
