# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Actual local transport observation.

"""Run the actual local transport API with declared initial profile controls.

The core solver and its numerical closures are used unchanged. B0 and delta
remain unmapped in this adapter; the domain box is not an equivalent TORAX
equilibrium, and matching input profiles is not physical-reference admission.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from scpn_control.core.integrated_transport_solver import TransportSolver
from validation.code_to_code_comparison import _sha256_payload
from validation.code_to_code_scenario import initial_profiles, validate_scenario


def run_local_transport(scenario: dict[str, Any]) -> dict[str, Any]:
    """Initialize and advance the real CONTROL transport solver.

    Parameters
    ----------
    scenario : dict
        Required fields documented by validate_scenario. n_rho sets both
        radial points and the equilibrium grid; dt is seconds, P_aux is MW,
        temperatures keV and local density 10^19 m^-3.

    Returns
    -------
    dict
        Actual initial/final rho-aligned vectors and initial D/T/He densities,
        solver conservation diagnostics, arithmetic profile means, elapsed
        evolution time and a digest of normalized declared inputs. Zero steps
        returns the actual initialized state. Wall time excludes initialization.

    Raises
    ------
    ValueError
        Input validation or the real solver refuses the request.
    OSError
        A unique temporary configuration cannot be written or removed.
    Exception
        A core evolution error propagates; no replacement result is fabricated.

    Notes
    -----
    A unique configuration directory is created in the caller's cwd and
    removed on normal/error exit. Existing fixed-name configs remain untouched.
    D and T each start at half the assigned electron density; helium is zero.
    The solver's current/geometry/gyro-Bohm/source models are unchanged.
    """
    s = validate_scenario(scenario)
    cfg = {
        "reactor_name": s["name"],
        "dimensions": {
            "R_min": s["R0"] - s["a"],
            "R_max": s["R0"] + s["a"],
            "Z_min": -s["a"] * s["kappa"],
            "Z_max": s["a"] * s["kappa"],
        },
        "grid_resolution": [s["n_rho"], s["n_rho"]],
        "physics": {"plasma_current_target": s["I_p"]},
    }
    with tempfile.TemporaryDirectory(prefix="scpn-c2c-", dir=Path.cwd()) as directory:
        config = Path(directory) / "config.json"
        config.write_text(json.dumps(cfg), encoding="utf-8")
        solver = TransportSolver(config, nr=s["n_rho"], multi_ion=True)
        profiles = initial_profiles(s, np.asarray(solver.rho, dtype=np.float64))
        solver.Te = profiles["Te_initial"].copy()
        solver.Ti = profiles["Ti_initial"].copy()
        solver.ne = profiles["ne_initial"].copy()
        solver.n_D = 0.5 * solver.ne.copy()
        solver.n_T = 0.5 * solver.ne.copy()
        solver.n_He = np.zeros(solver.nr, dtype=np.float64)
        initial_species = {
            "n_D_initial": solver.n_D.tolist(),
            "n_T_initial": solver.n_T.tolist(),
            "n_He_initial": solver.n_He.tolist(),
        }
        started = time.perf_counter()
        for _ in range(s["n_steps"]):
            solver.evolve_profiles(dt=s["dt"], P_aux=s["P_aux"])
        wall_time = time.perf_counter() - started
        return {
            "code": "scpn-control",
            "scenario": s["name"],
            "scenario_sha256": _sha256_payload(s),
            "rho": solver.rho.tolist(),
            **{name: vector.tolist() for name, vector in profiles.items()},
            **initial_species,
            "Te_final": solver.Te.tolist(),
            "Ti_final": solver.Ti.tolist(),
            "ne_final": solver.ne.tolist(),
            "Te_avg": float(np.mean(solver.Te)),
            "Ti_avg": float(np.mean(solver.Ti)),
            "energy_balance_error": solver.energy_balance_error,
            "particle_balance_error": solver.particle_balance_error,
            "wall_time_s": wall_time,
            "n_steps": s["n_steps"],
            "dt": s["dt"],
            "t_final": s["n_steps"] * s["dt"],
            "transport_model": solver.transport_model,
            "unmapped_scenario_fields": ["B0", "delta"],
        }


_run_scpn_control = run_local_transport
