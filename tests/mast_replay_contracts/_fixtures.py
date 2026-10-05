# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Manufactured MAST channel contract fixtures.
"""Small manufactured arrays; no acquisition, provenance or training evidence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scpn_control._npz import save_npz_arrays
from validation.mast_replay_contracts._inputs import MEASURED_CHANNELS

STAMP = "2026-07-10T00:00:00+00:00"


def mirror() -> dict[str, NDArray[np.float64]]:
    """Return aligned native-summary/equilibrium/magnetics fixture arrays."""
    grid = np.linspace(0.0, 0.03, 8, dtype=np.float64)
    equilibrium = np.linspace(0.0, 0.03, 4, dtype=np.float64)
    fast = np.linspace(0.0, 0.03, 32, dtype=np.float64)
    angles = np.arange(12, dtype=np.float64) * 30.0
    return {
        "summary.time": grid,
        "summary.ip": np.full(8, 6.0e5, dtype=np.float64),
        "summary.line_average_n_e": np.full(8, 3.0e19, dtype=np.float64),
        "equilibrium.time": equilibrium,
        "equilibrium.q95": np.full(4, 3.8, dtype=np.float64),
        "equilibrium.beta_tor_normal": np.full(4, 1.5, dtype=np.float64),
        "equilibrium.bphi_rmag": np.full(4, -0.61, dtype=np.float64),
        "equilibrium.magnetic_axis_r": np.full(4, 0.8, dtype=np.float64),
        "equilibrium.z": np.zeros(4, dtype=np.float64),
        "magnetics.time_saddle": fast,
        "magnetics.time_mirnov": fast,
        "magnetics.b_field_tor_probe_saddle_field": np.outer(np.cos(np.deg2rad(angles)), 0.01 + 0.02 * fast),
        "magnetics.b_field_tor_probe_saddle_m_phi": np.tile(angles[:, None], (1, 2)),
        "magnetics.b_field_pol_probe_cc_field": np.tile(0.02 * fast, (2, 1)),
    }


def channels() -> dict[str, NDArray[np.float64]]:
    """Return eleven finite fixture vectors on a strictly increasing clock."""
    result: dict[str, NDArray[np.float64]] = {name: np.ones(8, dtype=np.float64) for name in MEASURED_CHANNELS}
    result["time_s"] = np.linspace(0.0, 0.03, 8, dtype=np.float64)
    return result


def write_channels(path: Path, *, identifiers: NDArray[np.float64] | NDArray[np.int64] | None = None) -> None:
    """Write genuine combined NPZ bytes through the project's public writer."""
    arrays = {f"101:{name}": value for name, value in channels().items()}
    if identifiers is None:
        identifiers = np.array([101], dtype=np.int64)
    save_npz_arrays(path, {**arrays, "shot_ids": identifiers}, allow_pickle=False)
