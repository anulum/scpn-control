# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — real OMAS IDS ecosystem contract tests.

"""Exercise real ODS structures beyond the equilibrium adapter's narrow slice."""

from __future__ import annotations

import numpy as np
import pytest

omas = pytest.importorskip(
    "omas",
    reason="real OMAS tests execute in the Ubuntu 3.12 fusion-data CI lane",
)
ODS = omas.ODS

NRHO = 50


def _write_core_profiles(ods: ODS) -> dict[str, np.ndarray]:
    rho = np.linspace(0.0, 1.0, NRHO)
    te = 10e3 * (1.0 - rho**2)
    ti = 8e3 * (1.0 - rho**2)
    ne = 1e20 * (1.0 - 0.8 * rho**2)
    q = 1.0 + 2.0 * rho**2
    profile = ods["core_profiles.profiles_1d.0"]
    profile["grid.rho_tor_norm"] = rho
    profile["electrons.temperature"] = te
    profile["electrons.density_thermal"] = ne
    profile["ion.0.temperature"] = ti
    profile["q"] = q
    return {"rho": rho, "te": te, "ti": ti, "ne": ne, "q": q}


def test_real_omas_core_profiles_round_trip() -> None:
    """Round-trip representative core-profile arrays in a real ODS."""
    ods = ODS(consistency_check=True)
    expected = _write_core_profiles(ods)
    profile = ods["core_profiles.profiles_1d.0"]
    np.testing.assert_allclose(profile["grid.rho_tor_norm"], expected["rho"])
    np.testing.assert_allclose(profile["electrons.temperature"], expected["te"])
    np.testing.assert_allclose(profile["electrons.density_thermal"], expected["ne"])
    np.testing.assert_allclose(profile["ion.0.temperature"], expected["ti"])
    np.testing.assert_allclose(profile["q"], expected["q"])


def test_real_omas_equilibrium_1d_profiles_round_trip() -> None:
    """Round-trip representative equilibrium 1-D profiles in a real ODS."""
    ods = ODS(consistency_check=True)
    psi = np.linspace(-5.0, -1.0, NRHO)
    pressure = np.linspace(1.0e5, 0.0, NRHO)
    q = np.linspace(1.0, 4.0, NRHO)
    profile = ods["equilibrium.time_slice.0.profiles_1d"]
    profile["psi"] = psi
    profile["pressure"] = pressure
    profile["q"] = q
    np.testing.assert_allclose(profile["psi"], psi)
    np.testing.assert_allclose(profile["pressure"], pressure)
    np.testing.assert_allclose(profile["q"], q)


def test_real_omas_time_arrays_are_indexed_consistently() -> None:
    """Keep related real ODS time arrays aligned across several slices."""
    ods = ODS(consistency_check=True)
    for index, time_s in enumerate((0.1, 0.4, 0.9)):
        ods.set_time_array("equilibrium.time", index, time_s)
        ods.set_time_array("equilibrium.vacuum_toroidal_field.b0", index, -2.0 - index * 0.1)
        ods[f"equilibrium.time_slice.{index}.time"] = time_s
    np.testing.assert_allclose(ods["equilibrium.time"], [0.1, 0.4, 0.9])
    np.testing.assert_allclose(ods["equilibrium.vacuum_toroidal_field.b0"], [-2.0, -2.1, -2.2])
    np.testing.assert_allclose(ods["equilibrium.time_slice"].time(), [0.1, 0.4, 0.9])
