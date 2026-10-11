# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Synthetic diagnostics contract tests.

"""Exercise the legacy noisy diagnostic examples and their explicitly ignored inputs."""

from __future__ import annotations

import numpy as np
import pytest

from validation.multi_machine_validation import (
    iter_15ma,
)


def test_synthetic_diagnostics() -> None:
    """Exercise actual synthetic profile sampling; shapes/positivity do not prove diagnostic fidelity."""
    machine = iter_15ma()
    rho = np.linspace(0, 1, 50, dtype=np.float64)
    temperature = machine.Te_profile(rho)
    density = machine.ne_profile(rho)
    thomson = machine.diagnostics.thomson_scattering(temperature, density, 20)
    assert len(thomson["Te_keV"]) == len(thomson["ne_19"]) == 20
    sxr = machine.diagnostics.soft_xray(temperature, density, rho, 40)
    assert len(sxr) == 40 and np.all(sxr > 0)


def test_ece_legacy_samples_ignore_magnetic_profile() -> None:
    """Keep the documented temperature-only sampling distinct from an ECE forward model."""
    diagnostics = iter_15ma().diagnostics
    temperature = np.linspace(1.0, 10.0, 50, dtype=np.float64)
    field = np.full(50, 5.3)
    state = np.random.get_state()
    try:
        np.random.seed(17)
        first = diagnostics.ece_radiometer(temperature, field, 12)
        np.random.seed(17)
        second = diagnostics.ece_radiometer(temperature, field * 2.0, 12)
    finally:
        np.random.set_state(state)
    assert first.shape == (12,)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(temperature, np.linspace(1.0, 10.0, 50, dtype=np.float64))


@pytest.mark.parametrize("diagnostic", ["interferometer", "bolometer"])
def test_legacy_chord_examples_scale_with_radius_and_ignore_rho(diagnostic: str) -> None:
    """Exercise the documented geometric proxy without claiming radial integration fidelity."""
    diagnostics = iter_15ma().diagnostics
    sample = diagnostics.interferometer if diagnostic == "interferometer" else diagnostics.bolometer
    profile = np.linspace(1.0, 10.0, 50, dtype=np.float64)
    rho = np.linspace(0.0, 1.0, 50, dtype=np.float64)
    state = np.random.get_state()
    try:
        np.random.seed(17)
        first = sample(profile, rho, 0.5, 8)
        np.random.seed(17)
        doubled_radius = sample(profile, rho, 1.0, 8)
        np.random.seed(17)
        reversed_grid = sample(profile, rho[::-1], 0.5, 8)
    finally:
        np.random.set_state(state)
    assert first.shape == (8,)
    np.testing.assert_allclose(doubled_radius, first * 2.0)
    np.testing.assert_array_equal(first, reversed_grid)


def test_legacy_magnetic_examples_ignore_machine_geometry() -> None:
    """Random constants retain their stated shapes and cannot establish machine-dependent magnetics."""
    diagnostics = iter_15ma().diagnostics
    state = np.random.get_state()
    try:
        np.random.seed(17)
        first = diagnostics.magnetics(6.2, 2.0)
        np.random.seed(17)
        second = diagnostics.magnetics(1.67, 0.67)
    finally:
        np.random.set_state(state)
    assert first["flux_loops"].shape == (20,)
    assert first["b_probes"].shape == (30,)
    np.testing.assert_array_equal(first["flux_loops"], second["flux_loops"])
    np.testing.assert_array_equal(first["b_probes"], second["b_probes"])
    assert first["Ip"] == second["Ip"]
