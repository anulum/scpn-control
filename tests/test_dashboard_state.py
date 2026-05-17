# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Dashboard state tests
# © 1998–2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────
"""Tests for Streamlit-independent dashboard state preparation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dashboard.state import (
    MACHINE_PRESETS,
    derived_machine_metrics,
    shot_phase_label,
    synthetic_profiles,
)


def test_machine_presets_have_required_operational_fields() -> None:
    required = {
        "R0",
        "a",
        "B0",
        "Ip",
        "kappa",
        "delta",
        "q95",
        "ne_1e19",
        "Te0_keV",
        "P_aux_MW",
        "description",
    }

    assert {"DIII-D", "SPARC", "ITER", "NSTX-U", "JET"} <= set(MACHINE_PRESETS)
    for name, machine in MACHINE_PRESETS.items():
        assert required <= machine.keys(), name
        assert machine["R0"] > machine["a"] > 0.0
        assert machine["B0"] > 0.0
        assert machine["Ip"] > 0.0
        assert machine["Te0_keV"] > 0.0
        assert machine["ne_1e19"] > 0.0


@pytest.mark.parametrize(
    ("frac", "phase"),
    [
        (0.0, "STARTUP"),
        (0.10, "RAMP-UP"),
        (0.25, "FLATTOP"),
        (0.75, "RAMP-DOWN"),
        (0.90, "TERMINATION"),
    ],
)
def test_shot_phase_boundaries(frac: float, phase: str) -> None:
    assert shot_phase_label(frac) == phase


def test_derived_machine_metrics_are_dimensionally_consistent() -> None:
    metrics = derived_machine_metrics(MACHINE_PRESETS["ITER"])

    assert metrics["epsilon"] == pytest.approx(2.0 / 6.2)
    assert metrics["aspect_ratio"] == pytest.approx(6.2 / 2.0)
    assert metrics["b_pol_t"] == pytest.approx(15.0 * 0.4 / 2.0)


def test_synthetic_profiles_are_finite_monotone_and_phase_scaled() -> None:
    machine = MACHINE_PRESETS["SPARC"]
    startup = synthetic_profiles(machine, n_rho=32, time_frac=0.02)
    flattop = synthetic_profiles(machine, n_rho=32, time_frac=0.5)

    for profiles in (startup, flattop):
        rho = profiles["rho"]
        assert rho.shape == (32,)
        assert np.all(np.diff(rho) > 0.0)
        assert np.all((rho >= 0.0) & (rho <= 1.0))
        for key in ("Te_keV", "Ti_keV", "ne_1e19"):
            assert profiles[key].shape == (32,)
            assert np.all(np.isfinite(profiles[key]))
            assert np.all(profiles[key] > 0.0)
            assert profiles[key][0] > profiles[key][-1]

    assert float(np.mean(flattop["Te_keV"])) > float(np.mean(startup["Te_keV"]))


@pytest.mark.parametrize("n_rho", [0, 1])
def test_synthetic_profiles_reject_invalid_grid_size(n_rho: int) -> None:
    with pytest.raises(ValueError, match="n_rho"):
        synthetic_profiles(MACHINE_PRESETS["DIII-D"], n_rho=n_rho)


@pytest.mark.parametrize("time_frac", [-0.01, math.nan, math.inf, 1.01])
def test_synthetic_profiles_reject_invalid_time_fraction(time_frac: float) -> None:
    with pytest.raises(ValueError, match="time_frac"):
        synthetic_profiles(MACHINE_PRESETS["DIII-D"], time_frac=time_frac)
