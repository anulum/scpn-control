# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Multi-Ion Species Evolution Tests
"""Tests for the multi-ion (D/T/He-ash) species evolution kernel.

Covers the fusion helium source, helium pumping, the quasineutral electron
density and effective charge, tungsten line radiation, the CFL sub-stepping
stability, and the edge boundary conditions extracted from the integrated
transport solver.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.core.plasma_power_terms import bosch_hale_dt_reactivity
from scpn_control.core.species_evolution import (
    SpeciesEvolutionResult,
    evolve_multi_ion_species,
)


def _state() -> dict[str, object]:
    """Build a well-conditioned multi-ion evolution input state."""
    rho = np.linspace(0.0, 1.0, 21)
    drho = float(rho[1] - rho[0])
    return {
        "n_D": 0.5 * np.ones_like(rho),
        "n_T": 0.5 * np.ones_like(rho),
        "n_He": 0.05 * np.ones_like(rho),
        "Ti": 15.0 * (1.0 - rho**2) + 0.5,
        "Te": 15.0 * (1.0 - rho**2) + 0.5,
        "n_impurity": 0.001 * np.ones_like(rho),
        "dV": 4.0 * np.pi**2 * 3.0 * 1.0**2 * rho * drho,
        "rho": rho,
        "a_minor": 1.0,
        "drho": drho,
        "D_species": 0.3,
        "tau_He": 2.5,
        "dt": 0.001,
    }


def _evolve_state(state: dict[str, object]) -> SpeciesEvolutionResult:
    """Call the public kernel with an editable input matrix, including invalid cases."""
    return evolve_multi_ion_species(
        n_D=cast(AnyFloatArray, state["n_D"]),
        n_T=cast(AnyFloatArray, state["n_T"]),
        n_He=cast(AnyFloatArray, state["n_He"]),
        Ti=cast(AnyFloatArray, state["Ti"]),
        Te=cast(AnyFloatArray, state["Te"]),
        n_impurity=cast(AnyFloatArray, state["n_impurity"]),
        dV=cast(AnyFloatArray, state["dV"]),
        rho=cast(AnyFloatArray, state["rho"]),
        drho=cast(float, state["drho"]),
        a_minor=cast(float, state["a_minor"]),
        D_species=cast(float | AnyFloatArray, state["D_species"]),
        tau_He=cast(float, state["tau_He"]),
        dt=cast(float, state["dt"]),
    )


class TestEvolveMultiIonSpecies:
    """Exercise species sources, boundaries and physical diffusion geometry."""

    def test_returns_finite_result_of_expected_shape(self) -> None:
        """A well-conditioned step returns finite densities and diagnostics."""
        state = _state()
        result = _evolve_state(state)
        assert isinstance(result, SpeciesEvolutionResult)
        shape = np.asarray(state["n_D"]).shape
        for arr in (result.n_D, result.n_T, result.n_He, result.ne, result.S_He, result.P_rad_line):
            assert arr.shape == shape
            assert np.all(np.isfinite(arr))
        assert np.isfinite(result.Z_eff)
        assert np.isfinite(result.particle_balance_error)

    def test_electron_density_follows_quasineutrality(self) -> None:
        """Electron density follows D + T + 2·He + Z_W·impurity without inventing a minimum charge density."""
        state = _state()
        result = _evolve_state(state)
        expected = result.n_D + result.n_T + 2.0 * result.n_He + 10.0 * np.maximum(np.asarray(state["n_impurity"]), 0.0)
        np.testing.assert_allclose(result.ne, expected, rtol=1e-12)

    def test_helium_source_is_integrated_burn_rate(self) -> None:
        """S_He averages the integrated burn instead of extrapolating its initial rate."""
        state = _state()
        state["D_species"] = 0.0
        n_D = np.asarray(state["n_D"])
        n_T = np.asarray(state["n_T"])
        sigmav = bosch_hale_dt_reactivity(np.asarray(state["Ti"]))
        expected_S_He = n_D * n_T * 1e19 * sigmav / (1.0 + n_D * 1e19 * sigmav * 0.001)
        result = _evolve_state(state)
        np.testing.assert_allclose(result.S_He, expected_S_He, rtol=1e-12)
        assert np.all(result.S_He > 0.0)  # burning plasma produces helium

    def test_z_eff_is_clipped_to_physical_range(self) -> None:
        """Z_eff stays within [1, 10] even for a heavy-impurity state."""
        state = _state()
        state["n_impurity"] = 5.0 * np.ones_like(np.asarray(state["n_D"]))  # heavy tungsten load
        result = _evolve_state(state)
        assert 1.0 <= result.Z_eff <= 10.0

    def test_tungsten_radiation_scales_with_impurity(self) -> None:
        """More tungsten impurity radiates more line power."""
        low = _state()
        high = _state()
        high["n_impurity"] = 10.0 * np.asarray(low["n_impurity"])
        r_low = _evolve_state(low)
        r_high = _evolve_state(high)
        assert float(np.sum(r_high.P_rad_line)) > float(np.sum(r_low.P_rad_line))

    def test_helium_pumping_removes_ash_without_fusion(self) -> None:
        """With no fusion (T absent) helium ash is depleted by pumping."""
        state = _state()
        state["n_T"] = np.zeros_like(np.asarray(state["n_D"]))  # no D-T reactions
        state["tau_He"] = 0.5  # strong pumping
        n_He_before = float(np.sum(np.asarray(state["n_He"])))
        result = _evolve_state(state)
        assert np.all(result.S_He == 0.0)  # no fusion -> no helium source
        assert float(np.sum(result.n_He)) < n_He_before

    def test_edge_boundary_conditions(self) -> None:
        """Deuterium/tritium hold an edge recycling floor; helium edge is pumped to zero."""
        state = _state()
        result = _evolve_state(state)
        assert result.n_D[-1] == 0.01
        assert result.n_T[-1] == 0.01
        assert result.n_He[-1] == 0.0

    def test_cfl_substepping_stays_stable_for_large_dt(self) -> None:
        """A large dt triggers many CFL sub-steps yet stays finite and bounded."""
        state = _state()
        state["dt"] = 1.0  # >> dt_CFL, forces hundreds of sub-steps
        result = _evolve_state(state)
        assert np.all(np.isfinite(result.n_D))
        assert np.all(np.isfinite(result.n_He))
        assert np.all(result.n_D >= 0.001)
        assert np.all(result.n_He >= 0.0)

    def test_zero_diffusivity_is_handled(self) -> None:
        """Zero diffusivity requires no diffusion substeps."""
        state = _state()
        state["D_species"] = 0.0
        result = _evolve_state(state)
        assert np.all(np.isfinite(result.n_D))
        assert np.all(np.isfinite(result.n_He))


@pytest.mark.parametrize("a_minor", [0.05, 0.1, 2.0])
def test_diffusion_substeps_preserve_density_maximum(a_minor: float) -> None:
    """Diffusion with nonnegative burn losses obeys the density maximum principle."""
    rho = np.linspace(0.0, 1.0, 21)
    initial = 0.01 + 2.5 * (1.0 - rho**2)
    initial[0] = initial[1]
    result = evolve_multi_ion_species(
        n_D=initial,
        n_T=np.zeros(21),
        n_He=np.zeros(21),
        Ti=np.ones(21),
        Te=np.ones(21),
        n_impurity=np.zeros(21),
        dV=4.0 * np.pi**2 * 6.0 * a_minor**2 * rho * 0.05,
        rho=rho,
        drho=0.05,
        a_minor=a_minor,
        D_species=0.3,
        tau_He=2.5,
        dt=0.01,
    )
    assert np.all(result.n_D >= 0.01)
    assert np.all(result.n_D <= initial.max())
    assert result.n_D[10] < initial[10]
    assert np.all(result.S_He >= 0.0)
    np.testing.assert_allclose(result.fusion_reactions, 0.01 * result.S_He, rtol=1e-13)


@pytest.mark.parametrize(
    "field,value",
    [
        ("a_minor", 0.0),
        ("a_minor", -1.0),
        ("a_minor", np.nan),
        ("D_species", -1.0),
        ("D_species", np.inf),
        ("dt", -0.1),
        ("dt", np.nan),
        ("drho", 0.1),
        ("rho", np.linspace(0.0, 1.0, 21) ** 2),
    ],
)
def test_invalid_diffusion_geometry_is_rejected(field: str, value: object) -> None:
    """Malformed physical and normalized geometry cannot produce a species result."""
    rho = np.linspace(0.0, 1.0, 21)
    numeric = float(np.asarray(value)) if field != "rho" else 0.0
    with pytest.raises(ValueError):
        evolve_multi_ion_species(
            n_D=np.ones(21),
            n_T=np.ones(21),
            n_He=np.zeros(21),
            Ti=np.ones(21),
            Te=np.ones(21),
            n_impurity=np.zeros(21),
            dV=4.0 * np.pi**2 * 6.0 * rho * 0.05,
            rho=np.asarray(value, dtype=np.float64) if field == "rho" else rho,
            drho=numeric if field == "drho" else 0.05,
            a_minor=numeric if field == "a_minor" else 1.0,
            D_species=numeric if field == "D_species" else 0.3,
            tau_He=2.5,
            dt=numeric if field == "dt" else 0.001,
        )


@pytest.mark.parametrize("deuterium,tritium", [(1.0, 1.0), (0.01, 10.0), (10.0, 0.01)])
def test_burn_cannot_create_helium_beyond_available_fuel(deuterium: float, tritium: float) -> None:
    """The public source step preserves D/T/He stoichiometry through strong burn."""
    rho = np.linspace(0.0, 1.0, 21)
    temperature = np.full(21, 15.0)
    rate = float(bosch_hale_dt_reactivity(temperature)[10]) * 1e19
    result = evolve_multi_ion_species(
        n_D=np.full(21, deuterium),
        n_T=np.full(21, tritium),
        n_He=np.zeros(21),
        Ti=temperature,
        Te=temperature,
        n_impurity=np.zeros(21),
        dV=4.0 * np.pi**2 * 6.0 * rho * 0.05,
        rho=rho,
        drho=0.05,
        a_minor=1.0,
        D_species=0.0,
        tau_He=np.inf,
        dt=10.0 / rate,
    )
    assert 0.0 <= result.n_He[10] <= min(deuterium, tritium)
    assert deuterium - result.n_D[10] == pytest.approx(result.n_He[10], abs=1e-14)
    assert tritium - result.n_T[10] == pytest.approx(result.n_He[10], abs=1e-14)

    assert result.fusion_reactions[10] == pytest.approx(result.n_He[10], rel=1e-13)
    assert result.helium_pumped[10] == 0.0
    if deuterium == tritium:
        assert result.n_D[10] == pytest.approx(deuterium / (1.0 + 10.0 * deuterium), rel=1e-13)
    else:
        log_ratio = np.log(result.n_D[10] / result.n_T[10])
        assert log_ratio == pytest.approx(np.log(deuterium / tritium) + (deuterium - tritium) * 10.0, rel=1e-13)


def test_helium_pumping_matches_exponential_survival() -> None:
    """Source-free ash decays exponentially even when dt exceeds the pumping time."""
    rho = np.linspace(0.0, 1.0, 21)
    result = evolve_multi_ion_species(
        n_D=np.ones(21),
        n_T=np.zeros(21),
        n_He=np.ones(21),
        Ti=np.ones(21),
        Te=np.ones(21),
        n_impurity=np.zeros(21),
        dV=4.0 * np.pi**2 * 6.0 * rho * 0.05,
        rho=rho,
        drho=0.05,
        a_minor=1.0,
        D_species=0.0,
        tau_He=0.5,
        dt=1.0,
    )
    assert result.n_He[10] == pytest.approx(np.exp(-2.0), rel=1e-13)
    assert result.n_T[10] == 0.0
    assert result.helium_pumped[10] == pytest.approx(1.0 - np.exp(-2.0), rel=1e-13)
    assert result.fusion_reactions[10] == 0.0


def test_zero_timestep_preserves_species_and_reports_no_reactions() -> None:
    """A zero-duration call neither resets edge densities nor produces particles."""
    rho = np.linspace(0.0, 1.0, 21)
    deuterium, tritium, helium = np.full(21, 2.0), np.ones(21), np.full(21, 0.1)
    result = evolve_multi_ion_species(
        n_D=deuterium,
        n_T=tritium,
        n_He=helium,
        Ti=np.ones(21),
        Te=np.ones(21),
        n_impurity=np.zeros(21),
        dV=4.0 * np.pi**2 * 6.0 * rho * 0.05,
        rho=rho,
        drho=0.05,
        a_minor=1.0,
        D_species=0.3,
        tau_He=2.5,
        dt=0.0,
    )
    for initial, output in ((deuterium, result.n_D), (tritium, result.n_T), (helium, result.n_He)):
        np.testing.assert_array_equal(output, initial)
        assert not np.shares_memory(output, initial)
    np.testing.assert_array_equal(result.fusion_reactions, np.zeros(21))
    np.testing.assert_array_equal(result.helium_pumped, np.zeros(21))
    np.testing.assert_array_equal(result.S_He, np.zeros(21))
    assert result.particle_balance_error == 0.0


@pytest.mark.parametrize("tau", [0.0, -1.0, np.nan])
def test_invalid_pumping_time_is_rejected(tau: float) -> None:
    """An invalid removal timescale cannot produce a species-evolution result."""
    rho = np.linspace(0.0, 1.0, 21)
    with pytest.raises(ValueError, match="tau_He"):
        evolve_multi_ion_species(
            n_D=np.ones(21),
            n_T=np.ones(21),
            n_He=np.zeros(21),
            Ti=np.ones(21),
            Te=np.ones(21),
            n_impurity=np.zeros(21),
            dV=4.0 * np.pi**2 * 6.0 * rho * 0.05,
            rho=rho,
            drho=0.05,
            a_minor=1.0,
            D_species=0.0,
            tau_He=tau,
            dt=1.0,
        )


def test_split_burn_and_pumping_converge_to_coupled_source_reference() -> None:
    """Public repeated steps converge at first order to independently integrated ash survival."""
    from scipy.integrate import quad

    rho = np.linspace(0.0, 1.0, 21)
    temperature = np.full(21, 15.0)
    rate = float(bosch_hale_dt_reactivity(temperature)[10]) * 1e19
    integral, _ = quad(lambda time: np.exp(time) / (1.0 + time) ** 2, 0.0, 1.0, epsabs=1e-12)
    reference = np.exp(-1.0) * (0.1 + integral)
    errors = []
    for steps in (8, 16, 32):
        deuterium: FloatArray = np.ones(21)
        tritium: FloatArray = np.ones(21)
        helium: FloatArray = np.full(21, 0.1)
        reactions, pumped = 0.0, 0.0
        for _ in range(steps):
            result = evolve_multi_ion_species(
                n_D=deuterium,
                n_T=tritium,
                n_He=helium,
                Ti=temperature,
                Te=temperature,
                n_impurity=np.zeros(21),
                dV=4.0 * np.pi**2 * 6.0 * rho * 0.05,
                rho=rho,
                drho=0.05,
                a_minor=1.0,
                D_species=0.0,
                tau_He=1.0 / rate,
                dt=1.0 / (rate * steps),
            )
            reactions += result.fusion_reactions[10]
            pumped += result.helium_pumped[10]
            deuterium, tritium, helium = result.n_D, result.n_T, result.n_He
        assert deuterium[10] == pytest.approx(0.5, rel=1e-13)
        assert tritium[10] == pytest.approx(0.5, rel=1e-13)
        assert reactions == pytest.approx(0.5, rel=1e-13)
        assert helium[10] + pumped == pytest.approx(0.1 + reactions, rel=1e-13)
        errors.append(reference - helium[10])
    assert all(error > 0.0 for error in errors)
    assert 1.8 < errors[0] / errors[1] < 2.2
    assert 1.8 < errors[1] / errors[2] < 2.2


@pytest.mark.parametrize("slope", [-0.5, 0.5])
def test_open_particle_boundary_is_not_a_conservation_error(slope: float) -> None:
    """The cylindrical diffusive flux and imposed recycling explain open-system inventory changes."""
    rho = np.linspace(0.0, 1.0, 21)
    density = 2.0 + slope * rho**2
    density[0] = density[1]
    result = evolve_multi_ion_species(
        n_D=density,
        n_T=np.zeros(21),
        n_He=np.zeros(21),
        Ti=np.ones(21),
        Te=np.ones(21),
        n_impurity=np.zeros(21),
        dV=4.0 * np.pi**2 * 6.0 * 2.0**2 * rho * 0.05,
        rho=rho,
        drho=0.05,
        a_minor=2.0,
        D_species=0.3,
        tau_He=np.inf,
        dt=0.001,
    )
    assert result.particle_balance_error < 1e-12
    record = result.particle_balance
    volumes = 4.0 * np.pi**2 * 6.0 * 2.0**2 * rho * 0.05
    expected_initial = float(np.sum(density * 1e19 * volumes))
    outer_face = 0.5 * (rho[-2] + rho[-1])
    expected_flux = 0.001 * 1e19 * 4.0 * np.pi**2 * 6.0 * (2.0 * 0.3 * slope * outer_face**2)
    expected_edge = 1e19 * volumes[-1] * (0.02 - density[-1])
    assert record.initial_count == pytest.approx(expected_initial, rel=1e-13)
    assert record.diffusive_boundary_count == pytest.approx(expected_flux, rel=1e-12)
    assert record.prescribed_boundary_count == pytest.approx(expected_edge, rel=1e-13)
    assert record.final_count == pytest.approx(expected_initial + expected_flux + expected_edge, rel=1e-13)
    assert record.fusion_count == 0.0 and record.pumped_count == 0.0
    assert record.numerical_correction_count == 0.0
    assert record.relative_error == result.particle_balance_error


@pytest.mark.parametrize("a_minor", [0.5, 2.0])
@pytest.mark.parametrize("dt", [0.001, 0.1])
def test_reactive_substeps_close_particle_inventory(a_minor: float, dt: float) -> None:
    """Actual burn, pumping and open boundaries account for the inventory across CFL substeps."""
    rho = np.linspace(0.0, 1.0, 21)
    volumes = 4.0 * np.pi**2 * 6.0 * a_minor**2 * rho * 0.05
    result = evolve_multi_ion_species(
        n_D=2.0 - rho**2,
        n_T=0.6 + 0.2 * rho**2,
        n_He=np.full(21, 0.2),
        Ti=np.full(21, 15.0),
        Te=np.full(21, 5.0),
        n_impurity=np.zeros(21),
        dV=volumes,
        rho=rho,
        drho=0.05,
        a_minor=a_minor,
        D_species=0.3,
        tau_He=0.5,
        dt=dt,
    )
    record = result.particle_balance
    assert record.fusion_count == pytest.approx(float(np.sum(result.fusion_reactions * 1e19 * volumes)), rel=1e-13)
    assert record.pumped_count == pytest.approx(float(np.sum(result.helium_pumped * 1e19 * volumes)), rel=1e-13)
    assert record.fusion_count > 0.0 and record.pumped_count > 0.0
    expected = (
        record.initial_count
        - record.fusion_count
        - record.pumped_count
        + record.diffusive_boundary_count
        + record.prescribed_boundary_count
    )
    assert record.final_count == pytest.approx(expected, rel=1e-12)
    assert record.numerical_correction_count == 0.0
    assert record.relative_error < 1e-12


@pytest.mark.parametrize("volume_shape", ["zero", "flat", "nonzero_axis", "wrong_edge"])
def test_inconsistent_cylindrical_volume_weights_are_rejected(volume_shape: str) -> None:
    """Particle evidence rejects quadrature that cannot match the radial face geometry."""
    rho = np.linspace(0.0, 1.0, 21)
    volumes = 4.0 * np.pi**2 * 6.0 * rho * 0.05
    if volume_shape == "zero":
        volumes[:] = 0.0
    elif volume_shape == "flat":
        volumes[:] = 1.0
    elif volume_shape == "nonzero_axis":
        volumes[0] = 1.0
    else:
        volumes[-1] *= 2.0
    with pytest.raises(ValueError, match="dV"):
        evolve_multi_ion_species(
            n_D=np.ones(21),
            n_T=np.ones(21),
            n_He=np.zeros(21),
            Ti=np.ones(21),
            Te=np.ones(21),
            n_impurity=np.zeros(21),
            dV=volumes,
            rho=rho,
            drho=0.05,
            a_minor=1.0,
            D_species=0.3,
            tau_He=0.5,
            dt=0.001,
        )


@pytest.mark.parametrize("density", [0.0, 1e-5, 0.01])
@pytest.mark.parametrize("dt", [0.0, 0.001])
def test_dilute_species_preserve_charge_density(density: float, dt: float) -> None:
    """Vacuum and dilute inputs retain their actual ion charge, including at zero dt."""
    state = _state()
    state.update(
        n_D=np.full(21, density), n_T=np.zeros(21), n_He=np.zeros(21), n_impurity=np.zeros(21), D_species=0.0, dt=dt
    )
    result = _evolve_state(state)
    np.testing.assert_array_equal(result.ne, result.n_D + result.n_T + 2 * result.n_He)
    np.testing.assert_array_equal(result.ne[1:-1], np.full(19, density))


@pytest.mark.parametrize("beta", [-0.5, 0.5])
@pytest.mark.parametrize("radius", [0.5, 2.0])
def test_radial_diffusivity_matches_manufactured_operator(beta: float, radius: float) -> None:
    """Quadratic D and density have an independently derived cylindrical increment."""
    rho = np.linspace(0.0, 1.0, 41)
    spacing, dt, slope, base = 1 / 40, 1e-5, 0.5, 0.3
    density = 2 + slope * rho**2
    density[0] = density[1]
    zeros = np.zeros(41)
    diffusivity = base * (1 + beta * rho**2)
    result = evolve_multi_ion_species(
        n_D=density,
        n_T=zeros,
        n_He=zeros,
        Ti=np.ones(41),
        Te=np.ones(41),
        n_impurity=zeros,
        dV=4 * np.pi**2 * 6 * radius**2 * rho * spacing,
        rho=rho,
        drho=spacing,
        a_minor=radius,
        D_species=diffusivity,
        tau_He=np.inf,
        dt=dt,
    )
    # Arithmetic face D adds 3/4 beta*drho² to the continuous quadratic operator.
    rate = 4 * slope * base / radius**2 * (1 + 2 * beta * rho**2 + 0.75 * beta * spacing**2)
    np.testing.assert_allclose(result.n_D[2:-1], (density + dt * rate)[2:-1], rtol=1e-13, atol=1e-14)
    assert result.particle_balance.relative_error < 1e-12
    assert result.particle_balance.numerical_correction_count == 0.0
    np.testing.assert_array_equal(diffusivity, base * (1 + beta * rho**2))


@pytest.mark.parametrize("coefficient", [0.0, 0.3, 3.0])
def test_constant_diffusivity_profile_matches_scalar(coefficient: float) -> None:
    """A constant profile preserves the scalar species source and CFL trajectory."""
    state = _state()
    state["D_species"] = coefficient
    scalar = _evolve_state(state)
    state["D_species"] = np.full(21, coefficient)
    profile = _evolve_state(state)
    for name in ("n_D", "n_T", "n_He", "ne", "fusion_reactions", "helium_pumped"):
        np.testing.assert_array_equal(getattr(scalar, name), getattr(profile, name))
    assert scalar.particle_balance == profile.particle_balance


def test_sharp_diffusivity_profile_respects_physical_cfl() -> None:
    """High local diffusivity must not cause clipping hidden by nonnegativity recovery."""
    state = _state()
    state.update(
        D_species=np.where(np.arange(21) % 2, 3.0, 0.0),
        a_minor=0.5,
        dt=0.02,
        n_D=np.where(np.arange(21) % 2, 2.0, 0.02),
        n_T=np.zeros(21),
        n_He=np.zeros(21),
    )
    result = _evolve_state(state)
    assert np.min(result.n_D) >= 0.0
    assert np.max(result.n_D) <= 2.0
    assert result.particle_balance.numerical_correction_count == 0.0
    assert result.particle_balance.relative_error < 1e-12


@pytest.mark.parametrize(
    "diffusivity",
    [
        np.ones(20),
        np.ones((21, 1)),
        np.full(21, -0.1),
        np.full(21, np.nan),
        np.ones(21, dtype=bool),
        np.ones(21, dtype=complex),
    ],
)
def test_invalid_diffusivity_profiles_are_rejected(diffusivity: np.ndarray[Any, np.dtype[Any]]) -> None:
    """Diffusivity must be a real nonnegative scalar or a matching real profile."""
    state = _state()
    state["D_species"] = diffusivity
    with pytest.raises(ValueError, match="D_species"):
        _evolve_state(state)


@pytest.mark.parametrize("profile", [False, True])
def test_diffusivity_outside_working_range_is_rejected(profile: bool) -> None:
    """Finite wider coefficients must fail before float64 conversion or state mutation."""
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("longdouble has no wider finite range on this platform")
    state = _state()
    value = np.longdouble(np.finfo(np.float64).max) * 2
    coefficient = np.full(21, value, dtype=np.longdouble) if profile else value
    state["D_species"] = coefficient
    before = {name: value.copy() for name, value in state.items() if isinstance(value, np.ndarray)}
    assert np.all(np.isfinite(coefficient))
    with np.errstate(over="raise", invalid="raise"), pytest.raises(ValueError, match="D_species"):
        _evolve_state(state)
    for name, original in before.items():
        np.testing.assert_array_equal(state[name], original)


@pytest.mark.parametrize("profile", [False, True])
def test_representable_wider_diffusivity_preserves_trajectory(profile: bool) -> None:
    """Representable longdouble inputs retain the float64 physical trajectory."""
    state = _state()
    state["D_species"] = 0.25
    expected = _evolve_state(state)
    state["D_species"] = np.full(21, 0.25, dtype=np.longdouble) if profile else np.longdouble(0.25)
    actual = _evolve_state(state)
    for name in ("n_D", "n_T", "n_He", "ne", "fusion_reactions", "helium_pumped"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert actual.particle_balance == expected.particle_balance


@pytest.mark.parametrize("field", ["n_D", "n_T", "n_He", "Ti", "Te", "n_impurity", "dV"])
@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, True, 1.0 + 0.0j])
def test_invalid_profile_values_refuse_without_mutating_any_input(field: str, value: float | bool | complex) -> None:
    """Every physical profile rejects nonfinite, negative or non-real data before evolution."""
    state = _state()
    state[field] = np.full(21, value)
    before = {name: array.copy() for name, array in state.items() if isinstance(array, np.ndarray)}
    with pytest.raises(ValueError, match=field):
        _evolve_state(state)
    for name, original in before.items():
        np.testing.assert_array_equal(state[name], original)


@pytest.mark.parametrize("field", ["n_D", "n_T", "n_He", "Ti", "Te", "n_impurity", "dV"])
@pytest.mark.parametrize("shape", [(), (20,), (21, 1)])
def test_profile_shape_mismatch_refuses_without_mutating_any_input(field: str, shape: tuple[int, ...]) -> None:
    """Scalars, truncated profiles and column vectors cannot broadcast into radial state."""
    state = _state()
    state[field] = np.ones(shape)
    before = {name: array.copy() for name, array in state.items() if isinstance(array, np.ndarray)}
    expected = "rho and drho" if field == "n_D" else field
    with pytest.raises(ValueError, match=expected):
        _evolve_state(state)
    for name, original in before.items():
        np.testing.assert_array_equal(state[name], original)
