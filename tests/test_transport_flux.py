# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Conservative signed face transport tests.

"""Check cylindrical divergence, signed inventory exchange and rejection without clipping."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, cast

import numpy as np
import pytest

from scpn_control._typing import AnyFloatArray
from scpn_control.core.transport_flux import (
    TransportFaceFlux,
    TransportFaceGeometry,
    TransportFluxState,
    advance_face_flux,
)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_internal_face_transfer_conserves_particles_and_energy(direction: float) -> None:
    """One interior face transfers equal counts/J between neighbours, with zero boundary exchange."""
    rho = np.linspace(0, 1, 7)
    density = np.stack([np.full(7, x) for x in (5.0, 3.0, 1.0, 0.5)])
    temperature = np.stack((np.full(7, 2.0), np.full(7, 3.0)))
    particle = np.zeros((4, 6))
    particle[:2, 2] = direction * 1e20
    heat = np.zeros((2, 6))
    heat[:, 2] = direction * np.array([1e5, 2e5])
    result = advance_face_flux(
        rho=rho,
        major_radius_m=6.0,
        minor_radius_m=2.0,
        density=density,
        temperature=temperature,
        impurity_density=np.zeros(7),
        flux=TransportFaceFlux(particle, heat),
        dt=1e-3,
    )
    volume = 4 * np.pi**2 * 6 * 4 * rho / 6
    area = 4 * np.pi**2 * 6 * 2 * (rho[2] + rho[3]) / 2
    expected = density.copy()
    expected[:2, 2] -= 1e-3 * area * direction * 1e20 / (volume[2] * 1e19)
    expected[:2, 3] += 1e-3 * area * direction * 1e20 / (volume[3] * 1e19)
    np.testing.assert_allclose(result.density, expected, rtol=1e-14)
    assert result.balance.particle_boundary == (0.0, 0.0, 0.0, 0.0)
    assert result.balance.energy_boundary_j == 0
    assert result.balance.particle_relative_error < 1e-14
    assert result.balance.energy_relative_error < 1e-14
    np.testing.assert_array_equal(density[1], np.full(7, 3.0))
    np.testing.assert_array_equal(temperature[0], np.full(7, 2.0))


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_linear_radial_flux_matches_constant_divergence(direction: float) -> None:
    """F=C*rho yields independently known dn/dt=-2*C/a in the cylindrical interior."""
    rho = np.linspace(0, 1, 9)
    particle = np.zeros((4, 8))
    particle[:2] = direction * 1e19 * (rho[1:] + rho[:-1]) / 2
    density = np.stack([np.full(9, x) for x in (5.0, 5.0, 0.0, 0.0)])
    result = advance_face_flux(
        rho=rho,
        major_radius_m=6.0,
        minor_radius_m=2.0,
        density=density,
        temperature=np.full((2, 9), 2.0),
        impurity_density=np.zeros(9),
        flux=TransportFaceFlux(particle, np.zeros((2, 8))),
        dt=0.01,
    )
    np.testing.assert_allclose(result.density[:2, 1:-1], 5.0 - direction * 0.01, rtol=1e-14)
    np.testing.assert_allclose(result.density[0, 1:-1] * result.temperature[0, 1:-1], 10.0, rtol=1e-14)
    assert np.sign(result.balance.particle_boundary[0]) == -direction
    assert result.balance.particle_relative_error < 1e-14
    assert result.balance.energy_relative_error < 1e-14


def test_nonambipolar_flux_is_rejected_without_mutation() -> None:
    """A charge-imbalanced particle prescription must not be repaired by overwriting electrons."""
    density = np.stack([np.full(5, x) for x in (5.0, 5.0, 0.0, 0.0)])
    before = density.copy()
    flux = np.zeros((4, 4))
    flux[1] = 1e19
    with pytest.raises(ValueError, match="ambipolar"):
        advance_face_flux(
            rho=np.linspace(0, 1, 5),
            major_radius_m=6.0,
            minor_radius_m=2.0,
            density=density,
            temperature=np.ones((2, 5)),
            impurity_density=np.zeros(5),
            flux=TransportFaceFlux(flux, np.zeros((2, 4))),
            dt=0.01,
        )
    np.testing.assert_array_equal(density, before)


def test_zero_time_preserves_temperatures_and_profiles_exactly() -> None:
    """Zero-time validation does not impose the axis constraint or round-trip temperatures."""
    density = np.stack([np.linspace(1, 2, 5), np.linspace(1, 2, 5), np.zeros(5), np.zeros(5)])
    temperature = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.3, 0.5, 0.7, 0.9, 1.1]])
    result = advance_face_flux(
        rho=np.linspace(0, 1, 5),
        major_radius_m=6.0,
        minor_radius_m=2.0,
        density=density,
        temperature=temperature,
        impurity_density=np.zeros(5),
        flux=TransportFaceFlux(np.zeros((4, 4)), np.zeros((2, 4))),
        dt=0.0,
    )
    np.testing.assert_array_equal(result.density, density)
    np.testing.assert_array_equal(result.temperature, temperature)


@pytest.mark.parametrize(
    ("major_radius", "minor_radius", "temperature"),
    [(1e308, 2.0, 1.0), (6.0, 1e200, 1.0), (6.0, 1e-200, 1.0), (6.0, 2.0, 1e308)],
)
def test_unrepresentable_geometry_or_energy_is_rejected_without_mutation(
    major_radius: float, minor_radius: float, temperature: float
) -> None:
    """Finite public inputs must not leak arithmetic exceptions or silently accept zero cell volumes."""
    density = np.stack([np.full(5, x) for x in (5.0, 5.0, 0.0, 0.0)])
    temperatures = np.full((2, 5), temperature)
    before = density.copy()
    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="not representable"):
            advance_face_flux(
                rho=np.linspace(0, 1, 5),
                major_radius_m=major_radius,
                minor_radius_m=minor_radius,
                density=density,
                temperature=temperatures,
                impurity_density=np.zeros(5),
                flux=TransportFaceFlux(np.zeros((4, 4)), np.zeros((2, 4))),
                dt=0.0,
            )
        assert all(value == "raise" for value in np.geterr().values())
    np.testing.assert_array_equal(density, before)
    np.testing.assert_array_equal(temperatures, np.full((2, 5), temperature))


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("relative_offset", [0.0, 5e-13])
def test_near_depletion_checks_final_charge(direction: float, relative_offset: float) -> None:
    """Initially tolerated charge error must not become admissible large relative error after depletion."""
    rho = np.linspace(0, 1, 5)
    density = np.stack([np.full(5, x) for x in (5 * (1 + relative_offset), 5.0, 0.0, 0.0)])
    before = density.copy()
    volume = 4 * np.pi**2 * 6 * 4 * rho / 4
    area = 4 * np.pi**2 * 6 * 2 * (rho[1:] + rho[:-1]) / 2
    particle = np.zeros((4, 4))
    donor = 1 if direction > 0 else 2
    particle[:2, 1] = direction * (5 - 1e-14) * 1e19 * volume[donor] / area[1]

    def advance() -> TransportFluxState:
        """Exercise the public step with identical physical input for acceptance and refusal cases."""
        return advance_face_flux(
            rho=rho,
            major_radius_m=6.0,
            minor_radius_m=2.0,
            density=density,
            temperature=np.ones((2, 5)),
            impurity_density=np.zeros(5),
            flux=TransportFaceFlux(particle, np.zeros((2, 4))),
            dt=1.0,
        )

    if relative_offset > 0:
        with pytest.raises(ValueError, match="Final.*quasineutral"):
            advance()
    else:
        result = advance()
        np.testing.assert_allclose(result.density[0], result.density[1], rtol=1e-12, atol=0)
        assert result.balance.particle_relative_error < 1e-12
        assert result.balance.energy_relative_error < 1e-12
    np.testing.assert_array_equal(density, before)


@pytest.fixture
def face_state() -> dict[str, Any]:
    """Supply a neutral electron/deuterium state with zero face exchange on five nodes."""
    return {
        "rho": np.linspace(0, 1, 5),
        "major_radius_m": 6.0,
        "minor_radius_m": 2.0,
        "density": np.stack([np.full(5, value) for value in (5.0, 5.0, 0.0, 0.0)]),
        "temperature": np.ones((2, 5)),
        "impurity_density": np.zeros(5),
        "flux": TransportFaceFlux(np.zeros((4, 4)), np.zeros((2, 4))),
        "dt": 0.01,
    }


@pytest.mark.parametrize("field", ["major_radius_m", "minor_radius_m", "dt"])
@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
def test_invalid_geometry_scalars_refuse_before_profile_mutation(
    face_state: dict[str, Any], field: str, value: float
) -> None:
    """Negative and nonfinite dimensions or timesteps never enter the conservative update."""
    face_state[field] = value
    before = face_state["density"].copy()
    with pytest.raises(ValueError, match="geometry or timestep"):
        advance_face_flux(**face_state)
    np.testing.assert_array_equal(face_state["density"], before)


@pytest.mark.parametrize("field", ["rho", "density", "temperature", "impurity_density", "particle_m2_s", "energy_w_m2"])
@pytest.mark.parametrize("imaginary", [0.0, 1.0])
def test_complex_face_inputs_refuse_before_real_conversion(
    face_state: dict[str, Any], field: str, imaginary: float
) -> None:
    """Complex profiles and fluxes cannot become a different real-valued transport problem through casting."""
    flux = face_state["flux"]
    if field in ("particle_m2_s", "energy_w_m2"):
        values = getattr(flux, field).astype(np.complex128)
        values.flat[0] += imaginary * 1j
        face_state["flux"] = TransportFaceFlux(
            values if field == "particle_m2_s" else flux.particle_m2_s,
            values if field == "energy_w_m2" else flux.energy_w_m2,
        )
    else:
        face_state[field] = face_state[field].astype(np.complex128)
        face_state[field].flat[0] += imaginary * 1j
    arrays = [face_state[name] for name in ("rho", "density", "temperature", "impurity_density")]
    arrays.extend((face_state["flux"].particle_m2_s, face_state["flux"].energy_w_m2))
    before = [array.copy() for array in arrays]
    with pytest.raises(ValueError, match="profiles and fluxes must be real"):
        advance_face_flux(**face_state)
    for actual, expected in zip(arrays, before, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("rho", [np.array(0.0), np.array([0.0, 1.0]), np.zeros((5, 1)), np.linspace(0, 2, 5)])
def test_invalid_radial_grid_refuses_before_profile_mutation(
    face_state: dict[str, Any], rho: np.ndarray[Any, np.dtype[Any]]
) -> None:
    """A scalar, short, multidimensional or unnormalized grid cannot define face positions."""
    face_state["rho"] = rho
    before = face_state["density"].copy()
    with pytest.raises(ValueError, match="geometry or timestep"):
        advance_face_flux(**face_state)
    np.testing.assert_array_equal(face_state["density"], before)


@pytest.mark.parametrize("field", ["volume", "area"])
def test_extended_precision_geometry_underflow_refuses_without_mutation(face_state: dict[str, Any], field: str) -> None:
    """Representable high-precision weights may not collapse to zero at float64 admission."""
    volume = np.array([0, 1, 1, 1, 1], dtype=object)
    area = np.array([1, 1, 1, 1], dtype=object)
    if field == "volume":
        volume[2] = Decimal("1e-400")
    else:
        area[1] = Decimal("1e-400")
    face_state["geometry"] = TransportFaceGeometry(cast(AnyFloatArray, volume), cast(AnyFloatArray, area))
    density_before = face_state["density"].copy()
    temperature_before = face_state["temperature"].copy()
    with pytest.raises(ValueError, match="geometry underflows to zero"):
        advance_face_flux(**face_state)
    np.testing.assert_array_equal(face_state["density"], density_before)
    np.testing.assert_array_equal(face_state["temperature"], temperature_before)


def test_complex_geometry_refuses_before_real_conversion(face_state: dict[str, Any]) -> None:
    """A complex cell weight cannot be silently projected onto a real transport grid."""
    volume = np.array([0, 1, 1, 1, 1], dtype=np.complex128)
    area = np.ones(4)
    face_state["geometry"] = TransportFaceGeometry(cast(AnyFloatArray, volume), area)
    before = face_state["density"].copy()
    with pytest.raises(ValueError, match="geometry must be real"):
        advance_face_flux(**face_state)
    np.testing.assert_array_equal(face_state["density"], before)


@pytest.mark.parametrize("field", ["density", "temperature", "impurity_density"])
@pytest.mark.parametrize("kind", ["shape", "negative", "nan", "infinite"])
def test_invalid_profile_refuses_without_changing_input_arrays(
    face_state: dict[str, Any], field: str, kind: str
) -> None:
    """Malformed state cannot broadcast, lose finiteness or inject negative density/temperature."""
    value = face_state[field].copy()
    if kind == "shape":
        value = value[..., :-1]
    else:
        value.flat[0] = {"negative": -1.0, "nan": np.nan, "infinite": np.inf}[kind]
    face_state[field] = value
    before = {name: array.copy() for name, array in face_state.items() if isinstance(array, np.ndarray)}
    with pytest.raises(ValueError, match="state or array shape"):
        advance_face_flux(**face_state)
    for name, original in before.items():
        np.testing.assert_array_equal(face_state[name], original)


@pytest.mark.parametrize("channel", ["particle_m2_s", "energy_w_m2"])
@pytest.mark.parametrize("kind", ["shape", "nan", "infinite"])
def test_invalid_face_flux_refuses_without_changing_profiles_or_flux(
    face_state: dict[str, Any], channel: str, kind: str
) -> None:
    """Each supplied flux channel must have the complete finite face matrix."""
    original = face_state["flux"]
    values = getattr(original, channel).copy()
    if kind == "shape":
        values = values[:, :-1]
    else:
        values.flat[0] = np.nan if kind == "nan" else np.inf
    flux = TransportFaceFlux(
        values if channel == "particle_m2_s" else original.particle_m2_s,
        values if channel == "energy_w_m2" else original.energy_w_m2,
    )
    face_state["flux"] = flux
    before = face_state["density"].copy()
    saved = values.copy()
    with pytest.raises(ValueError, match="state or array shape"):
        advance_face_flux(**face_state)
    np.testing.assert_array_equal(face_state["density"], before)
    np.testing.assert_array_equal(getattr(flux, channel), saved)


@pytest.mark.parametrize("offset", [-1e-6, 1e-6])
def test_initial_charge_error_is_not_projected_onto_electrons(face_state: dict[str, Any], offset: float) -> None:
    """Both signs of excess initial charge are refused without repairing the caller's state."""
    face_state["density"][0, 2] += offset
    before = face_state["density"].copy()
    with pytest.raises(ValueError, match="Initial densities must be quasineutral"):
        advance_face_flux(**face_state)
    np.testing.assert_array_equal(face_state["density"], before)


@pytest.mark.parametrize("channel", ["particle", "energy"])
def test_finite_face_flux_overflow_refuses_without_mutation(face_state: dict[str, Any], channel: str) -> None:
    """Finite physical flux that overflows its integrated face rate yields a public ValueError."""
    particles = np.zeros((4, 4))
    energy = np.zeros((2, 4))
    if channel == "particle":
        particles[:2, 1] = 1e308
    else:
        energy[:, 1] = 1e308
    face_state["flux"] = TransportFaceFlux(particles, energy)
    before = face_state["density"].copy()
    saved = (particles.copy(), energy.copy())
    with pytest.raises(ValueError, match="Face-flux update is not representable"):
        advance_face_flux(**face_state)
    np.testing.assert_array_equal(face_state["density"], before)
    np.testing.assert_array_equal(particles, saved[0])
    np.testing.assert_array_equal(energy, saved[1])


@pytest.mark.parametrize("failure", ["particle_depletion", "energy_depletion", "zero_capacity"])
def test_inadmissible_updated_state_is_refused_without_mutation(face_state: dict[str, Any], failure: str) -> None:
    """An excessive finite transfer or empty interior cannot return clipped or undefined state."""
    particle = np.zeros((4, 4))
    energy = np.zeros((2, 4))
    if failure == "particle_depletion":
        particle[:2, 1] = 1e25
    elif failure == "energy_depletion":
        energy[:, 1] = 1e20
    else:
        face_state["density"][:] = 0.0
    face_state["flux"] = TransportFaceFlux(particle, energy)
    before = {name: array.copy() for name, array in face_state.items() if isinstance(array, np.ndarray)}
    with pytest.raises(ValueError, match="negative state or zero thermal capacity"):
        advance_face_flux(**face_state)
    for name, original in before.items():
        np.testing.assert_array_equal(face_state[name], original)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("dt", [0.25, 1.0])
def test_subnormal_density_loss_is_bounded_by_inventory_tolerance(direction: float, dt: float) -> None:
    """Accumulated lost subnormal updates above the ledger tolerance are refused, not clipped.

    This is an extreme floating-point contract probe on caller-supplied positive
    weights, not a calibrated equilibrium. Per-cell increments round to zero;
    the independently computed net boundary count remains representable.
    """
    size = 20001
    rho = np.linspace(0, 1, size)
    density = np.stack([np.full(size, value) for value in (1e-320, 1e-320, 0.0, 0.0)])
    temperature = np.ones((2, size))
    impurity = np.zeros(size)
    volume = np.full(size, 1e289)
    volume[[0, -1]] = 0.0
    particle = np.zeros((4, size - 1))
    particle[:2] = direction * np.arange(size - 1) * 1e-16
    energy = np.zeros((2, size - 1))
    geometry = TransportFaceGeometry(volume, np.ones(size - 1))
    saved = [array.copy() for array in (density, temperature, impurity, volume, particle, energy)]
    expected_boundary = -dt * direction * (size - 2) * 1e-16

    def advance() -> TransportFluxState:
        """Run the public conservative step with the retained extreme inputs."""
        return advance_face_flux(
            rho=rho,
            major_radius_m=6.0,
            minor_radius_m=2.0,
            density=density,
            temperature=temperature,
            impurity_density=impurity,
            flux=TransportFaceFlux(particle, energy),
            dt=dt,
            geometry=geometry,
        )

    if abs(expected_boundary) > 1e-12:
        with pytest.raises(ValueError, match="inventory is nonfinite or fails conservation"):
            advance()
    else:
        result = advance()
        np.testing.assert_array_equal(result.density, density)
        assert result.balance.particle_boundary[:2] == pytest.approx((expected_boundary, expected_boundary), abs=0)
        assert result.balance.particle_relative_error == pytest.approx(abs(expected_boundary), abs=0)
        assert result.balance.particle_relative_error < 1e-12
    for array, original in zip((density, temperature, impurity, volume, particle, energy), saved, strict=True):
        np.testing.assert_array_equal(array, original)
