# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Signed TGLF physical unit conversion tests.

"""Check SI dimensions and reference scaling on actual captured provider output."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scpn_control.core.tglf_flux import read_tglf_fluxes
from scpn_control.core.tglf_units import TGLFReferenceUnits, physical_tglf_flux

_FIXTURE = Path(__file__).parent / "data/tglf/default"
_REFERENCE = TGLFReferenceUnits(5e19, 2.0, 1.0, 2 * 1.67262192369e-27, 2.0)


def test_real_flux_si_conversion_preserves_inward_particle_transport() -> None:
    """Independent gyrofrequency derivation agrees without division by a gradient."""
    raw = read_tglf_fluxes(_FIXTURE)
    converted = physical_tglf_flux(raw, _REFERENCE)
    temperature_j = 2000 * 1.602176634e-19
    velocity = np.sqrt(temperature_j / _REFERENCE.mass_kg)
    gyrofrequency = 1.602176634e-19 * 2 / _REFERENCE.mass_kg
    particle_unit = 5e19 * velocity * (velocity / gyrofrequency) ** 2
    np.testing.assert_allclose(
        converted.particle_m2_s, np.array([-1.9938544745721216, -1.9938544745721214]) * particle_unit
    )
    np.testing.assert_allclose(
        converted.energy_w_m2, np.array([16.209564881227283, 38.411116503174036]) * particle_unit * temperature_j
    )
    assert converted.particle_m2_s[0] < 0 < converted.energy_w_m2[0]
    assert raw.particle_flux_gb == pytest.approx((-1.9938544745721216, -1.9938544745721214), rel=1e-14)


@pytest.mark.parametrize(
    ("field", "factor", "particle_ratio", "energy_ratio"),
    [
        ("density_m3", 2.0, 2.0, 2.0),
        ("temperature_kev", 4.0, 8.0, 32.0),
        ("length_m", 2.0, 0.25, 0.25),
        ("mass_kg", 4.0, 2.0, 2.0),
        ("magnetic_field_t", 2.0, 0.25, 0.25),
    ],
)
def test_reference_scale_dimensions(field: str, factor: float, particle_ratio: float, energy_ratio: float) -> None:
    """The physical flux follows independent n*T^(3/2)*sqrt(m)/(B²*a²) scaling."""
    raw = read_tglf_fluxes(_FIXTURE)
    first = physical_tglf_flux(raw, _REFERENCE)
    second = physical_tglf_flux(raw, replace(_REFERENCE, **{field: getattr(_REFERENCE, field) * factor}))
    np.testing.assert_allclose(second.particle_m2_s, np.asarray(first.particle_m2_s) * particle_ratio)
    np.testing.assert_allclose(second.energy_w_m2, np.asarray(first.energy_w_m2) * energy_ratio)


@pytest.mark.parametrize("field", ["density_m3", "temperature_kev", "length_m", "mass_kg", "magnetic_field_t"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_invalid_reference_is_rejected(field: str, value: float) -> None:
    """A missing or nonphysical reference must not become a guessed normalization."""
    with pytest.raises(ValueError, match=field):
        replace(_REFERENCE, **{field: value})


def test_overflowing_physical_scale_is_rejected() -> None:
    """Finite extreme references must not emit infinite physical fluxes."""
    raw = read_tglf_fluxes(_FIXTURE)
    with pytest.raises(ValueError, match="flux"):
        physical_tglf_flux(raw, replace(_REFERENCE, density_m3=1e308))


@pytest.mark.parametrize("channel", ["particle_flux_gb", "energy_flux_gb"])
def test_nonzero_si_moment_underflow_is_rejected(channel: str) -> None:
    """Actual provider data with one tiny signed channel cannot silently lose that channel."""
    original = read_tglf_fluxes(_FIXTURE)
    raw = (
        replace(original, particle_flux_gb=(-1e-100, 1e-100))
        if channel == "particle_flux_gb"
        else replace(original, energy_flux_gb=(-1e-100, 1e-100))
    )
    with pytest.raises(ValueError, match="underflows to zero"):
        physical_tglf_flux(raw, replace(_REFERENCE, density_m3=1e-300))


def test_representable_subnormal_si_flux_and_exact_zero_are_preserved() -> None:
    """Working subnormal flux is admissible without an arbitrary epsilon floor."""
    raw = replace(read_tglf_fluxes(_FIXTURE), particle_flux_gb=(-1e-4, 0.0), energy_flux_gb=(-1e-4, 0.0))
    result = physical_tglf_flux(raw, replace(_REFERENCE, density_m3=1e-300))
    assert 0 < abs(result.energy_w_m2[0]) < np.finfo(float).tiny
    assert result.energy_w_m2[0] < 0 and result.particle_m2_s[0] < 0
    assert result.particle_m2_s[1] == result.energy_w_m2[1] == 0


@pytest.mark.parametrize("field", ["density_m3", "temperature_kev", "length_m", "mass_kg", "magnetic_field_t"])
@pytest.mark.parametrize("value", [False, True])
def test_boolean_reference_is_not_a_physical_scale(field: str, value: bool) -> None:
    """Boolean values cannot silently become zero or one in physical normalization."""
    with pytest.raises(ValueError, match=field):
        replace(_REFERENCE, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [("length_m", 1e-200), ("magnetic_field_t", float.fromhex("0x0.0000000000001p-1022"))],
)
def test_reference_arithmetic_failure_has_a_public_value_error(field: str, value: float) -> None:
    """Finite positive references that overflow a square or lose the denominator are refused."""
    raw = read_tglf_fluxes(_FIXTURE)
    reference = replace(_REFERENCE, **{field: value})
    with pytest.raises(ValueError, match="reference flux scales are not representable"):
        physical_tglf_flux(raw, reference)
    assert raw == read_tglf_fluxes(_FIXTURE)
    assert getattr(reference, field) == value


@pytest.mark.parametrize(
    ("particle", "energy"),
    [((), ()), ((1.0,), (1.0,)), ((1.0, -1.0), (1.0,)), ((1.0, -1.0), (1.0, 2.0, 3.0))],
)
def test_species_cardinality_is_rejected_before_conversion(
    particle: tuple[float, ...], energy: tuple[float, ...]
) -> None:
    """Missing electron/ion pairs and unequal moment cardinalities never yield partial SI output."""
    raw = replace(read_tglf_fluxes(_FIXTURE), particle_flux_gb=particle, energy_flux_gb=energy)
    with pytest.raises(ValueError, match="must have the same species"):
        physical_tglf_flux(raw, _REFERENCE)
    assert raw.particle_flux_gb == particle
    assert raw.energy_flux_gb == energy


@pytest.mark.parametrize("channel", ["particle_flux_gb", "energy_flux_gb"])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_finite_provider_moment_cannot_overflow_si_output(channel: str, sign: float) -> None:
    """Finite signed normalized input that overflows a valid SI scale is refused without mutation."""
    original = read_tglf_fluxes(_FIXTURE)
    moments = (sign * 1e308, 0.0)
    raw = replace(original, **{channel: moments})
    with pytest.raises(ValueError, match="physical flux is not finite"):
        physical_tglf_flux(raw, _REFERENCE)
    assert getattr(raw, channel) == moments
    assert original == read_tglf_fluxes(_FIXTURE)
