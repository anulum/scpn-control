# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Physical Miller input and real GACODE qualification.

"""Exercise physical deck construction, analytic shape derivatives and actual provider execution."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scpn_control.core.tglf_miller import TGLFMillerGeometry, TGLFSpecies, miller_tglf_deck
from scpn_control.core.tglf_units import TGLFReferenceUnits, physical_tglf_flux
from scpn_control.core.transport_flux import TransportFaceFlux, advance_face_flux
from validation.tglf_launcher import TGLFFluxSolver

_REFERENCE = TGLFReferenceUnits(5e19, 2.0, 2.0, 2 * 1.67262192369e-27, 2.0)
_ELECTRON = TGLFSpecies(-1, 9.1093837139e-31, 5e19, 2.0, -2.5e19, -3.0)
_MAIN = TGLFSpecies(1, _REFERENCE.mass_kg, 5e19, 3.0, -2.5e19, -3.0)
_GEOMETRY = TGLFMillerGeometry(1.0, 6.0, 2.0, 2.0, 1.7, 0.3)
_COLLISION_RATE = 34195.32091515598


def _parse_deck(text: str) -> dict[str, str]:
    """Read the public serialized deck for independent dimensional assertions."""
    return dict(line.split("=", 1) for line in text.splitlines())


def test_physical_mapping_matches_independent_gaussian_units() -> None:
    """Reference ratios, gradients, Debye length and pressure normalization agree with cgs arithmetic."""
    text = miller_tglf_deck(_REFERENCE, (_ELECTRON, _MAIN), _GEOMETRY, electron_collision_rate_s=_COLLISION_RATE)
    deck = _parse_deck(text)
    assert text == miller_tglf_deck(
        _REFERENCE, (_ELECTRON, _MAIN), _GEOMETRY, electron_collision_rate_s=_COLLISION_RATE
    )
    assert float(deck["RLNS_1"]) == 1
    assert float(deck["RLTS_1"]) == 3
    assert float(deck["RLTS_2"]) == 2
    assert float(deck["Q_PRIME_LOC"]) == 16
    assert float(deck["AS_1"]) == float(deck["AS_2"]) == 1
    assert float(deck["TAUS_2"]) == 1.5
    assert float(deck["RMIN_LOC"]) == 0.5 and float(deck["RMAJ_LOC"]) == 3
    ne_cm3, a_cm, b_gauss, q, r_cm = 5e13, 200.0, 2e4, 2.0, 100.0
    erg_per_ev = 1.602176634e-12
    pressure_gradient_ev_cm4 = -(ne_cm3 / a_cm) * (2000 * 4 + 3000 * 3)
    expected_p = erg_per_ev / b_gauss**2 * q * a_cm**2 / r_cm * pressure_gradient_ev_cm4
    expected_beta = 8 * np.pi * ne_cm3 * 2000 * erg_per_ev / b_gauss**2
    assert float(deck["P_PRIME_LOC"]) == pytest.approx(expected_p, rel=1e-14)
    assert float(deck["BETAE"]) == pytest.approx(expected_beta, rel=1e-14)
    cs_cm_s = np.sqrt(2000 * erg_per_ev / (_REFERENCE.mass_kg * 1000))
    assert float(deck["XNUE"]) == pytest.approx(_COLLISION_RATE * a_cm / cs_cm_s, rel=1e-14)
    # lambda_D/rho_s cancels electron temperature and elementary charge.
    expected_debye = _REFERENCE.magnetic_field_t * np.sqrt(
        1 / (4 * np.pi * 1e-7 * 299792458.0**2 * _REFERENCE.density_m3 * _REFERENCE.mass_kg)
    )
    assert float(deck["DEBYE"]) == pytest.approx(expected_debye, rel=1e-14)
    assert deck["UNITS"] == "GYRO" and deck["SAT_RULE"] == "0"


def test_nonzero_shape_shears_match_physical_finite_differences() -> None:
    """The emitted current-GACODE shape derivatives reproduce dR/dr and dZ/dr of physical Miller surfaces."""
    geometry = replace(_GEOMETRY, elongation_gradient_m=0.2, triangularity_gradient_m=0.1, major_radius_gradient=0.05)
    deck = _parse_deck(miller_tglf_deck(_REFERENCE, (_ELECTRON, _MAIN), geometry, electron_collision_rate_s=0))
    theta = np.linspace(0, 2 * np.pi, 37)
    step = 1e-5
    curves = []
    for offset in (-step, step):
        radius = geometry.minor_radius_m + offset
        major = geometry.major_radius_m + geometry.major_radius_gradient * offset
        kappa = geometry.elongation + geometry.elongation_gradient_m * offset
        delta = geometry.triangularity + geometry.triangularity_gradient_m * offset
        curves.append(
            np.stack(
                (major + radius * np.cos(theta + np.arcsin(delta) * np.sin(theta)), kappa * radius * np.sin(theta))
            )
        )
    derivative = (curves[1] - curves[0]) / (2 * step)
    angle = theta + np.arcsin(geometry.triangularity) * np.sin(theta)
    encoded_r = (
        float(deck["DRMAJDX_LOC"])
        + np.cos(angle)
        - np.sin(angle) * np.sin(theta) * float(deck["S_DELTA_LOC"]) / np.sqrt(1 - geometry.triangularity**2)
    )
    encoded_z = float(deck["KAPPA_LOC"]) * np.sin(theta) * (1 + float(deck["S_KAPPA_LOC"]))
    np.testing.assert_allclose(np.stack((encoded_r, encoded_z)), derivative, rtol=1e-8, atol=1e-9)
    assert float(deck["S_KAPPA_LOC"]) == pytest.approx(0.2 / 1.7)
    assert float(deck["S_DELTA_LOC"]) == pytest.approx(0.1)


@pytest.mark.parametrize("field", ["mass_kg", "density_m3", "temperature_kev"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, True])
def test_invalid_species_scale_is_rejected(field: str, value: float) -> None:
    """Physical species scales cannot become guessed or singular normalized input."""
    with pytest.raises(ValueError):
        replace(_ELECTRON, **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("minor_radius_m", 0),
        ("major_radius_m", 0.5),
        ("safety_factor", 0),
        ("elongation", 0),
        ("triangularity", 1),
        ("triangularity", -1),
        ("elongation_gradient_m", np.inf),
        ("triangularity_gradient_m", np.nan),
    ],
)
def test_invalid_geometry_is_rejected(field: str, value: float) -> None:
    """Refuse axis, invalid aspect/shape and nonfinite radial derivatives."""
    with pytest.raises(ValueError):
        replace(_GEOMETRY, **{field: value})


@pytest.mark.parametrize("fault", ["density", "gradient", "order", "fractional_charge", "no_ion", "too_many"])
def test_invalid_composition_cannot_generate_a_deck(fault: str) -> None:
    """Charge and derivative neutrality are independent input invariants, with explicit species ordering."""
    species = (_ELECTRON, _MAIN)
    if fault == "density":
        species = (_ELECTRON, replace(_MAIN, density_m3=4e19))
    elif fault == "gradient":
        species = (_ELECTRON, replace(_MAIN, density_gradient_m4=-2e19))
    elif fault == "order":
        species = (_MAIN, _ELECTRON)
    elif fault == "fractional_charge":
        species = (_ELECTRON, replace(_MAIN, charge_e=1.5))
    elif fault == "no_ion":
        species = (_ELECTRON,)
    else:
        species = (_ELECTRON,) + (_MAIN,) * 7
    with pytest.raises(ValueError):
        miller_tglf_deck(_REFERENCE, species, _GEOMETRY, electron_collision_rate_s=0)


@pytest.mark.parametrize("radius", [1e-6, 3.0])
def test_provider_radius_clamp_or_outside_edge_is_refused(radius: float) -> None:
    """Do not hand the provider a radius that it silently clamps or lies outside the declared edge."""
    with pytest.raises(ValueError, match="r/a"):
        miller_tglf_deck(
            _REFERENCE, (_ELECTRON, _MAIN), replace(_GEOMETRY, minor_radius_m=radius), electron_collision_rate_s=0
        )


@pytest.mark.parametrize("rate", [-1.0, np.nan, np.inf, True])
def test_invalid_collision_frequency_is_refused(rate: float) -> None:
    """Collision frequency is explicit finite nonnegative model input, not a fallback estimate."""
    with pytest.raises(ValueError):
        miller_tglf_deck(_REFERENCE, (_ELECTRON, _MAIN), _GEOMETRY, electron_collision_rate_s=rate)


@pytest.mark.parametrize(
    "field,value",
    [("mass_kg", 1e-320), ("density_m3", 1e-310), ("temperature_kev", 1e-310), ("magnetic_field_t", 1e308)],
)
def test_unrepresentable_reference_mapping_is_refused(field: str, value: float) -> None:
    """Finite but unrepresentable reference arithmetic cannot emit zero or infinite physical coefficients."""
    with pytest.raises(ValueError, match="range"):
        miller_tglf_deck(
            replace(_REFERENCE, **{field: value}), (_ELECTRON, _MAIN), _GEOMETRY, electron_collision_rate_s=0
        )


@pytest.mark.parametrize("modes", [1, 3, 5, True, 2.0])
def test_provider_rewritten_mode_count_is_not_requested(modes: int) -> None:
    """SAT0 uses the documented two/four-mode fits instead of relying on preset rewrites."""
    with pytest.raises(ValueError, match="modes"):
        miller_tglf_deck(_REFERENCE, (_ELECTRON, _MAIN), _GEOMETRY, electron_collision_rate_s=0, modes=modes)


@pytest.mark.parametrize("shaped,modes,fields", [(False, 2, 1), (True, 2, 1), (True, 4, 2), (True, 2, 3)])
def test_real_generated_miller_input_and_conservative_flux(
    tmp_path: Path, shaped: bool, modes: int, fields: int
) -> None:
    """Execute the generated three-species deck, check resolved inputs and advance physical profiles without charge repair."""
    binary = os.environ.get("SCPN_TGLF_BINARY")
    if not binary:
        pytest.skip("SCPN_TGLF_BINARY must name an actual configured GACODE launcher")
    environment = json.loads(os.environ.get("SCPN_TGLF_ENV_JSON", "{}"))
    trace = TGLFSpecies(2, 2 * _REFERENCE.mass_kg, 0.031 * 5e19, 3.0, -0.25 * 0.031 * 5e19, -3.0)
    main = replace(
        _MAIN,
        density_m3=0.938 * 5e19,
        density_gradient_m4=_ELECTRON.density_gradient_m4 - 2 * trace.density_gradient_m4,
    )
    geometry = (
        replace(_GEOMETRY, elongation_gradient_m=0.2, triangularity_gradient_m=0.1, major_radius_gradient=0.05)
        if shaped
        else _GEOMETRY
    )
    text = miller_tglf_deck(
        _REFERENCE,
        (_ELECTRON, main, trace),
        geometry,
        electron_collision_rate_s=_COLLISION_RATE,
        modes=modes,
        use_bper=fields >= 2,
        use_bpar=fields == 3,
    )
    path = tmp_path / "physical.tglf"
    path.write_text(text)
    raw = TGLFFluxSolver(tmp_path / "runs", binary=binary, environment=environment).run(path)
    resolved = {line.split()[1]: line.split()[0] for line in (raw.run_dir / "input.tglf.gen").read_text().splitlines()}
    for key, value in _parse_deck(text).items():
        assert resolved[key] == value
    assert len(raw.particle_flux_gb) == 3 and len(raw.growth_rate[0]) == modes
    physical = physical_tglf_flux(raw, _REFERENCE)
    particle = np.zeros((4, 4))
    particle[[0, 1, 3]] = np.asarray(physical.particle_m2_s)[:, None]
    heat = np.repeat([[physical.energy_w_m2[0]], [sum(physical.energy_w_m2[1:])]], 4, axis=1)
    result = advance_face_flux(
        rho=np.linspace(0, 1, 5),
        major_radius_m=6.0,
        minor_radius_m=2.0,
        density=np.repeat([[5.0], [4.69], [0.0], [0.155]], 5, axis=1),
        temperature=np.repeat([[2.0], [3.0]], 5, axis=1),
        impurity_density=np.zeros(5),
        flux=TransportFaceFlux(particle, heat),
        dt=1e-6,
    )
    assert result.balance.particle_relative_error < 1e-12
    assert result.balance.energy_relative_error < 1e-12
    (raw.run_dir / "physical_mapping_probe.json").write_text(
        json.dumps(
            {
                "shaped": shaped,
                "fields": fields,
                "modes": modes,
                "particle_gb": raw.particle_flux_gb,
                "energy_gb": raw.energy_flux_gb,
                "particle_relative_error": result.balance.particle_relative_error,
                "energy_relative_error": result.balance.energy_relative_error,
                "boundary": "Generated local Maxwellian Miller deck with prescribed cylindrical face harness; no Miller metric coupling, radial sampling or equilibrium certificate",
            },
            indent=2,
        )
    )


def test_signed_profile_gradients_are_not_clipped() -> None:
    """Outward-rising density/temperature profiles produce negative RLN/RLT and positive pressure derivative."""
    species = tuple(
        replace(
            sp, density_gradient_m4=-sp.density_gradient_m4, temperature_gradient_kev_m=-sp.temperature_gradient_kev_m
        )
        for sp in (_ELECTRON, _MAIN)
    )
    deck = _parse_deck(miller_tglf_deck(_REFERENCE, species, _GEOMETRY, electron_collision_rate_s=0))
    assert float(deck["RLNS_1"]) == -1
    assert float(deck["RLTS_1"]) == -3
    assert float(deck["P_PRIME_LOC"]) == pytest.approx(0.013618501389)
    assert float(deck["XNUE"]) == 0


@pytest.mark.parametrize("nky", [0, -1, 1.5, True])
def test_invalid_ky_grid_setting_is_rejected(nky: int) -> None:
    """Require an actual positive high-k grid integer instead of silent truncation."""
    with pytest.raises(ValueError, match="nky"):
        miller_tglf_deck(_REFERENCE, (_ELECTRON, _MAIN), _GEOMETRY, electron_collision_rate_s=0, nky=nky)


@pytest.mark.parametrize("sign", [0, 2, True, 1.0])
def test_invalid_orientation_is_rejected(sign: int) -> None:
    """Only explicit integer field/current orientations are serialized."""
    with pytest.raises(ValueError, match="signs"):
        miller_tglf_deck(_REFERENCE, (_ELECTRON, _MAIN), _GEOMETRY, electron_collision_rate_s=0, sign_bt=sign)


def test_parallel_magnetic_field_requires_perpendicular_field() -> None:
    """Reject an unsupported BPAR-only model choice rather than enabling another field implicitly."""
    with pytest.raises(ValueError, match="BPAR"):
        miller_tglf_deck(_REFERENCE, (_ELECTRON, _MAIN), _GEOMETRY, electron_collision_rate_s=0, use_bpar=True)


def test_zero_electron_gradient_can_balance_opposing_ion_gradients() -> None:
    """Gradient neutrality checks the signed species sum even when electron gradient is exactly zero."""
    electron = replace(_ELECTRON, density_gradient_m4=0)
    ions = (
        replace(_MAIN, density_m3=2.5e19, density_gradient_m4=1e19),
        replace(_MAIN, density_m3=2.5e19, density_gradient_m4=-1e19),
    )
    deck = _parse_deck(miller_tglf_deck(_REFERENCE, (electron, *ions), _GEOMETRY, electron_collision_rate_s=0))
    assert float(deck["RLNS_1"]) == 0
    assert float(deck["RLNS_2"]) == -float(deck["RLNS_3"]) != 0


@pytest.mark.parametrize("radius,major,length", [(5e-324, 6.0, 2.0), (0.05, 1e308, 0.1)])
def test_unrepresentable_geometry_normalization_is_rejected(radius: float, major: float, length: float) -> None:
    """Finite positive geometry cannot serialize a nonzero ratio as zero or infinity."""
    geometry = replace(_GEOMETRY, minor_radius_m=radius, major_radius_m=major)
    reference = replace(_REFERENCE, length_m=length)
    with pytest.raises(ValueError, match="Miller input ratio is outside the working range"):
        miller_tglf_deck(reference, (_ELECTRON, _MAIN), geometry, electron_collision_rate_s=0)


def test_overflowing_species_gradient_sum_preserves_arithmetic_cause() -> None:
    """Individually finite ion gradients cannot overflow their sum into an admissible deck."""
    electron = replace(_ELECTRON, density_gradient_m4=0)
    ion = replace(_MAIN, density_m3=_ELECTRON.density_m3 / 2, density_gradient_m4=1e308)
    with pytest.raises(ValueError, match="Miller physical inputs are outside the working range") as failure:
        miller_tglf_deck(_REFERENCE, (electron, ion, ion), _GEOMETRY, electron_collision_rate_s=0)
    assert isinstance(failure.value.__cause__, OverflowError)
