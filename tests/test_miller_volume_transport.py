# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Miller volume metrics and conservative transport.

"""Compare shape integration with analytic volumes and exercise metric-aware public transport."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.special import jv

from scpn_control.core.integrated_transport_solver import TransportSolver
from scpn_control.core.tglf_miller import TGLFMillerGeometry, miller_volume_metric
from scpn_control.core.transport_flux import TransportFaceFlux, TransportFaceGeometry, advance_face_flux


@pytest.mark.parametrize("elongation", [1.0, 1.7, 2.4])
def test_elliptical_volume_and_derivative_are_exact(elongation: float) -> None:
    """Constant elliptical tori have V=2*pi²*R*kappa*r² and V'=4*pi²*R*kappa*r."""
    volume, derivative = miller_volume_metric(TGLFMillerGeometry(1.2, 6.0, 2.0, 0, elongation))
    assert volume == pytest.approx(2 * np.pi**2 * 6 * elongation * 1.2**2, rel=1e-12)
    assert derivative == pytest.approx(4 * np.pi**2 * 6 * elongation * 1.2, rel=1e-12)


def _bessel_volume(g: TGLFMillerGeometry) -> float:
    """Evaluate the independently integrated closed Bessel expression for enclosed Miller volume."""
    alpha = np.arcsin(g.triangularity)
    return float(
        2 * np.pi**2 * g.major_radius_m * g.elongation * g.minor_radius_m**2 * (jv(0, alpha) + jv(2, alpha))
        - np.pi**2 * g.elongation * g.minor_radius_m**3 / 2 * (jv(1, 2 * alpha) + jv(3, 2 * alpha))
    )


@pytest.mark.parametrize("delta", [-0.6, -0.2, 0.3, 0.7])
def test_triangular_volume_and_shape_derivative_match_independent_reference(delta: float) -> None:
    """Bessel volume and its finite difference include elongation, triangularity and major-radius variation."""
    g = TGLFMillerGeometry(1.0, 6.0, 2.0, 2.0, 1.7, delta, 0.2, 0.1, 0.05)
    volume, derivative = miller_volume_metric(g)
    assert volume == pytest.approx(_bessel_volume(g), rel=1e-12)
    h = 1e-5
    shifted = [
        replace(
            g,
            minor_radius_m=g.minor_radius_m + offset,
            major_radius_m=g.major_radius_m + offset * g.major_radius_gradient,
            elongation=g.elongation + offset * g.elongation_gradient_m,
            triangularity=g.triangularity + offset * g.triangularity_gradient_m,
        )
        for offset in (-h, h)
    ]
    expected = (_bessel_volume(shifted[1]) - _bessel_volume(shifted[0])) / (2 * h)
    assert derivative == pytest.approx(expected, rel=1e-9)


def test_locally_folding_shape_is_refused() -> None:
    """A negative radial Jacobian cannot be hidden by a positive integrated volume."""
    with pytest.raises(ValueError, match="nesting"):
        miller_volume_metric(TGLFMillerGeometry(1.0, 6.0, 2.0, 2.0, elongation_gradient_m=-3.0))


def test_shaped_metric_transport_has_second_order_spatial_refinement() -> None:
    """For kappa=1+s*r and F=C*r, cell-volume divergence converges to the analytic local derivative."""
    errors = []
    for count in (9, 17, 33):
        rho = np.linspace(0, 1, count)
        radii = rho[1:] + rho[:-1]
        metrics = np.array(
            [
                miller_volume_metric(TGLFMillerGeometry(float(r), 6, 2, 0, 1 + 0.2 * r, elongation_gradient_m=0.2))
                for r in radii
            ]
        )
        volumes = np.r_[0, np.diff(metrics[:, 0]), 0]
        geometry = TransportFaceGeometry(volumes, metrics[:, 1])
        particles = np.zeros((4, count - 1))
        particles[:2] = 1e19 * radii
        result = advance_face_flux(
            rho=rho,
            major_radius_m=6,
            minor_radius_m=2,
            density=np.repeat([[5.0], [5.0], [0.0], [0.0]], count, axis=1),
            temperature=np.full((2, count), 2.0),
            impurity_density=np.zeros(count),
            flux=TransportFaceFlux(particles, np.zeros((2, count - 1))),
            dt=0.01,
            geometry=geometry,
        )
        expected = 5 - 0.01 * (4 + 9 * 0.2) / (2 + 3 * 0.2)
        errors.append(abs(result.density[0, count // 2] - expected))
        assert result.balance.particle_relative_error < 1e-12
        assert result.balance.energy_relative_error < 1e-12
        np.testing.assert_allclose(result.density[0], result.density[1], rtol=1e-12)
    assert 3.8 < errors[0] / errors[1] < 4.2
    assert 3.8 < errors[1] / errors[2] < 4.2


@pytest.mark.parametrize("fault", ["axis", "volume", "metric", "shape", "nan"])
def test_invalid_explicit_geometry_preserves_inputs(fault: str) -> None:
    """Refuse malformed metric weights before any profile update."""
    volume = np.array([0.0, 1, 1, 1, 0])
    metric = np.ones(4)
    if fault == "axis":
        volume[0] = 1
    elif fault == "volume":
        volume[2] = 0
    elif fault == "metric":
        metric[0] = -1
    elif fault == "shape":
        metric = np.ones(3)
    else:
        volume[2] = np.nan
    density = np.repeat([[5.0], [5.0], [0.0], [0.0]], 5, axis=1)
    original = density.copy()
    with pytest.raises(ValueError, match="geometry"):
        advance_face_flux(
            rho=np.linspace(0, 1, 5),
            major_radius_m=6,
            minor_radius_m=2,
            density=density,
            temperature=np.ones((2, 5)),
            impurity_density=np.zeros(5),
            flux=TransportFaceFlux(np.zeros((4, 4)), np.zeros((2, 4))),
            dt=1,
            geometry=TransportFaceGeometry(volume, metric),
        )
    np.testing.assert_array_equal(density, original)


def test_transport_solver_uses_explicit_face_metric() -> None:
    """The public stateful solver forwards supplied dV/dr and volumes rather than reverting to circular weights."""
    root = Path(__file__).parents[1]
    solver = TransportSolver(root / "validation/iter_validated_config.json", nr=5, multi_ion=True)
    solver.ne = solver.n_D = np.full(5, 5.0)
    solver.n_T = np.zeros(5)
    solver.n_He = np.zeros(5)
    solver.n_impurity = np.zeros(5)
    solver.Te = np.full(5, 2.0)
    solver.Ti = np.full(5, 3.0)
    metric = np.array([1.0, 2.0, 4.0, 8.0])
    particle = np.zeros((4, 4))
    particle[:2] = 1e19
    result = solver.evolve_fluxes(
        0.01,
        TransportFaceFlux(particle, np.zeros((2, 4))),
        geometry=TransportFaceGeometry(np.array([0.0, 1.0, 1.0, 1.0, 0.0]), metric),
    )
    assert result.particle_boundary[0] == pytest.approx(-0.07 * 1e19, rel=1e-12)
    np.testing.assert_allclose(solver.ne[1:-1], [4.99, 4.98, 4.96], rtol=1e-12)
    assert solver.last_flux_balance is result


def test_real_tglf_on_distinct_miller_faces_advances_metric_profiles(tmp_path: Path) -> None:
    """Advance three stages with fresh physical decks at four shaped faces and retain each stage's provenance."""
    import hashlib
    import json
    import os

    from scpn_control.core.tglf_miller import TGLFSpecies, miller_tglf_deck
    from scpn_control.core.tglf_units import TGLFReferenceUnits, physical_tglf_flux
    from validation.tglf_launcher import TGLFFluxSolver

    binary = os.environ.get("SCPN_TGLF_BINARY")
    if not binary:
        pytest.skip("SCPN_TGLF_BINARY must name an actual configured GACODE launcher")
    provider = TGLFFluxSolver(
        tmp_path / "runs", binary=binary, environment=json.loads(os.environ.get("SCPN_TGLF_ENV_JSON", "{}"))
    )
    rho = np.linspace(0, 1, 5)
    radii = rho[1:] + rho[:-1]
    particle, heat, volumes, metrics, runs = [], [], [], [], []
    for index, radius in enumerate(radii):
        ne, te, ti = 5e19 * np.exp(-radius / 2), 2 * np.exp(-1.5 * radius), 3 * np.exp(-radius)
        reference = TGLFReferenceUnits(float(ne), float(te), 2.0, 2 * 1.67262192369e-27, 2.0)
        shape = TGLFMillerGeometry(float(radius), 6, float(1 + radius), 1, float(1.7 + 0.2 * radius), 0.3, 0.2)
        electron = TGLFSpecies(-1, 9.1093837139e-31, float(ne), float(te), float(-ne / 2), float(-1.5 * te))
        ion = TGLFSpecies(1, reference.mass_kg, float(ne), float(ti), float(-ne / 2), float(-ti))
        deck = tmp_path / f"face-{index}.tglf"
        deck.write_text(miller_tglf_deck(reference, (electron, ion), shape, electron_collision_rate_s=0))
        raw = provider.run(deck)
        physical = physical_tglf_flux(raw, reference)
        particle.append((*physical.particle_m2_s, 0.0, 0.0))
        heat.append(physical.energy_w_m2)
        volume, metric = miller_volume_metric(shape)
        volumes.append(volume)
        metrics.append(metric)
        runs.append(str(raw.run_dir))
    density = np.zeros((4, 5))
    density[:2] = 5 * np.exp(-rho)
    temperature = np.stack((2 * np.exp(-3 * rho), 3 * np.exp(-2 * rho)))
    original = density.copy()
    flux = TransportFaceFlux(np.asarray(particle).T, np.asarray(heat).T)
    geometry = TransportFaceGeometry(np.r_[0, np.diff(volumes), 0], np.asarray(metrics))
    result = advance_face_flux(
        rho=rho,
        major_radius_m=6.0,
        minor_radius_m=2.0,
        density=density,
        temperature=temperature,
        impurity_density=np.zeros(5),
        flux=flux,
        dt=1e-8,
        geometry=geometry,
    )
    assert result.balance.particle_relative_error < 1e-12 and result.balance.energy_relative_error < 1e-12
    assert len(set(runs)) == 4 and np.ptp(flux.energy_w_m2[0]) > 0
    assert not np.array_equal(result.temperature[:, 1:-1], temperature[:, 1:-1])
    np.testing.assert_array_equal(density, original)
    (tmp_path / "radial_provider_receipt.json").write_text(
        json.dumps(
            {
                "runs": runs,
                "radii_m": radii.tolist(),
                "cell_volumes_m3": geometry.cell_volume_m3.tolist(),
                "face_dV_dr_m2": metrics,
                "particle_relative_error": result.balance.particle_relative_error,
                "energy_relative_error": result.balance.energy_relative_error,
                "boundary": "Analytic prescribed profiles, collisionless local SAT0 at four Miller faces; one frozen-flux step, not equilibrium or coupled temporal convergence",
            },
            indent=2,
        )
    )

    balances = [result.balance]
    stages = []
    for step in (1, 2):
        previous = result
        particle, heat = [], []
        stage_runs, deck_hashes = [], []
        profiles = np.stack((previous.density[0] * 1e19, *previous.temperature))
        for index, radius in enumerate(radii):
            left, right = profiles[:, index], profiles[:, index + 1]
            face = np.sqrt(left * right)
            gradient = face * np.log(right / left) / (2 * (rho[index + 1] - rho[index]))
            ne, te, ti = (float(value) for value in face)
            dn, dte, dti = (float(value) for value in gradient)
            reference = TGLFReferenceUnits(ne, te, 2.0, 2 * 1.67262192369e-27, 2.0)
            shape = TGLFMillerGeometry(float(radius), 6, float(1 + radius), 1, float(1.7 + 0.2 * radius), 0.3, 0.2)
            electron = TGLFSpecies(-1, 9.1093837139e-31, ne, te, dn, dte)
            ion = TGLFSpecies(1, reference.mass_kg, ne, ti, dn, dti)
            deck = tmp_path / f"step-{step}-face-{index}.tglf"
            deck.write_text(miller_tglf_deck(reference, (electron, ion), shape, electron_collision_rate_s=0))
            previous_deck = tmp_path / (f"face-{index}.tglf" if step == 1 else f"step-{step - 1}-face-{index}.tglf")
            assert deck.read_bytes() != previous_deck.read_bytes()
            raw = provider.run(deck)
            receipt = json.loads((raw.run_dir / "execution.json").read_text())
            assert receipt["input_sha256"] == hashlib.sha256(deck.read_bytes()).hexdigest()
            assert receipt["output_validated"] is True and receipt["returncode"] == 0
            assert (raw.run_dir / "input.tglf").read_bytes() == deck.read_bytes()
            physical = physical_tglf_flux(raw, reference)
            particle.append((*physical.particle_m2_s, 0.0, 0.0))
            heat.append(physical.energy_w_m2)
            stage_runs.append(str(raw.run_dir))
            deck_hashes.append(receipt["input_sha256"])
        result = advance_face_flux(
            rho=rho,
            major_radius_m=6.0,
            minor_radius_m=2.0,
            density=previous.density,
            temperature=previous.temperature,
            impurity_density=np.zeros(5),
            flux=TransportFaceFlux(np.asarray(particle).T, np.asarray(heat).T),
            dt=1e-8,
            geometry=geometry,
        )
        np.testing.assert_allclose(result.balance.particle_initial, previous.balance.particle_final, rtol=1e-14)
        assert result.balance.energy_initial_j == pytest.approx(previous.balance.energy_final_j, rel=1e-14)
        assert result.balance.particle_relative_error < 1e-12
        assert result.balance.energy_relative_error < 1e-12
        assert not np.array_equal(result.temperature[:, 1:-1], previous.temperature[:, 1:-1])
        np.testing.assert_array_equal(result.density[:, -1], density[:, -1])
        np.testing.assert_array_equal(result.temperature[:, -1], temperature[:, -1])
        balances.append(result.balance)
        runs.extend(stage_runs)
        stages.append({"step": step, "profiles": profiles.tolist(), "runs": stage_runs, "input_sha256": deck_hashes})
    assert len(set(runs)) == 12
    initial_inventory = np.asarray(balances[0].particle_initial)
    final_inventory = np.asarray(balances[-1].particle_final)
    exchange = np.sum([balance.particle_boundary for balance in balances], axis=0)
    np.testing.assert_allclose(final_inventory, initial_inventory + exchange, rtol=1e-12, atol=0)
    assert balances[-1].energy_final_j == pytest.approx(
        balances[0].energy_initial_j + sum(balance.energy_boundary_j for balance in balances), rel=1e-12
    )
    (tmp_path / "temporal_provider_receipt.json").write_text(
        json.dumps(
            {
                "stages": stages,
                "all_runs": runs,
                "dt_s": 1e-8,
                "boundary": "Three explicit frozen-flux steps; log-linear face reconstruction from updated positive node profiles, collisionless e/D only, fixed shape/Bunit; not coupled temporal convergence or equilibrium validation",
            },
            indent=2,
        )
    )


def test_small_radius_metric_avoids_major_radius_cancellation() -> None:
    """Analytically removing constant contour terms retains circular volume at extreme aspect ratio."""
    radius, major = 1e-8, 6e4
    volume, derivative = miller_volume_metric(TGLFMillerGeometry(radius, major, 2, 0))
    assert volume == pytest.approx(2 * np.pi**2 * major * radius**2, rel=1e-12, abs=0)
    assert derivative == pytest.approx(4 * np.pi**2 * major * radius, rel=1e-12, abs=0)


@pytest.mark.parametrize("fault", ["underflow", "complex"])
def test_metric_conversion_cannot_discard_supplied_values(fault: str) -> None:
    """Even an unevolved edge weight must not lose a nonzero real value or an imaginary component."""
    if fault == "underflow":
        if np.finfo(np.longdouble).tiny == np.finfo(np.float64).tiny:
            pytest.skip("extended precision is needed to supply a nonzero value below binary64")
        volume = np.array([0, 1, 1, 1, np.longdouble("1e-400")], dtype=np.longdouble)
    else:
        volume = np.array([0, 1, 1, 1, 1j])
    with pytest.raises(ValueError, match="geometry"):
        advance_face_flux(
            rho=np.linspace(0, 1, 5),
            major_radius_m=6,
            minor_radius_m=2,
            density=np.repeat([[5.0], [5.0], [0.0], [0.0]], 5, axis=1),
            temperature=np.ones((2, 5)),
            impurity_density=np.zeros(5),
            flux=TransportFaceFlux(np.zeros((4, 4)), np.zeros((2, 4))),
            dt=0,
            geometry=TransportFaceGeometry(volume, np.ones(4)),
        )


@pytest.mark.parametrize(
    "radius,major,message",
    [
        (1e-200, 6.0, "Miller volume metric is not finite and positive"),
        (1e150, 1e200, "Miller volume arithmetic is not representable"),
        (1e-160, 6.0, "Miller volume quadrature did not converge"),
    ],
)
def test_unrepresentable_volume_metric_is_refused(radius: float, major: float, message: str) -> None:
    """Actual quadrature refuses zero volume, overflowing products and unresolved subnormal refinement."""
    geometry = TGLFMillerGeometry(radius, major, 2.0, 0.0)
    with pytest.raises(ValueError, match=message) as failure:
        miller_volume_metric(geometry)
    if radius == 1e150:
        assert isinstance(failure.value.__cause__, FloatingPointError)
    else:
        assert failure.value.__cause__ is None
