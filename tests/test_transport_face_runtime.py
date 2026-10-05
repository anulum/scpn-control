# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Transport Face Runtime tests
"""Exercise the named transport face runtime surface through the integrated solver."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpn_control.core.integrated_transport_solver import (
    TransportSolver,
)
from scpn_control.core.transport_flux import TransportFluxBalance


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_public_signed_flux_step_closes_actual_tglf_inventory(direction: float) -> None:
    """Captured provider->SI->public flux step preserves signed reservoir counts and energy."""
    from scpn_control.core.tglf_flux import read_tglf_fluxes
    from scpn_control.core.tglf_units import TGLFReferenceUnits, physical_tglf_flux
    from scpn_control.core.transport_flux import TransportFaceFlux

    root = Path(__file__).parents[1]
    ts = TransportSolver(root / "validation/iter_validated_config.json", nr=9, multi_ion=True)
    ts.n_D = np.full(9, 5.0)
    ts.n_T = np.zeros(9)
    ts.n_He = np.zeros(9)
    ts.n_impurity = np.zeros(9)
    ts.ne = np.full(9, 5.0)
    ts.Te = np.full(9, 2.0)
    ts.Ti = np.full(9, 2.0)
    raw = read_tglf_fluxes(root / "tests/data/tglf/default")
    physical = physical_tglf_flux(raw, TGLFReferenceUnits(5e19, 2.0, ts.a, 2 * 1.67262192369e-27, 2.0))
    particle = np.zeros((4, 8))
    particle[:2] = direction * np.asarray(physical.particle_m2_s)[:, None]
    heat = direction * np.repeat(np.asarray(physical.energy_w_m2)[:, None], 8, axis=1)
    dt = 1e-4
    dims = ts.cfg["dimensions"]
    radius = (dims["R_min"] + dims["R_max"]) / 2
    faces = (ts.rho[1:] + ts.rho[:-1]) / 2
    areas = 4 * np.pi**2 * radius * ts.a * faces
    expected_n = dt * (particle[:, 0] * areas[0] - particle[:, -1] * areas[-1])
    expected_e = dt * np.sum(heat[:, 0] * areas[0] - heat[:, -1] * areas[-1])
    record = ts.evolve_fluxes(dt, TransportFaceFlux(particle, heat))
    np.testing.assert_allclose(record.particle_boundary, expected_n, rtol=1e-14)
    assert record.energy_boundary_j == pytest.approx(expected_e, rel=1e-14)
    assert record.particle_relative_error < 1e-14
    assert record.energy_relative_error < 1e-14
    assert ts.last_flux_balance is record
    np.testing.assert_allclose(ts.ne, ts.n_D, rtol=1e-14)
    before = [x.copy() for x in (ts.ne, ts.n_D, ts.Te, ts.Ti)]
    with pytest.raises(ValueError, match="negative state"):
        ts.evolve_fluxes(1e9, TransportFaceFlux(particle, heat))
    assert ts.last_flux_balance is None
    for actual, original in zip((ts.ne, ts.n_D, ts.Te, ts.Ti), before):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_public_near_depletion_preserves_state_on_final_charge_refusal(direction: float) -> None:
    """Final charge validation precedes profile mutation and clears an earlier valid balance on refusal."""
    from scpn_control.core.transport_flux import TransportFaceFlux

    ts = TransportSolver(Path(__file__).parents[1] / "validation/iter_validated_config.json", nr=5, multi_ion=True)
    ts.n_D = np.full(5, 5.0)
    ts.n_T = np.zeros(5)
    ts.n_He = np.zeros(5)
    ts.n_impurity = np.zeros(5)
    ts.ne = np.full(5, 5 * (1 + 5e-13))
    ts.Te = np.ones(5)
    ts.Ti = np.ones(5)
    zero = TransportFaceFlux(np.zeros((4, 4)), np.zeros((2, 4)))
    assert ts.evolve_fluxes(0, zero) is ts.last_flux_balance
    radius = (ts.cfg["dimensions"]["R_min"] + ts.cfg["dimensions"]["R_max"]) / 2
    volume = 4 * np.pi**2 * radius * ts.a**2 * ts.rho / 4
    areas = 4 * np.pi**2 * radius * ts.a * (ts.rho[1:] + ts.rho[:-1]) / 2
    particle = np.zeros((4, 4))
    donor = 1 if direction > 0 else 2
    particle[:2, 1] = direction * (5 - 1e-14) * 1e19 * volume[donor] / areas[1]
    before = [x.copy() for x in (ts.ne, ts.n_D, ts.n_T, ts.n_He, ts.n_impurity, ts.Te, ts.Ti)]
    with pytest.raises(ValueError, match="Final.*quasineutral"):
        ts.evolve_fluxes(1, TransportFaceFlux(particle, np.zeros((2, 4))))
    assert ts.last_flux_balance is None
    for actual, original in zip((ts.ne, ts.n_D, ts.n_T, ts.n_He, ts.n_impurity, ts.Te, ts.Ti), before, strict=True):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_public_cubic_face_flux_has_second_order_spatial_convergence(direction: float) -> None:
    """F=C*rho^3 has analytic cylindrical divergence 4*C*rho^2/a in the evolved interior."""
    from scpn_control.core.transport_flux import TransportFaceFlux

    errors = []
    for nr in (9, 17, 33, 65):
        ts = TransportSolver(Path(__file__).parents[1] / "validation/iter_validated_config.json", nr=nr, multi_ion=True)
        ts.n_D = 5 + ts.rho**2
        ts.ne = ts.n_D.copy()
        ts.n_T = np.zeros(nr)
        ts.n_He = np.zeros(nr)
        ts.n_impurity = np.zeros(nr)
        ts.Te = 2 + ts.rho**2
        ts.Ti = 3 + 0.5 * ts.rho**2
        heat_capacity = 1.5e19 * 1.602176634e-16
        initial_density = ts.ne.copy()
        initial_energy = heat_capacity * np.stack((ts.ne * ts.Te, ts.n_D * ts.Ti))
        faces = (ts.rho[1:] + ts.rho[:-1]) / 2
        particle = np.zeros((4, nr - 1))
        particle[:2] = direction * 1e20 * faces**3
        heat_coefficients = direction * np.array([2e5, 1e5])
        dt = 1e-3
        record = ts.evolve_fluxes(dt, TransportFaceFlux(particle, heat_coefficients[:, None] * faces**3))
        expected_density = initial_density - dt * direction * 40 * ts.rho**2 / ts.a
        expected_energy = initial_energy - dt * 4 * heat_coefficients[:, None] * ts.rho**2 / ts.a
        actual_energy = heat_capacity * np.stack((ts.ne * ts.Te, ts.n_D * ts.Ti))
        errors.append(
            (
                np.max(np.abs(ts.ne[1:-1] - expected_density[1:-1])),
                np.max(np.abs(actual_energy[:, 1:-1] - expected_energy[:, 1:-1])),
            )
        )
        assert record.particle_relative_error < 1e-12
        assert record.energy_relative_error < 1e-12
    ratios = np.asarray(errors[:-1]) / np.asarray(errors[1:])
    np.testing.assert_allclose(ratios, 4, rtol=1e-5, atol=0)


@pytest.mark.parametrize("missing", ["n_D", "n_T", "n_He", "single_ion"])
def test_incomplete_species_state_refuses_face_step_without_mutation(config_file: Path, missing: str) -> None:
    """Absent species and single-ion mode cannot reuse a flux balance or alter retained profiles."""
    from scpn_control.core.transport_flux import TransportFaceFlux

    ts = TransportSolver(config_file, nr=5, multi_ion=missing != "single_ion")
    zero = TransportFaceFlux(np.zeros((4, 4)), np.zeros((2, 4)))
    if missing != "single_ion":
        ts.n_D = np.full(5, 5.0)
        ts.n_T = np.zeros(5)
        ts.n_He = np.zeros(5)
        ts.n_impurity = np.zeros(5)
        ts.ne = np.full(5, 5.0)
        assert ts.evolve_fluxes(0, zero) is ts.last_flux_balance
        assert ts.last_flux_balance is not None
        setattr(ts, missing, None)
        with pytest.raises(ValueError, match="multi-ion heat capacity requires D, T and He profiles"):
            _ = ts.ion_density
    names = ("ne", "n_D", "n_T", "n_He", "n_impurity", "Te", "Ti", "rho", "Psi")
    before = {name: None if getattr(ts, name) is None else getattr(ts, name).copy() for name in names}
    with pytest.raises(ValueError, match="Signed face transport requires multi-ion species state"):
        ts.evolve_fluxes(1e-3, zero)
    assert ts.last_flux_balance is None
    for name, original in before.items():
        if original is None:
            assert getattr(ts, name) is None
        else:
            np.testing.assert_array_equal(getattr(ts, name), original)
    np.testing.assert_array_equal(zero.particle_m2_s, np.zeros((4, 4)))
    np.testing.assert_array_equal(zero.energy_w_m2, np.zeros((2, 4)))


@pytest.mark.parametrize("elongation,shear", [(1.0, 0.0), (1.7, 0.2)])
def test_refreshed_face_transport_converges_in_time_and_recovers_after_rejection(
    config_file: Path, elongation: float, shear: float
) -> None:
    """State-dependent outward loss approaches exponential decay with telescoping inventories after a rejected step."""
    from math import fsum

    from scpn_control.core.tglf_miller import TGLFMillerGeometry, miller_volume_metric
    from scpn_control.core.transport_flux import TransportFaceFlux, TransportFaceGeometry

    errors = []
    heat_capacity = 1.5e19 * 1.602176634e-16
    decay_rate = 0.4
    for steps in (8, 16, 32, 64):
        ts = TransportSolver(config_file, nr=9, multi_ion=True)
        ts.set_neoclassical(R0=6.0, a=2.0, B0=5.3)
        ts.ne = np.full(9, 5.0)
        ts.n_D = np.full(9, 3.0)
        ts.n_T = np.full(9, 1.0)
        ts.n_He = np.full(9, 0.5)
        ts.n_impurity = np.zeros(9)
        ts.Te = np.full(9, 2.0)
        ts.Ti = np.full(9, 3.0)
        radii = ts.a * (ts.rho[1:] + ts.rho[:-1]) / 2
        metrics = np.asarray(
            [
                miller_volume_metric(
                    TGLFMillerGeometry(
                        float(radius),
                        6.0,
                        2.0,
                        0.0,
                        elongation + shear * radius,
                        elongation_gradient_m=shear,
                    )
                )
                for radius in radii
            ]
        )
        geometry = TransportFaceGeometry(np.r_[0, np.diff(metrics[:, 0]), 0], metrics[:, 1])
        volume = geometry.cell_volume_m3
        initial_density = np.stack((ts.ne, ts.n_D, ts.n_T, ts.n_He))
        initial_particles = (initial_density * volume).sum(axis=1) * 1e19
        initial_energy = float(np.sum(heat_capacity * (ts.ne * ts.Te + ts.ion_density * ts.Ti) * volume))
        records: list[TransportFluxBalance] = []
        for step in range(steps):
            density = np.stack((ts.ne, ts.n_D, ts.n_T, ts.n_He))
            thermal = heat_capacity * np.array([ts.ne[4] * ts.Te[4], ts.ion_density[4] * ts.Ti[4]])
            radial_factor = decay_rate * metrics[:, 0] / metrics[:, 1]
            flux = TransportFaceFlux(
                1e19 * density[:, 4, None] * radial_factor,
                thermal[:, None] * radial_factor,
            )
            if step == 3:
                before = {
                    name: getattr(ts, name).copy()
                    for name in ("ne", "n_D", "n_T", "n_He", "n_impurity", "Te", "Ti", "Psi")
                }
                previous_balance = ts.last_flux_balance
                assert previous_balance is records[-1]
                with pytest.raises(ValueError, match="negative state or zero thermal capacity"):
                    ts.evolve_fluxes(100.0, flux, geometry=geometry)
                assert ts.last_flux_balance is None
                for name, value in before.items():
                    np.testing.assert_array_equal(getattr(ts, name), value)
                assert previous_balance is records[-1]
            record = ts.evolve_fluxes(1.0 / steps, flux, geometry=geometry)
            assert record is ts.last_flux_balance
            if records:
                np.testing.assert_allclose(record.particle_initial, records[-1].particle_final, rtol=1e-14)
                assert record.energy_initial_j == pytest.approx(records[-1].energy_final_j, rel=1e-14)
            records.append(record)
            np.testing.assert_allclose(ts.Te, 2.0, rtol=1e-13)
            np.testing.assert_allclose(ts.Ti, 3.0, rtol=1e-13)
            np.testing.assert_allclose(ts.ne, ts.n_D + ts.n_T + 2 * ts.n_He, rtol=1e-13)
            np.testing.assert_array_equal(
                np.array([ts.ne[-1], ts.n_D[-1], ts.n_T[-1], ts.n_He[-1]]), initial_density[:, -1]
            )
        final_particles = (np.stack((ts.ne, ts.n_D, ts.n_T, ts.n_He)) * volume).sum(axis=1) * 1e19
        exchange = np.array([fsum(record.particle_boundary[channel] for record in records) for channel in range(4)])
        np.testing.assert_allclose(final_particles - initial_particles, exchange, rtol=1e-12, atol=0)
        final_energy = float(np.sum(heat_capacity * (ts.ne * ts.Te + ts.ion_density * ts.Ti) * volume))
        assert final_energy - initial_energy == pytest.approx(
            fsum(record.energy_boundary_j for record in records), rel=1e-12, abs=0
        )
        for profile, initial in zip((ts.ne, ts.n_D, ts.n_T, ts.n_He), initial_density[:, 0], strict=True):
            np.testing.assert_allclose(profile[1:-1], profile[4], rtol=1e-13)
            assert abs(profile[4] / initial - np.exp(-decay_rate)) < 0.01
        errors.append(abs(ts.ne[4] / 5.0 - np.exp(-decay_rate)))
    assert np.all((np.asarray(errors[:-1]) / errors[1:] > 1.9) & (np.asarray(errors[:-1]) / errors[1:] < 2.1))


@pytest.mark.parametrize("field", ["particle_m2_s", "energy_w_m2"])
def test_complex_public_face_flux_clears_balance_without_changing_state(
    solver_multi: TransportSolver, field: str
) -> None:
    """Public multi-ion evolution refuses a complex provider flux before committing a different real-valued stage."""
    from scpn_control.core.transport_flux import TransportFaceFlux

    ts = solver_multi
    flux = TransportFaceFlux(np.zeros((4, ts.nr - 1)), np.zeros((2, ts.nr - 1)))
    prior = ts.evolve_fluxes(0, flux)
    values = getattr(flux, field).astype(np.complex128)
    values.flat[0] = 1j
    invalid = TransportFaceFlux(
        values if field == "particle_m2_s" else flux.particle_m2_s,
        values if field == "energy_w_m2" else flux.energy_w_m2,
    )
    before = {name: getattr(ts, name).copy() for name in ("ne", "n_D", "n_T", "n_He", "n_impurity", "Te", "Ti", "Psi")}
    with pytest.raises(ValueError, match="profiles and fluxes must be real"):
        ts.evolve_fluxes(0.001, invalid)
    assert ts.last_flux_balance is None
    assert prior.particle_boundary == (0.0, 0.0, 0.0, 0.0)
    for name, expected in before.items():
        np.testing.assert_array_equal(getattr(ts, name), expected)
    np.testing.assert_array_equal(getattr(invalid, field), values)
