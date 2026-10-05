# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Transport Species Runtime tests
"""Exercise the named transport species runtime surface through the integrated solver."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from scpn_control.core.integrated_transport_solver import (
    PhysicsError,
    TransportSolver,
)


class TestMultiIon:
    """Check fuel burn, helium production, heating split and charge accounting."""

    def test_multi_ion_he_ash_grows(self, solver_multi: TransportSolver) -> None:
        """With multi_ion=True, after N steps, n_He should increase."""
        assert solver_multi.n_He is not None
        he_initial = np.sum(solver_multi.n_He)
        for _ in range(20):
            solver_multi.update_transport_model(50.0)
            solver_multi.evolve_profiles(dt=0.01, P_aux=50.0)
        he_final = np.sum(solver_multi.n_He)
        # He-ash is produced by fusion reactions
        assert he_final > he_initial

    def test_multi_ion_fuel_depletes(self, solver_multi: TransportSolver) -> None:
        """With multi_ion=True, sum of n_D should decrease over time."""
        assert solver_multi.n_D is not None
        d_initial = np.sum(solver_multi.n_D)
        for _ in range(20):
            solver_multi.update_transport_model(50.0)
            solver_multi.evolve_profiles(dt=0.01, P_aux=50.0)
        d_final = np.sum(solver_multi.n_D)
        # Fuel is consumed by fusion reactions
        assert d_final < d_initial

    def test_multi_ion_tritium_depletes(self, solver_multi: TransportSolver) -> None:
        """Tritium should also deplete over time due to fusion burn."""
        assert solver_multi.n_T is not None
        t_initial = np.sum(solver_multi.n_T)
        for _ in range(20):
            solver_multi.update_transport_model(50.0)
            solver_multi.evolve_profiles(dt=0.01, P_aux=50.0)
        t_final = np.sum(solver_multi.n_T)
        assert t_final < t_initial

    def test_aux_heating_source_power_split_multi_ion(self, solver_multi: TransportSolver) -> None:
        """Multi-ion lane should split auxiliary power between ions/electrons."""
        solver_multi.aux_heating_electron_fraction = 0.5
        with pytest.raises(PhysicsError, match="positive finite electron density"):
            solver_multi._compute_aux_heating_sources(40.0)
        solver_multi.ne[-1] = 0.02
        s_i, s_e = solver_multi._compute_aux_heating_sources(40.0)
        assert np.all(np.isfinite(s_i))
        assert np.all(np.isfinite(s_e))

        dV = solver_multi._rho_volume_element()
        e_keV_J = 1.602176634e-16
        ne_m3 = solver_multi.ne * 1e19

        rec_i = 1.5 * np.sum(ne_m3 * s_i * e_keV_J * dV) / 1e6
        rec_e = 1.5 * np.sum(ne_m3 * s_e * e_keV_J * dV) / 1e6
        assert rec_i == pytest.approx(20.0, rel=1e-6, abs=1e-6)
        assert rec_e == pytest.approx(20.0, rel=1e-6, abs=1e-6)
        assert solver_multi._last_aux_heating_balance["reconstructed_total_MW"] == pytest.approx(
            40.0,
            rel=1e-6,
            abs=1e-6,
        )

    def test_multi_ion_quasineutrality(self, solver_multi: TransportSolver) -> None:
        """After evolving, ne should be updated from quasineutrality."""
        assert solver_multi.n_D is not None and solver_multi.n_T is not None and solver_multi.n_He is not None
        for _ in range(5):
            solver_multi.update_transport_model(50.0)
            solver_multi.evolve_profiles(dt=0.01, P_aux=50.0)
        # ne = n_D + n_T + 2*n_He + Z_W * n_impurity
        ne_check = (
            solver_multi.n_D
            + solver_multi.n_T
            + 2.0 * solver_multi.n_He
            + 10.0 * np.maximum(solver_multi.n_impurity, 0.0)
        )
        np.testing.assert_allclose(solver_multi.ne, ne_check, rtol=1e-10)


class TestSpeciesAndBalanceDiagnostics:
    """Check the single-ion species no-op and scalar balance properties."""

    def test_evolve_species_is_noop_in_single_ion_mode(self, solver: TransportSolver) -> None:
        """The single-ion helper returns zero sources and zero particle-balance error."""
        s_he, p_rad = solver._evolve_species(0.01)
        assert np.all(s_he == 0.0)
        assert np.all(p_rad == 0.0)
        assert solver.particle_balance_error == 0.0

    def test_balance_error_properties_return_floats(self, solver: TransportSolver) -> None:
        """Public energy and particle balance properties return scalar floats."""
        assert isinstance(solver.energy_balance_error, float)
        assert isinstance(solver.particle_balance_error, float)


@pytest.mark.parametrize("a_minor", [0.5, 2.0, 4.0])
def test_species_diffusion_uses_physical_cylindrical_geometry(tmp_path: Path, a_minor: float) -> None:
    """A parabolic density loses 4 D C / a^2 through the public thermal step."""
    config = json.loads((Path(__file__).parents[1] / "iter_config.json").read_text())
    config["dimensions"]["R_min"] = 6.0 - a_minor
    config["dimensions"]["R_max"] = 6.0 + a_minor
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(config))
    ts = TransportSolver(str(path), nr=25, multi_ion=True)
    initial = 0.01 + 2.5 * (1.0 - ts.rho**2)
    initial[0] = initial[1]
    ts.n_D = initial.copy()
    ts.n_T = initial.copy()
    ts.n_He = np.zeros(ts.nr)
    ts.n_impurity = np.zeros(ts.nr)
    ts.ne = np.maximum(2.0 * initial, 0.1)
    ts.Ti = np.full(ts.nr, 0.2)
    ts.Te = np.full(ts.nr, 0.2)
    ts.D_species = 0.3
    dt = 1e-5
    from scpn_control.core.plasma_power_terms import bosch_hale_dt_reactivity

    burn = initial**2 * 1e19 * bosch_hale_dt_reactivity(ts.Ti)
    ts.evolve_profiles(dt, P_aux=0.0)
    expected = dt * (-4.0 * 2.5 * 0.3 / a_minor**2 - burn[10])
    assert ts.n_D is not None
    assert ts.n_D[10] - initial[10] == pytest.approx(expected, rel=1e-8, abs=1e-14)


def test_absent_tritium_is_not_created_by_thermal_sanitization() -> None:
    """The public thermal step keeps absent core fuel absent when diffusion is disabled."""
    config = Path(__file__).parents[1] / "validation/iter_validated_config.json"
    ts = TransportSolver(config, nr=25, multi_ion=True)
    ts.n_D = np.full(25, 2.0)
    ts.n_T = np.zeros(25)
    ts.n_He = np.full(25, 1.0)
    ts.n_impurity = np.zeros(25)
    ts.ne = np.full(25, 4.0)
    ts.Ti = np.full(25, 2.0)
    ts.Te = np.full(25, 2.0)
    ts.D_species = 0.0
    ts.evolve_profiles(0.01, 0.0)
    assert ts.n_T[10] == 0.0
    assert ts.n_D[10] == 2.0
    assert ts.n_T[-1] == 0.01  # Explicit recycling boundary remains a physical input.
    assert 0.0 < ts.n_He[10] < 1.0


def test_empty_ion_population_refuses_thermal_evolution() -> None:
    """No thermal result is admitted for a cell with zero ion heat capacity."""
    config = Path(__file__).parents[1] / "validation/iter_validated_config.json"
    ts = TransportSolver(config, nr=25, multi_ion=True)
    ts.n_D = np.zeros(25)
    ts.n_T = np.zeros(25)
    ts.n_He = np.zeros(25)
    ts.n_impurity = np.zeros(25)
    ts.D_species = 0.0
    with pytest.raises(PhysicsError, match="positive ion density"):
        ts.evolve_profiles(0.01, 0.0)
    assert ts.last_energy_balance is None


@pytest.mark.parametrize("enforce", [False, True])
def test_species_record_remains_inspectable_after_thermal_rejection(enforce: bool) -> None:
    """Species-stage inventory is immutable and distinct from later thermal admission."""
    from scpn_control.core.pedestal import PedestalParams, PedestalProfile

    config = Path(__file__).parents[1] / "validation/iter_validated_config.json"
    ts = TransportSolver(config, nr=25, multi_ion=True)
    assert ts.last_particle_balance is None
    ts.n_D = np.full(25, 2.0)
    ts.n_T = np.full(25, 1.0)
    ts.n_He = np.full(25, 0.5)
    ts.n_impurity = np.zeros(25)
    ts.ne = np.full(25, 4.0)
    ts.Ti = np.full(25, 2.0)
    ts.Te = np.full(25, 1.0)
    volumes = 4.0 * np.pi**2 * 6.0 * 4.0**2 * ts.rho * ts.drho
    initial = float(np.sum((ts.n_D + ts.n_T + ts.n_He) * 1e19 * volumes))
    pedestal = PedestalProfile(PedestalParams(f_ped=20.0, f_sep=0.08))
    if enforce:
        with pytest.raises(PhysicsError, match="Energy conservation violated"):
            ts.evolve_profiles(0.001, 0.0, enforce_conservation=True, ped_te=pedestal)
    else:
        ts.evolve_profiles(0.001, 0.0, ped_te=pedestal)
    record = ts.last_particle_balance
    assert record is not None
    assert record.initial_count == pytest.approx(initial, rel=1e-13)
    assert record.final_count == pytest.approx(float(np.sum((ts.n_D + ts.n_T + ts.n_He) * 1e19 * volumes)), rel=1e-13)
    assert record.relative_error == ts.particle_balance_error
    assert record.relative_error < 1e-12
    assert ts.last_energy_balance is not None and ts.last_energy_balance.relative_error > 0.01
    with pytest.raises(FrozenInstanceError):
        record.__setattr__("initial_count", 0.0)
    ts.n_D[:] = 99.0
    assert record.initial_count == pytest.approx(initial, rel=1e-13)
    with pytest.raises(ValueError):
        ts.evolve_profiles(-0.1, 0.0)
    assert ts.last_particle_balance is None
    assert ts.particle_balance_error == 0.0
    ts.evolve_profiles(0.0, 0.0)
    assert ts.last_particle_balance is None


@pytest.mark.parametrize("density", [0.005, 0.02])
@pytest.mark.parametrize("power", [0.0, 0.01])
def test_dilute_runtime_preserves_charge_and_actual_source_power(density: float, power: float) -> None:
    """Low-density charge and heat capacity must not be replaced by a numerical floor."""
    from scpn_control.core.plasma_power_terms import bremsstrahlung_power_density

    config = Path(__file__).parents[1] / "validation/iter_validated_config.json"
    ts = TransportSolver(config, nr=25, multi_ion=True)
    ts.n_D = np.full(25, density)
    ts.n_T = np.zeros(25)
    ts.n_He = np.zeros(25)
    ts.n_impurity = np.zeros(25)
    ts.ne = np.full(25, density)
    ts.Ti = np.full(25, 2.0)
    ts.Te = np.full(25, 1.0)
    ts.D_species = 0.0
    volumes = 4 * np.pi**2 * 6.0 * 4.0**2 * ts.rho * ts.drho
    expected_initial = float(1.5 * 1e19 * 1.602176634e-16 * np.sum(3 * density * volumes))
    dt = 1e-5
    ts.evolve_profiles(dt, power)
    np.testing.assert_allclose(ts.ne, ts.n_D + ts.n_T + 2 * ts.n_He, rtol=0, atol=0)
    record = ts.last_energy_balance
    assert record is not None
    assert record.initial_energy_j == pytest.approx(expected_initial, rel=1e-13)
    brem = bremsstrahlung_power_density(ts.ne, np.ones(25), 1.0)
    expected_source = dt * (power * 1e6 - float(np.sum(brem * volumes)))
    assert record.source_energy_j == pytest.approx(expected_source, rel=1e-12, abs=1e-12)


def test_public_species_step_accepts_radial_diffusivity() -> None:
    """The public coupled solver routes a spatial species coefficient without flattening it."""
    config = Path(__file__).parents[1] / "validation/iter_validated_config.json"
    ts = TransportSolver(config, nr=25, multi_ion=True)
    density = 2 + 0.5 * ts.rho**2
    density[0] = density[1]
    ts.n_D = density.copy()
    ts.n_T = np.zeros(25)
    ts.n_He = np.zeros(25)
    ts.n_impurity = np.zeros(25)
    ts.ne = density.copy()
    ts.Ti = np.full(25, 2.0)
    ts.Te = np.ones(25)
    profile = 0.3 * (1 + 0.5 * ts.rho**2)
    ts.D_species = profile
    dt = 1e-5
    expected_rate = 4 * 0.5 * 0.3 / 4**2 * (1 + ts.rho**2 + 0.75 * 0.5 * ts.drho**2)
    ts.evolve_profiles(dt, 0.0, enforce_conservation=True)
    np.testing.assert_allclose(ts.n_D[2:-1], (density + dt * expected_rate)[2:-1], rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(profile, 0.3 * (1 + 0.5 * ts.rho**2))
    assert ts.particle_balance_error < 1e-12
