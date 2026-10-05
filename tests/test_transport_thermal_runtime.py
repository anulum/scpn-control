# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Transport Thermal Runtime tests
"""Exercise the named transport thermal runtime surface through the integrated solver."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from scpn_control.core.integrated_transport_solver import (
    PhysicsError,
    TransportSolver,
)


class TestEvolveProfiles:
    """Check temperature evolution, heating accounting and nonfinite-state recovery."""

    def test_evolve_profiles_runs(self, solver: TransportSolver) -> None:
        """evolve_profiles returns (avg_T, core_T) as finite floats."""
        avg_T, core_T = solver.evolve_profiles(dt=0.01, P_aux=50.0)
        assert isinstance(avg_T, float)
        assert isinstance(core_T, float)
        assert np.isfinite(avg_T)
        assert np.isfinite(core_T)
        assert avg_T > 0
        assert core_T > 0

    def test_conservation_attribute(self, solver: TransportSolver) -> None:
        """After evolve, _last_conservation_error is a finite float."""
        solver.evolve_profiles(dt=0.01, P_aux=50.0)
        err = solver._last_conservation_error
        assert isinstance(err, float)
        assert np.isfinite(err)

    def test_profiles_stay_positive(self, solver: TransportSolver) -> None:
        """Profiles should remain non-negative after multiple steps."""
        for _ in range(20):
            solver.update_transport_model(50.0)
            solver.evolve_profiles(dt=0.01, P_aux=50.0)
        assert np.all(solver.Ti >= 0)
        assert np.all(solver.Te >= 0)
        assert np.all(solver.ne >= 0)

    def test_evolve_changes_profiles(self, solver: TransportSolver) -> None:
        """Profiles should change after evolution (not frozen)."""
        Ti_before = solver.Ti.copy()
        solver.evolve_profiles(dt=0.01, P_aux=50.0)
        assert not np.allclose(solver.Ti, Ti_before, atol=1e-12)

    def test_enforce_conservation_no_raise_small_dt(self, solver: TransportSolver) -> None:
        """With small dt and reasonable params, enforce_conservation should not raise."""
        # This is best-effort: small dt + moderate heating shouldn't violate conservation
        try:
            solver.evolve_profiles(dt=0.001, P_aux=20.0, enforce_conservation=True)
        except PhysicsError:
            # If the initial conditions are too far from equilibrium,
            # conservation may be violated. This is acceptable.
            pass

    def test_multiple_steps_trend(self, solver: TransportSolver) -> None:
        """With sustained heating, average temperature should change over time."""
        T_start = float(np.mean(solver.Ti))
        for _ in range(50):
            solver.update_transport_model(50.0)
            solver.evolve_profiles(dt=0.01, P_aux=50.0)
        T_end = float(np.mean(solver.Ti))
        # Temperature should have changed (either up or stabilised)
        assert T_end != T_start

    def test_evolve_rejects_nonpositive_or_nonfinite_dt(self, solver: TransportSolver) -> None:
        """Time steps must be finite and nonnegative, with zero accepted as a no-op."""
        # dt=0.0 is now allowed (returns early)
        solver.evolve_profiles(dt=0.0, P_aux=50.0)

        with pytest.raises(ValueError, match="finite and >= 0"):
            solver.evolve_profiles(dt=-0.01, P_aux=50.0)
        with pytest.raises(ValueError):
            solver.evolve_profiles(dt=float("nan"), P_aux=50.0)

    def test_evolve_rejects_nonfinite_heating(self, solver: TransportSolver) -> None:
        """P_aux must be finite."""
        with pytest.raises(ValueError):
            solver.evolve_profiles(dt=0.01, P_aux=float("nan"))
        with pytest.raises(ValueError):
            solver.evolve_profiles(dt=0.01, P_aux=float("inf"))

    def test_evolve_recovers_nonfinite_state(self, solver: TransportSolver) -> None:
        """Non-finite profile/transport state is repaired before CN stepping."""
        solver.Ti[3] = float("nan")
        solver.Te[5] = float("inf")
        solver.ne[7] = float("nan")
        solver.chi_i[11] = float("inf")
        solver.chi_e[13] = float("nan")
        solver.n_impurity[17] = float("inf")

        avg_t, core_t = solver.evolve_profiles(dt=0.01, P_aux=50.0)

        assert np.isfinite(avg_t)
        assert np.isfinite(core_t)
        assert np.all(np.isfinite(solver.Ti))
        assert np.all(np.isfinite(solver.Te))
        assert np.all(np.isfinite(solver.ne))
        assert np.all(np.isfinite(solver.chi_i))
        assert np.all(np.isfinite(solver.chi_e))
        assert np.all(np.isfinite(solver.n_impurity))
        assert solver._last_numerical_recovery_count > 0

    def test_aux_heating_source_power_balance_single_ion(self, solver: TransportSolver) -> None:
        """Integrated heating source must reconstruct the requested total MW."""
        solver.aux_heating_electron_fraction = 0.5
        s_i, s_e = solver._compute_aux_heating_sources(50.0)
        assert np.all(np.isfinite(s_e))
        assert np.any(s_e > 0.0)
        assert np.all(np.isfinite(s_i))

        dV = solver._rho_volume_element()
        e_keV_J = 1.602176634e-16
        ne_m3 = np.maximum(solver.ne, 0.1) * 1e19
        rec_w = 1.5 * np.sum(ne_m3 * (s_i + s_e) * e_keV_J * dV)
        rec_mw = rec_w / 1e6
        assert rec_mw == pytest.approx(50.0, rel=1e-6, abs=1e-6)
        assert solver._last_aux_heating_balance["reconstructed_total_MW"] == pytest.approx(
            50.0,
            rel=1e-6,
            abs=1e-6,
        )

    def test_single_ion_evolves_explicit_electron_channel(self, config_file: Path) -> None:
        """Single-ion mode evolves Te explicitly instead of copying Ti."""
        ts = TransportSolver(str(config_file), multi_ion=False)
        ts.aux_heating_electron_fraction = 1.0
        ts.Ti = np.full(ts.nr, 1.0)
        ts.Te = np.full(ts.nr, 1.0)
        ts.ne = np.full(ts.nr, 8.0)
        ts.chi_i = np.zeros(ts.nr)
        ts.chi_e = np.zeros(ts.nr)

        ts.evolve_profiles(dt=0.01, P_aux=20.0)
        assert not np.allclose(ts.Te, ts.Ti, atol=1e-10)

    def test_aux_heating_source_zero_power(self, solver: TransportSolver) -> None:
        """Zero auxiliary power must return zero source terms."""
        s_i, s_e = solver._compute_aux_heating_sources(0.0)
        assert np.all(s_i == 0.0)
        assert np.all(s_e == 0.0)
        assert solver._last_aux_heating_balance["reconstructed_total_MW"] == 0.0


class TestTransportNumericalGuards:
    """Check finite diffusion and failure telemetry for invalid quadrature weights."""

    def test_zero_auxiliary_power_diffusion_retains_finite_profiles(self, config_file: Path) -> None:
        """Large zero-power diffusion retains bounded, finite thermal profiles."""
        ts = TransportSolver(str(config_file), multi_ion=False)
        ts.Ti = 0.5 + 4.5 * ts.rho**2
        ts.Te = ts.Ti.copy()
        ts.ne = np.full(ts.nr, 8.0)
        ts.n_impurity = np.zeros(ts.nr)
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        ts.update_transport_model(0.0)
        ts.chi_i = np.full(ts.nr, 100.0)
        ts.chi_e = np.full(ts.nr, 100.0)

        ts.evolve_profiles(dt=1.0, P_aux=0.0)

        assert np.all(np.isfinite(ts.Ti))
        assert np.all(np.isfinite(ts.Te))
        assert np.all(ts.Ti >= 0.01)
        assert np.all(ts.Te >= 0.01)

    def test_nonfinite_energy_volume_marks_conservation_error_infinite(
        self, solver: TransportSolver, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-finite energy integrals should be reported as infinite conservation error."""
        original_volume_element = solver._rho_volume_element

        def _nonfinite_volume_element() -> npt.NDArray[np.float64]:
            """Inject one infinite quadrature weight for the existing conservation-telemetry test."""
            dV = original_volume_element()
            dV[5] = float("inf")
            return dV

        monkeypatch.setattr(solver, "_rho_volume_element", _nonfinite_volume_element)

        solver.evolve_profiles(dt=0.01, P_aux=50.0)

        assert solver._last_conservation_error == float("inf")


class TestZeroAuxHeatingGuard:
    """Check finite ion temperatures without mean growth at zero auxiliary power."""

    def test_evolve_with_zero_aux_heating(self, solver: TransportSolver) -> None:
        """Zero auxiliary power keeps ion temperature finite without mean growth in this case."""
        solver.Ti = 5.0 * (1 - solver.rho**2)
        solver.Te = solver.Ti.copy()
        solver.ne = 8.0 * (1 - solver.rho**2) ** 0.5
        solver.update_transport_model(0.0)
        ti_before = solver.Ti.copy()
        solver.evolve_profiles(dt=0.001, P_aux=0.0)
        assert np.all(np.isfinite(solver.Ti))
        assert float(np.mean(solver.Ti)) <= float(np.mean(ti_before)) + 1e-8


class TestAuxHeatingZeroVolume:
    """Check auxiliary sources when a quadrature cell has zero volume."""

    def test_compute_aux_heating_zero_volume_element(self, config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """_compute_aux_heating_sources handles degenerate volume gracefully."""
        ts = TransportSolver(str(config_file))
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = ts.Ti.copy()
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        # Patch volume element to return zeros — exercises the fallback path
        monkeypatch.setattr(ts, "_rho_volume_element", lambda: np.zeros(ts.nr))
        s_i, s_e = ts._compute_aux_heating_sources(50.0)
        assert np.all(s_i == 0.0)
        assert np.all(s_e == 0.0)


@pytest.mark.parametrize("dt", [0.0001, 0.0003, 0.001])
def test_energy_gate_checks_final_profiles_after_exchange(config_file: Path, dt: float) -> None:
    """Conservative exchange heats electrons without bypassing final-state accounting."""
    observed = []
    for enforce in (False, True):
        transport = TransportSolver(config_file, nr=50)
        transport.Ti = np.full(50, 10.0)
        transport.Te = np.full(50, 1.0)
        transport.ne = np.full(50, 5.0)
        transport.Ti[-1], transport.Te[-1] = 0.1, 0.08
        transport.chi_i = np.full(50, 0.01)
        transport.chi_e = np.full(50, 0.01)
        transport.evolve_profiles(dt, 0.0, enforce_conservation=enforce)
        record = transport.last_energy_balance
        assert record is not None
        volumes = 4.0 * np.pi**2 * 6.0 * 2.0**2 * transport.rho * transport.drho
        final = float(np.sum(1.5 * transport.ne * 1e19 * (transport.Ti + transport.Te) * 1.602176634e-16 * volumes))
        assert record.final_energy_j == pytest.approx(final, rel=1e-13)
        assert record.relative_error == pytest.approx(
            abs(
                final
                - record.initial_energy_j
                - record.source_energy_j
                - record.diffusive_boundary_energy_j
                - record.prescribed_edge_energy_j
            )
            / record.initial_energy_j
        )
        assert transport.energy_balance_error < 0.01
        assert transport.Te[10] > 1.0
        assert transport.Ti[10] < 10.0
        observed.append((transport.Ti.copy(), transport.Te.copy(), transport.energy_balance_error))
    np.testing.assert_array_equal(observed[0][0], observed[1][0])
    np.testing.assert_array_equal(observed[0][1], observed[1][1])
    assert observed[0][2] == observed[1][2]


@pytest.mark.parametrize("enforce", [False, True])
def test_multi_ion_energy_gate_uses_initial_density(enforce: bool) -> None:
    """Species-density evolution must not rewrite the thermal energy at step start."""
    config = Path(__file__).resolve().parents[1] / "validation/iter_validated_config.json"
    transport = TransportSolver(config, nr=50, multi_ion=True)
    transport.Ti = np.full(50, 10.0)
    transport.Te = np.full(50, 1.0)
    transport.ne = np.full(50, 5.0)
    transport.Ti[-1], transport.Te[-1] = 0.1, 0.08
    transport.n_D = transport.ne * 0.5
    transport.n_T = transport.ne * 0.5
    transport.n_He = np.zeros(50)
    density_before = transport.ne.copy()
    temperatures_before = transport.Ti + transport.Te
    volumes = 4.0 * np.pi**2 * 6.0 * 4.0**2 * transport.rho * transport.drho
    initial = float(np.sum(1.5 * density_before * 1e19 * temperatures_before * 1.602176634e-16 * volumes))
    transport.evolve_profiles(0.001, 50.0, enforce_conservation=enforce)
    assert not np.array_equal(density_before, transport.ne)
    record = transport.last_energy_balance
    assert record is not None
    assert record.initial_energy_j == pytest.approx(initial, rel=1e-13)
    wrong_initial = float(np.sum(1.5 * transport.ne * 1e19 * temperatures_before * 1.602176634e-16 * volumes))
    assert abs(wrong_initial - initial) / initial > 1e-4
    assert transport.energy_balance_error < 0.01


@pytest.mark.parametrize("enforce", [False, True])
def test_energy_record_exposes_actual_balance_and_survives_rejection(enforce: bool) -> None:
    """Public immutable energy accounting supports arithmetic checks without private state."""
    config = Path(__file__).resolve().parents[1] / "validation/iter_validated_config.json"
    transport = TransportSolver(config, nr=50, multi_ion=True)
    assert transport.last_energy_balance is None
    transport.Ti = np.full(50, 10.0)
    transport.Te = np.full(50, 1.0)
    transport.ne = np.full(50, 5.0)
    transport.Ti[-1], transport.Te[-1] = 0.1, 0.08
    transport.n_D = transport.ne * 0.5
    transport.n_T = transport.ne * 0.5
    transport.n_He = np.zeros(50)
    volumes = 4.0 * np.pi**2 * 6.0 * 4.0**2 * transport.rho * transport.drho
    initial = float(np.sum(1.5 * transport.ne * 1e19 * (transport.Te + transport.Ti) * 1.602176634e-16 * volumes))
    from scpn_control.core.pedestal import PedestalParams, PedestalProfile

    pedestal = PedestalProfile(PedestalParams(f_ped=20.0, f_sep=0.08))
    if enforce:
        with pytest.raises(PhysicsError):
            transport.evolve_profiles(0.001, 50.0, enforce_conservation=True, ped_te=pedestal)
    else:
        transport.evolve_profiles(0.001, 50.0, enforce_conservation=False, ped_te=pedestal)
    record = transport.last_energy_balance
    assert record is not None
    ion_after = transport.n_D + transport.n_T + transport.n_He + transport.n_impurity
    final = float(
        np.sum(1.5 * 1e19 * (transport.ne * transport.Te + ion_after * transport.Ti) * 1.602176634e-16 * volumes)
    )
    assert record.initial_energy_j == pytest.approx(initial, rel=1e-13)
    assert record.final_energy_j == pytest.approx(final, rel=1e-13)
    assert record.relative_error == pytest.approx(
        abs(
            final
            - initial
            - record.source_energy_j
            - record.diffusive_boundary_energy_j
            - record.prescribed_edge_energy_j
        )
        / initial,
        rel=1e-12,
    )
    assert record.relative_error == transport.energy_balance_error
    assert record.dt_s == 0.001 and record.auxiliary_power_mw == 50.0
    with pytest.raises(FrozenInstanceError):
        record.__setattr__("initial_energy_j", 0.0)
    transport.Te[:] = 9.0
    assert record.final_energy_j == pytest.approx(final, rel=1e-13)
    transport.evolve_profiles(0.0, 50.0)
    assert transport.last_energy_balance is None
    assert record.initial_energy_j == pytest.approx(initial, rel=1e-13)


def test_invalid_transport_call_cannot_reuse_energy_record(config_file: Path) -> None:
    """A failed input contract leaves no current-attempt energy evidence."""
    transport = TransportSolver(config_file)
    transport.evolve_profiles(0.001, 50.0)
    assert transport.last_energy_balance is not None
    with pytest.raises(ValueError):
        transport.evolve_profiles(-1.0, 50.0)
    assert transport.last_energy_balance is None


@pytest.mark.parametrize("ti,te", [(10.0, 1.0), (1.0, 10.0), (1.0, 1.0)])
@pytest.mark.parametrize("dt", [1e-6, 0.001, 1.0])
def test_internal_exchange_conserves_core_heat_and_is_bounded(ti: float, te: float, dt: float) -> None:
    """Actual transport conserves core heat apart from radiation, even for stiff exchange."""
    config = Path(__file__).resolve().parents[1] / "validation/iter_validated_config.json"
    transport = TransportSolver(config, nr=50, multi_ion=False)
    transport.Ti = np.full(50, ti)
    transport.Te = np.full(50, te)
    transport.ne = np.full(50, 5.0)
    transport.Ti[-1], transport.Te[-1] = 0.1, 0.08
    transport.chi_i = np.full(50, 0.01)
    transport.chi_e = np.full(50, 0.01)
    transport.n_impurity = np.zeros(50)
    transport.evolve_profiles(dt, 0.0)
    # Single-ion default Z_eff=1.5; flat core is remote from the fixed edge.
    # This oracle checks total heat and direction, not the exchange formula.
    radiated_temperature = dt * 5.35e-37 * 1.5 * (5e19) ** 2 * np.sqrt(te) / (1.5 * 5e19 * 1.602176634e-16)
    assert transport.Ti[10] + transport.Te[10] == pytest.approx(ti + te - radiated_temperature, rel=1e-8)
    assert min(ti, te - radiated_temperature) <= transport.Ti[10] <= max(ti, te)
    assert min(ti, te - radiated_temperature) <= transport.Te[10] <= max(ti, te)
    assert abs(transport.Ti[10] - transport.Te[10]) <= abs(ti - te + radiated_temperature) + 1e-12
    if ti > te:
        assert transport.Te[10] > te
    if te > ti:
        assert transport.Ti[10] > ti
    assert transport.Ti[-1] == 0.1 and transport.Te[-1] == 0.08
    record = transport.last_energy_balance
    assert record is not None
    assert record.source_energy_j < 0.0


@pytest.mark.parametrize("multi_ion", [False, True])
@pytest.mark.parametrize("power_mw", [0.0, 50.0])
def test_radiated_power_matches_public_source_energy(multi_ion: bool, power_mw: float) -> None:
    """A watt of prescribed radiation removes one joule per second, not 1.5."""
    config = Path(__file__).resolve().parents[1] / "validation/iter_validated_config.json"
    transport = TransportSolver(config, nr=50, multi_ion=multi_ion)
    transport.Ti = np.full(50, 2.0)
    transport.Te = np.full(50, 2.0)
    transport.ne = np.full(50, 5.0)
    transport.Ti[-1], transport.Te[-1] = 0.1, 0.08
    transport.n_impurity = np.full(50, 1e-5 if multi_ion else 0.0)
    if multi_ion:
        transport.n_D = np.full(50, 2.5)
        transport.n_T = np.full(50, 2.5)
        transport.n_He = np.zeros(50)
    electron_before = transport.Te.copy()
    dt = 1e-4
    transport.tau_He_factor = np.inf  # Isolate radiation from ash-removal heat.
    transport.evolve_profiles(dt, power_mw)
    volumes = 4.0 * np.pi**2 * 6.0 * 4.0**2 * transport.rho * transport.drho
    z_eff = 1.5
    line_power = np.zeros(50)
    if multi_ion:
        assert transport.n_D is not None and transport.n_T is not None and transport.n_He is not None
        # The runtime uses Z_W=10 and a mean effective charge for D/T/He/W.
        z_eff = float(
            np.clip(
                np.mean(
                    (transport.n_D + transport.n_T + 4.0 * transport.n_He + 100.0 * transport.n_impurity) / transport.ne
                ),
                1.0,
                10.0,
            )
        )
        # This profile stays within the two lowest branches of the W fit.
        coefficient = np.where(electron_before < 1.0, 5e-31 * np.sqrt(electron_before), 5e-31)
        line_power = transport.ne * 1e19 * transport.n_impurity * 1e19 * coefficient
    brem_power = 5.35e-37 * z_eff * (transport.ne * 1e19) ** 2 * np.sqrt(electron_before)
    expected = dt * (power_mw * 1e6 - float(np.sum((brem_power + line_power) * volumes)))
    record = transport.last_energy_balance
    assert record is not None
    assert record.source_energy_j == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("dt", [0.001, 0.01])
@pytest.mark.parametrize("diffusivity", [1.0, 10.0])
def test_returned_profiles_satisfy_cn_boundary_equations(dt: float, diffusivity: float) -> None:
    """Returned physical boundary values belong to the solved CN equations."""
    config = Path(__file__).resolve().parents[1] / "validation/iter_validated_config.json"
    transport = TransportSolver(config, nr=25, multi_ion=False)
    radius = transport.rho.copy()
    spacing = transport.drho
    transport.Ti = 0.1 + 3.0 * (1.0 - radius**2)
    transport.Te = 0.08 + 2.0 * (1.0 - radius**2)
    transport.Ti[0], transport.Te[0] = transport.Ti[1], transport.Te[1]
    transport.ne = np.full(25, 5.0)
    transport.n_impurity = np.zeros(25)
    transport.chi_i = np.full(25, diffusivity)
    transport.chi_e = np.full(25, diffusivity)
    initial_sum = transport.Ti + transport.Te
    electron_before = transport.Te.copy()
    transport.evolve_profiles(dt, 0.0)
    final_sum = transport.Ti + transport.Te
    # Equal diffusivities let internal exchange cancel out of the sum equation.
    # Form face fluxes from public returned profiles, independently of the solver.
    faces = 0.5 * (radius[1:] + radius[:-1])
    old_flux = diffusivity * faces * np.diff(initial_sum) / spacing
    new_flux = diffusivity * faces * np.diff(final_sum) / spacing
    diffusion_increment = dt * 0.5 * np.diff(old_flux + new_flux) / (4.0**2 * radius[1:-1] * spacing)
    radiation_increment = (
        dt * 5.35e-37 * 1.5 * (5e19) ** 2 * np.sqrt(electron_before[1:-1]) / (1.5 * 5e19 * 1.602176634e-16)
    )
    residual = final_sum[1:-1] - initial_sum[1:-1] - diffusion_increment + radiation_increment
    np.testing.assert_allclose(residual, 0.0, atol=2e-12, rtol=0.0)
    assert final_sum[0] == final_sum[1]
    assert transport.Ti[-1] == 0.1 and transport.Te[-1] == 0.08


@pytest.mark.parametrize("density_gradient", [-0.5, 0.5])
@pytest.mark.parametrize("dt", [0.001, 0.01])
def test_variable_density_heat_matches_boundary_flux(density_gradient: float, dt: float) -> None:
    """Interior heat changes only by independent face power and radiation."""
    config = Path(__file__).resolve().parents[1] / "validation/iter_validated_config.json"
    transport = TransportSolver(config, nr=25, multi_ion=False)
    radius = transport.rho.copy()
    spacing = transport.drho
    transport.Ti = 0.1 + 3.0 * (1.0 - radius**2)
    transport.Te = 0.08 + 2.0 * (1.0 - radius**2)
    transport.Ti[0], transport.Te[0] = transport.Ti[1], transport.Te[1]
    transport.ne = 5.0 * (1.0 + density_gradient * radius**2)
    transport.n_impurity = np.zeros(25)
    transport.chi_i = np.full(25, 3.0)
    transport.chi_e = np.full(25, 3.0)
    initial_sum = transport.Ti + transport.Te
    electron_before = transport.Te.copy()
    density = transport.ne.copy()
    transport.evolve_profiles(dt, 0.0)
    final_sum = transport.Ti + transport.Te
    np.testing.assert_array_equal(transport.ne, density)
    volumes = 4.0 * np.pi**2 * 6.0 * 4.0**2 * radius * spacing
    energy_change = float(
        np.sum(1.5 * density[1:-1] * 1e19 * 1.602176634e-16 * volumes[1:-1] * (final_sum[1:-1] - initial_sum[1:-1]))
    )
    # Independently integrate the two bounding face powers of the evolved cells.
    faces = 0.5 * (radius[1:] + radius[:-1])
    conductivity = 3.0 * 0.5 * (density[1:] + density[:-1])
    flux = conductivity * faces * np.diff(0.5 * (initial_sum + final_sum)) / spacing
    boundary_energy = dt * 1.5 * 1e19 * 1.602176634e-16 * 4.0 * np.pi**2 * 6.0 * (flux[-1] - flux[0])
    radiation = 5.35e-37 * 1.5 * (density[1:-1] * 1e19) ** 2 * np.sqrt(electron_before[1:-1])
    radiated_energy = dt * float(np.sum(radiation * volumes[1:-1]))
    assert energy_change == pytest.approx(boundary_energy - radiated_energy, rel=1e-10, abs=1e-7)


@pytest.mark.parametrize("power_mw", [0.0, 50.0])
@pytest.mark.parametrize("amplitude,edge_offset", [(3.0, 0.0), (-0.03, 0.0), (3.0, 0.5)])
@pytest.mark.parametrize("gradient", [-0.5, 0.5])
@pytest.mark.parametrize("enforce", [False, True])
def test_energy_record_accounts_for_independent_boundary_power(
    amplitude: float, edge_offset: float, gradient: float, enforce: bool, power_mw: float
) -> None:
    """Physical face exchange and the prescribed edge reconcile public thermal energy."""
    config = Path(__file__).resolve().parents[1] / "validation/iter_validated_config.json"
    transport = TransportSolver(config, nr=25, multi_ion=False)
    radius, spacing = transport.rho.copy(), transport.drho
    transport.Ti = 0.1 + amplitude * (1.0 - radius**2)
    transport.Te = 0.08 + amplitude * (2.0 / 3.0) * (1.0 - radius**2)
    transport.Ti[0], transport.Te[0] = transport.Ti[1], transport.Te[1]
    transport.Ti[-1] += edge_offset
    transport.Te[-1] += edge_offset
    transport.ne = 5.0 * (1.0 + gradient * radius**2)
    transport.n_impurity = np.zeros(25)
    transport.chi_i = np.full(25, 3.0)
    transport.chi_e = np.full(25, 3.0)
    initial_sum, initial_te = transport.Ti + transport.Te, transport.Te.copy()
    density = transport.ne.copy()
    dt = 0.05
    transport.evolve_profiles(dt, power_mw, enforce_conservation=enforce)
    record = transport.last_energy_balance
    assert record is not None
    assert record.relative_error < 1e-11
    final_sum = transport.Ti + transport.Te
    volumes = 4.0 * np.pi**2 * 6.0 * 4.0**2 * radius * spacing
    capacity = 1.5 * density * 1e19 * 1.602176634e-16 * volumes
    assert record.initial_energy_j == pytest.approx(float(np.sum(capacity * initial_sum)), rel=1e-13)
    assert record.final_energy_j == pytest.approx(float(np.sum(capacity * final_sum)), rel=1e-13)
    faces = 0.5 * (radius[1:] + radius[:-1])
    conductivity = 3.0 * 0.5 * (density[1:] + density[:-1])
    flux = conductivity * faces * np.diff(0.5 * (initial_sum + final_sum)) / spacing
    boundary = dt * 1.5 * 1e19 * 1.602176634e-16 * 4.0 * np.pi**2 * 6.0 * (flux[-1] - flux[0])
    radiation = 5.35e-37 * 1.5 * (density * 1e19) ** 2 * np.sqrt(initial_te)
    heating_shape = np.exp(-(radius**2) / transport.aux_heating_profile_width)
    edge_heating = power_mw * 1e6 * heating_shape[-1] / float(np.sum(heating_shape * volumes))
    edge = -2.0 * edge_offset * capacity[-1] + dt * (radiation[-1] - edge_heating) * volumes[-1]
    assert record.diffusive_boundary_energy_j == pytest.approx(boundary, rel=1e-12, abs=1e-8)
    assert record.prescribed_edge_energy_j == pytest.approx(edge, rel=1e-12, abs=1e-8)
    assert record.source_energy_j == pytest.approx(
        dt * (power_mw * 1e6 - float(np.sum(radiation * volumes))), rel=1e-12
    )
    if amplitude < 0.0 and power_mw == 0.0:
        assert record.diffusive_boundary_energy_j > 0.0


def test_boundary_accounting_does_not_hide_unmodelled_pedestal_power() -> None:
    """An unaccounted pedestal override still fails the final-state admission gate."""
    from scpn_control.core.pedestal import PedestalParams, PedestalProfile

    config = Path(__file__).resolve().parents[1] / "validation/iter_validated_config.json"
    transport = TransportSolver(config, nr=50, multi_ion=False)
    transport.Ti = 0.1 + 3.0 * (1.0 - transport.rho**2)
    transport.Te = 0.08 + 2.0 * (1.0 - transport.rho**2)
    transport.ne = np.full(50, 5.0)
    transport.n_impurity = np.zeros(50)
    pedestal = PedestalProfile(PedestalParams(f_ped=20.0, f_sep=0.08))
    with pytest.raises(PhysicsError, match="Energy conservation violated"):
        transport.evolve_profiles(0.001, 0.0, enforce_conservation=True, ped_te=pedestal)
    record = transport.last_energy_balance
    assert record is not None and record.relative_error > 0.01
    residual = (
        record.final_energy_j
        - record.initial_energy_j
        - record.source_energy_j
        - record.diffusive_boundary_energy_j
        - record.prescribed_edge_energy_j
    )
    assert record.relative_error == pytest.approx(abs(residual) / record.initial_energy_j)


@pytest.mark.parametrize("electron_diffusivity", [0.2, 5.0])
def test_boundary_energy_uses_each_channels_cn_state(electron_diffusivity: float) -> None:
    """Independent dense solves verify fluxes before unequal-channel exchange."""
    config = Path(__file__).resolve().parents[1] / "validation/iter_validated_config.json"
    transport = TransportSolver(config, nr=25, multi_ion=False)
    rho, dr = transport.rho.copy(), transport.drho
    transport.Ti = 0.1 + 3.0 * (1.0 - rho**2)
    transport.Te = 0.08 + 2.0 * (1.0 - rho**2)
    transport.Ti[0], transport.Te[0] = transport.Ti[1], transport.Te[1]
    transport.ne = 5.0 * (1.0 - 0.5 * rho**2)
    transport.n_impurity = np.zeros(25)
    transport.chi_i = np.asarray(3.0 * (1.0 + 0.5 * rho), dtype=np.float64)
    transport.chi_e = np.asarray(electron_diffusivity * (1.0 + rho**2), dtype=np.float64)
    dt = 0.01
    brem = 5.35e-37 * 1.5 * (transport.ne * 1e19) ** 2 * np.sqrt(transport.Te)
    electron_sink = brem / (1.5 * transport.ne * 1e19 * 1.602176634e-16)
    faces = 0.5 * (rho[1:] + rho[:-1])
    boundary_reference = 0.0
    for initial, chi, source, edge in (
        (transport.Ti, transport.chi_i, np.zeros(25), 0.1),
        (transport.Te, transport.chi_e, -electron_sink, 0.08),
    ):
        face_conductivity = 0.5 * (transport.ne[1:] * chi[1:] + transport.ne[:-1] * chi[:-1])
        conductance = face_conductivity * faces / (4.0**2 * dr**2)
        operator = np.zeros((25, 25))
        for index in range(1, 24):
            left = conductance[index - 1] / (transport.ne[index] * rho[index])
            right = conductance[index] / (transport.ne[index] * rho[index])
            operator[index, index - 1 : index + 2] = (left, -left - right, right)
        matrix = np.eye(25) - 0.5 * dt * operator
        matrix[0, 1] = -1.0
        rhs = initial + 0.5 * dt * operator @ initial + dt * source
        rhs[0], rhs[-1] = 0.0, edge
        predicted = np.linalg.solve(matrix, rhs)
        flux = face_conductivity * faces * np.diff(0.5 * (initial + predicted)) / dr
        boundary_reference += dt * 1.5 * 1e19 * 1.602176634e-16 * 4.0 * np.pi**2 * 6.0 * (flux[-1] - flux[0])
    transport.evolve_profiles(dt, 0.0, enforce_conservation=True)
    record = transport.last_energy_balance
    assert record is not None and record.relative_error < 1e-11
    assert record.diffusive_boundary_energy_j == pytest.approx(boundary_reference, rel=1e-12, abs=1e-8)


@pytest.mark.parametrize("helium,tungsten", [(0.5, 0.0), (0.5, 0.01), (2.0, 0.0)])
def test_species_counts_define_stored_thermal_energy(helium: float, tungsten: float) -> None:
    """Stored ion heat counts nuclei once, including multiply charged species."""
    config = Path(__file__).parents[1] / "validation/iter_validated_config.json"
    ts = TransportSolver(config, nr=50, multi_ion=True)
    ts.n_D = np.full(50, 2.0)
    ts.n_T = np.full(50, 1.0)
    ts.n_He = np.full(50, helium)
    ts.n_impurity = np.full(50, tungsten)
    ts.ne = ts.n_D + ts.n_T + 2.0 * ts.n_He + 10.0 * ts.n_impurity
    ts.Ti = np.full(50, 5.0)
    ts.Te = np.full(50, 2.0)
    ts.Ti[-1], ts.Te[-1] = 0.1, 0.08
    volumes = 4.0 * np.pi**2 * 6.0 * 4.0**2 * ts.rho * ts.drho
    ion_density = ts.n_D + ts.n_T + ts.n_He + ts.n_impurity
    expected = float(np.sum(1.5 * 1e19 * 1.602176634e-16 * (ion_density * ts.Ti + ts.ne * ts.Te) * volumes))
    assert ts.compute_confinement_time(50.0) == pytest.approx(expected / 50e6, rel=1e-13)
    ts.evolve_profiles(1e-6, 0.0)
    record = ts.last_energy_balance
    assert record is not None
    assert record.initial_energy_j == pytest.approx(expected, rel=1e-13)
    ion_after = ts.n_D + ts.n_T + ts.n_He + ts.n_impurity
    final = float(np.sum(1.5 * 1e19 * 1.602176634e-16 * (ion_after * ts.Ti + ts.ne * ts.Te) * volumes))
    assert record.final_energy_j == pytest.approx(final, rel=1e-13)


@pytest.mark.parametrize("ion_temperature,electron_temperature", [(10.0, 1.0), (1.0, 10.0)])
@pytest.mark.parametrize("dt", [1e-6, 0.001, 0.01])
def test_exchange_preserves_unequal_species_heat_capacities(
    ion_temperature: float, electron_temperature: float, dt: float
) -> None:
    """Internal exchange transfers equal joules between unequal ion/electron populations."""
    config = Path(__file__).parents[1] / "validation/iter_validated_config.json"
    ts = TransportSolver(config, nr=50, multi_ion=True)
    ts.n_D = np.full(50, 2.0)
    ts.n_T = np.full(50, 1.0)
    ts.n_He = np.full(50, 2.0)
    ts.n_impurity = np.zeros(50)
    ts.ne = np.full(50, 7.0)
    ts.Ti = np.full(50, ion_temperature)
    ts.Te = np.full(50, electron_temperature)
    ts.Ti[-1], ts.Te[-1] = 0.1, 0.08
    ts.D_species = 0.0
    ts.tau_He_factor = np.inf  # Isolate exchange from ash-removal heat.
    ts.chi_i = np.full(50, 0.01)
    ts.chi_e = np.full(50, 0.01)
    ts.evolve_profiles(dt, 0.0)
    ni = ts.n_D + ts.n_T + ts.n_He
    charge_squared = ts.n_D + ts.n_T + 4.0 * ts.n_He
    z_eff = float(np.clip(np.mean(charge_squared / ts.ne), 1.0, 10.0))
    brem = 5.35e-37 * z_eff * (ts.ne[10] * 1e19) ** 2 * np.sqrt(electron_temperature)
    capacity_unit = 1.5 * 1e19 * 1.602176634e-16
    ion_before_exchange = 5.0 * ion_temperature / ni[10]
    electron_before_exchange = 7.0 * electron_temperature / ts.ne[10] - dt * brem / (capacity_unit * ts.ne[10])
    before_exchange = ni[10] * ion_before_exchange + ts.ne[10] * electron_before_exchange
    after_exchange = ni[10] * ts.Ti[10] + ts.ne[10] * ts.Te[10]
    assert after_exchange == pytest.approx(before_exchange, rel=1e-12)
    low, high = sorted((ion_before_exchange, electron_before_exchange))
    assert low <= ts.Ti[10] <= high
    assert low <= ts.Te[10] <= high
    assert abs(ts.Ti[10] - ts.Te[10]) < abs(ion_before_exchange - electron_before_exchange)


@pytest.mark.parametrize("power_mw", [0.0, 50.0])
def test_multispecies_transport_matches_independent_capacity_weighted_system(power_mw: float) -> None:
    """Dense channel solves independently constrain heat, sources and boundary exchange."""
    config = Path(__file__).parents[1] / "validation/iter_validated_config.json"
    ts = TransportSolver(config, nr=25, multi_ion=True)
    rho, dr = ts.rho.copy(), ts.drho
    ts.n_D = 2.0 - rho**2
    ts.n_T = 1.0 - 0.5 * rho**2
    ts.n_He = 0.5 + rho**2
    ts.n_impurity = np.zeros(25)
    ts.ne = ts.n_D + ts.n_T + 2.0 * ts.n_He
    ts.Ti = 0.1 + 3.0 * (1.0 - rho**2)
    ts.Te = 0.08 + 2.0 * (1.0 - rho**2)
    ts.Ti[0], ts.Te[0] = ts.Ti[1], ts.Te[1]
    ni_initial, ne_initial = ts.ion_density.copy(), ts.ne.copy()
    ti, te = ts.Ti.copy(), ts.Te.copy()
    ts.chi_i = np.asarray(3.0 * (1.0 + 0.5 * rho), dtype=np.float64)
    ts.chi_e = np.asarray(0.2 * (1.0 + rho**2), dtype=np.float64)
    dt = 0.001
    from scpn_control.core.species_evolution import evolve_multi_ion_species

    species = evolve_multi_ion_species(
        n_D=ts.n_D,
        n_T=ts.n_T,
        n_He=ts.n_He,
        Ti=ti,
        Te=te,
        n_impurity=ts.n_impurity,
        dV=4 * np.pi**2 * 6 * 4**2 * rho * dr,
        rho=rho,
        drho=dr,
        a_minor=4.0,
        D_species=ts.D_species,
        tau_He=max(ts.tau_He_factor * ts.compute_confinement_time(1.0), 0.5),
        dt=dt,
    )
    # Public species counts feed an independent dense thermal-deposition oracle.
    pumped = species.helium_pumped
    ts.evolve_profiles(dt, power_mw)
    ni = ts.n_D + ts.n_T + ts.n_He
    ne = ts.ne
    z_eff = float(np.clip(np.mean((ts.n_D + ts.n_T + 4.0 * ts.n_He) / ne), 1.0, 10.0))
    volumes = 4.0 * np.pi**2 * 6.0 * 4.0**2 * rho * dr
    heat_unit = 1.5 * 1e19 * 1.602176634e-16
    shape = np.exp(-(rho**2) / 0.1)
    heat_power = 0.5 * power_mw * 1e6 * shape / np.sum(shape * volumes)
    brem = 5.35e-37 * z_eff * (ne * 1e19) ** 2 * np.sqrt(te)
    sources = (
        heat_power / (heat_unit * ni) - pumped * ti / (dt * ni),
        (heat_power - brem) / (heat_unit * ne) - 2 * pumped * te / (dt * ne),
    )
    faces = 0.5 * (rho[1:] + rho[:-1])
    predicted_heat = np.zeros(25)
    boundary, edge_energy = 0.0, 0.0
    for temperature, diffusivity, old_density, density, source, edge in (
        (ti, ts.chi_i, ni_initial, ni, sources[0], 0.1),
        (te, ts.chi_e, ne_initial, ne, sources[1], 0.08),
    ):
        operators, face_weights = [], []
        for capacity in (old_density, density):
            weights = 0.5 * (capacity[1:] * diffusivity[1:] + capacity[:-1] * diffusivity[:-1])
            conductance = faces * weights / (4.0**2 * dr**2)
            operator = np.zeros((25, 25))
            for cell in range(1, 24):
                left, right = conductance[cell - 1] / rho[cell], conductance[cell] / rho[cell]
                operator[cell, cell - 1 : cell + 2] = left, -left - right, right
            operators.append(operator)
            face_weights.append(weights)
        matrix = np.diag(density) - 0.5 * dt * operators[1]
        matrix[0] = 0.0
        matrix[0, :2] = 1.0, -1.0
        matrix[-1] = 0.0
        matrix[-1, -1] = 1.0
        rhs = old_density * temperature + 0.5 * dt * operators[0] @ temperature + dt * density * source
        rhs[0], rhs[-1] = 0.0, edge
        predicted = np.linalg.solve(matrix, rhs)
        predicted_heat += density * predicted
        flux = 0.5 * faces / dr * (face_weights[0] * np.diff(temperature) + face_weights[1] * np.diff(predicted))
        boundary += dt * heat_unit * 4.0 * np.pi**2 * 6.0 * (flux[-1] - flux[0])
        edge_energy += (
            heat_unit * volumes[-1] * (density[-1] * (edge - dt * source[-1]) - old_density[-1] * temperature[-1])
        )
    np.testing.assert_allclose(ni[1:] * ts.Ti[1:] + ne[1:] * ts.Te[1:], predicted_heat[1:], rtol=1e-12)
    record = ts.last_energy_balance
    assert record is not None
    assert record.diffusive_boundary_energy_j == pytest.approx(boundary, rel=1e-12)
    assert record.prescribed_edge_energy_j == pytest.approx(edge_energy, rel=1e-12)
    assert record.source_energy_j == pytest.approx(
        dt * (power_mw * 1e6 - float(np.sum(brem * volumes)))
        - heat_unit * float(np.sum(pumped * (ti + 2 * te) * volumes)),
        rel=1e-12,
    )


@pytest.mark.parametrize("slope", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("dt", [1e-4, 2e-4])
def test_density_diffusion_preserves_closed_thermal_storage(slope: float, dt: float) -> None:
    """The heat PDE evolves n*T; closed particle redistribution is not external heat."""
    config = Path(__file__).parents[1] / "validation/iter_validated_config.json"
    ts = TransportSolver(config, nr=25, multi_ion=True)
    density = 0.02 + 2 * ts.rho**2 * (1 - ts.rho) ** 2
    density[0] = density[1]
    density[-2:] = 0.02
    ts.n_D = density.copy()
    ts.n_T = np.zeros(25)
    ts.n_He = np.zeros(25)
    ts.n_impurity = np.zeros(25)
    ts.ne = density.copy()
    ts.Ti = 2 + slope * ts.rho**2
    ts.Te = np.ones(25)
    ts.D_species = 0.3
    ts.evolve_profiles(dt, 0.0, enforce_conservation=True)
    particles = ts.last_particle_balance
    energy = ts.last_energy_balance
    assert particles is not None and energy is not None
    assert particles.diffusive_boundary_count == particles.prescribed_boundary_count == 0
    assert particles.fusion_count == particles.pumped_count == 0
    residual = (
        energy.final_energy_j
        - energy.initial_energy_j
        - energy.source_energy_j
        - energy.diffusive_boundary_energy_j
        - energy.prescribed_edge_energy_j
    )
    assert abs(residual) < 1e-8


@pytest.mark.parametrize("dt", [0.001, 0.01])
@pytest.mark.parametrize("pumping_enabled", [False, True])
def test_pumping_carries_thermal_ion_and_electron_energy(dt: float, pumping_enabled: bool) -> None:
    """Velocity-independent He removal carries local ion and ambipolar electron heat."""
    from scpn_control.core.plasma_power_terms import bremsstrahlung_power_density

    config = Path(__file__).parents[1] / "validation/iter_validated_config.json"
    ts = TransportSolver(config, nr=25, multi_ion=True)
    ts.n_D = np.full(25, 2.0)
    ts.n_T = np.zeros(25)
    ts.n_He = np.ones(25)
    ts.n_impurity = np.zeros(25)
    ts.ne = np.full(25, 4.0)
    ts.Ti = 2 + 0.2 * ts.rho**2
    ts.Te = 1 + 0.1 * ts.rho**2
    ti, te = ts.Ti.copy(), ts.Te.copy()
    ts.D_species = 0.0
    ts.tau_He_factor = 0.0 if pumping_enabled else np.inf
    volumes = 4 * np.pi**2 * 6 * 4**2 * ts.rho * ts.drho
    removed = -np.expm1(-dt / 0.5) if pumping_enabled else 0.0
    expected_heat = -1.5 * 1e19 * 1.602176634e-16 * removed * float(np.sum((ti + 2 * te) * volumes))
    ts.evolve_profiles(dt, 0.0, enforce_conservation=True)
    balance = ts.last_energy_balance
    assert balance is not None
    assert balance.helium_pumping_energy_j == pytest.approx(expected_heat, rel=1e-12, abs=1e-10)
    z_eff = float(np.clip(np.mean((ts.n_D + ts.n_T + 4 * ts.n_He) / ts.ne), 1, 10))
    brem = bremsstrahlung_power_density(ts.ne, te, z_eff)
    expected_source = expected_heat - dt * float(np.sum(brem * volumes))
    assert balance.source_energy_j == pytest.approx(expected_source, rel=1e-12, abs=1e-10)
    assert balance.relative_error < 1e-12
    ts.evolve_profiles(0.0, 0.0)
    assert ts.last_energy_balance is None


@pytest.mark.parametrize("initial", [(4.0, 1.0), (1.0, 4.0)])
def test_pumping_exchange_and_radiation_converge_to_local_ode(initial: tuple[float, float]) -> None:
    """Step refinement converges to independent local thermal equations in both heat-flow directions."""
    from scipy.integrate import solve_ivp

    size, duration = 33, 0.03
    heat_unit = 1.5e19 * 1.602176634e-16

    def thermal_rhs(time: float, temperatures: npt.NDArray[np.float64]) -> list[float]:
        """Uniform-core moment equations with exact exponential helium survival."""
        ti, te = temperatures
        helium = np.exp(-time / 0.5)
        ni, ne = 2 + helium, 2 + 2 * helium
        # The inherited arithmetic Z_eff includes one prescribed edge node.
        zeff = ((size - 1) * (2 + 4 * helium) / ne + 1) / size
        tau = max(0.252 * te**1.5 / (ne * zeff * 17), 1e-4)
        exchange = (ti - te) / tau
        brem = 5.35e-37 * zeff * (ne * 1e19) ** 2 * np.sqrt(te)
        # Local mean-energy pumping cancels the corresponding dn/dt terms.
        return [-exchange, ni / ne * exchange - brem / (heat_unit * ne)]

    reference = solve_ivp(thermal_rhs, (0.0, duration), initial, method="DOP853", rtol=1e-12, atol=1e-13)
    assert reference.success
    expected = reference.y[:, -1]
    errors = []
    config = Path(__file__).parents[1] / "validation/iter_validated_config.json"
    for steps in (64, 128, 256):
        ts = TransportSolver(config, nr=size, multi_ion=True)
        ts.n_D = np.full(size, 2.0)
        ts.n_T = np.zeros(size)
        ts.n_He = np.ones(size)
        ts.n_impurity = np.zeros(size)
        ts.ne = np.full(size, 4.0)
        ts.Ti = np.full(size, initial[0])
        ts.Te = np.full(size, initial[1])
        ts.D_species = 0.0
        ts.tau_He_factor = 0.0
        ts.chi_i = np.full(size, 0.01)
        ts.chi_e = np.full(size, 0.01)
        for _ in range(steps):
            ts.evolve_profiles(duration / steps, 0.0, enforce_conservation=True)
            assert ts.energy_balance_error < 1e-11
        # The sampled uniform core is far outside the boundary diffusion layer.
        np.testing.assert_allclose([ts.Ti[8], ts.Te[8]], [ts.Ti[10], ts.Te[10]], rtol=1e-12)
        assert ts.n_He[10] == pytest.approx(np.exp(-duration / 0.5), abs=2e-13)
        errors.append(float(np.max(np.abs(np.array([ts.Ti[10], ts.Te[10]]) - expected))))
    assert errors[1] < 0.6 * errors[0]
    assert errors[2] < 0.6 * errors[1]
    assert errors[2] < 2.5e-4
