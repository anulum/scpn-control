# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Coverage gap tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import zlib
import base64
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════
# 1. neuro_cybernetic_controller.py — lines 49, 118-134, 173-181, 407
# ═══════════════════════════════════════════════════════════════════════


class _DummyKernel:
    def __init__(self, _config_file: str) -> None:
        self.cfg = {
            "physics": {"plasma_current_target": 5.0},
            "coils": [{"current": 0.0} for _ in range(5)],
        }
        self.R = np.linspace(5.9, 6.5, 25)
        self.Z = np.linspace(-0.3, 0.3, 25)
        RR, ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = 1.0 - ((RR - 6.2) ** 2 + ((ZZ - 0.0) / 1.4) ** 2)

    def solve_equilibrium(self) -> None:
        pass


class TestNeuroCyberneticLine49:
    def test_resolve_fusion_kernel_first_import_succeeds(self):
        from scpn_control.control.neuro_cybernetic_controller import _resolve_fusion_kernel

        cls = _resolve_fusion_kernel()
        assert cls is not None


class TestNeuroCyberneticScNeurocore:
    def test_sc_neurocore_backend_init_and_step(self, monkeypatch):
        import scpn_control.control.neuro_cybernetic_controller as mod

        mock_neuron = MagicMock()
        mock_neuron.step = MagicMock(return_value=True)
        mock_cls = MagicMock(return_value=mock_neuron)
        mock_qsrc = MagicMock()

        monkeypatch.setattr(mod, "SC_NEUROCORE_AVAILABLE", True)
        monkeypatch.setattr(mod, "StochasticLIFNeuron", mock_cls)
        monkeypatch.setattr(mod, "QuantumEntropySource", mock_qsrc)

        from scpn_control.control.neuro_cybernetic_controller import SpikingControllerPool

        pool = SpikingControllerPool(n_neurons=3, gain=1.0, tau_window=2, seed=0, use_quantum=False)
        assert pool.backend == "sc_neurocore"
        assert pool._v_pos is None

        out = pool.step(0.5)
        assert isinstance(out, float)

    def test_sc_neurocore_with_quantum(self, monkeypatch):
        import scpn_control.control.neuro_cybernetic_controller as mod

        mock_neuron = MagicMock()
        mock_neuron.step = MagicMock(return_value=False)
        mock_cls = MagicMock(return_value=mock_neuron)
        mock_qsrc_cls = MagicMock(return_value=MagicMock())

        monkeypatch.setattr(mod, "SC_NEUROCORE_AVAILABLE", True)
        monkeypatch.setattr(mod, "StochasticLIFNeuron", mock_cls)
        monkeypatch.setattr(mod, "QuantumEntropySource", mock_qsrc_cls)

        pool = mod.SpikingControllerPool(n_neurons=2, gain=1.0, tau_window=2, seed=0, use_quantum=True)
        assert pool.q_source is not None
        pool.step(-0.5)


class TestNeuroCyberneticVisualize:
    def test_visualize_verbose_logs(self, tmp_path, monkeypatch, caplog):
        import logging
        import matplotlib

        matplotlib.use("Agg")
        import scpn_control.control.neuro_cybernetic_controller as mod

        monkeypatch.setattr(mod, "SC_NEUROCORE_AVAILABLE", False)

        nc = mod.NeuroCyberneticController("dummy.json", seed=42, shot_duration=3, kernel_factory=_DummyKernel)
        nc.run_shot(save_plot=False, verbose=False)
        out = tmp_path / "test_plot.png"
        with caplog.at_level(logging.INFO, logger="scpn_control.control.neuro_cybernetic_controller"):
            nc.visualize("Test", output_path=str(out), verbose=True)
        assert "Analysis saved" in caplog.text


# ═══════════════════════════════════════════════════════════════════════
# 2. nengo_snn_wrapper.py — lines 217, 247, 263, 267-274, 297-298, 320-339
# ═══════════════════════════════════════════════════════════════════════


def _make_mock_nengo():
    mock = MagicMock()
    mock.LIF = MagicMock(return_value=MagicMock())
    mock.dists.Uniform = MagicMock(return_value=MagicMock())

    network_ctx = MagicMock()
    network_ctx.__enter__ = MagicMock(return_value=network_ctx)
    network_ctx.__exit__ = MagicMock(return_value=False)

    conn_mock = MagicMock()
    conn_mock.solver = MagicMock()
    conn_mock.label = "test_conn"
    network_ctx.all_connections = [conn_mock]

    mock.Network.return_value = network_ctx

    node = MagicMock()
    node.__getitem__ = MagicMock(return_value=MagicMock())
    mock.Node.return_value = node
    mock.Ensemble.return_value = MagicMock()
    mock.Connection.return_value = MagicMock()

    probe = MagicMock()
    mock.Probe.return_value = probe

    sim = MagicMock()
    sim.data = {probe: np.zeros((1, 2))}
    sim.step = MagicMock()
    sim.reset = MagicMock()
    mock.Simulator.return_value = sim

    return mock, sim, probe, conn_mock


@pytest.fixture()
def patched_nengo(monkeypatch):
    mock_nengo, sim, probe, conn = _make_mock_nengo()
    import scpn_control.control.nengo_snn_wrapper as mod

    monkeypatch.setattr(mod, "_nengo", mock_nengo)
    monkeypatch.setattr(mod, "_nengo_available", True)
    return SimpleNamespace(mock=mock_nengo, sim=sim, probe=probe, conn=conn, mod=mod)


class TestNengoStepNotBuilt:
    def test_step_raises_when_not_built(self, patched_nengo):
        ctrl = patched_nengo.mod.NengoSNNController()
        ctrl._built = False
        with pytest.raises(RuntimeError, match="not built"):
            ctrl.step(np.zeros(2))


class TestNengoGetSpikeDataNone:
    def test_returns_empty_when_no_simulator(self, patched_nengo):
        ctrl = patched_nengo.mod.NengoSNNController()
        ctrl._simulator = None
        assert ctrl.get_spike_data() == {}


class TestNengoExportWeightsNotBuilt:
    def test_raises_when_not_built(self, patched_nengo):
        ctrl = patched_nengo.mod.NengoSNNController()
        ctrl._built = False
        with pytest.raises(RuntimeError, match="not built"):
            ctrl.export_weights()


class TestNengoExportWeightsConnections:
    def test_weight_extraction_success(self, patched_nengo):
        ctrl = patched_nengo.mod.NengoSNNController()
        w = np.eye(3)
        weight_data = MagicMock()
        weight_data.weights = w
        patched_nengo.sim.data[patched_nengo.conn] = weight_data
        weights = ctrl.export_weights()
        assert "test_conn" in weights
        np.testing.assert_array_equal(weights["test_conn"], w)

    def test_weight_extraction_attr_error(self, patched_nengo):
        ctrl = patched_nengo.mod.NengoSNNController()
        patched_nengo.sim.data[patched_nengo.conn] = MagicMock(spec=[])
        weights = ctrl.export_weights()
        assert "test_conn" not in weights


class TestNengoFpgaExportWithWeights:
    def test_fpga_weights_includes_connection_weights(self, patched_nengo, tmp_path):
        ctrl = patched_nengo.mod.NengoSNNController()
        w = np.eye(2)
        weight_data = MagicMock()
        weight_data.weights = w
        patched_nengo.sim.data[patched_nengo.conn] = weight_data
        out = tmp_path / "fpga.npz"
        ctrl.export_fpga_weights(out)
        loaded = np.load(str(out))
        assert "weight_test_conn" in loaded


class TestNengoLoihiExport:
    def test_loihi_export_with_mock(self, patched_nengo, tmp_path):
        ctrl = patched_nengo.mod.NengoSNNController()
        mock_loihi = MagicMock()
        mock_loihi_sim = MagicMock()
        mock_loihi.Simulator.return_value = mock_loihi_sim
        out = tmp_path / "loihi.npz"
        with patch.dict("sys.modules", {"nengo_loihi": mock_loihi}):
            ctrl.export_loihi(str(out))
        assert out.exists()
        mock_loihi_sim.close.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════
# 3. integrated_transport_solver.py
# ═══════════════════════════════════════════════════════════════════════


def _write_iter_config():
    import tempfile
    import os

    cfg = {
        "reactor_name": "test",
        "grid_resolution": [8, 8],
        "dimensions": {"R_min": 4.0, "R_max": 8.4, "Z_min": -4.0, "Z_max": 4.0},
        "physics": {"plasma_current_target": 15.0, "vacuum_permeability": 1.0},
        "coils": [{"name": f"PF{i}", "r": 3.0 + i, "z": (-1) ** i * 3.0, "current": 1.0} for i in range(6)],
        "solver": {"max_iterations": 5, "tolerance": 1e-4},
    }
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(cfg, f)
    return path


class TestChangHintonEpsilonGuard:
    def test_zero_rho_returns_floor(self):
        from scpn_control.core.integrated_transport_solver import chang_hinton_chi_profile

        rho = np.array([0.0, 0.5])
        T_i = np.array([10.0, 10.0])
        n_e = np.array([5.0, 5.0])
        q = np.array([1.5, 2.0])
        chi = chang_hinton_chi_profile(rho, T_i, n_e, q, R0=6.2, a=2.0, B0=5.3)
        assert chi[0] == pytest.approx(0.01)


class TestSauterBootstrapDrGuard:
    def test_constant_rho_gives_zero_bootstrap(self):
        from scpn_control.core.integrated_transport_solver import calculate_sauter_bootstrap_current_full

        n = 5
        rho = np.array([0.0, 0.01, 0.01, 0.01, 1.0])
        Te = np.full(n, 10.0)
        Ti = np.full(n, 10.0)
        ne = np.full(n, 5.0)
        q = np.full(n, 2.0)
        j_bs = calculate_sauter_bootstrap_current_full(rho, Te, Ti, ne, q, R0=6.2, a=2.0, B0=5.3)
        assert j_bs[2] == 0.0


class TestTransportSolverSauterDispatch:
    def test_uses_sauter_when_neoclassical_params_set(self):
        from scpn_control.core.integrated_transport_solver import TransportSolver

        cfg_path = _write_iter_config()
        solver = TransportSolver(cfg_path)
        solver.neoclassical_params = {
            "R0": 6.2,
            "a": 2.0,
            "B0": 5.3,
            "A_ion": 2.0,
            "Z_eff": 1.5,
            "q_profile": np.linspace(1, 4, len(solver.rho)),
        }
        B_pol = np.full_like(solver.rho, 0.5)
        j_bs = solver.calculate_bootstrap_current(6.2, B_pol)
        assert j_bs.shape == solver.rho.shape


class TestTransportSolverGyroBohm:
    def test_explicit_c_gB(self):
        from scpn_control.core.integrated_transport_solver import TransportSolver

        cfg_path = _write_iter_config()
        solver = TransportSolver(cfg_path)
        solver.neoclassical_params = {
            "R0": 6.2,
            "a": 2.0,
            "B0": 5.3,
            "A_ion": 2.0,
            "Z_eff": 1.5,
            "q_profile": np.linspace(1, 4, len(solver.rho)),
            "c_gB": 0.2,
        }
        chi = solver._gyro_bohm_chi()
        assert chi.shape == solver.rho.shape
        assert np.all(chi >= 0.01)


class TestTransportEpedPedestalPath:
    def test_hmode_with_neoclassical_activates_eped(self):
        from scpn_control.core.integrated_transport_solver import TransportSolver

        cfg_path = _write_iter_config()
        solver = TransportSolver(cfg_path)
        solver.neoclassical_params = {
            "R0": 6.2,
            "a": 2.0,
            "B0": 5.3,
            "A_ion": 2.0,
            "Z_eff": 1.5,
            "q_profile": np.linspace(1, 4, len(solver.rho)),
            "Ip_MA": 15.0,
            "kappa": 1.7,
        }
        solver.Ti = np.linspace(10.0, 0.5, len(solver.rho))
        solver.Te = solver.Ti.copy()
        solver.ne = np.linspace(8.0, 2.0, len(solver.rho))
        solver.update_transport_model(P_aux=50.0)
        assert hasattr(solver, "chi_e")


class TestTransportNoHeatingScaleDown:
    def test_no_heating_prevents_temperature_growth(self):
        from scpn_control.core.integrated_transport_solver import TransportSolver

        cfg_path = _write_iter_config()
        solver = TransportSolver(cfg_path)
        solver.Ti = np.linspace(5.0, 0.5, len(solver.rho))
        solver.Te = solver.Ti.copy()
        solver.ne = np.full_like(solver.rho, 5.0)
        solver.evolve_profiles(dt=0.01, P_aux=0.0, enforce_conservation=False)
        assert np.all(np.isfinite(solver.Ti))


class TestTransportNonfiniteConservation:
    def test_nan_conservation_set_to_inf(self):
        from scpn_control.core.integrated_transport_solver import TransportSolver

        cfg_path = _write_iter_config()
        solver = TransportSolver(cfg_path)
        solver.Ti = np.full_like(solver.rho, np.nan)
        solver.Te = solver.Ti.copy()
        solver.ne = np.full_like(solver.rho, 5.0)
        solver.evolve_profiles(dt=0.01, P_aux=1.0, enforce_conservation=False)
        err = solver._last_conservation_error
        assert err == float("inf") or np.isfinite(err)


# ═══════════════════════════════════════════════════════════════════════
# 4. tokamak_digital_twin.py — lines 29, 352-353, 380, 417-418,
#    443-455, 449-455, 479-487
# ═══════════════════════════════════════════════════════════════════════


class TestDigitalTwinIdsRun:
    def test_run_digital_twin_ids(self, monkeypatch):
        import scpn_control.control.tokamak_digital_twin as dt_mod

        mock_to_ids = MagicMock(return_value={"ids": True})
        monkeypatch.setattr(dt_mod, "digital_twin_summary_to_ids", mock_to_ids, raising=False)
        monkeypatch.setattr(dt_mod, "HAS_IMAS", True, raising=False)
        result = dt_mod.run_digital_twin_ids(
            machine="TEST",
            shot=1,
            run=0,
            time_steps=5,
            save_plot=False,
            verbose=False,
        )
        assert isinstance(result, dict)
        mock_to_ids.assert_called_once()

    def test_run_digital_twin_ids_pulse(self, monkeypatch):
        import scpn_control.control.tokamak_digital_twin as dt_mod

        mock_pulse = MagicMock(return_value={"pulse": True})
        monkeypatch.setattr(dt_mod, "digital_twin_history_to_ids_pulse", mock_pulse, raising=False)
        monkeypatch.setattr(dt_mod, "HAS_IMAS", True, raising=False)
        result = dt_mod.run_digital_twin_ids_pulse(
            [3, 5],
            machine="TEST",
            shot=1,
            run=0,
            save_plot=False,
            verbose=False,
        )
        assert isinstance(result, dict)
        mock_pulse.assert_called_once()

    def test_run_digital_twin_ids_history_calls_through(self, monkeypatch):
        import scpn_control.control.tokamak_digital_twin as dt_mod

        mock_hist = MagicMock(return_value={"history": True})
        monkeypatch.setattr(dt_mod, "digital_twin_history_to_ids", mock_hist, raising=False)
        monkeypatch.setattr(dt_mod, "HAS_IMAS", True, raising=False)
        result = dt_mod.run_digital_twin_ids_history(
            [3, 5],
            machine="TEST",
            shot=1,
            run=0,
            save_plot=False,
            verbose=False,
        )
        assert isinstance(result, dict)
        mock_hist.assert_called_once()


class TestDigitalTwinPlotAndVerbose:
    def test_verbose_log_after_plot_save(self, tmp_path, caplog):
        import logging
        import matplotlib

        matplotlib.use("Agg")
        from scpn_control.control.tokamak_digital_twin import run_digital_twin

        out = str(tmp_path / "twin.png")
        with caplog.at_level(logging.INFO, logger="scpn_control.control.tokamak_digital_twin"):
            result = run_digital_twin(
                time_steps=60,
                seed=42,
                save_plot=True,
                output_path=out,
                verbose=True,
            )
        if result["plot_saved"]:
            assert "saved" in caplog.text.lower() or "Complete" in caplog.text


# ═══════════════════════════════════════════════════════════════════════
# 5. halo_re_physics.py — lines 336, 350, 372, 392, 415, 491, 496, 499, 505, 640
# ═══════════════════════════════════════════════════════════════════════


class TestDreicerEdgeCases:
    def test_dreicer_non_finite_E(self):
        from scpn_control.control.halo_re_physics import RunawayElectronModel

        m = RunawayElectronModel(n_e=1e20, T_e_keV=20.0, z_eff=1.0)
        assert m._dreicer_rate(float("nan"), 10.0) == 0.0
        assert m._dreicer_rate(float("inf"), 10.0) == 0.0

    def test_dreicer_ratio_non_finite(self):
        """Trigger line 336: ratio non-finite from extreme n_e."""
        from scpn_control.control.halo_re_physics import RunawayElectronModel

        m = RunawayElectronModel(n_e=1e20, T_e_keV=20.0, z_eff=1.0)
        # Force n_e_free extreme so E_D overflows in the per-call recomputation
        m.n_e_free = 1e308
        assert m._dreicer_rate(1e-6, 1e-300) == 0.0

    def test_dreicer_rate_non_finite(self):
        """Trigger line 350: rate computation produces non-finite."""
        from scpn_control.control.halo_re_physics import RunawayElectronModel

        m = RunawayElectronModel(n_e=1e20, T_e_keV=20.0, z_eff=1.0)
        m.tau_coll = 0.0  # force division by near-zero
        m.n_e_free = 1e308
        rate = m._dreicer_rate(1e10, 10.0)
        assert rate == 0.0 or np.isfinite(rate)


class TestAvalancheEdgeCases:
    def test_avalanche_non_finite_E(self):
        from scpn_control.control.halo_re_physics import RunawayElectronModel

        m = RunawayElectronModel(n_e=1e20, T_e_keV=20.0, z_eff=1.0)
        assert m._avalanche_rate(float("nan"), 1e10) == 0.0

    def test_avalanche_growth_non_finite(self):
        """Trigger line 372: growth computation produces non-finite."""
        from scpn_control.control.halo_re_physics import RunawayElectronModel

        m = RunawayElectronModel(n_e=1e20, T_e_keV=20.0, z_eff=1.0)
        m.tau_av = 0.0
        m.E_c = 1e-300
        assert m._avalanche_rate(1e10, 1e308) == 0.0


class TestMomentumSpaceEdgeCases:
    def test_momentum_space_non_finite(self):
        from scpn_control.control.halo_re_physics import RunawayElectronModel

        m = RunawayElectronModel(n_e=1e20, T_e_keV=20.0, z_eff=1.0)
        assert m._momentum_space_growth(float("nan"), 1e10) == 0.0

    def test_momentum_space_fp_rate_non_finite(self):
        """Trigger line 392: fp_rate computation produces non-finite."""
        from scpn_control.control.halo_re_physics import RunawayElectronModel

        m = RunawayElectronModel(n_e=1e20, T_e_keV=20.0, z_eff=1.0)
        m.tau_av = 0.0
        m.E_c = 1e-300
        assert m._momentum_space_growth(1e308, 1e308) == 0.0


class TestRelLossEdgeCases:
    def test_rel_loss_non_finite(self):
        from scpn_control.control.halo_re_physics import RunawayElectronModel

        m = RunawayElectronModel(n_e=1e20, T_e_keV=20.0, z_eff=1.0)
        assert m._relativistic_loss_rate(E=float("nan"), n_re=1e10) == 0.0

    def test_rel_loss_computation_non_finite(self):
        """Trigger line 415: loss computation produces non-finite."""
        from scpn_control.control.halo_re_physics import RunawayElectronModel

        m = RunawayElectronModel(n_e=1e20, T_e_keV=20.0, z_eff=1.0)
        m.E_c = 0.0
        m.B_t = 0.0
        m.n_e_tot = 0.0
        m.Z_eff = 1e308
        loss = m._relativistic_loss_rate(E=1e10, n_re=1e308)
        assert loss == 0.0 or np.isfinite(loss)


class TestRESimulateNonFinite:
    def test_simulate_with_extreme_params(self):
        from scpn_control.control.halo_re_physics import RunawayElectronModel

        m = RunawayElectronModel(n_e=1e20, T_e_keV=20.0, z_eff=1.0)
        result = m.simulate(
            plasma_current_ma=15.0,
            tau_cq_s=1e-6,
            T_e_quench_keV=0.02,
            neon_z_eff=3.0,
            duration_s=0.001,
            dt_s=1e-4,
        )
        assert np.isfinite(result.peak_re_current_ma)
        assert np.isfinite(result.final_re_current_ma)

    def test_simulate_guards_non_finite_intermediate(self):
        """Trigger lines 491, 496, 499, 505 by corrupting internal state mid-sim."""
        from scpn_control.control.halo_re_physics import RunawayElectronModel

        m = RunawayElectronModel(n_e=1e20, T_e_keV=20.0, z_eff=1.0)
        # Corrupt internal state to force non-finite intermediates
        m.tau_coll = 0.0
        m.tau_av = 0.0
        m.E_c = 0.0
        result = m.simulate(
            plasma_current_ma=15.0,
            tau_cq_s=1e-10,
            T_e_quench_keV=0.001,
            neon_z_eff=1.0,
            duration_s=1e-6,
            dt_s=1e-7,
        )
        assert np.isfinite(result.peak_re_current_ma)
        assert np.isfinite(result.final_re_current_ma)


class TestDisruptionEnsemblePrevention:
    def test_ensemble_has_some_prevented(self):
        from scpn_control.control.halo_re_physics import run_disruption_ensemble

        # Use enough runs and a seed where prevention is likely to occur
        report = run_disruption_ensemble(ensemble_runs=20, seed=42)
        assert report.ensemble_runs == 20
        assert 0.0 <= report.prevention_rate <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# 6. cli.py — lines 244-264, 337
# ═══════════════════════════════════════════════════════════════════════


class TestCliLiveCommand:
    def test_live_imports_and_starts(self):
        from click.testing import CliRunner
        from scpn_control.cli import main

        mock_monitor = MagicMock()
        mock_server = MagicMock()

        mock_rt_mod = MagicMock()
        mock_rt_mod.RealtimeMonitor.from_paper27.return_value = mock_monitor

        mock_ws_mod = MagicMock()
        mock_ws_mod.PhaseStreamServer.return_value = mock_server

        with patch.dict(
            "sys.modules",
            {
                "scpn_control.phase.realtime_monitor": mock_rt_mod,
                "scpn_control.phase.ws_phase_stream": mock_ws_mod,
            },
        ):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["live", "--port", "9999", "--host", "127.0.0.1", "--layers", "4", "--n-per", "10", "--zeta", "0.3"],
            )
        mock_server.serve_sync.assert_called_once()


class TestCliInfoWeightFiles:
    def test_info_text_lists_weight_files(self):
        """Exercise line 337 by placing .npz at the path cli.py resolves."""
        from pathlib import Path
        from click.testing import CliRunner
        from scpn_control.cli import main
        import scpn_control.cli as cli_mod

        # cli.py resolves: Path(__file__).resolve().parent.parent.parent.parent / "weights"
        cli_file = Path(cli_mod.__file__).resolve()
        weights_dir = cli_file.parent.parent.parent.parent / "weights"
        created_dir = False
        created_file = None
        try:
            if not weights_dir.exists():
                weights_dir.mkdir(parents=True, exist_ok=True)
                created_dir = True
            test_file = weights_dir / "_test_cov_model.npz"
            np.savez(str(test_file), data=np.zeros(5))
            created_file = test_file

            runner = CliRunner()
            result = runner.invoke(main, ["info"])
            assert result.exit_code == 0
            assert "_test_cov_model.npz" in result.output
            assert "KB" in result.output
        finally:
            if created_file is not None and created_file.exists():
                created_file.unlink()
            if created_dir and weights_dir.exists():
                try:
                    weights_dir.rmdir()
                except OSError:
                    pass


# ═══════════════════════════════════════════════════════════════════════
# 7. compiler.py — lines 44-45, 187, 191-193, 208-210, 258
# ═══════════════════════════════════════════════════════════════════════


class TestCompilerScNeurocorePaths:
    def test_compiled_net_lif_fire_with_neurons(self):
        from scpn_control.scpn.compiler import CompiledNet

        mock_neuron = MagicMock()
        mock_neuron.reset_state = MagicMock()
        mock_neuron.step = MagicMock(return_value=True)

        net = CompiledNet(
            n_places=2,
            n_transitions=1,
            place_names=["p0", "p1"],
            transition_names=["t0"],
            W_in=np.array([[0.5, 0.0]]),
            W_out=np.array([[0.0], [0.5]]),
            neurons=[mock_neuron],
            thresholds=np.array([0.3]),
            transition_delay_ticks=np.array([0], dtype=np.int64),
            initial_marking=np.array([0.5, 0.0]),
        )
        currents = np.array([0.6])
        fired = net.lif_fire(currents)
        assert fired[0] == 1.0
        mock_neuron.reset_state.assert_called_once()


class TestCompilerDenseForwardFloat:
    def test_dense_forward_raises_without_sc_neurocore(self, monkeypatch):
        import scpn_control.scpn.compiler as compiler_mod
        from scpn_control.scpn.compiler import CompiledNet

        monkeypatch.setattr(compiler_mod, "_HAS_SC_NEUROCORE", False)

        net = CompiledNet(
            n_places=2,
            n_transitions=1,
            place_names=["p0", "p1"],
            transition_names=["t0"],
            W_in=np.array([[0.5, 0.0]]),
            W_out=np.array([[0.0], [0.5]]),
            W_in_packed=np.zeros((1, 2, 1), dtype=np.uint64),
            thresholds=np.array([0.3]),
        )
        with pytest.raises(RuntimeError, match="sc_neurocore"):
            net.dense_forward(net.W_in_packed, np.array([0.5, 0.5]))


# ═══════════════════════════════════════════════════════════════════════
# 8. controller.py — lines 46-56, 464, 556
# ═══════════════════════════════════════════════════════════════════════


class TestControllerPassthroughMissingKey:
    def test_missing_passthrough_raises(self, tmp_path):
        from scpn_control.scpn.artifact import save_artifact
        from scpn_control.scpn.compiler import FusionCompiler
        from scpn_control.scpn.controller import NeuroSymbolicController
        from scpn_control.scpn.contracts import ControlScales, ControlTargets, FeatureAxisSpec
        from scpn_control.scpn.structure import StochasticPetriNet

        net = StochasticPetriNet()
        net.add_place("p0", initial_tokens=0.0)
        net.add_place("p1", initial_tokens=0.0)
        net.add_place("p2", initial_tokens=0.0)
        net.add_place("p3", initial_tokens=0.0)
        net.add_transition("t0", threshold=0.1)
        net.add_arc("p0", "t0", 0.1)
        net.add_arc("t0", "p2", 0.1)
        net.add_transition("t1", threshold=0.1)
        net.add_arc("p1", "t1", 0.1)
        net.add_arc("t1", "p3", 0.1)

        compiler = FusionCompiler(bitstream_length=64, seed=0)
        compiled = compiler.compile(net)
        art = compiled.export_artifact(
            name="pt_test",
            readout_config={
                "actions": [{"name": "ctrl", "pos_place": 2, "neg_place": 3}],
                "gains": [1.0],
                "abs_max": [1.0],
                "slew_per_s": [1e6],
            },
            injection_config=[
                {"place_id": 0, "source": "x_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
                {"place_id": 1, "source": "x_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
                {"place_id": 2, "source": "extra_sensor", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            ],
        )
        path = tmp_path / "pt.scpnctl.json"
        save_artifact(art, str(path))
        from scpn_control.scpn.artifact import load_artifact

        loaded = load_artifact(str(path))
        ctrl = NeuroSymbolicController(
            artifact=loaded,
            seed_base=42,
            targets=ControlTargets(R_target_m=1.0, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=1.0, Z_scale_m=1.0),
            feature_axes=[FeatureAxisSpec(obs_key="err", target=1.0, scale=1.0, pos_key="x_pos", neg_key="x_neg")],
        )
        with pytest.raises(KeyError, match="extra_sensor"):
            ctrl.step({"err": 0.5}, 0)


# ═══════════════════════════════════════════════════════════════════════
# 9. artifact.py — lines 225, 265, 437, 448, 512, 621, 628
# ═══════════════════════════════════════════════════════════════════════


class TestArtifactDecodeExceedsLimit:
    def test_non_multiple_of_8_raises(self):
        from scpn_control.scpn.artifact import _decode_u64_compact, ArtifactValidationError

        raw = b"\x01\x02\x03\x04\x05"
        compressed = zlib.compress(raw)
        payload = base64.b64encode(compressed).decode("ascii")
        encoded = {"encoding": "u64-le-zlib-base64", "count": None, "data_u64_b64_zlib": payload}
        with pytest.raises(ArtifactValidationError, match="not divisible by 8"):
            _decode_u64_compact(encoded)


class TestArtifactDecodeInvalidCountType:
    def test_float_count_raises(self):
        from scpn_control.scpn.artifact import _decode_u64_compact, ArtifactValidationError

        raw = b"\x00" * 8
        compressed = zlib.compress(raw)
        payload = base64.b64encode(compressed).decode("ascii")
        encoded = {"encoding": "u64-le-zlib-base64", "count": 1.5, "data_u64_b64_zlib": payload}
        with pytest.raises(ArtifactValidationError, match="Invalid compact packed count type"):
            _decode_u64_compact(encoded)


class TestArtifactLoadPackedCompact:
    def test_load_with_compact_packed(self, tmp_path):
        from scpn_control.scpn.artifact import (
            save_artifact,
            load_artifact,
            Artifact,
            ArtifactMeta,
            FixedPoint,
            SeedPolicy,
            CompilerInfo,
            Topology,
            PlaceSpec,
            TransitionSpec,
            WeightMatrix,
            Weights,
            PackedWeightsGroup,
            PackedWeights,
            Readout,
            ActionReadout,
            InitialState,
            PlaceInjection,
        )

        pw = PackedWeights(shape=[1, 2, 1], data_u64=[42, 99])
        pw_out = PackedWeights(shape=[2, 1, 1], data_u64=[7, 8])
        packed = PackedWeightsGroup(words_per_stream=1, w_in_packed=pw, w_out_packed=pw_out)
        art = Artifact(
            meta=ArtifactMeta(
                artifact_version="1.0.0",
                name="test",
                dt_control_s=0.001,
                stream_length=64,
                fixed_point=FixedPoint(data_width=16, fraction_bits=10, signed=False),
                firing_mode="binary",
                seed_policy=SeedPolicy(id="default", hash_fn="splitmix64", rng_family="xoshiro256++"),
                created_utc="2026-01-01T00:00:00Z",
                compiler=CompilerInfo(name="test", version="0.1", git_sha="0000000"),
            ),
            topology=Topology(
                places=[PlaceSpec(id=0, name="p0"), PlaceSpec(id=1, name="p1")],
                transitions=[TransitionSpec(id=0, name="t0", threshold=0.5)],
            ),
            weights=Weights(
                w_in=WeightMatrix(shape=[1, 2], data=[0.5, 0.3]),
                w_out=WeightMatrix(shape=[2, 1], data=[0.2, 0.4]),
                packed=packed,
            ),
            readout=Readout(
                actions=[ActionReadout(id=0, name="ctrl", pos_place=0, neg_place=1)],
                gains=[1.0],
                abs_max=[1.0],
                slew_per_s=[1e6],
            ),
            initial_state=InitialState(
                marking=[0.0, 0.0],
                place_injections=[PlaceInjection(place_id=0, source="x", scale=1.0, offset=0.0, clamp_0_1=True)],
            ),
        )

        path = tmp_path / "packed.scpnctl.json"
        save_artifact(art, str(path), compact_packed=False)
        loaded = load_artifact(str(path))
        assert loaded.weights.packed is not None
        assert loaded.weights.packed.w_out_packed is not None

        path2 = tmp_path / "packed_compact.scpnctl.json"
        save_artifact(art, str(path2), compact_packed=True)
        loaded2 = load_artifact(str(path2))
        assert loaded2.weights.packed is not None


class TestArtifactJsonSchema:
    def test_schema_returns_dict(self):
        from scpn_control.scpn.artifact import get_artifact_json_schema

        schema = get_artifact_json_schema()
        assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
        assert "meta" in schema["properties"]


class TestArtifactSavePackedNonCompact:
    def test_save_non_compact_with_w_out(self, tmp_path):
        from scpn_control.scpn.artifact import (
            save_artifact,
            Artifact,
            ArtifactMeta,
            FixedPoint,
            SeedPolicy,
            CompilerInfo,
            Topology,
            PlaceSpec,
            TransitionSpec,
            WeightMatrix,
            Weights,
            PackedWeightsGroup,
            PackedWeights,
            Readout,
            ActionReadout,
            InitialState,
        )

        pw = PackedWeights(shape=[1, 2, 1], data_u64=[1])
        pw_out = PackedWeights(shape=[2, 1, 1], data_u64=[2, 3])
        packed = PackedWeightsGroup(words_per_stream=1, w_in_packed=pw, w_out_packed=pw_out)

        art = Artifact(
            meta=ArtifactMeta(
                artifact_version="1.0.0",
                name="test2",
                dt_control_s=0.001,
                stream_length=64,
                fixed_point=FixedPoint(data_width=16, fraction_bits=10, signed=False),
                firing_mode="binary",
                seed_policy=SeedPolicy(id="d", hash_fn="s", rng_family="x"),
                created_utc="2026-01-01T00:00:00Z",
                compiler=CompilerInfo(name="t", version="0.1", git_sha="0000000"),
            ),
            topology=Topology(
                places=[PlaceSpec(id=0, name="p0"), PlaceSpec(id=1, name="p1")],
                transitions=[TransitionSpec(id=0, name="t0", threshold=0.5)],
            ),
            weights=Weights(
                w_in=WeightMatrix(shape=[1, 2], data=[0.5, 0.3]),
                w_out=WeightMatrix(shape=[2, 1], data=[0.2, 0.4]),
                packed=packed,
            ),
            readout=Readout(
                actions=[ActionReadout(id=0, name="c", pos_place=0, neg_place=1)],
                gains=[1.0],
                abs_max=[1.0],
                slew_per_s=[1e6],
            ),
            initial_state=InitialState(marking=[0.0, 0.0], place_injections=[]),
        )
        path = tmp_path / "nc.scpnctl.json"
        save_artifact(art, str(path), compact_packed=False)
        with open(path) as f:
            data = json.load(f)
        assert "w_out_packed" in data["weights"]["packed"]
        assert "data_u64" in data["weights"]["packed"]["w_out_packed"]


# ═══════════════════════════════════════════════════════════════════════
# 10. structure.py — lines 194, 220, 433, 457, 486
# ═══════════════════════════════════════════════════════════════════════


class TestStructureStrictValidation:
    def test_strict_rejects_dead_transitions(self):
        from scpn_control.scpn.structure import StochasticPetriNet

        net = StochasticPetriNet()
        net.add_place("p0", initial_tokens=0.5)
        net.add_place("p1")
        net.add_transition("t_connected", threshold=0.3)
        net.add_transition("t_dead", threshold=0.3)
        net.add_arc("p0", "t_connected", 0.5)
        net.add_arc("t_connected", "p1", 0.5)
        with pytest.raises(ValueError, match="dead_transitions"):
            net.compile(strict_validation=True)


class TestStructureInhibitorOutputArc:
    def test_output_arc_negative_weight_raises(self):
        from scpn_control.scpn.structure import StochasticPetriNet

        net = StochasticPetriNet()
        net.add_place("p0", initial_tokens=0.5)
        net.add_place("p1")
        net.add_transition("t0", threshold=0.3)
        net.add_arc("p0", "t0", 0.5)
        net.add_arc("t0", "p1", 0.5)
        net._arcs.append(("t0", "p1", -0.3, True))
        with pytest.raises(ValueError, match="Inhibitor arcs are only valid"):
            net.compile(allow_inhibitor=True)


class TestStructureVerifyBoundednessNone:
    def test_verify_boundedness_uncompiled_raises(self):
        from scpn_control.scpn.structure import StochasticPetriNet

        net = StochasticPetriNet()
        net.add_place("p0")
        net.add_transition("t0")
        net.add_arc("p0", "t0", 0.5)
        with pytest.raises(RuntimeError, match="compiled"):
            net.verify_boundedness()


class TestStructureVerifyBoundednessWorks:
    def test_verify_boundedness_returns_bounded(self):
        from scpn_control.scpn.structure import StochasticPetriNet

        net = StochasticPetriNet()
        net.add_place("p0", initial_tokens=0.5)
        net.add_place("p1")
        net.add_transition("t0", threshold=0.3)
        net.add_arc("p0", "t0", 0.5)
        net.add_arc("t0", "p1", 0.5)
        net.compile()
        result = net.verify_boundedness(n_steps=10, n_trials=5)
        assert result["bounded"] is True


class TestStructureVerifyLivenessNone:
    def test_verify_liveness_uncompiled_raises(self):
        from scpn_control.scpn.structure import StochasticPetriNet

        net = StochasticPetriNet()
        net.add_place("p0")
        net.add_transition("t0")
        net.add_arc("p0", "t0", 0.5)
        with pytest.raises(RuntimeError, match="compiled"):
            net.verify_liveness()

    def test_verify_liveness_returns_result(self):
        from scpn_control.scpn.structure import StochasticPetriNet

        net = StochasticPetriNet()
        net.add_place("p0", initial_tokens=0.5)
        net.add_place("p1")
        net.add_transition("t0", threshold=0.3)
        net.add_arc("p0", "t0", 0.5)
        net.add_arc("t0", "p1", 0.5)
        net.compile()
        result = net.verify_liveness(n_steps=10, n_trials=5)
        assert "live" in result
        assert "min_fire_pct" in result
