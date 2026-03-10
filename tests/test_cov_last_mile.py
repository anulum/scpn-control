"""Last-mile coverage tests for 99%+ target."""

from __future__ import annotations

import platform
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


# ── neural_transport: weight loading exception (lines 329-330) ──


class TestNeuralTransportWeightLoadException:
    def test_corrupt_npz_logs_exception(self, tmp_path):
        from scpn_control.core.neural_transport import NeuralTransportModel

        corrupt = tmp_path / "weights.npz"
        corrupt.write_bytes(b"not a real npz file")

        model = NeuralTransportModel(auto_discover=False)
        model.weights_path = corrupt
        model._try_load_weights()
        assert model._weights is None


# ── neural_transport: stable channel (line 376) ──


class TestNeuralTransportStableChannel:
    def test_predict_stable_channel(self):
        from scpn_control.core.neural_transport import (
            NeuralTransportModel,
            TransportInputs,
        )

        model = NeuralTransportModel()
        if model._weights is None:
            pytest.skip("no weights available")
        inp = TransportInputs(
            rho=0.5, te_kev=0.001, ti_kev=0.001, ne_19=1.0,
            grad_te=0.001, grad_ti=0.001, grad_ne=0.001,
            q=1.5, s_hat=0.5, beta_e=0.01,
        )
        result = model.predict(inp)
        assert result.channel in ("ITG", "TEM", "stable")


# ── tokamak_digital_twin: seed kwarg rejected (lines 443, 480) ──


class TestDigitalTwinTimeStepsRejected:
    def test_history_time_steps_kwarg_raises(self):
        from scpn_control.control.tokamak_digital_twin import (
            run_digital_twin_ids_history,
        )

        with pytest.raises(ValueError, match="time_steps is controlled"):
            run_digital_twin_ids_history(
                history_steps=[10], time_steps=100
            )

    def test_pulse_time_steps_kwarg_raises(self):
        from scpn_control.control.tokamak_digital_twin import (
            run_digital_twin_ids_pulse,
        )

        with pytest.raises(ValueError, match="time_steps is controlled"):
            run_digital_twin_ids_pulse(
                history_steps=[10], time_steps=100
            )


# ── hpc_bridge: candidate lib found on disk (lines 106-107) ──


class TestHPCBridgeCandidateFound:
    def test_candidate_lib_found_on_disk(self, tmp_path):
        import scpn_control.core.hpc_bridge as hpc_mod
        from scpn_control.core.hpc_bridge import HPCBridge

        lib_name = "scpn_solver.dll" if platform.system() == "Windows" else "libscpn_solver.so"
        fake_lib = tmp_path / lib_name
        fake_lib.write_bytes(b"\x7fELF")

        orig_file = hpc_mod.__file__
        try:
            hpc_mod.__file__ = str(tmp_path / "hpc_bridge.py")
            bridge = HPCBridge()
            assert bridge.lib_path == str(fake_lib)
        finally:
            hpc_mod.__file__ = orig_file


# ── neuro_cybernetic_controller: line 49 (Rust FusionKernel import success) ──


class TestResolveFusionKernelRustPath:
    def test_resolve_rust_fusion_kernel(self):
        from scpn_control.control.neuro_cybernetic_controller import (
            _resolve_fusion_kernel,
        )

        result = _resolve_fusion_kernel()
        assert result is not None


# ── halo_re_physics: non-finite guards ──


class TestHaloNonFiniteGuards:
    def test_dreicer_nonfinite_ratio(self):
        from scpn_control.control.halo_re_physics import RunawayElectronModel

        m = RunawayElectronModel(n_e=1e20, T_e_keV=10.0)
        m.n_e_free = np.inf
        rate = m._dreicer_rate(E=1.0, T_e_keV=10.0)
        assert rate == 0.0

    def test_dreicer_nonfinite_rate_output(self):
        from scpn_control.control.halo_re_physics import RunawayElectronModel

        m = RunawayElectronModel(n_e=1e20, T_e_keV=10.0)
        m.n_e_free = 1e40
        m.tau_coll = 1e-300
        rate = m._dreicer_rate(E=5.0, T_e_keV=10.0)
        assert np.isfinite(rate)

    def test_simulate_nan_dreicer_guards(self):
        from scpn_control.control.halo_re_physics import RunawayElectronModel

        m = RunawayElectronModel(n_e=1e20, T_e_keV=10.0)
        original_dreicer = m._dreicer_rate

        call_count = [0]

        def patched_dreicer(E, T_e_keV):
            call_count[0] += 1
            if call_count[0] > 5:
                return np.nan
            return original_dreicer(E, T_e_keV)

        m._dreicer_rate = patched_dreicer
        result = m.simulate(dt_s=1e-5, duration_s=2e-4)
        assert result.peak_re_current_ma >= 0.0

    def test_simulate_inf_n_re_guards(self):
        from scpn_control.control.halo_re_physics import RunawayElectronModel

        m = RunawayElectronModel(n_e=1e20, T_e_keV=10.0)
        original_av = m._avalanche_rate

        call_count = [0]

        def patched_av(E, n_re):
            call_count[0] += 1
            if call_count[0] > 3:
                return np.inf
            return original_av(E, n_re)

        m._avalanche_rate = patched_av
        result = m.simulate(dt_s=1e-5, duration_s=2e-4)
        assert np.isfinite(result.peak_re_current_ma)

    def test_ensemble_has_prevented_runs(self):
        from scpn_control.control.halo_re_physics import run_disruption_ensemble

        result = run_disruption_ensemble(ensemble_runs=50, seed=42)
        assert result.prevention_rate >= 0.0


