# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Deep Coverage Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Tests targeting deep coverage gaps: disruption_episode, IDS helpers,
train_predictor, evaluate_predictor, coil warning, visualize."""
from __future__ import annotations

import numpy as np
import pytest


# ── run_disruption_episode (disruption_contracts.py lines 235-397) ───

from scpn_control.control.advanced_soc_fusion_learning import FusionAIAgent
from scpn_control.control.disruption_contracts import (
    GlobalDesignExplorer,
    run_disruption_episode,
)

_SKIP_NO_GDE = GlobalDesignExplorer is None


class TestRunDisruptionEpisode:
    @pytest.mark.skipif(_SKIP_NO_GDE, reason="GlobalDesignExplorer not available")
    def test_returns_valid_summary(self):
        rng = np.random.default_rng(42)
        agent = FusionAIAgent(epsilon=0.1)
        explorer = GlobalDesignExplorer()
        result = run_disruption_episode(
            rng=rng, rl_agent=agent, base_tbr=1.05, explorer=explorer,
        )
        assert 0.0 <= result["risk_before"] <= 1.0
        assert 0.0 <= result["risk_after"] <= 1.0
        assert "wall_damage_index" in result
        assert "prevented" in result

    @pytest.mark.skipif(_SKIP_NO_GDE, reason="GlobalDesignExplorer not available")
    def test_deterministic_with_same_seed(self):
        def _run(seed):
            rng = np.random.default_rng(seed)
            agent = FusionAIAgent(epsilon=0.0)
            explorer = GlobalDesignExplorer()
            return run_disruption_episode(
                rng=rng, rl_agent=agent, base_tbr=1.05, explorer=explorer,
            )

        r1 = _run(123)
        r2 = _run(123)
        assert r1["disturbance"] == r2["disturbance"]
        assert r1["risk_before"] == r2["risk_before"]


# ── run_real_shot_replay edge cases ──────────────────────────────────

from scpn_control.control.disruption_contracts import run_real_shot_replay


class TestRealShotReplayDeep:
    def _build_shot(self, n, disruptive=True):
        t = np.linspace(0.0, 0.01 * n, n, dtype=np.float64)
        return {
            "time_s": t,
            "Ip_MA": np.full(n, 12.0),
            "beta_N": np.full(n, 2.0),
            "n1_amp": np.full(n, 0.05),
            "n2_amp": np.full(n, 0.02),
            "dBdt_gauss_per_s": np.full(n, 0.3),
            "is_disruption": disruptive,
            "disruption_time_idx": n - 3 if disruptive else -1,
        }

    def test_disruptive_shot_has_risk_fields(self):
        agent = FusionAIAgent(epsilon=0.0)
        shot = self._build_shot(50, disruptive=True)
        out = run_real_shot_replay(
            shot_data=shot, rl_agent=agent,
            risk_threshold=0.01, spi_trigger_risk=0.02, window_size=8,
        )
        assert out["n_steps"] == 50
        assert isinstance(out["spi_triggered"], bool)
        assert "detection_lead_ms" in out

    def test_safe_shot_no_spi(self):
        agent = FusionAIAgent(epsilon=0.0)
        shot = self._build_shot(50, disruptive=False)
        out = run_real_shot_replay(
            shot_data=shot, rl_agent=agent,
            risk_threshold=0.99, spi_trigger_risk=0.99, window_size=8,
        )
        assert out["n_steps"] == 50
        assert out["spi_triggered"] is False


# ── digital twin IDS helpers (skip if HAS_IMAS=False) ────────────────

from scpn_control.control.tokamak_digital_twin import (
    run_digital_twin_ids,
    run_digital_twin_ids_history,
    run_digital_twin_ids_pulse,
)

try:
    from scpn_control.io.imas_connector import digital_twin_summary_to_ids
    _HAS_IMAS = True
except ImportError:
    _HAS_IMAS = False


class TestDigitalTwinIDS:
    @pytest.mark.skipif(not _HAS_IMAS, reason="IMAS connector not available")
    def test_ids_roundtrip(self):
        result = run_digital_twin_ids(
            time_steps=10, seed=0, save_plot=False, verbose=False,
        )
        assert isinstance(result, dict)

    @pytest.mark.skipif(not _HAS_IMAS, reason="IMAS connector not available")
    def test_ids_history(self):
        result = run_digital_twin_ids_history(
            [5, 10], seed=0, save_plot=False, verbose=False,
        )
        assert isinstance(result, dict)

    @pytest.mark.skipif(not _HAS_IMAS, reason="IMAS connector not available")
    def test_ids_pulse(self):
        result = run_digital_twin_ids_pulse(
            [5, 10], seed=0, save_plot=False, verbose=False,
        )
        assert isinstance(result, dict)

    def test_ids_history_rejects_time_steps_kwarg(self):
        with pytest.raises(ValueError, match="time_steps"):
            run_digital_twin_ids_history(
                [5], time_steps=20, seed=0, save_plot=False, verbose=False,
            )

    def test_ids_pulse_rejects_time_steps_kwarg(self):
        with pytest.raises(ValueError, match="time_steps"):
            run_digital_twin_ids_pulse(
                [5], time_steps=20, seed=0, save_plot=False, verbose=False,
            )

    def test_ids_history_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            run_digital_twin_ids_history(
                [], seed=0, save_plot=False, verbose=False,
            )


# ── fusion_control_room: coil update TypeError warning ───────────────

from scpn_control.control.fusion_control_room import run_control_room


class _KernelBadCoils:
    """Kernel where coil 'current' is None — float(None) raises TypeError."""
    def __init__(self, _cfg=None):
        self.cfg = {"coils": [
            {"current": None}, {"current": 0}, {"current": 0},
            {"current": 0}, {"current": None},
        ]}
        self.R = np.linspace(5.8, 6.4, 10)
        self.Z = np.linspace(-0.3, 0.3, 10)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = 1.0 - ((self.RR - 6.2) ** 2 + self.ZZ ** 2)

    def solve_equilibrium(self):
        pass


class TestCoilWarningPath:
    def test_bad_coil_type_triggers_warning(self):
        s = run_control_room(
            sim_duration=3, seed=0,
            save_animation=False, save_report=False,
            kernel_factory=_KernelBadCoils,
        )
        assert s["steps"] == 3


# ── neuro_cybernetic visualize ────────────────────────────────────────

mpl = pytest.importorskip("matplotlib")
import matplotlib
matplotlib.use("Agg")

import scpn_control.control.neuro_cybernetic_controller as nc_mod
from scpn_control.control.neuro_cybernetic_controller import (
    NeuroCyberneticController,
    run_neuro_cybernetic_control,
)


class _NCKernel:
    def __init__(self, _cfg: str) -> None:
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


class TestNeuroCyberneticVisualize:
    @pytest.fixture(autouse=True)
    def _no_neurocore(self, monkeypatch):
        monkeypatch.setattr(nc_mod, "SC_NEUROCORE_AVAILABLE", False)

    def test_visualize_produces_file(self, tmp_path):
        nc = NeuroCyberneticController(
            "dummy.json", seed=42, shot_duration=5, kernel_factory=_NCKernel,
        )
        nc.run_shot(save_plot=False, verbose=False)
        out = str(tmp_path / "nc_viz.png")
        result = nc.visualize("Test Shot", output_path=out, verbose=False)
        assert result == out

    def test_run_with_save_plot_true(self, tmp_path):
        out = str(tmp_path / "nc_run.png")
        s = run_neuro_cybernetic_control(
            config_file="dummy.json", shot_duration=5, seed=42,
            save_plot=True, verbose=False, output_path=out,
            kernel_factory=_NCKernel,
        )
        assert s["plot_saved"] is True


# ── evaluate_predictor ───────────────────────────────────────────────

from scpn_control.control.disruption_predictor import evaluate_predictor


class _DummyPredictor:
    def predict(self, seq):
        return float(np.mean(seq))


class TestEvaluatePredictor:
    def test_basic_evaluation(self):
        model = _DummyPredictor()
        X_test = [np.ones(32) * 0.8, np.zeros(32), np.ones(32) * 0.9]
        y_test = [1, 0, 1]
        result = evaluate_predictor(model, X_test, y_test, threshold=0.5)
        assert 0.0 <= result["accuracy"] <= 1.0
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert "confusion_matrix" in result

    def test_with_times(self):
        model = _DummyPredictor()
        X_test = [np.ones(32) * 0.8, np.zeros(32)]
        y_test = [1, 0]
        times = [0.05, 0.01]
        result = evaluate_predictor(model, X_test, y_test, times_test=times)
        assert "recall_at_10ms" in result
        assert "recall_at_50ms" in result

    def test_all_correct(self):
        model = _DummyPredictor()
        X_test = [np.ones(32), np.zeros(32)]
        y_test = [1, 0]
        result = evaluate_predictor(model, X_test, y_test, threshold=0.5)
        assert result["accuracy"] == 1.0


# ── train_predictor (with torch) ─────────────────────────────────────

class TestTrainPredictor:
    def test_train_minimal(self, tmp_path):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        from scpn_control.control.disruption_predictor import train_predictor
        model_path = str(tmp_path / "test_model.pth")
        model, info = train_predictor(
            seq_len=32, n_shots=8, epochs=2,
            model_path=model_path, seed=0, save_plot=False,
        )
        assert model is not None
        assert info["seq_len"] == 32
        assert info["epochs"] == 2
        assert (tmp_path / "test_model.pth").exists()

    def test_load_or_train_retrain_path(self, tmp_path):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        from scpn_control.control.disruption_predictor import load_or_train_predictor
        model_path = str(tmp_path / "retrain_model.pth")
        model, meta = load_or_train_predictor(
            model_path=model_path,
            seq_len=32,
            force_retrain=False,
            train_if_missing=True,
            train_kwargs={"n_shots": 8, "epochs": 2, "save_plot": False},
            allow_fallback=True,
        )
        assert model is not None
        assert meta.get("trained") is True
        assert meta.get("fallback") is False

    def test_load_checkpoint(self, tmp_path):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        from scpn_control.control.disruption_predictor import (
            train_predictor,
            load_or_train_predictor,
        )
        path = str(tmp_path / "ckpt.pth")
        train_predictor(seq_len=32, n_shots=8, epochs=2, model_path=path, seed=0, save_plot=False)
        model, meta = load_or_train_predictor(
            model_path=path, seq_len=32, allow_fallback=False,
        )
        assert model is not None
        assert meta.get("fallback") is False

    def test_corrupt_checkpoint_fallback(self, tmp_path):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        from scpn_control.control.disruption_predictor import load_or_train_predictor
        bad_path = tmp_path / "corrupt.pth"
        torch.save({"state_dict": {"bogus_key": torch.tensor([1.0])}}, bad_path)
        model, meta = load_or_train_predictor(
            model_path=str(bad_path), seq_len=32, allow_fallback=True,
        )
        assert model is None
        assert meta["fallback"] is True
        assert "checkpoint_load_failed" in meta["reason"]
