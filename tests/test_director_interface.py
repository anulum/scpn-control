# Tests for Director Interface and RuleBasedDirector fallback.

import numpy as np
import pytest

from scpn_control.control.director_interface import (
    DirectorInterface,
    _RuleBasedDirector,
)


class TestRuleBasedDirector:
    def test_init_default(self):
        d = _RuleBasedDirector()
        assert d.entropy_threshold == 0.3
        assert d.history_window == 10

    def test_init_rejects_bad_threshold(self):
        with pytest.raises(ValueError, match="finite"):
            _RuleBasedDirector(entropy_threshold=float("nan"))
        with pytest.raises(ValueError, match="finite and > 0"):
            _RuleBasedDirector(entropy_threshold=0.0)

    def test_init_rejects_bad_window(self):
        with pytest.raises(ValueError, match="history_window must be >= 1"):
            _RuleBasedDirector(history_window=0)

    def test_approve_stable_low_entropy(self):
        d = _RuleBasedDirector(entropy_threshold=0.3)
        prompt = "Time=0, Ip=5.0, Stability=Stable, BrainEntropy=0.10"
        approved, score = d.review_action(prompt, "Increase Ip")
        assert approved is True
        assert score >= 0.0

    def test_deny_unstable(self):
        d = _RuleBasedDirector(entropy_threshold=0.3)
        prompt = "Time=0, Ip=5.0, Stability=Unstable, BrainEntropy=0.10"
        approved, _ = d.review_action(prompt, "Increase Ip")
        assert approved is False

    def test_deny_high_entropy(self):
        d = _RuleBasedDirector(entropy_threshold=0.3)
        prompt = "Time=0, Ip=5.0, Stability=Stable, BrainEntropy=0.80"
        approved, score = d.review_action(prompt, "Increase Ip")
        assert approved is False
        assert score > 1.0

    def test_history_window_limits(self):
        d = _RuleBasedDirector(entropy_threshold=0.3, history_window=3)
        prompt = "Time=0, Ip=5.0, Stability=Stable, BrainEntropy=0.10"
        for _ in range(10):
            d.review_action(prompt, "action")
        assert len(d._scores) == 3


class TestDirectorInterface:
    def test_init_with_fallback(self):
        di = DirectorInterface.__new__(DirectorInterface)
        # Construct manually to avoid requiring config file
        d = _RuleBasedDirector()
        di.director = d
        di.director_backend = "fallback_rule_based"
        di.step_count = 0
        di.log = []
        assert di.director_backend == "fallback_rule_based"

    def test_format_state_stable(self):
        di = DirectorInterface.__new__(DirectorInterface)
        di.nc = None  # not needed for format_state_for_director
        prompt = di.format_state_for_director(
            t=10, ip=5.0, err_r=0.01, err_z=0.01, brain_activity=[0.1, 0.2]
        )
        assert "Stability=Stable" in prompt
        assert "BrainEntropy=" in prompt

    def test_format_state_unstable(self):
        di = DirectorInterface.__new__(DirectorInterface)
        prompt = di.format_state_for_director(
            t=10, ip=5.0, err_r=0.2, err_z=0.0, brain_activity=[0.0, 0.0]
        )
        assert "Stability=Unstable" in prompt

    def test_format_state_critical(self):
        di = DirectorInterface.__new__(DirectorInterface)
        prompt = di.format_state_for_director(
            t=10, ip=5.0, err_r=0.6, err_z=0.0, brain_activity=[0.0]
        )
        assert "Stability=Critical" in prompt

    def test_format_state_rejects_nonfinite(self):
        di = DirectorInterface.__new__(DirectorInterface)
        with pytest.raises(ValueError, match="ip must be finite"):
            di.format_state_for_director(
                t=0, ip=float("nan"), err_r=0, err_z=0, brain_activity=[0.0]
            )
        with pytest.raises(ValueError, match="brain_activity"):
            di.format_state_for_director(
                t=0, ip=5.0, err_r=0, err_z=0, brain_activity=[float("inf")]
            )

    def test_format_state_rejects_nonfinite_err_r(self):
        di = DirectorInterface.__new__(DirectorInterface)
        with pytest.raises(ValueError, match="err_r must be finite"):
            di.format_state_for_director(
                t=0, ip=5.0, err_r=float("inf"), err_z=0, brain_activity=[0.0]
            )

    def test_format_state_rejects_nonfinite_err_z(self):
        di = DirectorInterface.__new__(DirectorInterface)
        with pytest.raises(ValueError, match="err_z must be finite"):
            di.format_state_for_director(
                t=0, ip=5.0, err_r=0, err_z=float("nan"), brain_activity=[0.0]
            )


# ── DirectorInterface with mock controller ─────────────────────────

class _MockKernel:
    """Minimal kernel stand-in for run_directed_mission."""
    def __init__(self):
        self.cfg = {
            "physics": {"plasma_current_target": 5.0},
            "coils": [{"current": 0.0} for _ in range(5)],
        }
        self.R = np.linspace(1.0, 5.0, 20)
        self.Z = np.linspace(-3.0, 3.0, 20)
        RR, ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = -((RR - 3.0) ** 2 + ZZ ** 2)

    def solve_equilibrium(self):
        pass


class _MockBrain:
    """Minimal SNN brain stand-in."""
    def step(self, error):
        return -0.1 * error


class _MockNeuroCyberneticController:
    """Stand-in for NeuroCyberneticController."""
    def __init__(self, _config_path):
        self.kernel = _MockKernel()
        self.brain_R = _MockBrain()
        self.brain_Z = _MockBrain()

    def initialize_brains(self, use_quantum=True):
        pass


class TestDirectorInterfaceRunMission:
    def test_run_directed_mission_returns_keys(self):
        di = DirectorInterface(
            "mock.json",
            controller_factory=_MockNeuroCyberneticController,
        )
        result = di.run_directed_mission(
            duration=20, save_plot=False, verbose=False,
        )
        for key in ("backend", "steps", "final_target_ip", "mean_abs_err_r",
                     "intervention_count", "plot_saved", "plot_error"):
            assert key in result
        assert result["steps"] == 20
        assert result["backend"] == "fallback_rule_based"
        assert result["plot_saved"] is False

    def test_run_directed_mission_deterministic(self):
        a = DirectorInterface(
            "mock.json", controller_factory=_MockNeuroCyberneticController,
        ).run_directed_mission(duration=15, rng_seed=42, save_plot=False, verbose=False)
        b = DirectorInterface(
            "mock.json", controller_factory=_MockNeuroCyberneticController,
        ).run_directed_mission(duration=15, rng_seed=42, save_plot=False, verbose=False)
        assert a["final_target_ip"] == b["final_target_ip"]
        assert a["intervention_count"] == b["intervention_count"]

    def test_run_directed_mission_rejects_bad_duration(self):
        di = DirectorInterface(
            "mock.json", controller_factory=_MockNeuroCyberneticController,
        )
        with pytest.raises(ValueError, match="duration must be >= 1"):
            di.run_directed_mission(duration=0, save_plot=False, verbose=False)

    def test_run_directed_mission_rejects_bad_glitch_start(self):
        di = DirectorInterface(
            "mock.json", controller_factory=_MockNeuroCyberneticController,
        )
        with pytest.raises(ValueError, match="glitch_start_step must be >= 0"):
            di.run_directed_mission(
                duration=10, glitch_start_step=-1, save_plot=False, verbose=False,
            )

    def test_run_directed_mission_rejects_bad_glitch_std(self):
        di = DirectorInterface(
            "mock.json", controller_factory=_MockNeuroCyberneticController,
        )
        with pytest.raises(ValueError, match="glitch_std"):
            di.run_directed_mission(
                duration=10, glitch_std=float("nan"), save_plot=False, verbose=False,
            )

    def test_injected_director(self):
        director = _RuleBasedDirector(entropy_threshold=0.5)
        di = DirectorInterface(
            "mock.json",
            controller_factory=_MockNeuroCyberneticController,
            director=director,
        )
        assert di.director_backend == "injected"

    def test_no_fallback_raises(self):
        with pytest.raises(ImportError, match="allow_fallback=False"):
            DirectorInterface(
                "mock.json",
                allow_fallback=False,
                controller_factory=_MockNeuroCyberneticController,
            )

    def test_glitch_triggers_intervention(self):
        di = DirectorInterface(
            "mock.json", controller_factory=_MockNeuroCyberneticController,
        )
        result = di.run_directed_mission(
            duration=100, glitch_start_step=10, glitch_std=5000.0,
            save_plot=False, verbose=False,
        )
        assert result["intervention_count"] > 0

    def test_verbose_output(self, capsys):
        di = DirectorInterface(
            "mock.json", controller_factory=_MockNeuroCyberneticController,
        )
        di.run_directed_mission(
            duration=20, save_plot=False, verbose=True,
        )
        out = capsys.readouterr().out
        assert "DIRECTOR-GHOSTED FUSION MISSION" in out
        assert "fallback_rule_based" in out

    def test_verbose_intervention_prints(self, capsys):
        di = DirectorInterface(
            "mock.json", controller_factory=_MockNeuroCyberneticController,
        )
        di.run_directed_mission(
            duration=100, glitch_start_step=5, glitch_std=50000.0,
            save_plot=False, verbose=True, rng_seed=0,
        )
        out = capsys.readouterr().out
        assert "APPROVED" in out or "DENIED" in out

    def test_verbose_with_denied_intervention(self, capsys):
        class _HighEntropyDirector:
            def review_action(self, prompt, action):
                return False, 9.9

        di = DirectorInterface(
            "mock.json",
            controller_factory=_MockNeuroCyberneticController,
            director=_HighEntropyDirector(),
        )
        result = di.run_directed_mission(
            duration=20, save_plot=False, verbose=True,
        )
        out = capsys.readouterr().out
        assert "DENIED" in out
        assert "INTERVENTION" in out
        assert result["intervention_count"] > 0
