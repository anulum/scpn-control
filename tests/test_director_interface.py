# Tests for Director Interface and RuleBasedDirector fallback.

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
