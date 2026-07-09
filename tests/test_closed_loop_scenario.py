# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Closed-loop scenario tests

"""Tests for bounded integrated-scenario closed-loop wiring."""

from __future__ import annotations

from typing import Any, cast

import pytest

from scpn_control.control.closed_loop_scenario import (
    closed_loop_scenario_result_to_dict,
    run_integrated_scenario_closed_loop,
)
from scpn_control.core.integrated_scenario import ScenarioConfig


def _fast_config() -> ScenarioConfig:
    """Return a fast deterministic scenario for closed-loop tests."""
    return ScenarioConfig(
        R0=0.93,
        a=0.58,
        B0=1.0,
        kappa=2.0,
        delta=0.4,
        Ip_MA=1.0,
        P_aux_MW=0.5,
        P_eccd_MW=0.0,
        P_nbi_MW=0.0,
        t_start=0.0,
        t_end=0.03,
        dt=0.01,
        include_sawteeth=False,
        include_ntm=False,
        include_sol=False,
        include_elm=False,
        include_stability=True,
    )


def test_closed_loop_scenario_runs_controller_plant_and_audit() -> None:
    """Closed-loop run emits controller steps and a passing replay audit."""
    result = run_integrated_scenario_closed_loop(_fast_config(), max_steps=3)
    payload = closed_loop_scenario_result_to_dict(result)

    assert result.schema_version == "closed-loop-integrated-scenario-v1"
    assert result.coupling_audit.passed is True
    assert len(result.steps) == 3
    assert result.initial_w_thermal_j > 0.0
    assert result.final_w_thermal_j > 0.0
    assert result.final_abs_error_fraction >= 0.0
    assert all(step.applied_p_aux_mw >= 0.0 for step in result.steps)
    audit = cast(dict[str, Any], payload["coupling_audit"])
    steps = cast(list[dict[str, Any]], payload["steps"])
    assert audit["passed"] is True
    assert steps[0]["step_index"] == 0


def test_closed_loop_scenario_default_config_is_bounded() -> None:
    """Default helper keeps the demo short and claim-bounded."""
    result = run_integrated_scenario_closed_loop(max_steps=2)

    assert len(result.steps) == 2
    assert "bounded controller-to-plant" in result.claim_status
    assert result.coupling_audit.metadata.scenario_name == "closed_loop_integrated_scenario"


def test_closed_loop_scenario_clips_upper_auxiliary_power() -> None:
    """Positive feedback trim is bounded by the actuator power limit."""
    result = run_integrated_scenario_closed_loop(
        _fast_config(),
        target_w_thermal_j=1.0e30,
        feedback_gain_mw=50.0,
        p_aux_bounds_mw=(0.0, 1.0),
        max_steps=1,
    )

    assert result.steps[0].commanded_p_aux_mw > 1.0
    assert result.steps[0].applied_p_aux_mw == pytest.approx(1.0)


def test_closed_loop_scenario_clips_lower_auxiliary_power() -> None:
    """Negative feedback trim is bounded at the actuator lower limit."""
    result = run_integrated_scenario_closed_loop(
        _fast_config(),
        target_w_thermal_j=1.0,
        feedback_gain_mw=50.0,
        p_aux_bounds_mw=(0.0, 1.0),
        max_steps=1,
    )

    assert result.steps[0].commanded_p_aux_mw < 0.0
    assert result.steps[0].applied_p_aux_mw == pytest.approx(0.0)


def test_closed_loop_scenario_rejects_invalid_max_steps() -> None:
    """Closed-loop max step domains fail closed."""
    with pytest.raises(ValueError, match="max_steps"):
        run_integrated_scenario_closed_loop(_fast_config(), max_steps=0)


def test_closed_loop_scenario_rejects_invalid_target() -> None:
    """Closed-loop target domains fail closed."""
    with pytest.raises(ValueError, match="target_w_thermal_j"):
        run_integrated_scenario_closed_loop(_fast_config(), target_w_thermal_j=0.0)


def test_closed_loop_scenario_rejects_nonfinite_target() -> None:
    """Closed-loop finite-scalar guard rejects NaN targets."""
    with pytest.raises(ValueError, match="target_w_thermal_j must be finite"):
        run_integrated_scenario_closed_loop(_fast_config(), target_w_thermal_j=float("nan"))


def test_closed_loop_scenario_rejects_invalid_feedback_gain() -> None:
    """Closed-loop feedback-gain domains fail closed."""
    with pytest.raises(ValueError, match="feedback_gain_mw"):
        run_integrated_scenario_closed_loop(_fast_config(), feedback_gain_mw=-1.0)


def test_closed_loop_scenario_rejects_reversed_power_bounds() -> None:
    """Closed-loop power bounds must be ordered."""
    with pytest.raises(ValueError, match="upper bound"):
        run_integrated_scenario_closed_loop(_fast_config(), p_aux_bounds_mw=(2.0, 1.0))


def test_closed_loop_scenario_rejects_negative_power_bound() -> None:
    """Closed-loop power bounds must be non-negative."""
    with pytest.raises(ValueError, match="p_aux_bounds_mw\\[0\\]"):
        run_integrated_scenario_closed_loop(_fast_config(), p_aux_bounds_mw=(-1.0, 1.0))
