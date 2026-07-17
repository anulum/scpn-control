# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Contracts
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Petri Net Contract Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Tests for SPN logic invariants and formal data contracts.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.scpn.contracts import (
    DEFAULT_PHYSICS_INVARIANTS,
    ActionSpec,
    PhysicsInvariant,
    check_all_invariants,
    check_boundedness,
    check_deadlock,
    check_marking_conservation,
    check_physics_invariant,
    decode_actions,
    evaluate_safety_invariants,
    should_trigger_mitigation,
    _is_satisfied,
    _seed64,
)


def test_marking_conservation() -> None:
    """Verify that token conservation is correctly detected."""
    m1 = [1.0, 2.0, 3.0]
    m2 = [2.0, 2.0, 2.0]  # Sum is 6.0 for both

    assert check_marking_conservation(m1, m2) is None

    m3 = [1.0, 2.0, 3.1]  # Sum is 6.1
    violation = check_marking_conservation(m1, m3)
    assert violation is not None
    assert violation.invariant.name == "marking_conservation"
    assert "Token sum mismatch" in violation.message


def test_deadlock_detection() -> None:
    """Verify that deadlock (no enabled transitions) is correctly detected."""
    # Incidence matrix: (places, transitions)
    # W_minus = max(0, -incidence_matrix)
    # T0 consumes 1 token from P0
    # T1 consumes 1 token from P1
    incidence = np.array(
        [
            [-1, 0],  # P0
            [0, -1],  # P1
            [1, 1],  # P2 (output)
        ]
    )

    # Enabled case: tokens in P0
    assert check_deadlock([1.0, 0.0, 0.0], incidence) is None

    # Deadlock case: no tokens in P0 or P1
    violation = check_deadlock([0.0, 0.0, 5.0], incidence)
    assert violation is not None
    assert violation.invariant.name == "deadlock"

    # Terminal case: no tokens but marked as terminal
    assert check_deadlock([0.0, 0.0, 5.0], incidence, is_terminal=True) is None


def test_boundedness_check() -> None:
    """Verify that place capacity limits are enforced."""
    marking = [10.0, 50.0, 100.0]

    # Nominal case
    assert check_boundedness(marking, max_capacity=200.0) is None

    # Violation case
    violation = check_boundedness(marking, max_capacity=80.0)
    assert violation is not None
    assert violation.invariant.name == "boundedness"
    assert "Place 2 exceeds capacity" in violation.message


def test_physics_invariant_invalid_comparator() -> None:
    with pytest.raises(ValueError, match="Invalid comparator"):
        PhysicsInvariant(name="x", description="d", threshold=1.0, comparator="eq")


def test_physics_invariant_non_finite_threshold() -> None:
    with pytest.raises(ValueError, match="finite"):
        PhysicsInvariant(name="x", description="d", threshold=float("inf"), comparator="gt")


def test_is_satisfied_gte() -> None:
    assert _is_satisfied("gte", 1.0, 1.0) is True
    assert _is_satisfied("gte", 0.9, 1.0) is False


def test_is_satisfied_lte() -> None:
    assert _is_satisfied("lte", 1.0, 1.0) is True
    assert _is_satisfied("lte", 1.1, 1.0) is False


def test_is_satisfied_gt_lt() -> None:
    assert _is_satisfied("gt", 2.0, 1.0) is True
    assert _is_satisfied("lt", 0.5, 1.0) is True


def test_is_satisfied_rejects_unknown_comparator() -> None:
    """Unknown comparators fail closed if an invalid value reaches the helper."""
    with pytest.raises(ValueError, match="Unknown comparator"):
        _is_satisfied("eq", 1.0, 1.0)


def test_seed64_is_deterministic_u64() -> None:
    """Controller stochastic seeds derive deterministically from base and stream id."""
    seed = _seed64(42, "shot-a")
    assert seed == _seed64(42, "shot-a")
    assert seed != _seed64(42, "shot-b")
    assert 0 <= seed < 2**64


def test_decode_actions_applies_gain_slew_and_absolute_limits() -> None:
    """Marking readout mutates the previous-action vector after limiting."""
    previous = [0.0]
    result = decode_actions(
        marking=[0.8, 0.2],
        actions_spec=[ActionSpec(name="coil_current", pos_place=0, neg_place=1)],
        gains=[2.0],
        abs_max=[1.0],
        slew_per_s=[10.0],
        dt=0.1,
        prev=previous,
    )
    assert result == {"coil_current": pytest.approx(1.0)}
    assert previous == [pytest.approx(1.0)]


def test_decode_actions_rejects_vector_length_mismatch() -> None:
    """Action specification and limiter vectors are a single aligned contract."""
    with pytest.raises(ValueError, match="equal lengths"):
        decode_actions(
            marking=[0.0, 1.0],
            actions_spec=[ActionSpec(name="coil_current", pos_place=0, neg_place=1)],
            gains=[],
            abs_max=[1.0],
            slew_per_s=[1.0],
            dt=0.1,
            prev=[0.0],
        )


def test_decode_actions_rejects_invalid_dt() -> None:
    """Control-tick duration must be finite and positive before slew limiting."""
    with pytest.raises(ValueError, match="dt must be finite"):
        decode_actions(
            marking=[0.0, 1.0],
            actions_spec=[ActionSpec(name="coil_current", pos_place=0, neg_place=1)],
            gains=[1.0],
            abs_max=[1.0],
            slew_per_s=[1.0],
            dt=0.0,
            prev=[0.0],
        )


def test_decode_actions_rejects_invalid_place_indices() -> None:
    """Action readout rejects negative and out-of-bounds place indices."""
    with pytest.raises(ValueError, match=">= 0"):
        decode_actions(
            marking=[0.0, 1.0],
            actions_spec=[ActionSpec(name="coil_current", pos_place=-1, neg_place=1)],
            gains=[1.0],
            abs_max=[1.0],
            slew_per_s=[1.0],
            dt=0.1,
            prev=[0.0],
        )
    with pytest.raises(ValueError, match="out of bounds"):
        decode_actions(
            marking=[0.0, 1.0],
            actions_spec=[ActionSpec(name="coil_current", pos_place=0, neg_place=2)],
            gains=[1.0],
            abs_max=[1.0],
            slew_per_s=[1.0],
            dt=0.1,
            prev=[0.0],
        )


@pytest.mark.parametrize(
    ("marking", "gains", "abs_max", "slew", "prev"),
    [
        ([float("nan"), 0.2], [2.0], [1.0], [10.0], [0.0]),  # non-finite used positive place
        ([0.8, float("inf")], [2.0], [1.0], [10.0], [0.0]),  # non-finite used negative place
        ([0.8, 0.2], [float("nan")], [1.0], [10.0], [0.0]),  # non-finite gain
        ([0.8, 0.2], [2.0], [float("inf")], [10.0], [0.0]),  # non-finite abs_max
        ([0.8, 0.2], [2.0], [1.0], [float("nan")], [0.0]),  # non-finite slew
        ([0.8, 0.2], [2.0], [1.0], [10.0], [float("nan")]),  # non-finite prev
    ],
)
def test_decode_actions_rejects_non_finite_inputs(marking, gains, abs_max, slew, prev) -> None:
    """A non-finite marking/gain/limit/slew/prev cannot produce a safe command."""
    with pytest.raises(ValueError, match="must be finite"):
        decode_actions(
            marking=marking,
            actions_spec=[ActionSpec(name="coil_current", pos_place=0, neg_place=1)],
            gains=gains,
            abs_max=abs_max,
            slew_per_s=slew,
            dt=0.1,
            prev=prev,
        )


def test_decode_actions_leaves_prev_unchanged_on_non_finite_reject() -> None:
    """A rejected decode does not mutate the previous-action vector (no NaN poison)."""
    previous = [0.5]
    with pytest.raises(ValueError, match="must be finite"):
        decode_actions(
            marking=[float("nan"), 0.2],
            actions_spec=[ActionSpec(name="coil_current", pos_place=0, neg_place=1)],
            gains=[2.0],
            abs_max=[1.0],
            slew_per_s=[10.0],
            dt=0.1,
            prev=previous,
        )
    assert previous == [0.5]


def test_decode_actions_ignores_non_finite_in_unused_marking_place() -> None:
    """A non-finite value in a place the action never reads is not a barrier."""
    previous = [0.0]
    result = decode_actions(
        marking=[0.8, 0.2, float("nan")],  # place 2 is unused by this action
        actions_spec=[ActionSpec(name="coil_current", pos_place=0, neg_place=1)],
        gains=[2.0],
        abs_max=[1.0],
        slew_per_s=[10.0],
        dt=0.1,
        prev=previous,
    )
    assert result == {"coil_current": pytest.approx(1.0)}


@pytest.mark.parametrize(
    ("abs_max", "slew"),
    [
        ([-1.0], [10.0]),  # negative saturation inverts the abs clamp
        ([1.0], [-10.0]),  # negative slew rate inverts the rate clamp
    ],
)
def test_decode_actions_rejects_negative_limits(abs_max, slew) -> None:
    """A negative abs_max or slew rate would invert the clamp, so it is rejected."""
    with pytest.raises(ValueError, match="decode limits must be non-negative"):
        decode_actions(
            marking=[0.8, 0.2],
            actions_spec=[ActionSpec(name="coil_current", pos_place=0, neg_place=1)],
            gains=[2.0],
            abs_max=abs_max,
            slew_per_s=slew,
            dt=0.1,
            prev=[0.0],
        )


def test_decode_actions_allows_zero_limits() -> None:
    """Zero abs_max/slew is a valid (degenerate) clamp, not rejected as an inversion."""
    result = decode_actions(
        marking=[0.8, 0.2],
        actions_spec=[ActionSpec(name="coil_current", pos_place=0, neg_place=1)],
        gains=[2.0],
        abs_max=[0.0],
        slew_per_s=[0.0],
        dt=0.1,
        prev=[0.5],
    )
    assert result["coil_current"] == pytest.approx(0.0)


def test_check_physics_invariant_non_finite_value() -> None:
    inv = PhysicsInvariant(name="q", description="d", threshold=1.0, comparator="gt")
    result = check_physics_invariant(inv, float("nan"))
    assert result is not None
    assert result.severity == "critical"
    assert result.margin == float("inf")


def test_check_physics_invariant_violation_warning() -> None:
    inv = PhysicsInvariant(name="q", description="d", threshold=1.0, comparator="gt")
    result = check_physics_invariant(inv, 0.9)
    assert result is not None
    assert result.severity == "warning"
    assert result.margin == pytest.approx(0.1)


def test_check_physics_invariant_violation_critical() -> None:
    inv = PhysicsInvariant(name="q", description="d", threshold=1.0, comparator="gt")
    result = check_physics_invariant(inv, 0.5)
    assert result is not None
    assert result.severity == "critical"


def test_check_physics_invariant_satisfied() -> None:
    inv = PhysicsInvariant(name="q", description="d", threshold=1.0, comparator="gt")
    assert check_physics_invariant(inv, 1.5) is None


def test_check_all_invariants_default() -> None:
    violations = check_all_invariants({"q_min": 0.5, "beta_N": 1.0})
    names = [v.invariant.name for v in violations]
    assert "q_min" in names
    assert "beta_N" not in names


def test_check_all_invariants_custom() -> None:
    inv = PhysicsInvariant(name="x", description="d", threshold=5.0, comparator="lt")
    violations = check_all_invariants({"x": 10.0}, invariants=[inv])
    assert len(violations) == 1
    assert violations[0].invariant.name == "x"


def test_check_all_invariants_no_violations() -> None:
    violations = check_all_invariants({"q_min": 2.0, "beta_N": 1.0})
    assert violations == []


def test_should_trigger_mitigation_critical() -> None:
    inv = PhysicsInvariant(name="q", description="d", threshold=1.0, comparator="gt")
    v = check_physics_invariant(inv, 0.5)
    assert v is not None
    assert should_trigger_mitigation([v]) is True


def test_should_trigger_mitigation_warning_only() -> None:
    inv = PhysicsInvariant(name="q", description="d", threshold=1.0, comparator="gt")
    v = check_physics_invariant(inv, 0.9)
    assert v is not None
    assert v.severity == "warning"
    assert should_trigger_mitigation([v]) is False


def test_should_trigger_mitigation_empty() -> None:
    assert should_trigger_mitigation([]) is False


# ── Fail-closed safety monitor (CF-5/SP-4) ──────────────────────────────────
# evaluate_safety_invariants treats every invariant channel as mandatory: a
# sensor dropout must surface as a critical violation, never a silent pass.

_NOMINAL_SNAPSHOT = {
    "q_min": 2.0,
    "beta_N": 1.5,
    "greenwald": 0.8,
    "T_i": 10.0,
    "energy_conservation_error": 0.001,
}


def test_evaluate_safety_invariants_fails_closed_on_full_dropout() -> None:
    """Empty snapshot (all sensors lost) yields a critical violation per invariant."""
    violations = evaluate_safety_invariants({})
    assert len(violations) == len(DEFAULT_PHYSICS_INVARIANTS)
    assert {v.invariant.name for v in violations} == {inv.name for inv in DEFAULT_PHYSICS_INVARIANTS}
    assert all(v.severity == "critical" for v in violations)
    assert should_trigger_mitigation(violations) is True


def test_evaluate_safety_invariants_missing_channel_records_nan_and_infinite_margin() -> None:
    """A missing channel is critical with NaN actual value and infinite margin."""
    violations = evaluate_safety_invariants({}, invariants=list(DEFAULT_PHYSICS_INVARIANTS[:1]))
    assert len(violations) == 1
    missing = violations[0]
    assert missing.severity == "critical"
    assert np.isnan(missing.actual_value)
    assert np.isinf(missing.margin)


def test_evaluate_safety_invariants_all_present_and_nominal_is_empty() -> None:
    """A complete, nominal snapshot raises no violations (no false positives)."""
    assert evaluate_safety_invariants(dict(_NOMINAL_SNAPSHOT)) == []


def test_evaluate_safety_invariants_partial_dropout_flags_missing_and_present() -> None:
    """Both a dropped channel and a present-but-violating channel are reported."""
    snapshot = {"q_min": 0.5, "beta_N": 1.5}  # q_min violates; three channels dropped
    violations = evaluate_safety_invariants(snapshot)
    by_name = {v.invariant.name: v for v in violations}
    assert by_name["q_min"].severity == "critical"  # 0.5 !> 1.0, far from threshold
    assert np.isnan(by_name["greenwald"].actual_value)  # dropped -> missing-channel critical
    assert by_name["greenwald"].severity == "critical"
    assert "beta_N" not in by_name  # present and nominal
    assert should_trigger_mitigation(violations) is True


def test_evaluate_safety_invariants_non_finite_present_channel_is_critical() -> None:
    """A present but non-finite measurement is critical, not skipped."""
    snapshot = dict(_NOMINAL_SNAPSHOT)
    snapshot["q_min"] = float("nan")
    violations = evaluate_safety_invariants(snapshot)
    assert [v.invariant.name for v in violations] == ["q_min"]
    assert violations[0].severity == "critical"


def test_evaluate_safety_invariants_custom_set_requires_every_channel() -> None:
    """Every invariant in a custom set is mandatory, not just the ones supplied."""
    inv = PhysicsInvariant(name="x", description="d", threshold=5.0, comparator="lt")
    violations = evaluate_safety_invariants({}, invariants=[inv])
    assert len(violations) == 1
    assert violations[0].invariant.name == "x"
    assert violations[0].severity == "critical"


def test_check_all_invariants_remains_a_subset_utility() -> None:
    """The legacy subset checker is deliberately unchanged: absent channels skip.

    ``check_all_invariants`` is a subset utility and must stay open on absent
    channels; ``evaluate_safety_invariants`` is the fail-closed safety monitor.
    """
    assert check_all_invariants({}) == []
    assert evaluate_safety_invariants({}) != []
