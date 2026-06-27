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
    ActionSpec,
    PhysicsInvariant,
    check_all_invariants,
    check_boundedness,
    check_deadlock,
    check_marking_conservation,
    check_physics_invariant,
    decode_actions,
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
