# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Petri Net Contract Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""
Tests for SPN logic invariants and formal data contracts.
"""

from __future__ import annotations

import numpy as np

from scpn_control.scpn.contracts import (
    check_boundedness,
    check_deadlock,
    check_marking_conservation,
)


def test_marking_conservation():
    """Verify that token conservation is correctly detected."""
    m1 = [1.0, 2.0, 3.0]
    m2 = [2.0, 2.0, 2.0]  # Sum is 6.0 for both

    assert check_marking_conservation(m1, m2) is None

    m3 = [1.0, 2.0, 3.1]  # Sum is 6.1
    violation = check_marking_conservation(m1, m3)
    assert violation is not None
    assert violation.invariant.name == "marking_conservation"
    assert "Token sum mismatch" in violation.message


def test_deadlock_detection():
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


def test_boundedness_check():
    """Verify that place capacity limits are enforced."""
    marking = [10.0, 50.0, 100.0]

    # Nominal case
    assert check_boundedness(marking, max_capacity=200.0) is None

    # Violation case
    violation = check_boundedness(marking, max_capacity=80.0)
    assert violation is not None
    assert violation.invariant.name == "boundedness"
    assert "Place 2 exceeds capacity" in violation.message
