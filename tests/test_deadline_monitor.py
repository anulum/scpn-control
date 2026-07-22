# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Runtime Deadline Monitor Tests
"""Unit coverage for the per-cycle runtime deadline monitor.

Covers within-deadline/overrun accounting, the fail-soft default vs strict-raise
policy, the boundary (elapsed == deadline), input validation, snapshot, and reset.
"""

from __future__ import annotations

import math

import pytest

from scpn_control.scpn.deadline_monitor import DeadlineMonitor, DeadlineOverrunError


def test_within_deadline_is_counted_without_overrun() -> None:
    monitor = DeadlineMonitor(deadline_us=500.0)
    assert monitor.record(300.0) is True
    assert monitor.cycles == 1
    assert monitor.overruns == 0
    assert monitor.last_cycle_us == 300.0
    assert monitor.max_cycle_us == 300.0


def test_boundary_elapsed_equals_deadline_is_within() -> None:
    monitor = DeadlineMonitor(deadline_us=500.0)
    assert monitor.record(500.0) is True
    assert monitor.overruns == 0


def test_overrun_is_counted_in_fail_soft_default() -> None:
    monitor = DeadlineMonitor(deadline_us=500.0)
    assert monitor.record(750.0) is False
    assert monitor.record(200.0) is True
    assert monitor.cycles == 2
    assert monitor.overruns == 1
    assert monitor.max_cycle_us == 750.0
    assert monitor.last_cycle_us == 200.0


def test_strict_mode_raises_on_overrun() -> None:
    monitor = DeadlineMonitor(deadline_us=500.0, strict=True)
    assert monitor.record(400.0) is True
    with pytest.raises(DeadlineOverrunError, match="exceeded deadline"):
        monitor.record(600.0)
    # The overrun is still counted before the raise.
    assert monitor.overruns == 1


def test_rejects_non_finite_or_non_positive_deadline() -> None:
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="deadline_us must be finite and > 0"):
            DeadlineMonitor(deadline_us=bad)


def test_record_rejects_non_finite_or_negative_elapsed() -> None:
    monitor = DeadlineMonitor(deadline_us=500.0)
    for bad in (-1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="elapsed_us must be finite and >= 0"):
            monitor.record(bad)


def test_as_dict_snapshots_state() -> None:
    monitor = DeadlineMonitor(deadline_us=500.0)
    monitor.record(700.0)
    snapshot = monitor.as_dict()
    assert snapshot == {
        "deadline_us": 500.0,
        "strict": False,
        "cycles": 1,
        "overruns": 1,
        "last_cycle_us": 700.0,
        "max_cycle_us": 700.0,
    }


def test_reset_clears_statistics() -> None:
    monitor = DeadlineMonitor(deadline_us=500.0)
    monitor.record(700.0)
    monitor.record(100.0)
    monitor.reset()
    assert monitor.cycles == 0
    assert monitor.overruns == 0
    assert monitor.last_cycle_us == 0.0
    assert monitor.max_cycle_us == 0.0
