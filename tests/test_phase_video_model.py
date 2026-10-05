# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Actual phase capture, frame sampling and public input contracts.

"""Exercise real monitor capture and declared public display-data refusals."""

from __future__ import annotations

import copy
import math
from dataclasses import replace
from typing import Any

import pytest

from scpn_control.phase.realtime_monitor import RealtimeMonitor
from tools.phase_video_model import PhaseTrajectory, capture_trajectory, sample_frame_indices


def test_capture_matches_actual_monitor_for_every_displayed_value() -> None:
    """Replay the unchanged seeded provider independently and compare actual model values, excluding timing."""
    trajectory = capture_trajectory(8, 2, 3, 0.5)
    trajectory.validate()
    monitor = RealtimeMonitor.from_paper27(L=2, N_per=3, zeta_uniform=0.5, psi_driver=0, seed=42)
    expected = [monitor.tick() for _ in range(8)]
    assert trajectory.dt == monitor.upde.dt == 0.001
    for actual, observed in zip(trajectory.snapshots, expected, strict=True):
        for key in ("tick", "R_global", "R_layer", "V_global", "lambda_exp", "guard_approved"):
            assert actual[key] == observed[key]
    assert {sample["guard_approved"] for sample in trajectory.snapshots} == {True, False}


def test_current_default_model_does_not_claim_old_caption() -> None:
    """Capture all500 original-default ticks without substituting a model or retuning convergence."""
    trajectory = capture_trajectory(500, 16, 50, 0.5)
    trajectory.validate()
    last = trajectory.snapshots[-1]
    assert last["tick"] == 500 and len(last["R_layer"]) == 16
    assert 0.15 < last["R_global"] < 0.16 and 0.84 < last["V_global"] < 0.85
    assert last["lambda_exp"] < 0 and any(not sample["guard_approved"] for sample in trajectory.snapshots)


@pytest.mark.parametrize(
    "ticks,layers,population,zeta",
    [
        (0, 2, 3, 0.5),
        (True, 2, 3, 0.5),
        (1, 0, 3, 0.5),
        (1, 2, 0, 0.5),
        (1, 2, 3, -1.0),
        (1, 2, 3, math.nan),
        (1, 2, 3, True),
        (1, 2, 3, 1 << 1024),
    ],
)
def test_capture_refuses_invalid_public_inputs(ticks: int, layers: int, population: int, zeta: float) -> None:
    """Refuse real public parameter inputs before constructing a monitor."""
    with pytest.raises(ValueError):
        capture_trajectory(ticks, layers, population, zeta)


@pytest.mark.parametrize("ticks,fps", [(0, 5), (1, 0), (1, True), (1, 101)])
def test_frame_policy_refuses_invalid_counts(ticks: int, fps: int) -> None:
    """The maintained public sampling policy refuses boolean/zero counts and unsupported GIF rates."""
    with pytest.raises(ValueError):
        sample_frame_indices(ticks, fps)


def test_frame_policy_has_initial_terminal_and_floor_stride() -> None:
    """Observe both terminal policies and the original floor stride through the public function."""
    assert sample_frame_indices(1, 100) == (0,)
    assert sample_frame_indices(8, 20) == tuple(range(8))
    assert sample_frame_indices(42, 2) == (*range(0, 42, 2), 41)
    assert sample_frame_indices(43, 2) == tuple(range(0, 43, 2))


@pytest.mark.parametrize(
    "key,value",
    [
        ("tick", 0),
        ("tick", True),
        ("tick", 1.0),
        ("tick", 2),
        ("R_global", None),
        ("R_global", True),
        ("R_global", math.inf),
        ("R_global", -1.0),
        ("V_global", 3.0),
        ("lambda_exp", math.nan),
        ("latency_us", -1),
        ("latency_us", 1 << 1024),
        ("guard_approved", 0),
        ("R_layer", None),
        ("R_layer", []),
        ("R_layer", [None, 0.5]),
        ("R_layer", [-1.0, 0.5]),
        ("R_layer", [math.inf, 0.5]),
    ],
)
def test_borrowed_display_data_is_revalidated(key: str, value: Any) -> None:
    """Mutate declared public display inputs and refuse invalid shapes/domains without cached admission."""
    original = capture_trajectory(1, 2, 3, 0.5)
    snapshots = copy.deepcopy(original.snapshots)
    snapshots[0][key] = value
    supplied = replace(original, snapshots=snapshots)
    with pytest.raises(ValueError):
        supplied.validate()
    original.validate()


@pytest.mark.parametrize("dt", [0.0, math.inf, True])
def test_trajectory_refuses_invalid_model_time(dt: float) -> None:
    """Validate actual captured data with a caller-supplied invalid model-time label."""
    with pytest.raises(ValueError):
        replace(capture_trajectory(1, 1, 1, 0), dt=dt).validate()


def test_actual_monitor_error_snapshot_refuses_ordinary_trajectory() -> None:
    """Use a real monitor's public invalid driver state and actual fail-closed tick output."""
    monitor = RealtimeMonitor.from_paper27(L=2, N_per=3)
    monitor.psi_driver = math.nan
    failed = monitor.tick()
    assert failed["guard_approved"] is False and failed["error_type"] == "ValueError"
    with pytest.raises(ValueError, match="Monitor tick error"):
        PhaseTrajectory(2, 3, 0.5, (failed,)).validate()


def test_empty_trajectory_refuses_render_data() -> None:
    """An empty caller-supplied public trajectory cannot index or label an absent terminal sample."""
    with pytest.raises(ValueError):
        PhaseTrajectory(2, 3, 0.5, ()).validate()
