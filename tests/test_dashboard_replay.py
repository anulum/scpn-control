# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Dashboard replay tests
# © 1998–2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────
"""Tests for Streamlit-independent dashboard shot replay preparation."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest

from dashboard.replay import available_signal_keys, build_replay_frame
from dashboard.state import MACHINE_PRESETS


def _shot(n: int = 8) -> dict[str, object]:
    time_s = np.linspace(0.0, 0.7, n, dtype=np.float64)
    return {
        "time_s": time_s,
        "Ip_MA": np.linspace(0.5, 1.5, n, dtype=np.float64),
        "beta_N": np.linspace(0.1, 2.2, n, dtype=np.float64),
        "q95": np.linspace(4.5, 3.1, n, dtype=np.float64),
        "ne_1e19": np.linspace(2.0, 6.0, n, dtype=np.float64),
        "ignored_scalar": 7.0,
        "is_disruption": True,
        "disruption_time_idx": n - 2,
        "disruption_type": "locked_mode",
    }


def test_available_signal_keys_filters_known_finite_same_length_vectors() -> None:
    shot = _shot()
    shot["n1_amp"] = np.full(8, np.nan, dtype=np.float64)
    shot["locked_mode_amp"] = np.arange(7, dtype=np.float64)

    assert available_signal_keys(shot) == ("Ip_MA", "beta_N", "q95", "ne_1e19")


def test_build_replay_frame_returns_valid_current_state_and_profiles() -> None:
    frame = build_replay_frame(
        shot_data=_shot(),
        machine=MACHINE_PRESETS["DIII-D"],
        shot_label="reference_001",
        step_idx=4,
        profile_points=16,
    )

    assert frame.shot_label == "reference_001"
    assert frame.current_time_s == pytest.approx(0.4)
    assert frame.duration_s == pytest.approx(0.7)
    assert frame.time_fraction == pytest.approx(4 / 7)
    assert frame.phase_label == "FLATTOP"
    assert frame.is_disruption is True
    assert frame.disruption_time_idx == 6
    assert frame.disruption_type == "locked_mode"
    assert tuple(frame.signals) == ("Ip_MA", "beta_N", "q95", "ne_1e19")
    assert frame.profiles["rho"].shape == (16,)
    assert frame.profiles["Te_keV"][0] > frame.profiles["Te_keV"][-1]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda shot: shot.update({"time_s": np.array([0.0])}),
        lambda shot: shot.update({"time_s": np.array([0.0, 0.2, 0.1])}),
        lambda shot: shot.update({"time_s": np.array([0.0, np.nan, 0.2])}),
        lambda shot: shot.update({"disruption_time_idx": -1}),
    ],
)
def test_build_replay_frame_rejects_invalid_shot_data(mutation: Callable[[dict[str, object]], None]) -> None:
    shot = _shot()
    mutation(shot)

    with pytest.raises(ValueError):
        build_replay_frame(shot, MACHINE_PRESETS["DIII-D"], "bad", step_idx=1)


@pytest.mark.parametrize("step_idx", [-1, 8, True])
def test_build_replay_frame_rejects_invalid_step_index(step_idx: Any) -> None:
    with pytest.raises(ValueError, match="step_idx"):
        build_replay_frame(_shot(), MACHINE_PRESETS["DIII-D"], "bad", step_idx=step_idx)
