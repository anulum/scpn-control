# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Dashboard replay state
# © 1998–2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────
"""Streamlit-independent shot replay preparation for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dashboard.state import MachinePreset, ProfileMap, shot_phase_label, synthetic_profiles


ShotData = dict[str, Any]
SignalMap = dict[str, NDArray[np.float64]]

SIGNAL_KEYS = ("Ip_MA", "beta_N", "q95", "ne_1e19", "n1_amp", "locked_mode_amp")


@dataclass(frozen=True)
class ReplayFrame:
    """Validated shot replay state for one selected timeline index."""

    shot_label: str
    time_s: NDArray[np.float64]
    step_idx: int
    current_time_s: float
    duration_s: float
    time_fraction: float
    phase_label: str
    is_disruption: bool
    disruption_time_idx: int
    disruption_type: str
    signals: SignalMap
    profiles: ProfileMap


def _time_vector(shot_data: ShotData) -> NDArray[np.float64]:
    if "time_s" not in shot_data:
        raise ValueError("shot_data must contain time_s.")
    time_s = np.asarray(shot_data["time_s"], dtype=np.float64)
    if time_s.ndim != 1 or time_s.size < 2:
        raise ValueError("time_s must be a one-dimensional vector with at least two samples.")
    if not np.all(np.isfinite(time_s)):
        raise ValueError("time_s must contain only finite values.")
    if not np.all(np.diff(time_s) > 0.0):
        raise ValueError("time_s must be strictly increasing.")
    return time_s


def _validate_step_idx(step_idx: int, n_steps: int) -> int:
    if isinstance(step_idx, bool) or not isinstance(step_idx, int):
        raise ValueError("step_idx must be an integer timeline index.")
    if step_idx < 0 or step_idx >= n_steps:
        raise ValueError(f"step_idx must be within [0, {n_steps - 1}].")
    return step_idx


def _disruption_time_idx(shot_data: ShotData, n_steps: int) -> int:
    raw = shot_data.get("disruption_time_idx", n_steps - 1)
    if isinstance(raw, bool):
        raise ValueError("disruption_time_idx must be an integer index.")
    idx = int(raw)
    if idx < 0 or idx >= n_steps:
        raise ValueError(f"disruption_time_idx must be within [0, {n_steps - 1}].")
    return idx


def available_signal_keys(shot_data: ShotData) -> tuple[str, ...]:
    """Return known finite vector signals with the same length as time_s."""
    time_s = _time_vector(shot_data)
    keys: list[str] = []
    for key in SIGNAL_KEYS:
        if key not in shot_data:
            continue
        values = np.asarray(shot_data[key], dtype=np.float64)
        if values.shape == time_s.shape and np.all(np.isfinite(values)):
            keys.append(key)
    return tuple(keys)


def _signals(shot_data: ShotData, keys: tuple[str, ...]) -> SignalMap:
    return {key: np.asarray(shot_data[key], dtype=np.float64) for key in keys}


def build_replay_frame(
    shot_data: ShotData,
    machine: MachinePreset,
    shot_label: str,
    step_idx: int,
    *,
    profile_points: int = 80,
) -> ReplayFrame:
    """Build validated dashboard state for the selected replay timeline index."""
    time_s = _time_vector(shot_data)
    n_steps = int(time_s.size)
    checked_step_idx = _validate_step_idx(step_idx, n_steps)
    time_fraction = checked_step_idx / max(n_steps - 1, 1)
    phase_label = shot_phase_label(time_fraction)
    disruption_time_idx = _disruption_time_idx(shot_data, n_steps)
    signal_keys = available_signal_keys(shot_data)

    return ReplayFrame(
        shot_label=str(shot_label),
        time_s=time_s,
        step_idx=checked_step_idx,
        current_time_s=float(time_s[checked_step_idx]),
        duration_s=float(time_s[-1]),
        time_fraction=float(time_fraction),
        phase_label=phase_label,
        is_disruption=bool(shot_data.get("is_disruption", False)),
        disruption_time_idx=disruption_time_idx,
        disruption_type=str(shot_data.get("disruption_type", "N/A")),
        signals=_signals(shot_data, signal_keys),
        profiles=synthetic_profiles(machine, n_rho=profile_points, time_frac=time_fraction),
    )


__all__ = [
    "ReplayFrame",
    "SIGNAL_KEYS",
    "ShotData",
    "SignalMap",
    "available_signal_keys",
    "build_replay_frame",
]
