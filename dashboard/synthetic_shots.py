# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Dashboard synthetic shots
# © 1998–2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────
"""Synthetic DIII-D-format shot generation for dashboard replay."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


ShotValue = NDArray[np.float64] | bool | int | str
SyntheticShot = dict[str, ShotValue]

DISRUPTION_TYPES = ("hmode", "locked_mode", "density_limit", "vde", "beta_limit")


def _require_int(name: str, value: int, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return value


def generate_synthetic_diiid_shot(
    *,
    shot_id: int = 999999,
    n_steps: int = 1000,
    disruption: bool = True,
    disruption_type: str = "hmode",
    seed: int = 42,
) -> SyntheticShot:
    """Return deterministic DIII-D-format arrays for dashboard shot replay."""
    _require_int("shot_id", shot_id, minimum=1)
    checked_steps = _require_int("n_steps", n_steps, minimum=128)
    checked_seed = _require_int("seed", seed, minimum=0)
    if not isinstance(disruption, bool):
        raise ValueError("disruption must be a bool.")
    if disruption_type not in DISRUPTION_TYPES:
        raise ValueError(f"disruption_type must be one of {DISRUPTION_TYPES}.")

    rng = np.random.default_rng(checked_seed)
    time_s = np.linspace(0.0, 5.0, checked_steps, dtype=np.float64)
    disruption_time_idx = int(checked_steps * 0.8) if disruption else checked_steps - 1

    ramp_samples = min(100, checked_steps // 4)
    ip = np.full(checked_steps, 1.5, dtype=np.float64)
    ip[:ramp_samples] = np.linspace(0.0, 1.5, ramp_samples, dtype=np.float64)
    if disruption:
        decay = np.exp(-np.linspace(0.0, 5.0, checked_steps - disruption_time_idx, dtype=np.float64))
        ip[disruption_time_idx:] *= decay

    bt = np.full(checked_steps, 2.1, dtype=np.float64)
    beta_n = np.maximum(0.05, 2.0 + 0.3 * rng.standard_normal(checked_steps))
    q95 = np.maximum(1.05, 3.5 + 0.2 * rng.standard_normal(checked_steps))
    ne = np.maximum(0.1, 5.0 + 0.5 * rng.standard_normal(checked_steps))

    n1 = 0.01 * np.abs(rng.standard_normal(checked_steps))
    n2 = 0.005 * np.abs(rng.standard_normal(checked_steps))
    locked_mode = 0.001 * np.abs(rng.standard_normal(checked_steps))
    if disruption:
        n1_window = min(50, disruption_time_idx)
        locked_window = min(30, disruption_time_idx)
        n1[disruption_time_idx - n1_window : disruption_time_idx] += (
            np.exp(np.linspace(0.0, 4.0, n1_window, dtype=np.float64)) * 0.1
        )
        locked_mode[disruption_time_idx - locked_window : disruption_time_idx] += (
            np.exp(np.linspace(0.0, 3.0, locked_window, dtype=np.float64)) * 0.05
        )

    dbdt = np.gradient(np.cumsum(n1 + n2), time_s) * 100.0
    vertical_position = 0.01 * rng.standard_normal(checked_steps)
    if disruption:
        vertical_position[disruption_time_idx:] += np.linspace(
            0.0,
            0.15,
            checked_steps - disruption_time_idx,
            dtype=np.float64,
        )

    return {
        "time_s": time_s,
        "Ip_MA": ip,
        "BT_T": bt,
        "beta_N": np.asarray(beta_n, dtype=np.float64),
        "q95": np.asarray(q95, dtype=np.float64),
        "ne_1e19": np.asarray(ne, dtype=np.float64),
        "n1_amp": np.asarray(n1, dtype=np.float64),
        "n2_amp": np.asarray(n2, dtype=np.float64),
        "locked_mode_amp": np.asarray(locked_mode, dtype=np.float64),
        "dBdt_gauss_per_s": np.asarray(dbdt, dtype=np.float64),
        "vertical_position_m": np.asarray(vertical_position, dtype=np.float64),
        "is_disruption": disruption,
        "disruption_time_idx": disruption_time_idx,
        "disruption_type": disruption_type if disruption else "safe",
    }


__all__ = ["DISRUPTION_TYPES", "ShotValue", "SyntheticShot", "generate_synthetic_diiid_shot"]
