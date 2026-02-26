# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Mock DIII-D Shot Generator
# ──────────────────────────────────────────────────────────────────────
"""Generate synthetic DIII-D-format shot data for CI end-to-end tests.

Each shot has 1000 timesteps at 1 kHz with the same field structure as
real DIII-D disruption shots (time_s, Ip_MA, BT_T, beta_N, q95, etc.).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def generate_mock_shot(
    shot_id: int = 999999,
    n_steps: int = 1000,
    disruption: bool = True,
    disruption_type: str = "hmode",
    seed: int = 42,
) -> dict:
    """Return a dict of arrays matching real DIII-D shot npz format."""
    rng = np.random.default_rng(seed)

    time_s = np.linspace(0.0, 5.0, n_steps)
    dt_idx = int(n_steps * 0.8) if disruption else n_steps - 1

    # Plasma current ramp-up → flat → (optional) disruption drop
    ip = np.ones(n_steps) * 1.5
    ip[:100] = np.linspace(0.0, 1.5, 100)
    if disruption:
        ip[dt_idx:] *= np.exp(-np.linspace(0, 5, n_steps - dt_idx))

    bt = np.full(n_steps, 2.1)
    beta_n = 2.0 + 0.3 * rng.standard_normal(n_steps)
    q95 = 3.5 + 0.2 * rng.standard_normal(n_steps)
    ne = 5.0 + 0.5 * rng.standard_normal(n_steps)

    # MHD activity: low baseline, spike at disruption
    n1 = 0.01 * np.abs(rng.standard_normal(n_steps))
    n2 = 0.005 * np.abs(rng.standard_normal(n_steps))
    lm = 0.001 * np.abs(rng.standard_normal(n_steps))
    if disruption:
        spike = np.exp(np.linspace(0, 4, 50))
        n1[dt_idx - 50:dt_idx] += spike * 0.1
        lm[dt_idx - 30:dt_idx] += np.exp(np.linspace(0, 3, 30)) * 0.05

    dbdt = np.gradient(np.cumsum(n1 + n2), time_s) * 100
    vpos = 0.01 * rng.standard_normal(n_steps)
    if disruption:
        vpos[dt_idx:] += np.linspace(0, 0.15, n_steps - dt_idx)

    return {
        "time_s": time_s,
        "Ip_MA": ip,
        "BT_T": bt,
        "beta_N": beta_n,
        "q95": q95,
        "ne_1e19": ne,
        "n1_amp": n1,
        "n2_amp": n2,
        "locked_mode_amp": lm,
        "dBdt_gauss_per_s": dbdt,
        "vertical_position_m": vpos,
        "is_disruption": np.bool_(disruption),
        "disruption_time_idx": np.int64(dt_idx),
        "disruption_type": np.str_(disruption_type),
    }


def save_mock_shot(directory: Path, shot_id: int = 999999, **kwargs) -> Path:
    """Generate and save a mock shot to npz."""
    directory.mkdir(parents=True, exist_ok=True)
    data = generate_mock_shot(shot_id=shot_id, **kwargs)
    path = directory / f"shot_{shot_id}_mock.npz"
    np.savez_compressed(path, **data)
    return path
