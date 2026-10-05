# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Resilience campaign scalar normalisation.
"""Coerce legacy adapter inputs without changing core campaign computations."""

from __future__ import annotations

import math


def _normalize_campaign_inputs(
    *,
    seed: int,
    episodes: int,
    window: int,
    noise_std: float,
    bit_flip_interval: int,
    recovery_window: int,
    recovery_epsilon: float,
) -> tuple[int, int, int, float, int, int, float]:
    """Coerce campaign scalars and enforce local range/finite checks.

    Parameters
    ----------
    seed, episodes, window, bit_flip_interval, recovery_window : int
        int() conversion occurs first, accepting integer-convertible inputs;
        episode/interval/recovery counts must be >= 1 and window >= 16.
    noise_std, recovery_epsilon : float
        float() conversion occurs first. Noise must be finite and >= 0;
        recovery tolerance must be finite and > 0.

    Returns
    -------
    tuple of int and float
        Normalised values in argument order. Seed validity is left to the
        public core campaign, which requires a non-negative integer.

    Raises
    ------
    ValueError
        Scalar conversion or one of the named range/finite checks fails.
    TypeError
        A value cannot be converted to its scalar type.
    OverflowError
        int()/float() conversion overflows. Errors are not suppressed.

    Notes
    -----
    This legacy adapter truncates integer-convertible floats and converts
    booleans; it does not mutate RNG or controller state.
    """
    seed_i = int(seed)
    episodes_i = int(episodes)
    window_i = int(window)
    noise = float(noise_std)
    bit_flip_i = int(bit_flip_interval)
    recovery_window_i = int(recovery_window)
    recovery_eps = float(recovery_epsilon)

    if episodes_i < 1:
        raise ValueError("episodes must be >= 1.")
    if window_i < 16:
        raise ValueError("window must be >= 16.")
    if not math.isfinite(noise) or noise < 0.0:
        raise ValueError("noise_std must be finite and >= 0.")
    if bit_flip_i < 1:
        raise ValueError("bit_flip_interval must be >= 1.")
    if recovery_window_i < 1:
        raise ValueError("recovery_window must be >= 1.")
    if not math.isfinite(recovery_eps) or recovery_eps <= 0.0:
        raise ValueError("recovery_epsilon must be finite and > 0.")

    return (
        seed_i,
        episodes_i,
        window_i,
        noise,
        bit_flip_i,
        recovery_window_i,
        recovery_eps,
    )
