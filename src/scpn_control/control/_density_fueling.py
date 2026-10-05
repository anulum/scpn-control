# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Pellet schedule optimisation

"""Pellet schedule optimization."""

from __future__ import annotations

from dataclasses import dataclass

from scpn_control._typing import AnyFloatArray


@dataclass
class PelletSchedule:
    """Timed pellet-injection sequence.

    Attributes
    ----------
    times
        Injection times in seconds.
    speeds
        Pellet velocities in m/s, one per injection.
    sizes
        Pellet radii in mm, one per injection.
    """

    times: list[float]
    speeds: list[float]
    sizes: list[float]


class FuelingOptimizer:
    """Plan evenly-spaced pellet-injection sequences toward a target density."""

    def optimize_pellet_sequence(
        self, ne_current: AnyFloatArray, ne_target: AnyFloatArray, n_pellets: int, time_horizon: float
    ) -> PelletSchedule:
        """Evenly-spaced pellet schedule over the given horizon."""
        if n_pellets <= 0:
            return PelletSchedule([], [], [])

        dt = time_horizon / (n_pellets + 1)
        times = [dt * (i + 1) for i in range(n_pellets)]
        speeds = [500.0] * n_pellets
        sizes = [2.0] * n_pellets
        return PelletSchedule(times, speeds, sizes)
