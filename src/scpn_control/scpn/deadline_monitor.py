# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Runtime Deadline Monitor
"""Per-cycle runtime deadline monitor for the control loop.

The admitted runtime-safety certificate declares a ``deadline_us`` in its timing
envelope, but that is a *static* schedulability assumption — it is validated at
admission and never checked against the actual control-cycle duration. This
monitor closes that gap: each measured cycle is compared against the declared
deadline. It is fail-soft by default (overruns are counted and exposed for
telemetry); ``strict=True`` turns a single overrun into a hard
:class:`DeadlineOverrunError` for callers that must treat a missed deadline as a
safety fault.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


class DeadlineOverrunError(RuntimeError):
    """Raised in strict mode when a control cycle exceeds its declared deadline."""


@dataclass
class DeadlineMonitor:
    """Compare each control-cycle duration against a certificate deadline.

    Parameters
    ----------
    deadline_us
        The per-cycle deadline in microseconds (from the admitted timing
        envelope). Must be finite and strictly positive.
    strict
        When ``True`` a single overrun raises :class:`DeadlineOverrunError`;
        otherwise overruns are only counted and exposed (fail-soft default).
    """

    deadline_us: float
    strict: bool = False
    _cycles: int = field(default=0, init=False)
    _overruns: int = field(default=0, init=False)
    _last_cycle_us: float = field(default=0.0, init=False)
    _max_cycle_us: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        """Reject a non-finite or non-positive deadline."""
        self.deadline_us = float(self.deadline_us)
        if not math.isfinite(self.deadline_us) or self.deadline_us <= 0.0:
            raise ValueError("deadline_us must be finite and > 0")

    def record(self, elapsed_us: float) -> bool:
        """Record one cycle's duration and return whether it met the deadline.

        Raises
        ------
        ValueError
            If ``elapsed_us`` is not finite and non-negative.
        DeadlineOverrunError
            In strict mode, when the cycle exceeds the deadline.
        """
        elapsed_us = float(elapsed_us)
        if not math.isfinite(elapsed_us) or elapsed_us < 0.0:
            raise ValueError("elapsed_us must be finite and >= 0")
        self._cycles += 1
        self._last_cycle_us = elapsed_us
        if elapsed_us > self._max_cycle_us:
            self._max_cycle_us = elapsed_us
        within = elapsed_us <= self.deadline_us
        if not within:
            self._overruns += 1
            if self.strict:
                raise DeadlineOverrunError(
                    f"control cycle {elapsed_us:.3f} us exceeded deadline {self.deadline_us:.3f} us"
                )
        return within

    @property
    def cycles(self) -> int:
        """Total number of recorded control cycles."""
        return self._cycles

    @property
    def overruns(self) -> int:
        """Number of cycles that exceeded the deadline."""
        return self._overruns

    @property
    def last_cycle_us(self) -> float:
        """Duration of the most recently recorded cycle in microseconds."""
        return self._last_cycle_us

    @property
    def max_cycle_us(self) -> float:
        """Longest recorded cycle duration in microseconds."""
        return self._max_cycle_us

    def as_dict(self) -> dict[str, float | int | bool]:
        """Return a JSON-serialisable snapshot of the monitor state."""
        return {
            "deadline_us": self.deadline_us,
            "strict": self.strict,
            "cycles": self._cycles,
            "overruns": self._overruns,
            "last_cycle_us": self._last_cycle_us,
            "max_cycle_us": self._max_cycle_us,
        }

    def reset(self) -> None:
        """Clear the accumulated cycle statistics."""
        self._cycles = 0
        self._overruns = 0
        self._last_cycle_us = 0.0
        self._max_cycle_us = 0.0
