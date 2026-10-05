# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Actual phase-model capture and display-input validation.

"""Capture unchanged monitor dynamics and validate the fields used by the video.

This frontend does not identify reactor parameters or admit physical reference,
monitoring, control or protection claims. Validation covers display inputs, not
the truth of supplied snapshot provenance or their numerical derivation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from scpn_control.phase.realtime_monitor import RealtimeMonitor


def _positive_int(value: object) -> bool:
    """Recognize a positive Python integer while refusing boolean count aliases."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _finite_number(value: object) -> bool:
    """Recognize finite JSON-compatible numeric display fields without boolean coercion."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def validate_model_inputs(n_ticks: int, L: int, N_per: int, zeta: float) -> None:
    """Refuse nonpositive/boolean counts and nonfinite/negative gain before capturing a model."""
    if not all(_positive_int(value) for value in (n_ticks, L, N_per)):
        raise ValueError("Ticks, layers and population must be positive nonboolean integers")
    if not _finite_number(zeta) or zeta < 0:
        raise ValueError("Model gain must be finite and nonnegative")


def sample_frame_indices(n_ticks: int, fps: int) -> tuple[int, ...]:
    """Return the original zero-based frame policy, including the terminal sample.

    Positive nonboolean n_ticks and integer fps in [1,100] are required. The
    stride is max(1, n_ticks // (fps*10)); select range(0,n_ticks,stride) and append
    n_ticks-1 when absent. This targets about ten playback seconds, not exactly
    ten. GIF durations are quantised to centiseconds by its encoder.
    """
    if not _positive_int(n_ticks) or not _positive_int(fps) or fps > 100:
        raise ValueError("Ticks must be positive and playback fps must be an integer in [1,100]")
    indices = tuple(range(0, n_ticks, max(1, n_ticks // (fps * 10))))
    if indices[-1] != n_ticks - 1:
        indices += (n_ticks - 1,)
    return indices


@dataclass(frozen=True)
class PhaseTrajectory:
    """Borrowed display snapshots with explicit model labels, not physical admission.

    L/N_per are positive population labels; zeta is nonnegative finite model
    gain and dt is positive finite model time per tick. snapshots contains
    consecutive ticks 1..n with R_global, R_layer(L,), V_global, lambda_exp,
    guard_approved and latency_us. Mappings remain mutable; render revalidates
    them on every call. Additional fields are ignored by the display projection.
    R and V are dimensionless; lambda is per model-time unit and latency is the
    monitor's measured model/guard tick microseconds, excluding rendering/encoding.
    """

    L: int
    N_per: int
    zeta: float
    snapshots: tuple[dict[str, Any], ...]
    dt: float = 1e-3

    def validate(self) -> None:
        """Refuse malformed display fields and actual monitor error snapshots before writing.

        R must lie within [0,1] and V within [0,2], with 1e-12 endpoint roundoff
        tolerance. Model guard refusal alone is valid display data, not a
        failed capture or a reactor safety decision. Numeric latency is >=0.
        """
        validate_model_inputs(len(self.snapshots), self.L, self.N_per, self.zeta)
        if not _finite_number(self.dt) or self.dt <= 0:
            raise ValueError("Model dt must be positive and finite")
        for tick, snapshot in enumerate(self.snapshots, start=1):
            if "error" in snapshot or "error_type" in snapshot:
                raise ValueError("Monitor tick error is not an ordinary model trajectory")
            if not _positive_int(snapshot.get("tick")) or snapshot.get("tick") != tick:
                raise ValueError("Display ticks must be consecutive from one")
            for key in ("R_global", "V_global", "lambda_exp", "latency_us"):
                if not _finite_number(snapshot.get(key)):
                    raise ValueError("Display scalars must be finite numbers")
            if not -1e-12 <= snapshot["R_global"] <= 1 + 1e-12 or not -1e-12 <= snapshot["V_global"] <= 2 + 1e-12:
                raise ValueError("Display coherence and Lyapunov values are outside their model ranges")
            if snapshot["latency_us"] < 0 or not isinstance(snapshot.get("guard_approved"), bool):
                raise ValueError("Display latency/guard fields are invalid")
            layers = snapshot.get("R_layer")
            if not isinstance(layers, (list, tuple)) or len(layers) != self.L:
                raise ValueError("Per-layer coherence shape must match L")
            if not all(_finite_number(value) and -1e-12 <= value <= 1 + 1e-12 for value in layers):
                raise ValueError("Per-layer coherence values must be finite and in the model range")


def capture_trajectory(n_ticks: int, L: int, N_per: int, zeta: float) -> PhaseTrajectory:
    """Capture n_ticks from the actual seeded Paper 27 RealtimeMonitor factory.

    Counts must be positive nonboolean integers and zeta finite/nonnegative.
    Seed42, external driver0, PAC0, dt0.001, guard window50/max violations3 and
    the actual available UPDE implementation are unchanged. Layers beyond16
    repeat the factory's frequency table modulo16. All snapshots are retained
    in memory with the monitor recorder; memory grows with ticks and layers.
    Model guard HALT is recorded. Provider error snapshots remain marked and
    are refused by PhaseTrajectory.validate/render rather than shown as success.
    """
    validate_model_inputs(n_ticks, L, N_per, zeta)
    monitor = RealtimeMonitor.from_paper27(L=L, N_per=N_per, zeta_uniform=zeta, psi_driver=0.0, seed=42)
    return PhaseTrajectory(L, N_per, zeta, tuple(monitor.tick() for _ in range(n_ticks)), monitor.upde.dt)
