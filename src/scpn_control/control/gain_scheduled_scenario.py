# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Gain-scheduled scenario waveforms.

"""Waveform and baseline-scenario utilities used by gain scheduling."""

from __future__ import annotations

import math

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray


def _validated_knots(times: AnyFloatArray, values: AnyFloatArray) -> tuple[FloatArray, FloatArray]:
    """Check the current waveform knot arrays before interpolation."""
    knots = np.asarray(times, dtype=float)
    samples = np.asarray(values, dtype=float)
    if knots.ndim != 1 or samples.ndim != 1 or knots.size == 0 or knots.size != samples.size:
        raise ValueError("times and values must be non-empty equal-length vectors")
    if not np.all(np.isfinite(knots)) or not np.all(np.isfinite(samples)):
        raise ValueError("times and values must be finite")
    if not np.all(knots[1:] > knots[:-1]):
        raise ValueError("times must be strictly increasing")
    return knots, samples


class ScenarioWaveform:
    """Piecewise-linear waveform with finite, increasing input knots."""

    def __init__(self, name: str, times: AnyFloatArray, values: AnyFloatArray, interp_kind: str = "linear") -> None:
        if interp_kind != "linear":
            raise ValueError("interp_kind must be linear")
        knots, samples = _validated_knots(times, values)
        self.name = name
        self.times = knots.copy()
        self.values = samples.copy()
        self.interp_kind = interp_kind

    def __call__(self, t: float) -> float:
        """Return the interpolated value at a finite time in seconds."""
        if not math.isfinite(t):
            raise ValueError("time must be finite")
        knots, samples = _validated_knots(self.times, self.values)
        if t <= knots[0]:
            return float(samples[0])
        if t >= knots[-1]:
            return float(samples[-1])
        right = int(np.searchsorted(knots, t, side="right"))
        left_time, right_time = float(knots[right - 1]), float(knots[right])
        span = right_time - left_time
        if math.isfinite(span):
            fraction = (t - left_time) / span
        else:
            scale = max(abs(left_time), abs(right_time))
            fraction = (t / scale - left_time / scale) / (right_time / scale - left_time / scale)
        value = (1.0 - fraction) * float(samples[right - 1]) + fraction * float(samples[right])
        return value


class ScenarioSchedule:
    """Collection of waveforms defining a full discharge scenario."""

    def __init__(self, waveforms: dict[str, ScenarioWaveform]) -> None:
        self.waveforms = waveforms

    def evaluate(self, t: float) -> dict[str, float]:
        """Return all waveform values at time ``t``.

        Parameters
        ----------
        t
            Time in seconds.

        Returns
        -------
        dict[str, float]
            Each waveform name mapped to its interpolated value.
        """
        return {name: wf(t) for name, wf in self.waveforms.items()}

    def duration(self) -> float:
        """Return the scenario duration in seconds (latest waveform end time)."""
        if not self.waveforms:
            return 0.0
        errors = self.validate()
        if errors:
            raise ValueError("; ".join(errors))
        return float(max(wf.times[-1] for wf in self.waveforms.values()))

    def validate(self) -> list[str]:
        """Validate the schedule waveforms.

        Returns
        -------
        list[str]
            Error messages for malformed waveform knots; empty when valid.
        """
        errors = []
        for name, wf in self.waveforms.items():
            try:
                _validated_knots(wf.times, wf.values)
            except ValueError as error:
                errors.append(f"Waveform {name}: {error}")
        return errors


def iter_baseline_schedule() -> ScenarioSchedule:
    """Return the ITER 15 MA inductive scenario baseline waveform.

    Timing and values follow ITER PCDH v3.1 (Polevoi et al. 2014,
    ITER Report ITR-18-001, §4.1, Table 4-1):
        t=0–10 s   : ramp-up  (I_p 0.5→5 MA)
        t=10–30 s  : ramp-up  (I_p 5→10 MA, auxiliary heating on)
        t=30–60 s  : ramp-up  (I_p 10→15 MA)
        t=60–400 s : flat top (I_p = 15 MA, NBI 33 MW, ECCD 17 MW)
        t=400–430 s: ramp-down start
        t=430–480 s: ramp-down to 2 MA
    """
    times = np.array([0, 10, 30, 60, 400, 430, 480], dtype=float)
    ip_vals = np.array([0.5, 5.0, 10.0, 15.0, 15.0, 10.0, 2.0])  # MA
    p_nbi = np.array([0.0, 0.0, 10.0, 33.0, 33.0, 10.0, 0.0])  # MW
    p_eccd = np.array([0.0, 0.0, 5.0, 17.0, 17.0, 5.0, 0.0])  # MW

    return ScenarioSchedule(
        {
            "Ip": ScenarioWaveform("Ip", times, ip_vals),
            "P_NBI": ScenarioWaveform("P_NBI", times, p_nbi),
            "P_ECCD": ScenarioWaveform("P_ECCD", times, p_eccd),
        }
    )
