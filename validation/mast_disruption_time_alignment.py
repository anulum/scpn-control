#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Mask-preserving FAIR-MAST disruption time alignment
"""Align verified scalar FAIR-MAST channels without fabricating samples.

The exact ``summary.time`` array is the reference grid. Bound scalar signals are
mapped to it within contiguous finite source runs only: no extrapolation, no
interpolation across a missing-data gap, and no zero fill. A faster source is
low-pass filtered with a versioned Kaiser-window FIR before interpolation; the
filter design, coefficient digest, measured response, masks, and aligned values
are all represented in the digest-bound alignment report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, cast

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.signal import convolve, firwin, freqz, kaiserord

from validation.mast_disruption_signal_binding import (
    MastSignalBindingSpec,
    SignalBinding,
    assess_artifact_signal_bindings,
    mast_level2_signal_binding_spec,
)
from validation.mast_source_artifact_reader import VerifiedSourceArtifact
from validation.mast_source_object_manifest import array_value_sha256, canonical_json_sha256

MAST_TIME_ALIGNMENT_SCHEMA = "scpn-control.mast-time-alignment.v1.0.0"
MAST_TIME_ALIGNMENT_VERSION = "1.0.0"
REFERENCE_CHANNEL = "time_s"
REFERENCE_SOURCE_KEY = "summary.time"

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class TimeAlignmentError(ValueError):
    """Raised when a reference grid or alignment policy is inadmissible."""


@dataclass(frozen=True)
class TimeAlignmentSpec:
    """Versioned alignment and anti-alias policy.

    Parameters
    ----------
    schema_version:
        Exact serialised contract discriminator.
    version:
        Semantic version of the alignment algorithm.
    reference_channel:
        Canonical channel whose exact source array is retained as the target.
    uniform_rtol:
        Maximum relative source-period deviation from its median.
    cutoff_target_nyquist_fraction:
        FIR cutoff as a fraction of the target Nyquist frequency.
    attenuation_db:
        Requested Kaiser stopband attenuation.
    max_filter_taps:
        Fail-closed upper bound on generated FIR length.
    """

    schema_version: str = MAST_TIME_ALIGNMENT_SCHEMA
    version: str = MAST_TIME_ALIGNMENT_VERSION
    reference_channel: str = REFERENCE_CHANNEL
    uniform_rtol: float = 1.0e-5
    cutoff_target_nyquist_fraction: float = 0.8
    attenuation_db: float = 80.0
    max_filter_taps: int = 4097

    def __post_init__(self) -> None:
        """Reject ambiguous or numerically unsafe policy values."""
        if self.schema_version != MAST_TIME_ALIGNMENT_SCHEMA:
            raise TimeAlignmentError(f"schema_version must equal {MAST_TIME_ALIGNMENT_SCHEMA!r}")
        if self.version != MAST_TIME_ALIGNMENT_VERSION or self.reference_channel != REFERENCE_CHANNEL:
            raise TimeAlignmentError("version and reference channel must identify the supported contract")
        if not math.isfinite(self.uniform_rtol) or self.uniform_rtol <= 0.0:
            raise TimeAlignmentError("uniform_rtol must be finite and > 0")
        if not 0.0 < self.cutoff_target_nyquist_fraction < 1.0:
            raise TimeAlignmentError("cutoff_target_nyquist_fraction must lie in (0, 1)")
        if not math.isfinite(self.attenuation_db) or self.attenuation_db < 20.0:
            raise TimeAlignmentError("attenuation_db must be finite and >= 20")
        if self.max_filter_taps < 3 or self.max_filter_taps % 2 == 0:
            raise TimeAlignmentError("max_filter_taps must be an odd integer >= 3")

    def to_dict(self) -> dict[str, object]:
        """Return a fresh self-digested JSON-ready policy record."""
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "version": self.version,
            "reference_channel": self.reference_channel,
            "reference_source_key": REFERENCE_SOURCE_KEY,
            "uniform_rtol": self.uniform_rtol,
            "cutoff_target_nyquist_fraction": self.cutoff_target_nyquist_fraction,
            "attenuation_db": self.attenuation_db,
            "max_filter_taps": self.max_filter_taps,
            "target_grid_policy": "exact_finite_strictly_increasing_summary_time",
            "edge_policy": "no_extrapolation; FIR transient samples invalid",
            "gap_policy": "align_contiguous_finite_runs_only; never_bridge_or_zero_fill",
            "payload_sha256": None,
        }
        payload["payload_sha256"] = canonical_json_sha256(payload)
        return payload


@dataclass(frozen=True)
class FIRFilterDesign:
    """One reproducible anti-alias FIR design and its measured response."""

    coefficients: FloatArray
    provenance: Mapping[str, object]


@dataclass(frozen=True)
class AlignedChannelSet:
    """Immutable aligned values, validity masks, and digest-bound report."""

    time_s: FloatArray
    values: Mapping[str, FloatArray]
    valid_masks: Mapping[str, BoolArray]
    report: Mapping[str, Any]


def _uniform_period(time: FloatArray, *, rtol: float, label: str) -> float:
    """Return the median period of a finite strictly increasing uniform grid."""
    if time.ndim != 1 or time.size < 2 or not bool(np.all(np.isfinite(time))):
        raise TimeAlignmentError(f"{label} must be a finite one-dimensional grid with at least two samples")
    delta = np.diff(time)
    if not bool(np.all(delta > 0.0)):
        raise TimeAlignmentError(f"{label} must be strictly increasing without duplicates")
    period = float(np.median(delta))
    tolerance = max(float(np.finfo(np.float64).eps) * 16.0, rtol * period)
    if float(np.max(np.abs(delta - period))) > tolerance:
        raise TimeAlignmentError(f"{label} is not uniform within relative tolerance {rtol}")
    return period


def design_antialias_filter(
    source_period_s: float,
    target_period_s: float,
    *,
    spec: TimeAlignmentSpec | None = None,
) -> FIRFilterDesign:
    """Design a Kaiser FIR that suppresses frequencies above target Nyquist.

    The function is valid only for downsampling. It uses ``kaiserord`` to choose
    an odd tap count for the requested attenuation and transition band from the
    configured cutoff to target Nyquist. Designs exceeding the policy tap bound
    fail closed instead of silently weakening the filter.
    """
    active = spec if spec is not None else TimeAlignmentSpec()
    for value, label in ((source_period_s, "source_period_s"), (target_period_s, "target_period_s")):
        if not math.isfinite(value) or value <= 0.0:
            raise TimeAlignmentError(f"{label} must be finite and > 0")
    source_rate_hz = 1.0 / source_period_s
    target_rate_hz = 1.0 / target_period_s
    if source_rate_hz <= target_rate_hz:
        raise TimeAlignmentError("anti-alias FIR is only defined when source rate exceeds target rate")
    target_nyquist_hz = 0.5 * target_rate_hz
    cutoff_hz = active.cutoff_target_nyquist_fraction * target_nyquist_hz
    transition_width = (target_nyquist_hz - cutoff_hz) / (0.5 * source_rate_hz)
    tap_count, beta = kaiserord(active.attenuation_db, transition_width)
    tap_count = max(3, int(tap_count)) | 1
    if tap_count > active.max_filter_taps:
        raise TimeAlignmentError(
            f"anti-alias design requires {tap_count} taps, exceeding max_filter_taps={active.max_filter_taps}"
        )
    coefficients = np.asarray(
        firwin(
            tap_count,
            cutoff_hz,
            window=("kaiser", float(beta)),
            pass_zero=True,
            scale=True,
            fs=source_rate_hz,
        ),
        dtype=np.float64,
    )
    frequencies, response = freqz(coefficients, worN=32768, fs=source_rate_hz)
    stopband = np.abs(response[frequencies >= target_nyquist_hz])
    stopband_max_db = 20.0 * math.log10(max(float(np.max(stopband)), float(np.finfo(np.float64).tiny)))
    coefficients.setflags(write=False)
    provenance: dict[str, object] = {
        "algorithm": "scipy.signal.firwin_kaiser_zero_phase_convolution",
        "scipy_version": scipy.__version__,
        "source_rate_hz": source_rate_hz,
        "target_rate_hz": target_rate_hz,
        "target_nyquist_hz": target_nyquist_hz,
        "cutoff_hz": cutoff_hz,
        "cutoff_target_nyquist_fraction": active.cutoff_target_nyquist_fraction,
        "transition_width_normalised_to_source_nyquist": transition_width,
        "requested_attenuation_db": active.attenuation_db,
        "measured_stopband_max_db_at_or_above_target_nyquist": stopband_max_db,
        "window": "kaiser",
        "kaiser_beta": float(beta),
        "numtaps": tap_count,
        "group_delay_source_samples": (tap_count - 1) // 2,
        "coefficient_sha256": array_value_sha256(coefficients),
    }
    return FIRFilterDesign(coefficients=coefficients, provenance=MappingProxyType(provenance))


def _finite_runs(valid: BoolArray) -> list[tuple[int, int]]:
    """Return half-open index runs for contiguous true samples."""
    padded = np.concatenate((np.asarray([False]), valid, np.asarray([False])))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop)) for start, stop in edges.reshape(-1, 2)]


def _transform_factor(binding: SignalBinding) -> float:
    """Resolve the only currently admitted scalar transforms."""
    if binding.transform in {"identity", "identity_reference_timebase"}:
        return 1.0
    if binding.transform == "scale":
        parameters = dict(binding.transform_parameters)
        try:
            factor = float(parameters["factor"])
        except (KeyError, ValueError) as exc:
            raise TimeAlignmentError(f"channel {binding.channel!r} has an invalid scale transform") from exc
        if not math.isfinite(factor):
            raise TimeAlignmentError(f"channel {binding.channel!r} scale factor must be finite")
        return factor
    raise TimeAlignmentError(f"channel {binding.channel!r} transform {binding.transform!r} is not alignable")


def _align_one(
    source: FloatArray,
    source_time: FloatArray,
    target_time: FloatArray,
    *,
    binding: SignalBinding,
    source_period_s: float,
    target_period_s: float,
    spec: TimeAlignmentSpec,
) -> tuple[FloatArray, BoolArray, dict[str, object]]:
    """Align one bound scalar channel within finite runs only."""
    output = np.full(target_time.shape, np.nan, dtype=np.float64)
    valid_output = np.zeros(target_time.shape, dtype=np.bool_)
    finite = np.isfinite(source) & np.isfinite(source_time)
    downsample = source_period_s < target_period_s * (1.0 - spec.uniform_rtol)
    design = design_antialias_filter(source_period_s, target_period_s, spec=spec) if downsample else None
    transient = cast(int, design.provenance["group_delay_source_samples"]) if design is not None else 0
    used_runs = 0
    rejected_short_runs = 0
    for start, stop in _finite_runs(finite):
        run_time = source_time[start:stop]
        run_values = source[start:stop]
        if design is not None:
            if run_values.size <= 2 * transient + 1:
                rejected_short_runs += 1
                continue
            filtered = np.asarray(
                convolve(run_values, design.coefficients, mode="same", method="direct"), dtype=np.float64
            )
            run_time = run_time[transient:-transient]
            run_values = filtered[transient:-transient]
        inside = (target_time >= run_time[0]) & (target_time <= run_time[-1])
        if run_values.size == 1:
            tolerance = max(float(np.finfo(np.float64).eps) * 16.0, spec.uniform_rtol * target_period_s)
            inside &= np.abs(target_time - run_time[0]) <= tolerance
            output[inside] = run_values[0]
        else:
            output[inside] = np.interp(target_time[inside], run_time, run_values)
        valid_output[inside] = True
        used_runs += 1
    output *= _transform_factor(binding)
    method = "kaiser_fir_then_piecewise_linear" if design is not None else "piecewise_linear_no_extrapolation"
    provenance: dict[str, object] = {
        "channel": binding.channel,
        "source_key": binding.source_key,
        "timebase_key": binding.timebase_key,
        "status": "aligned_with_validity_mask",
        "method": method,
        "source_period_s": source_period_s,
        "target_period_s": target_period_s,
        "source_rate_hz": 1.0 / source_period_s,
        "target_rate_hz": 1.0 / target_period_s,
        "finite_source_samples": int(np.count_nonzero(finite)),
        "contiguous_finite_runs": len(_finite_runs(finite)),
        "used_runs": used_runs,
        "short_runs_rejected_for_filter_transient": rejected_short_runs,
        "valid_target_samples": int(np.count_nonzero(valid_output)),
        "total_target_samples": int(target_time.size),
        "edge_policy": "inclusive finite-run support; no extrapolation",
        "gap_policy": "never interpolate across non-finite source runs",
        "transform": binding.transform,
        "transform_parameters": dict(binding.transform_parameters),
        "anti_alias_filter": dict(design.provenance) if design is not None else None,
        "values_sha256": array_value_sha256(output),
        "valid_mask_sha256": array_value_sha256(valid_output),
    }
    output.setflags(write=False)
    valid_output.setflags(write=False)
    return output, valid_output, provenance


def align_verified_bound_channels(
    artifact: VerifiedSourceArtifact,
    *,
    binding_spec: MastSignalBindingSpec | None = None,
    alignment_spec: TimeAlignmentSpec | None = None,
) -> AlignedChannelSet:
    """Align every source-metadata-verified scalar binding to ``summary.time``.

    Six unresolved canonical bindings remain report-only blockers. The returned
    values contain NaN wherever the corresponding validity mask is false.
    """
    active_binding = binding_spec if binding_spec is not None else mast_level2_signal_binding_spec()
    active_alignment = alignment_spec if alignment_spec is not None else TimeAlignmentSpec()
    assessment = assess_artifact_signal_bindings(artifact, spec=active_binding)
    assessment_by_channel = {
        cast(str, item["channel"]): item for item in cast(list[dict[str, object]], assessment["bindings"])
    }
    reference = np.asarray(artifact.arrays[REFERENCE_SOURCE_KEY], dtype=np.float64)
    target_period_s = _uniform_period(reference, rtol=active_alignment.uniform_rtol, label=REFERENCE_SOURCE_KEY)
    target_time = reference.copy()
    target_time.setflags(write=False)
    values: dict[str, FloatArray] = {}
    masks: dict[str, BoolArray] = {}
    channel_reports: list[dict[str, object]] = []
    for binding in active_binding.bindings:
        binding_assessment = assessment_by_channel[binding.channel]
        if binding_assessment["status"] != "source_metadata_verified":
            channel_reports.append(
                {
                    "channel": binding.channel,
                    "status": "not_aligned",
                    "reason_code": binding_assessment["reason_code"],
                    "binding_assessment_status": binding_assessment["status"],
                }
            )
            continue
        source_key = cast(str, binding.source_key)
        timebase_key = cast(str, binding.timebase_key)
        source = np.asarray(artifact.arrays[source_key], dtype=np.float64)
        source_time = np.asarray(artifact.arrays[timebase_key], dtype=np.float64)
        source_period_s = _uniform_period(
            source_time,
            rtol=active_alignment.uniform_rtol,
            label=timebase_key,
        )
        aligned, mask, provenance = _align_one(
            source,
            source_time,
            target_time,
            binding=binding,
            source_period_s=source_period_s,
            target_period_s=target_period_s,
            spec=active_alignment,
        )
        values[binding.channel] = aligned
        masks[binding.channel] = mask
        channel_reports.append(provenance)
    aligned_count = len(values)
    bound_count = sum(binding.status == "bound" for binding in active_binding.bindings)
    bound_alignment_complete = aligned_count == bound_count
    report: dict[str, Any] = {
        "schema_version": MAST_TIME_ALIGNMENT_SCHEMA,
        "status": "bound_scalar_alignment_complete" if bound_alignment_complete else "alignment_blocked",
        "alignment_spec_sha256": active_alignment.to_dict()["payload_sha256"],
        "binding_spec_sha256": active_binding.to_dict()["payload_sha256"],
        "binding_assessment_sha256": assessment["payload_sha256"],
        "shot_id": artifact.shot_id,
        "artifact_sha256": artifact.artifact_sha256,
        "reference_channel": REFERENCE_CHANNEL,
        "reference_source_key": REFERENCE_SOURCE_KEY,
        "target_time_sha256": array_value_sha256(target_time),
        "target_samples": int(target_time.size),
        "target_period_s": target_period_s,
        "target_rate_hz": 1.0 / target_period_s,
        "n_canonical_channels": len(active_binding.bindings),
        "n_bound_channels_aligned": aligned_count,
        "n_channels_not_aligned": len(active_binding.bindings) - aligned_count,
        "bound_scalar_alignment_complete": bound_alignment_complete,
        "full_canonical_extraction_admissible": aligned_count == len(active_binding.bindings),
        "channels": channel_reports,
        "payload_sha256": None,
    }
    report["payload_sha256"] = canonical_json_sha256(report)
    return AlignedChannelSet(
        time_s=target_time,
        values=MappingProxyType(values),
        valid_masks=MappingProxyType(masks),
        report=MappingProxyType(report),
    )
