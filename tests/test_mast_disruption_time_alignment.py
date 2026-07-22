# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Mask-preserving FAIR-MAST time-alignment tests
"""Real-surface tests for the L2F-11 scalar time-alignment contract."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Any, Mapping, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from validation.mast_disruption_signal_binding import MastSignalBindingSpec, mast_level2_signal_binding_spec
from validation.mast_disruption_time_alignment import (
    MAST_TIME_ALIGNMENT_SCHEMA,
    TimeAlignmentError,
    TimeAlignmentSpec,
    align_verified_bound_channels,
    design_antialias_filter,
)
from validation.mast_source_artifact_reader import VerifiedSourceArtifact
from validation.mast_source_object_manifest import canonical_json_sha256


def _metadata(dimensions: tuple[str, ...], units: str, *, status: str = "source_xarray") -> Mapping[str, Any]:
    """Build immutable verified-reader metadata for one source array."""
    return MappingProxyType(
        {
            "metadata_status": status,
            "dimensions": dimensions,
            "units": units,
            "timebase": dimensions[-1],
            "source_attributes": MappingProxyType({}),
        }
    )


def _artifact(
    *,
    target_time: NDArray[np.float64] | None = None,
    equilibrium_time: NDArray[np.float64] | None = None,
    q95: NDArray[np.float64] | None = None,
    q95_units: str = "",
) -> VerifiedSourceArtifact:
    """Build a verified source artifact containing the five bound scalar mappings."""
    summary_time = (
        np.asarray([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        if target_time is None
        else np.asarray(target_time, dtype=np.float64)
    )
    eq_time = (
        np.asarray([0.0, 0.2, 0.4], dtype=np.float64)
        if equilibrium_time is None
        else np.asarray(equilibrium_time, dtype=np.float64)
    )
    q_values = (
        np.linspace(3.0, 4.0, eq_time.size, dtype=np.float64) if q95 is None else np.asarray(q95, dtype=np.float64)
    )
    arrays: dict[str, NDArray[np.float64]] = {
        "summary.time": summary_time,
        "summary.ip": np.linspace(0.0, 8.0e5, summary_time.size, dtype=np.float64),
        "summary.line_average_n_e": np.linspace(1.0e19, 2.0e19, summary_time.size, dtype=np.float64),
        "equilibrium.time": eq_time,
        "equilibrium.q95": q_values,
        "equilibrium.magnetic_axis_z": np.linspace(-0.02, 0.02, eq_time.size, dtype=np.float64),
    }
    metadata: dict[str, Mapping[str, Any]] = {
        "summary.time": _metadata(("time",), "s"),
        "summary.ip": _metadata(("time",), "A"),
        "summary.line_average_n_e": _metadata(("time",), "1 / m ** 3"),
        "equilibrium.time": _metadata(("time",), "s"),
        "equilibrium.q95": _metadata(("time",), q95_units),
        "equilibrium.magnetic_axis_z": _metadata(("time",), "m"),
    }
    for array in arrays.values():
        array.setflags(write=False)
    return VerifiedSourceArtifact(
        shot_id=30421,
        artifact_kind="derived_npz_cache",
        local_path="shot_30421.npz",
        source_uri="s3://mast/level2/shots/30421.zarr",
        manifest_sha256="1" * 64,
        artifact_sha256="2" * 64,
        parent_digest="3" * 64,
        transform_digest="4" * 64,
        arrays=MappingProxyType(arrays),
        metadata=MappingProxyType(metadata),
    )


def _channel_report(result: Mapping[str, Any], channel: str) -> Mapping[str, Any]:
    """Return one channel's alignment provenance record."""
    records = cast(list[dict[str, Any]], result["channels"])
    return next(record for record in records if record["channel"] == channel)


def test_alignment_spec_is_versioned_self_digested_and_explicit() -> None:
    """The policy serialisation records every edge/gap/filter decision."""
    payload = TimeAlignmentSpec().to_dict()
    unsigned = dict(payload)
    unsigned["payload_sha256"] = None

    assert payload["schema_version"] == MAST_TIME_ALIGNMENT_SCHEMA
    assert payload["reference_source_key"] == "summary.time"
    assert "no_extrapolation" in cast(str, payload["edge_policy"])
    assert "never_bridge" in cast(str, payload["gap_policy"])
    assert payload["payload_sha256"] == canonical_json_sha256(unsigned)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": "wrong"}, "schema_version"),
        ({"version": "2"}, "version and reference"),
        ({"reference_channel": "q95"}, "version and reference"),
        ({"uniform_rtol": 0.0}, "uniform_rtol"),
        ({"uniform_rtol": float("nan")}, "uniform_rtol"),
        ({"cutoff_target_nyquist_fraction": 1.0}, "cutoff_target"),
        ({"attenuation_db": 19.0}, "attenuation_db"),
        ({"max_filter_taps": 4}, "max_filter_taps"),
    ],
)
def test_alignment_spec_rejects_ambiguous_policy(overrides: Mapping[str, object], message: str) -> None:
    """Every malformed policy field fails before array processing."""
    values: dict[str, object] = {
        "schema_version": MAST_TIME_ALIGNMENT_SCHEMA,
        "version": "1.0.0",
        "reference_channel": "time_s",
        "uniform_rtol": 1.0e-5,
        "cutoff_target_nyquist_fraction": 0.8,
        "attenuation_db": 80.0,
        "max_filter_taps": 4097,
    }
    values.update(overrides)
    with pytest.raises(TimeAlignmentError, match=message):
        TimeAlignmentSpec(**cast(Any, values))


@pytest.mark.parametrize(
    ("source_period", "target_period", "message"),
    [
        (0.0, 0.01, "source_period_s"),
        (0.001, float("nan"), "target_period_s"),
        (0.01, 0.001, "only defined"),
    ],
)
def test_antialias_design_rejects_invalid_or_non_downsampled_rates(
    source_period: float,
    target_period: float,
    message: str,
) -> None:
    """The FIR design cannot be invoked outside its physical rate domain."""
    with pytest.raises(TimeAlignmentError, match=message):
        design_antialias_filter(source_period, target_period)


def test_antialias_design_is_odd_bounded_digest_bound_and_measured() -> None:
    """A 1 kHz to 100 Hz design records a strong measured stopband."""
    design = design_antialias_filter(0.001, 0.01)

    assert design.coefficients.size % 2 == 1
    assert design.coefficients.size <= 4097
    assert design.provenance["numtaps"] == design.coefficients.size
    assert cast(float, design.provenance["measured_stopband_max_db_at_or_above_target_nyquist"]) < -70.0
    assert len(cast(str, design.provenance["coefficient_sha256"])) == 64
    assert not design.coefficients.flags.writeable


def test_antialias_design_fails_when_policy_tap_bound_would_be_weakened() -> None:
    """A too-small tap budget fails instead of lowering attenuation."""
    with pytest.raises(TimeAlignmentError, match="exceeding max_filter_taps"):
        design_antialias_filter(0.001, 0.01, spec=TimeAlignmentSpec(max_filter_taps=3))


def test_alignment_maps_five_bound_channels_with_units_and_masks() -> None:
    """The exact reference grid, scalar transforms and blocked-channel count survive."""
    aligned = align_verified_bound_channels(_artifact())

    assert tuple(aligned.values) == ("time_s", "Ip_MA", "q95", "ne_1e19", "vertical_position_m")
    assert np.array_equal(aligned.time_s, np.asarray([0.0, 0.1, 0.2, 0.3, 0.4]))
    assert np.allclose(aligned.values["Ip_MA"], np.linspace(0.0, 0.8, 5))
    assert np.allclose(aligned.values["ne_1e19"], np.linspace(1.0, 2.0, 5))
    assert np.allclose(aligned.values["q95"], np.linspace(3.0, 4.0, 5))
    assert all(bool(np.all(mask)) for mask in aligned.valid_masks.values())
    assert aligned.report["n_bound_channels_aligned"] == 5
    assert aligned.report["n_channels_not_aligned"] == 6
    assert aligned.report["bound_scalar_alignment_complete"] is True
    assert aligned.report["full_canonical_extraction_admissible"] is False


def test_alignment_never_bridges_nonfinite_source_gaps() -> None:
    """Two missing equilibrium samples remain false-mask NaNs, not interpolation."""
    time = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    aligned = align_verified_bound_channels(
        _artifact(equilibrium_time=time, q95=np.asarray([3.0, np.nan, np.nan, 3.8, 4.0]))
    )

    assert aligned.valid_masks["q95"].tolist() == [True, False, False, True, True]
    assert np.isnan(aligned.values["q95"][1:3]).all()
    report = _channel_report(aligned.report, "q95")
    assert report["contiguous_finite_runs"] == 2
    assert report["gap_policy"] == "never interpolate across non-finite source runs"


def test_alignment_does_not_extrapolate_beyond_source_support() -> None:
    """Reference-grid edges outside equilibrium support remain invalid."""
    eq_time = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
    aligned = align_verified_bound_channels(_artifact(equilibrium_time=eq_time))

    assert aligned.valid_masks["q95"].tolist() == [False, True, True, True, False]
    assert np.isnan(aligned.values["q95"][[0, 4]]).all()


def test_single_finite_source_sample_maps_only_to_its_exact_target() -> None:
    """A singleton finite run cannot smear across neighbouring target samples."""
    time = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    q95 = np.asarray([np.nan, np.nan, 4.2, np.nan, np.nan], dtype=np.float64)
    aligned = align_verified_bound_channels(_artifact(equilibrium_time=time, q95=q95))

    assert aligned.valid_masks["q95"].tolist() == [False, False, True, False, False]
    assert aligned.values["q95"][2] == pytest.approx(4.2)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (np.asarray([0.0]), "at least two"),
        (np.asarray([0.0, np.nan, 0.2]), "finite one-dimensional"),
        (np.asarray([0.0, 0.2, 0.1]), "strictly increasing"),
        (np.asarray([0.0, 0.1, 0.25]), "not uniform"),
    ],
)
def test_reference_grid_must_be_finite_strict_and_uniform(target: NDArray[np.float64], message: str) -> None:
    """Reference time corruption blocks the entire alignment contract."""
    with pytest.raises(TimeAlignmentError, match=message):
        align_verified_bound_channels(_artifact(target_time=target))


def test_binding_metadata_drift_is_not_laundered_by_alignment() -> None:
    """A q95 unit mismatch remains not-aligned while other bound channels proceed."""
    aligned = align_verified_bound_channels(_artifact(q95_units="T"))

    assert "q95" not in aligned.values
    q95_report = _channel_report(aligned.report, "q95")
    assert q95_report["status"] == "not_aligned"
    assert q95_report["binding_assessment_status"] == "source_metadata_mismatch"
    assert aligned.report["bound_scalar_alignment_complete"] is False


def test_downsampling_suppresses_above_nyquist_energy_and_records_filter() -> None:
    """A 200 Hz component is removed before a 1 kHz to 100 Hz alignment."""
    target = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    source = np.linspace(0.0, 1.0, 1001, dtype=np.float64)
    low = 3.5 + 0.2 * np.sin(2.0 * np.pi * 10.0 * source)
    high = np.sin(2.0 * np.pi * 200.0 * source)
    aligned = align_verified_bound_channels(_artifact(target_time=target, equilibrium_time=source, q95=low + high))
    mask = aligned.valid_masks["q95"]
    expected = 3.5 + 0.2 * np.sin(2.0 * np.pi * 10.0 * target[mask])

    assert int(np.count_nonzero(mask)) > 20
    assert float(np.sqrt(np.mean((aligned.values["q95"][mask] - expected) ** 2))) < 0.03
    provenance = _channel_report(aligned.report, "q95")
    assert provenance["method"] == "kaiser_fir_then_piecewise_linear"
    filter_report = cast(dict[str, object], provenance["anti_alias_filter"])
    assert cast(float, filter_report["measured_stopband_max_db_at_or_above_target_nyquist"]) < -70.0


def test_short_downsample_runs_are_masked_instead_of_unfiltered() -> None:
    """Finite bursts shorter than the FIR transient are rejected, never aliased."""
    target = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    source = np.linspace(0.0, 1.0, 1001, dtype=np.float64)
    q95 = np.full(source.shape, np.nan, dtype=np.float64)
    q95[450:551] = 4.0
    aligned = align_verified_bound_channels(_artifact(target_time=target, equilibrium_time=source, q95=q95))
    report = _channel_report(aligned.report, "q95")

    assert not bool(np.any(aligned.valid_masks["q95"]))
    assert np.isnan(aligned.values["q95"]).all()
    assert report["short_runs_rejected_for_filter_transient"] == 1


def test_outputs_are_immutable_fresh_and_digest_bound() -> None:
    """Callers cannot mutate aligned evidence and repeated reports are deterministic."""
    first = align_verified_bound_channels(_artifact())
    second = align_verified_bound_channels(_artifact())
    unsigned = dict(first.report)
    unsigned["payload_sha256"] = None

    assert first.report == second.report
    assert first.report is not second.report
    assert first.report["payload_sha256"] == canonical_json_sha256(unsigned)
    assert not first.time_s.flags.writeable
    assert all(not value.flags.writeable for value in first.values.values())
    assert all(not mask.flags.writeable for mask in first.valid_masks.values())
    with pytest.raises(ValueError, match="read-only"):
        first.values["q95"][0] = 99.0


def test_unsupported_or_malformed_transform_fails_closed() -> None:
    """Alignment never guesses a source-to-output transform."""
    base = mast_level2_signal_binding_spec()
    bindings = list(base.bindings)
    ip_index = next(index for index, binding in enumerate(bindings) if binding.channel == "Ip_MA")
    bindings[ip_index] = replace(bindings[ip_index], transform="mystery")
    unsupported = MastSignalBindingSpec(
        schema_version=base.schema_version,
        version=base.version,
        machine=base.machine,
        source_schema=base.source_schema,
        ingestion_commit_url=base.ingestion_commit_url,
        bindings=tuple(bindings),
    )
    with pytest.raises(TimeAlignmentError, match="not alignable"):
        align_verified_bound_channels(_artifact(), binding_spec=unsupported)

    bindings[ip_index] = replace(base.bindings[ip_index], transform_parameters=())
    malformed = replace(base, bindings=tuple(bindings))
    with pytest.raises(TimeAlignmentError, match="invalid scale"):
        align_verified_bound_channels(_artifact(), binding_spec=malformed)

    bindings[ip_index] = replace(base.bindings[ip_index], transform_parameters=(("factor", "nan"),))
    nonfinite = replace(base, bindings=tuple(bindings))
    with pytest.raises(TimeAlignmentError, match="factor must be finite"):
        align_verified_bound_channels(_artifact(), binding_spec=nonfinite)
