# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — FAIR-MAST disruption signal-binding contract tests
"""Real-surface tests for the versioned FAIR-MAST signal-binding contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from validation.mast_disruption_signal_binding import (
    CANONICAL_DISRUPTION_CHANNELS,
    MAST_INGESTION_COMMIT_URL,
    MAST_SIGNAL_BINDING_SCHEMA,
    MastSignalBindingSpec,
    SignalBinding,
    SignalBindingSpecError,
    assess_artifact_signal_bindings,
    mast_level2_signal_binding_spec,
)
from validation.mast_source_artifact_reader import VerifiedSourceArtifact, read_verified_npz_artifact
from validation.mast_source_object_manifest import canonical_json_sha256, file_sha256
from validation.migrate_mast_source_object_manifest import migrate_material_manifest_v1


def _source_xarray_metadata(dimensions: tuple[str, ...], units: str) -> Mapping[str, Any]:
    """Return immutable SourceObjectManifest-style xarray metadata."""
    return MappingProxyType(
        {
            "metadata_status": "source_xarray",
            "dimensions": dimensions,
            "units": units,
            "timebase": dimensions[-1],
            "source_attributes": MappingProxyType({}),
        }
    )


def _verified_artifact(
    *,
    ip_units: str = "A",
    equilibrium_time_units: str = "s",
    ip_values: NDArray[np.float64] | None = None,
) -> VerifiedSourceArtifact:
    """Build a verified-reader value with the exact bound MAST source schema."""
    summary_time = np.asarray([0.0, 0.1, 0.2], dtype=np.float64)
    equilibrium_time = np.asarray([0.0, 0.1, 0.2], dtype=np.float64)
    arrays: dict[str, NDArray[np.float64]] = {
        "summary.time": summary_time,
        "summary.ip": (np.asarray([0.0, 6.0e5, 5.5e5], dtype=np.float64) if ip_values is None else ip_values),
        "summary.line_average_n_e": np.asarray([np.nan, 2.1e19, 2.2e19], dtype=np.float64),
        "equilibrium.time": equilibrium_time,
        "equilibrium.q95": np.asarray([np.nan, 4.2, 4.0], dtype=np.float64),
        "equilibrium.magnetic_axis_z": np.asarray([np.nan, 0.01, 0.02], dtype=np.float64),
    }
    metadata: dict[str, Mapping[str, Any]] = {
        "summary.time": _source_xarray_metadata(("time",), "s"),
        "summary.ip": _source_xarray_metadata(("time",), ip_units),
        "summary.line_average_n_e": _source_xarray_metadata(("time",), "1 / m ** 3"),
        "equilibrium.time": _source_xarray_metadata(("time",), equilibrium_time_units),
        "equilibrium.q95": _source_xarray_metadata(("time",), ""),
        "equilibrium.magnetic_axis_z": _source_xarray_metadata(("time",), "m"),
    }
    for value in arrays.values():
        value.setflags(write=False)
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


def _binding_result(report: Mapping[str, object], channel: str) -> Mapping[str, object]:
    """Return one channel result from a public assessment report."""
    bindings = cast(list[dict[str, object]], report["bindings"])
    return next(binding for binding in bindings if binding["channel"] == channel)


def _save_named_arrays(path: Path, arrays: Mapping[str, object]) -> None:
    """Write dynamically named NPZ members through NumPy's public writer."""
    writer = cast(Callable[..., None], np.savez_compressed)
    writer(path, **arrays)


def test_spec_covers_every_canonical_channel_once_in_order() -> None:
    """The versioned spec neither omits nor silently adds a canonical channel."""
    spec = mast_level2_signal_binding_spec()

    assert spec.schema_version == MAST_SIGNAL_BINDING_SCHEMA
    assert spec.ingestion_commit_url == MAST_INGESTION_COMMIT_URL
    assert tuple(binding.channel for binding in spec.bindings) == CANONICAL_DISRUPTION_CHANNELS
    assert len(set(CANONICAL_DISRUPTION_CHANNELS)) == len(CANONICAL_DISRUPTION_CHANNELS)


def test_spec_preserves_exact_bound_units_timebases_signs_and_transforms() -> None:
    """Direct scalar mappings retain the official source semantics and conversions."""
    bindings = {binding.channel: binding for binding in mast_level2_signal_binding_spec().bindings}

    assert bindings["Ip_MA"].source_key == "summary.ip"
    assert bindings["Ip_MA"].source_units == "A"
    assert bindings["Ip_MA"].output_units == "MA"
    assert bindings["Ip_MA"].transform_parameters == (("factor", "1e-6"),)
    assert "anti-clockwise" in bindings["Ip_MA"].sign_convention
    assert bindings["q95"].source_key == "equilibrium.q95"
    assert bindings["q95"].timebase_key == "equilibrium.time"
    assert bindings["q95"].timebase_dimensions == ("time",)
    assert bindings["q95"].timebase_units == "s"
    assert bindings["q95"].source_units == ""
    assert bindings["ne_1e19"].transform_parameters == (("factor", "1e-19"),)
    assert bindings["vertical_position_m"].source_key == "equilibrium.magnetic_axis_z"


def test_unresolved_semantics_are_named_blockers_not_fallback_bindings() -> None:
    """BT, beta, modal, and magnetic-derivative ambiguity stays fail closed."""
    bindings = {binding.channel: binding for binding in mast_level2_signal_binding_spec().bindings}
    blockers = {
        channel: bindings[channel].blocker
        for channel in ("BT_T", "beta_N", "n1_amp", "n2_amp", "locked_mode_amp", "dBdt_gauss_per_s")
    }

    assert blockers == {
        "BT_T": "bvac_rmag_semantics_not_yet_approved_as_canonical_bt",
        "beta_N": "source_units_conflict_with_dimensionless_definition",
        "n1_amp": "saddle_geometry_angle_units_and_modal_reduction_unresolved",
        "n2_amp": "saddle_geometry_angle_units_and_modal_reduction_unresolved",
        "locked_mode_amp": "saddle_geometry_angle_units_and_modal_reduction_unresolved",
        "dBdt_gauss_per_s": "source_units_and_label_conflict",
    }
    assert all(bindings[channel].status == "blocked" for channel in blockers)
    assert all("never" in binding.missing_data_rule for binding in bindings.values())


def test_spec_payload_is_fresh_self_digested_json() -> None:
    """Serialisation is deterministic without returning shared mutable containers."""
    spec = mast_level2_signal_binding_spec()
    first = spec.to_dict()
    second = spec.to_dict()
    digest_payload = dict(first)
    digest_payload["payload_sha256"] = None

    assert first == second
    assert first is not second
    assert first["bindings"] is not second["bindings"]
    assert first["payload_sha256"] == canonical_json_sha256(digest_payload)


def test_assessment_verifies_only_fully_attested_bound_source_members() -> None:
    """Live-metadata mappings verify while six source-semantic blockers remain closed."""
    report = assess_artifact_signal_bindings(_verified_artifact())

    assert report["binding_contract_complete"] is True
    assert report["channel_extraction_admissible"] is False
    assert report["n_canonical_channels"] == 11
    assert report["n_source_metadata_verified"] == 5
    assert report["n_blocked"] == 6
    assert _binding_result(report, "Ip_MA")["status"] == "source_metadata_verified"
    assert _binding_result(report, "BT_T")["status"] == "binding_blocked"
    assert _binding_result(report, "ne_1e19")["finite_samples"] == 2


def test_assessment_rejects_source_unit_drift() -> None:
    """A unit change in the attested source metadata blocks the affected channel."""
    report = assess_artifact_signal_bindings(_verified_artifact(ip_units="kA"))
    ip_result = _binding_result(report, "Ip_MA")

    assert ip_result["status"] == "source_metadata_mismatch"
    assert ip_result["reason_code"] == "source_dimensions_or_units_mismatch"
    assert ip_result["actual_units"] == "kA"
    assert report["n_source_metadata_verified"] == 4


def test_assessment_rejects_signal_timebase_shape_mismatch() -> None:
    """A scalar signal that does not match its declared timebase fails closed."""
    report = assess_artifact_signal_bindings(_verified_artifact(ip_values=np.asarray([1.0, 2.0], dtype=np.float64)))
    ip_result = _binding_result(report, "Ip_MA")

    assert ip_result["status"] == "source_shape_mismatch"
    assert ip_result["source_shape"] == [2]
    assert ip_result["timebase_shape"] == [3]


def test_assessment_rejects_timebase_metadata_drift() -> None:
    """A time-coordinate unit drift blocks every mapping that depends on it."""
    report = assess_artifact_signal_bindings(_verified_artifact(equilibrium_time_units="ms"))

    for channel in ("q95", "vertical_position_m"):
        result = _binding_result(report, channel)
        assert result["status"] == "timebase_metadata_mismatch"
        assert result["reason_code"] == "timebase_dimensions_or_units_mismatch"
        assert result["actual_timebase_units"] == "ms"


def test_legacy_real_npz_reader_cannot_launder_missing_source_metadata(tmp_path: Path) -> None:
    """The v1 migration-reader-spec path keeps values-only metadata inadmissible."""
    path = tmp_path / "shot_30421.npz"
    _save_named_arrays(
        path,
        {
            "summary.time": np.asarray([0.0, 0.1], dtype=np.float64),
            "summary.ip": np.asarray([0.0, 6.0e5], dtype=np.float64),
            "summary.line_average_n_e": np.asarray([np.nan, 2.0e19], dtype=np.float64),
            "equilibrium.time": np.asarray([0.0, 0.1], dtype=np.float64),
            "equilibrium.q95": np.asarray([np.nan, 4.0], dtype=np.float64),
        },
    )
    legacy = {
        "schema_version": "scpn-control.mast-disruption-material.v1",
        "synthetic": False,
        "source": {"path_template": "s3://mast/level2/shots/{shot_id}.zarr"},
        "licence": "MIT",
        "retrieved_at": "2026-07-10T03:00:00Z",
        "shots": [
            {
                "shot_id": 30421,
                "status": "acquired",
                "npz": path.name,
                "checksum_sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
        ],
    }
    migrated = migrate_material_manifest_v1(legacy, artifact_root=tmp_path)
    artifact = read_verified_npz_artifact(migrated, artifact_root=tmp_path, shot_id=30421)

    report = assess_artifact_signal_bindings(artifact)

    assert report["channel_extraction_admissible"] is False
    assert report["n_source_metadata_verified"] == 0
    assert _binding_result(report, "Ip_MA")["status"] == "source_metadata_unverified"
    assert _binding_result(report, "Ip_MA")["reason_code"] == "manifest_does_not_attest_source_xarray_metadata"
    assert _binding_result(report, "vertical_position_m")["status"] == "source_member_absent"


def test_invalid_spec_and_binding_shapes_are_rejected_at_construction() -> None:
    """External callers cannot construct an incomplete versioned contract."""
    valid = mast_level2_signal_binding_spec()
    with pytest.raises(SignalBindingSpecError, match="canonical channels"):
        MastSignalBindingSpec(
            schema_version=valid.schema_version,
            version=valid.version,
            machine=valid.machine,
            source_schema=valid.source_schema,
            ingestion_commit_url=valid.ingestion_commit_url,
            bindings=valid.bindings[:-1],
        )
    with pytest.raises(SignalBindingSpecError, match="requires source key"):
        SignalBinding(
            channel="Ip_MA",
            status="bound",
            source_key=None,
            source_dimensions=(),
            source_units="A",
            output_units="MA",
            timebase_key=None,
            timebase_dimensions=(),
            timebase_units=None,
            sign_convention="known",
            validity_interval="finite",
            missing_data_rule="mask",
            transform="scale",
            transform_parameters=(("factor", "1e-6"),),
            citation=MAST_INGESTION_COMMIT_URL,
            uncertainty="unknown",
        )


def test_blocked_binding_requires_a_stable_reason_code() -> None:
    """A blocked record without its actionable reason is itself invalid."""
    with pytest.raises(SignalBindingSpecError, match="requires a reason code"):
        SignalBinding(
            channel="BT_T",
            status="blocked",
            source_key="equilibrium.bvac_rmag",
            source_dimensions=("time",),
            source_units="T",
            output_units="T",
            timebase_key="equilibrium.time",
            timebase_dimensions=("time",),
            timebase_units="s",
            sign_convention="unresolved",
            validity_interval="blocked",
            missing_data_rule="blocked",
            transform="candidate",
            transform_parameters=(),
            citation=MAST_INGESTION_COMMIT_URL,
            uncertainty="unknown",
            blocker=None,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"channel": ""}, "must be non-empty"),
        ({"output_units": ""}, "must be non-empty"),
        ({"citation": ""}, "must be non-empty"),
        ({"source_key": "summary.ip", "timebase_key": None}, "requires source key"),
        (
            {"source_key": "summary.ip", "timebase_key": "summary.time", "source_dimensions": ()},
            "requires source key",
        ),
        ({"blocker": "not_allowed_on_bound"}, "cannot carry a blocker"),
        ({"status": "unknown"}, "unsupported binding status"),
        (
            {"transform_parameters": (("z", "1"), ("a", "2"))},
            "transform parameters must be sorted",
        ),
    ],
)
def test_binding_constructor_rejects_each_inconsistent_contract_field(
    overrides: Mapping[str, object],
    message: str,
) -> None:
    """Each constructor invariant fails before a malformed mapping can escape."""
    values: dict[str, object] = {
        "channel": "Ip_MA",
        "status": "bound",
        "source_key": "summary.ip",
        "source_dimensions": ("time",),
        "source_units": "A",
        "output_units": "MA",
        "timebase_key": "summary.time",
        "timebase_dimensions": ("time",),
        "timebase_units": "s",
        "sign_convention": "known",
        "validity_interval": "finite",
        "missing_data_rule": "mask",
        "transform": "scale",
        "transform_parameters": (("factor", "1e-6"),),
        "citation": MAST_INGESTION_COMMIT_URL,
        "uncertainty": "unknown",
        "blocker": None,
    }
    values.update(overrides)

    with pytest.raises(SignalBindingSpecError, match=message):
        SignalBinding(**cast(Any, values))


@pytest.mark.parametrize(
    ("schema_version", "version", "machine", "message"),
    [
        ("wrong-schema", "1.0.0", "MAST", "schema_version"),
        (MAST_SIGNAL_BINDING_SCHEMA, "2.0.0", "MAST", "version and machine"),
        (MAST_SIGNAL_BINDING_SCHEMA, "1.0.0", "DIII-D", "version and machine"),
    ],
)
def test_spec_constructor_rejects_wrong_schema_version_or_machine(
    schema_version: str,
    version: str,
    machine: str,
    message: str,
) -> None:
    """A spec cannot masquerade as another schema, version, or machine."""
    valid = mast_level2_signal_binding_spec()
    with pytest.raises(SignalBindingSpecError, match=message):
        MastSignalBindingSpec(
            schema_version=schema_version,
            version=version,
            machine=machine,
            source_schema=valid.source_schema,
            ingestion_commit_url=valid.ingestion_commit_url,
            bindings=valid.bindings,
        )
