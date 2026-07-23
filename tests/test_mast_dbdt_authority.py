# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST dB/dt authority tests
"""Contract tests for the fail-closed MAST dB/dt source-authority gate."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pytest

import validation.mast_dbdt_authority as dbdt_module
from validation.mast_dbdt_authority import (
    DBDT_AUTHORITY_SCHEMA,
    EXPECTED_GEOMETRY_CHANNELS,
    EXPECTED_SOURCE_CHANNELS,
    EXPECTED_SOURCE_TO_GEOMETRY_INDICES,
    FAIR_MAST_DBDT_PROFILE_URL,
    FIELD_KEY,
    GEOMETRY_KEYS,
    MAPPING_SCALE,
    MAST_MAGNETICS_PAPER_URL,
    TIMEBASE_KEY,
    DbdtAuthorityError,
    assess_dbdt_authority,
    build_dbdt_authority_report,
    main,
    mast_dbdt_authority_spec,
)
from validation.mast_saddle_modal_authority import FAIR_MAST_MAPPING_COMMIT, FAIR_MAST_MAPPING_SHA256
from validation.mast_source_artifact_reader import VerifiedSourceArtifact
from validation.mast_source_object_manifest import canonical_json_sha256


def _metadata(
    *,
    dimensions: tuple[str, ...],
    units: str,
    attributes: Mapping[str, object] | None,
    value_sha256: str,
) -> Mapping[str, Any]:
    """Build immutable metadata matching the verified artefact reader."""
    return MappingProxyType(
        {
            "metadata_status": "source_xarray",
            "dimensions": dimensions,
            "units": units,
            "source_attributes": None if attributes is None else MappingProxyType(dict(attributes)),
            "value_sha256": value_sha256,
        }
    )


def _authority_contract(*, source_quantity: str = "magnetic_field") -> dict[str, object]:
    """Build one content-digested, source-quantity-specific authority contract."""
    units_and_transform = {
        "magnetic_field": ("T", "differentiate_once_then_multiply_by_1e4_G_per_T"),
        "magnetic_field_time_derivative": (
            "T/s",
            "multiply_by_1e4_G_per_T_without_differentiation",
        ),
    }
    source_units, transform = units_and_transform[source_quantity]
    contract: dict[str, object] = {
        "schema_version": DBDT_AUTHORITY_SCHEMA,
        "mapping_commit": FAIR_MAST_MAPPING_COMMIT,
        "mapping_file_sha256": FAIR_MAST_MAPPING_SHA256,
        "source_quantity": source_quantity,
        "source_units": source_units,
        "transform": transform,
        "source_channels": list(EXPECTED_SOURCE_CHANNELS),
        "geometry_channels": list(EXPECTED_GEOMETRY_CHANNELS),
        "source_to_geometry_indices": list(EXPECTED_SOURCE_TO_GEOMETRY_INDICES),
        "field_to_geometry_join_sha256": "a" * 64,
        "mapping_scale_evidence_sha256": "b" * 64,
        "calibration_evidence_sha256": "c" * 64,
        "mapping_scale": MAPPING_SCALE,
        "mapping_scale_units": "dimensionless source conversion factor",
        "measured_component": "poloidal magnetic field at the centre-column Mirnov array",
        "probe_orientation_policy": "apply the signed source geometry orientation per probe",
        "sign_convention": "positive follows each attested probe winding orientation",
        "probe_reduction_policy": "maximum absolute valid-probe value with row identity retained",
        "missing_data_policy": "preserve finite gaps and reject any affected reduction interval",
        "bad_channel_policy": "exclude only source-attested bad rows before the declared reduction",
        "filter_policy": "zero-phase source-approved low-pass in physical time",
        "edge_policy": "reject filter transients and never bridge a finite-data gap",
        "standard_uncertainty_gauss_per_s": 10.0,
        "low_pass_cutoff_hz": 100_000.0,
        "primary_source_citations": [FAIR_MAST_DBDT_PROFILE_URL, MAST_MAGNETICS_PAPER_URL],
        "authority_contract_sha256": None,
    }
    contract["authority_contract_sha256"] = canonical_json_sha256(contract)
    return contract


def _field_attributes(*, complete: bool, source_quantity: str) -> dict[str, object]:
    """Build current-conflicted or fully attested field attributes."""
    derivative = source_quantity == "magnetic_field_time_derivative"
    attributes: dict[str, object] = {
        "name": "b_field_pol_probe_cc_field",
        "uda_name": "xmc/CC/MV/201",
        "label": "Tesla/sec" if derivative or not complete else "Tesla",
        "units": "T/s" if derivative else "T",
        "description": "centre column poloidal Mirnov array",
    }
    if complete:
        attributes.update(
            {
                "mapping_scale": MAPPING_SCALE,
                "dbdt_authority_contract": _authority_contract(source_quantity=source_quantity),
            }
        )
    return attributes


def _geometry_attributes(*, complete: bool) -> dict[str, object]:
    """Build released or current development geometry metadata."""
    return {
        "status": "released" if complete else "development",
        "revision": 1 if complete else 0,
        "creatorCommitId": "d" * 40 if complete else "",
        "signedOffBy": "MAST Data Systems Team" if complete else "lkogan",
    }


def _artifact(
    *,
    shot_id: int = 30421,
    complete: bool = False,
    source_quantity: str = "magnetic_field",
    arrays: Mapping[str, np.ndarray[Any, Any]] | None = None,
    metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> VerifiedSourceArtifact:
    """Build one immutable five-row Mirnov source fixture."""
    time = np.arange(300, dtype=np.float64) * 2.0e-6
    field = np.arange(1500, dtype=np.float64).reshape(5, 300)
    if not complete:
        field[:, 0] = np.nan
    default_arrays: dict[str, np.ndarray[Any, Any]] = {
        FIELD_KEY: field,
        TIMEBASE_KEY: time,
        GEOMETRY_KEYS[0]: np.full(40, 270.0, dtype=np.float64),
        GEOMETRY_KEYS[1]: np.full(40, 0.1806, dtype=np.float64),
        GEOMETRY_KEYS[2]: np.linspace(-1.525, 1.44875, 40, dtype=np.float64),
    }
    source_units = "T/s" if source_quantity == "magnetic_field_time_derivative" else "T"
    default_metadata: dict[str, Mapping[str, Any]] = {
        FIELD_KEY: _metadata(
            dimensions=("b_field_pol_probe_cc_channel", "time_mirnov"),
            units=source_units,
            attributes=_field_attributes(complete=complete, source_quantity=source_quantity),
            value_sha256="e" * 64,
        ),
        TIMEBASE_KEY: _metadata(
            dimensions=("time_mirnov",),
            units="s",
            attributes={"units": "s"},
            value_sha256="f" * 64,
        ),
    }
    for key, units in zip(GEOMETRY_KEYS, ("degrees", "m", "m"), strict=True):
        default_metadata[key] = _metadata(
            dimensions=("b_field_pol_probe_cc_geometry_channel",),
            units=units if complete else "SI, degrees, m",
            attributes=_geometry_attributes(complete=complete),
            value_sha256="1" * 64,
        )
    active_arrays = dict(default_arrays if arrays is None else arrays)
    active_metadata = dict(default_metadata if metadata is None else metadata)
    for value in active_arrays.values():
        value.setflags(write=False)
    return VerifiedSourceArtifact(
        shot_id=shot_id,
        artifact_kind="derived_npz_cache",
        local_path=f"shot_{shot_id}.npz",
        source_uri=f"s3://mast/level2/shots/{shot_id}.zarr",
        manifest_sha256="2" * 64,
        artifact_sha256=str(shot_id % 10) * 64,
        parent_digest="3" * 64,
        transform_digest="4" * 64,
        arrays=MappingProxyType(active_arrays),
        metadata=MappingProxyType(active_metadata),
    )


def _replace_field_metadata(
    artifact: VerifiedSourceArtifact,
    *,
    units: str | None = None,
    attributes: Mapping[str, object] | None = None,
    dimensions: tuple[str, ...] = ("b_field_pol_probe_cc_channel", "time_mirnov"),
) -> dict[str, Mapping[str, Any]]:
    """Return metadata with one altered field descriptor."""
    metadata = dict(artifact.metadata)
    current = artifact.metadata[FIELD_KEY]
    metadata[FIELD_KEY] = _metadata(
        dimensions=dimensions,
        units=cast(str, current["units"]) if units is None else units,
        attributes=cast(Mapping[str, object] | None, current["source_attributes"])
        if attributes is None
        else attributes,
        value_sha256="e" * 64,
    )
    return metadata


def _digest(payload: Mapping[str, object]) -> str:
    """Recompute one self-digested payload."""
    copy = dict(payload)
    copy["payload_sha256"] = None
    return canonical_json_sha256(copy)


def test_spec_pins_conflict_and_exactly_two_transform_branches() -> None:
    """The contract makes double differentiation structurally inadmissible."""
    spec = mast_dbdt_authority_spec()

    assert spec["schema_version"] == DBDT_AUTHORITY_SCHEMA
    assert spec["mapping_scale"] == MAPPING_SCALE
    branches = cast(dict[str, str], spec["admissible_transform_branches"])
    assert branches == {
        "magnetic_field": "differentiate_once_then_multiply_by_1e4_G_per_T",
        "magnetic_field_time_derivative": "multiply_by_1e4_G_per_T_without_differentiation",
    }
    assert spec["payload_sha256"] == _digest(spec)


def test_current_source_conflict_is_measured_without_executing_transform() -> None:
    """The current T versus Tesla/sec conflict remains explicit and fail closed."""
    report = assess_dbdt_authority(_artifact())

    assert report["status"] == "blocked"
    assert report["canonical_dbdt_binding_admissible"] is False
    assert report["dbdt_transform_executed"] is False
    blockers = cast(list[str], report["blockers"])
    assert "dbdt_source_label_unit_conflict" in blockers
    assert "dbdt_mapping_scale_not_preserved" in blockers
    assert "dbdt_source_row_has_nonfinite_samples" in blockers
    assert "dbdt_authority_contract_absent" in blockers
    evidence = cast(dict[str, object], report["source_evidence"])
    assert evidence["sample_rate_hz"] == pytest.approx(500_000.0)
    assert evidence["field_nonfinite_positions_per_row"] == [[0], [0], [0], [0], [0]]
    assert report["payload_sha256"] == _digest(report)
    assert all(value is False for value in cast(dict[str, bool], report["claim_boundary"]).values())


@pytest.mark.parametrize("source_quantity", ["magnetic_field", "magnetic_field_time_derivative"])
def test_each_quantity_selects_one_valid_transform_branch(source_quantity: str) -> None:
    """Both attested source quantities can clear only through their own transform."""
    report = assess_dbdt_authority(_artifact(complete=True, source_quantity=source_quantity))

    assert report["status"] == "authority_verified"
    assert report["canonical_dbdt_binding_admissible"] is True
    assert report["dbdt_transform_executed"] is False
    assert report["blockers"] == []


@pytest.mark.parametrize(
    ("mutation", "blocker"),
    [
        ({"source_quantity": "voltage"}, "dbdt_source_quantity_not_attested"),
        ({"schema_version": "unknown"}, "dbdt_authority_contract_schema_mismatch"),
        ({"mapping_commit": "0" * 40}, "dbdt_mapping_lineage_mismatch"),
        ({"mapping_file_sha256": "0" * 64}, "dbdt_mapping_lineage_mismatch"),
        ({"source_units": "T/s"}, "dbdt_source_units_do_not_match_quantity"),
        (
            {"transform": "multiply_by_1e4_G_per_T_without_differentiation"},
            "dbdt_transform_does_not_match_source_quantity",
        ),
        ({"source_channels": list(reversed(EXPECTED_SOURCE_CHANNELS))}, "dbdt_source_row_identities_not_attested"),
        (
            {"geometry_channels": list(reversed(EXPECTED_GEOMETRY_CHANNELS))},
            "dbdt_geometry_row_identities_not_attested",
        ),
        (
            {"source_to_geometry_indices": [0, 1, 2, 3, 4]},
            "dbdt_field_geometry_join_indices_not_attested",
        ),
        ({"mapping_scale": 1.0}, "dbdt_mapping_scale_semantics_not_attested"),
        ({"mapping_scale_units": " "}, "dbdt_mapping_scale_semantics_not_attested"),
        ({"primary_source_citations": []}, "dbdt_primary_source_citations_absent"),
    ],
)
def test_identity_units_transform_scale_and_citations_have_independent_guards(
    mutation: Mapping[str, object],
    blocker: str,
) -> None:
    """Every source-identity and transform declaration is independently guarded."""
    base = _artifact(complete=True)
    attributes = _field_attributes(complete=True, source_quantity="magnetic_field")
    contract = _authority_contract()
    contract.update(mutation)
    contract["authority_contract_sha256"] = None
    contract["authority_contract_sha256"] = canonical_json_sha256(contract)
    attributes["dbdt_authority_contract"] = contract
    metadata = _replace_field_metadata(base, attributes=attributes)

    report = assess_dbdt_authority(_artifact(complete=True, arrays=base.arrays, metadata=metadata))

    assert blocker in cast(list[str], report["blockers"])


def test_source_metadata_units_must_match_contract_and_supported_quantity() -> None:
    """A contract cannot override or reinterpret the observed array units."""
    base = _artifact(complete=True)
    metadata = _replace_field_metadata(base, units="T/s")
    mismatch = assess_dbdt_authority(_artifact(complete=True, arrays=base.arrays, metadata=metadata))
    assert "dbdt_contract_units_do_not_match_source_metadata" in cast(list[str], mismatch["blockers"])

    metadata = _replace_field_metadata(base, units="V")
    unsupported = assess_dbdt_authority(_artifact(complete=True, arrays=base.arrays, metadata=metadata))
    blockers = cast(list[str], unsupported["blockers"])
    assert "dbdt_source_units_unsupported" in blockers
    assert "dbdt_contract_units_do_not_match_source_metadata" in blockers


@pytest.mark.parametrize(
    ("key", "blocker"),
    [
        ("field_to_geometry_join_sha256", "dbdt_field_geometry_join_evidence_absent"),
        ("mapping_scale_evidence_sha256", "dbdt_mapping_scale_evidence_absent"),
        ("calibration_evidence_sha256", "dbdt_calibration_evidence_absent"),
    ],
)
def test_lineage_evidence_requires_sha256_digests(key: str, blocker: str) -> None:
    """Named evidence cannot be replaced by a label or malformed digest."""
    base = _artifact(complete=True)
    attributes = _field_attributes(complete=True, source_quantity="magnetic_field")
    contract = _authority_contract()
    contract[key] = "invalid"
    contract["authority_contract_sha256"] = None
    contract["authority_contract_sha256"] = canonical_json_sha256(contract)
    attributes["dbdt_authority_contract"] = contract
    metadata = _replace_field_metadata(base, attributes=attributes)

    report = assess_dbdt_authority(_artifact(complete=True, arrays=base.arrays, metadata=metadata))

    assert blocker in cast(list[str], report["blockers"])


def test_authority_contract_digest_must_be_present_and_content_correct() -> None:
    """The contract's own digest detects silent post-attestation mutation."""
    base = _artifact(complete=True)
    for digest, blocker in (
        (None, "dbdt_authority_contract_digest_absent"),
        ("0" * 64, "dbdt_authority_contract_digest_mismatch"),
    ):
        attributes = _field_attributes(complete=True, source_quantity="magnetic_field")
        contract = _authority_contract()
        contract["authority_contract_sha256"] = digest
        attributes["dbdt_authority_contract"] = contract
        metadata = _replace_field_metadata(base, attributes=attributes)
        report = assess_dbdt_authority(_artifact(complete=True, arrays=base.arrays, metadata=metadata))
        assert blocker in cast(list[str], report["blockers"])


@pytest.mark.parametrize(
    ("key", "blocker"),
    [
        ("measured_component", "dbdt_measured_component_not_attested"),
        ("probe_orientation_policy", "dbdt_probe_orientation_not_attested"),
        ("sign_convention", "dbdt_sign_convention_not_attested"),
        ("probe_reduction_policy", "dbdt_probe_reduction_policy_absent"),
        ("missing_data_policy", "dbdt_missing_data_policy_absent"),
        ("bad_channel_policy", "dbdt_bad_channel_policy_absent"),
        ("filter_policy", "dbdt_filter_policy_absent"),
        ("edge_policy", "dbdt_edge_policy_absent"),
    ],
)
def test_physics_and_processing_policies_cannot_be_blank(key: str, blocker: str) -> None:
    """Each physical interpretation and processing policy must be substantive."""
    base = _artifact(complete=True)
    attributes = _field_attributes(complete=True, source_quantity="magnetic_field")
    contract = _authority_contract()
    contract[key] = " "
    contract["authority_contract_sha256"] = None
    contract["authority_contract_sha256"] = canonical_json_sha256(contract)
    attributes["dbdt_authority_contract"] = contract
    metadata = _replace_field_metadata(base, attributes=attributes)
    report = assess_dbdt_authority(_artifact(complete=True, arrays=base.arrays, metadata=metadata))
    assert blocker in cast(list[str], report["blockers"])


@pytest.mark.parametrize("bad_number", [None, True, 0.0, -1.0, float("inf")])
def test_cutoff_and_uncertainty_require_positive_finite_numbers(bad_number: object) -> None:
    """Absent, boolean, non-positive, and infinite quantities fail closed."""
    base = _artifact(complete=True)
    attributes = _field_attributes(complete=True, source_quantity="magnetic_field")
    contract = _authority_contract()
    contract["low_pass_cutoff_hz"] = bad_number
    contract["standard_uncertainty_gauss_per_s"] = bad_number
    contract["authority_contract_sha256"] = None
    contract["authority_contract_sha256"] = canonical_json_sha256(contract)
    attributes["dbdt_authority_contract"] = contract
    metadata = _replace_field_metadata(base, attributes=attributes)
    report = assess_dbdt_authority(_artifact(complete=True, arrays=base.arrays, metadata=metadata))
    blockers = cast(list[str], report["blockers"])
    assert "dbdt_low_pass_cutoff_not_attested" in blockers
    assert "dbdt_standard_uncertainty_absent" in blockers


def test_cutoff_must_be_below_observed_nyquist() -> None:
    """A declared physical cutoff cannot reach or exceed source Nyquist."""
    base = _artifact(complete=True)
    attributes = _field_attributes(complete=True, source_quantity="magnetic_field")
    contract = _authority_contract()
    contract["low_pass_cutoff_hz"] = 300_000.0
    contract["authority_contract_sha256"] = None
    contract["authority_contract_sha256"] = canonical_json_sha256(contract)
    attributes["dbdt_authority_contract"] = contract
    metadata = _replace_field_metadata(base, attributes=attributes)
    report = assess_dbdt_authority(_artifact(complete=True, arrays=base.arrays, metadata=metadata))
    assert "dbdt_low_pass_cutoff_not_below_nyquist" in cast(list[str], report["blockers"])


def test_source_members_shapes_dimensions_and_units_fail_closed() -> None:
    """Missing arrays and malformed source structure cannot reach authority logic."""
    base = _artifact(complete=True)
    arrays = {key: value for key, value in base.arrays.items() if key != GEOMETRY_KEYS[2]}
    missing = assess_dbdt_authority(_artifact(complete=True, arrays=arrays, metadata=base.metadata))
    assert "dbdt_source_members_absent" in cast(list[str], missing["blockers"])

    arrays = dict(base.arrays)
    arrays[FIELD_KEY] = np.zeros((4, 300), dtype=np.float64)
    arrays[GEOMETRY_KEYS[0]] = np.zeros((2, 20), dtype=np.float64)
    arrays[GEOMETRY_KEYS[1]] = np.full(40, np.nan, dtype=np.float64)
    malformed = assess_dbdt_authority(_artifact(complete=True, arrays=arrays, metadata=base.metadata))
    blockers = cast(list[str], malformed["blockers"])
    assert "dbdt_field_or_timebase_shape_mismatch" in blockers
    assert "dbdt_geometry_shape_mismatch" in blockers
    assert "dbdt_geometry_has_nonfinite_values" in blockers

    metadata = _replace_field_metadata(base, dimensions=("wrong",), units="V")
    metadata[TIMEBASE_KEY] = _metadata(dimensions=("wrong_time",), units="ms", attributes={}, value_sha256="f" * 64)
    malformed_metadata = assess_dbdt_authority(_artifact(complete=True, arrays=base.arrays, metadata=metadata))
    blockers = cast(list[str], malformed_metadata["blockers"])
    assert "dbdt_field_dimensions_mismatch" in blockers
    assert "dbdt_source_units_unsupported" in blockers
    assert "dbdt_timebase_metadata_mismatch" in blockers


def test_geometry_units_release_commit_signoff_and_digest_are_required() -> None:
    """Component units and released geometry provenance are independent gates."""
    base = _artifact(complete=True)
    metadata = dict(base.metadata)
    for key in GEOMETRY_KEYS:
        metadata[key] = _metadata(
            dimensions=("b_field_pol_probe_cc_geometry_channel",),
            units="SI, degrees, m",
            attributes={"status": "development", "revision": 0, "creatorCommitId": "", "signedOffBy": ""},
            value_sha256="invalid",
        )
    report = assess_dbdt_authority(_artifact(complete=True, arrays=base.arrays, metadata=metadata))
    blockers = cast(list[str], report["blockers"])
    assert "dbdt_geometry_units_not_component_specific" in blockers
    assert "dbdt_geometry_not_released" in blockers
    assert "dbdt_geometry_creator_commit_absent" in blockers
    assert "dbdt_geometry_signoff_absent" in blockers
    assert "dbdt_geometry_value_digest_absent" in blockers


def test_timebase_finiteness_order_and_uniformity_are_enforced() -> None:
    """Derivative and filter authority requires a finite, ordered, uniform clock."""
    base = _artifact(complete=True)
    arrays = dict(base.arrays)
    arrays[TIMEBASE_KEY] = np.asarray([0.0, np.nan, 0.1], dtype=np.float64)
    nonfinite = assess_dbdt_authority(_artifact(complete=True, arrays=arrays, metadata=base.metadata))
    assert "dbdt_timebase_not_finite" in cast(list[str], nonfinite["blockers"])

    arrays[TIMEBASE_KEY] = np.asarray([0.0, 0.2, 0.1], dtype=np.float64)
    unordered = assess_dbdt_authority(_artifact(complete=True, arrays=arrays, metadata=base.metadata))
    assert "dbdt_timebase_not_strictly_increasing" in cast(list[str], unordered["blockers"])

    time = np.arange(300, dtype=np.float64) * 2.0e-6
    time[200:] += 1.0e-7
    arrays = dict(base.arrays)
    arrays[TIMEBASE_KEY] = time
    nonuniform = assess_dbdt_authority(_artifact(complete=True, arrays=arrays, metadata=base.metadata))
    assert "dbdt_timebase_not_uniform" in cast(list[str], nonuniform["blockers"])


def test_multi_shot_report_is_deterministic_and_requires_unique_order() -> None:
    """Campaign evidence rejects empty, duplicate, or unsorted shot identity."""
    report = build_dbdt_authority_report([_artifact(shot_id=30421), _artifact(shot_id=30424)])
    assert report["status"] == "blocked"
    assert report["shot_count"] == 2
    assert report["payload_sha256"] == _digest(report)
    with pytest.raises(DbdtAuthorityError, match="at least one"):
        build_dbdt_authority_report([])
    with pytest.raises(DbdtAuthorityError, match="unique, ascending"):
        build_dbdt_authority_report([_artifact(shot_id=2), _artifact(shot_id=1)])
    with pytest.raises(DbdtAuthorityError, match="unique, ascending"):
        build_dbdt_authority_report([_artifact(shot_id=1), _artifact(shot_id=1)])


def test_cli_sorts_writes_exclusively_and_rejects_duplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command canonicalises shot order and never overwrites evidence."""
    seen: list[int] = []
    monkeypatch.setattr(dbdt_module, "load_verified_source_manifest", lambda *_args, **_kwargs: {})

    def _read(_manifest: Mapping[str, Any], *, artifact_root: Path, shot_id: int) -> VerifiedSourceArtifact:
        assert artifact_root == tmp_path
        seen.append(shot_id)
        return _artifact(shot_id=shot_id)

    monkeypatch.setattr(dbdt_module, "read_verified_npz_artifact", _read)
    output = tmp_path / "nested" / "report.json"
    args = [
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--artifact-root",
        str(tmp_path),
        "--shot-id",
        "30424",
        "--shot-id",
        "30421",
        "--json-out",
        str(output),
    ]
    assert main(args) == 0
    assert seen == [30421, 30424]
    assert json.loads(output.read_text(encoding="utf-8"))["shot_count"] == 2
    with pytest.raises(DbdtAuthorityError, match="refusing to overwrite"):
        main(args)
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--artifact-root",
                str(tmp_path),
                "--shot-id",
                "30421",
                "--shot-id",
                "30421",
                "--json-out",
                str(tmp_path / "duplicate.json"),
            ]
        )
    assert exc_info.value.code == 2


def test_cli_removes_partial_output_after_serialisation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A writer failure cannot leave an apparently valid report."""
    monkeypatch.setattr(dbdt_module, "load_verified_source_manifest", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(dbdt_module, "read_verified_npz_artifact", lambda *_args, **_kwargs: _artifact())

    def _fail_dump(*_args: object, **_kwargs: object) -> None:
        raise OSError("fixture serialisation failure")

    monkeypatch.setattr(json, "dump", _fail_dump)
    output = tmp_path / "report.json"
    with pytest.raises(OSError, match="fixture serialisation failure"):
        main(
            [
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--artifact-root",
                str(tmp_path),
                "--shot-id",
                "30421",
                "--json-out",
                str(output),
            ]
        )
    assert not output.exists()
