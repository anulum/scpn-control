# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST saddle-modal authority tests
"""Contract tests for the fail-closed MAST saddle-modal authority gate."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pytest

import validation.mast_saddle_modal_authority as modal_module
from validation.mast_saddle_modal_authority import (
    EXPECTED_CENTRES_DEG,
    EXPECTED_FIELD_CHANNELS,
    FIELD_KEY,
    GEOMETRY_KEYS,
    SADDLE_MODAL_AUTHORITY_SCHEMA,
    TIMEBASE_KEY,
    SaddleModalAuthorityError,
    assess_saddle_modal_authority,
    build_saddle_modal_authority_report,
    main,
    mast_saddle_modal_authority_spec,
)
from validation.mast_source_artifact_reader import VerifiedSourceArtifact
from validation.mast_source_object_manifest import canonical_json_sha256


def _geometry() -> np.ndarray[Any, np.dtype[np.float64]]:
    rows = []
    offsets = np.linspace(-10.0, 10.0, 28, dtype=np.float64)
    for centre in EXPECTED_CENTRES_DEG:
        rows.append(np.mod(centre + offsets, 360.0))
    return np.asarray(rows, dtype=np.float64)


def _metadata(
    *,
    dimensions: tuple[str, ...],
    units: str,
    attributes: Mapping[str, object] | None,
    value_sha256: str = "a" * 64,
    metadata_status: str = "source_xarray",
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "metadata_status": metadata_status,
            "dimensions": dimensions,
            "units": units,
            "source_attributes": None if attributes is None else MappingProxyType(dict(attributes)),
            "value_sha256": value_sha256,
        }
    )


def _field_attributes(*, complete: bool) -> dict[str, object]:
    attributes: dict[str, object] = {
        "name": "b_field_tor_probe_saddle_field",
        "uda_name": "ASM_SAD/M01",
        "label": "mT",
        "units": "T",
        "description": "",
    }
    if complete:
        attributes.update(
            {
                "source_channels": EXPECTED_FIELD_CHANNELS,
                "geometry_vertical_set": "middle",
                "geometry_value_sha256": "a" * 64,
                "row_join_evidence_sha256": "e" * 64,
                "calibration_evidence_sha256": "f" * 64,
                "authority_citation": "https://github.com/ukaea/fair-mast-ingestion/tree/authority",
                "standard_uncertainty_t": 1.0e-6,
                "baseline_policy": "subtract a source-attested pre-plasma baseline per row",
                "bad_channel_policy": "mask any row rejected by the source quality flag",
            }
        )
    return attributes


def _geometry_attributes(*, complete: bool) -> dict[str, object]:
    return {
        "status": "released" if complete else "development",
        "revision": 1 if complete else 0,
        "creatorCommitId": "b" * 40 if complete else "",
        "signedOffBy": "MAST Data Systems Team" if complete else "lkogan",
        "source": "https://git.ccfe.ac.uk/MAST-U/mast-geometry/sources/saddle.pdf",
        "units": "degrees" if complete else "SI, degrees, m",
    }


def _artifact(
    *,
    shot_id: int = 30421,
    complete: bool = False,
    arrays: Mapping[str, np.ndarray[Any, Any]] | None = None,
    metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> VerifiedSourceArtifact:
    geometry = _geometry()
    default_arrays: dict[str, np.ndarray[Any, Any]] = {
        FIELD_KEY: np.arange(48, dtype=np.float64).reshape(12, 4),
        TIMEBASE_KEY: np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float64),
        **{key: geometry.copy() for key in GEOMETRY_KEYS},
    }
    geometry_units = "degrees" if complete else "SI, degrees, m"
    default_metadata: dict[str, Mapping[str, Any]] = {
        FIELD_KEY: _metadata(
            dimensions=("b_field_tor_probe_saddle_field_channel", "time_saddle"),
            units="T",
            attributes=_field_attributes(complete=complete),
            value_sha256="c" * 64,
        ),
        TIMEBASE_KEY: _metadata(
            dimensions=("time_saddle",),
            units="s",
            attributes={"units": "s"},
            value_sha256="d" * 64,
        ),
    }
    for key, prefix in zip(GEOMETRY_KEYS, ("l", "m", "u"), strict=True):
        default_metadata[key] = _metadata(
            dimensions=(f"b_field_tor_probe_saddle_{prefix}_geometry_channel", "coordinate"),
            units=geometry_units,
            attributes=_geometry_attributes(complete=complete),
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
        manifest_sha256="1" * 64,
        artifact_sha256=str(shot_id % 10) * 64,
        parent_digest="3" * 64,
        transform_digest="4" * 64,
        arrays=MappingProxyType(active_arrays),
        metadata=MappingProxyType(active_metadata),
    )


def _digest(payload: Mapping[str, object]) -> str:
    copy = dict(payload)
    copy["payload_sha256"] = None
    return canonical_json_sha256(copy)


def test_spec_pins_mapping_channels_geometry_and_candidate_formula() -> None:
    """The contract fixes source identity while keeping the reduction blocked."""
    spec = mast_saddle_modal_authority_spec()

    assert spec["schema_version"] == SADDLE_MODAL_AUTHORITY_SCHEMA
    assert spec["expected_field_channels"] == list(EXPECTED_FIELD_CHANNELS)
    assert spec["expected_toroidal_centres_deg"] == list(EXPECTED_CENTRES_DEG)
    transform = cast(dict[str, object], spec["candidate_transform"])
    assert transform["orders"] == [1, 2]
    assert "inadmissible" in cast(str, transform["status"])
    assert spec["payload_sha256"] == _digest(spec)


def test_current_source_geometry_is_measured_but_not_promoted() -> None:
    """Regular polygons do not conceal missing row joins and release authority."""
    report = assess_saddle_modal_authority(_artifact())

    assert report["status"] == "blocked"
    assert report["canonical_modal_bindings_admissible"] is False
    assert report["modal_reduction_executed"] is False
    blockers = cast(list[str], report["blockers"])
    assert blockers == [
        "saddle_modal_source_metadata_mismatch",
        "saddle_field_row_identities_not_preserved",
        "saddle_field_geometry_vertical_set_not_attested",
        "saddle_field_row_join_evidence_absent",
        "saddle_field_calibration_evidence_absent",
        "saddle_field_authority_citation_absent",
        "saddle_field_standard_uncertainty_absent",
        "saddle_field_baseline_policy_absent",
        "saddle_field_bad_channel_policy_absent",
        "saddle_geometry_not_released",
        "saddle_geometry_creator_commit_absent",
    ]
    evidence = cast(dict[str, object], report["array_evidence"])
    centres = cast(dict[str, list[float]], evidence["geometry_centres_deg"])
    assert all(np.allclose(value, EXPECTED_CENTRES_DEG, rtol=0.0, atol=1.0e-9) for value in centres.values())
    assert report["payload_sha256"] == _digest(report)
    assert all(value is False for value in cast(dict[str, bool], report["claim_boundary"]).values())


def test_future_complete_source_metadata_can_admit_the_modal_join() -> None:
    """A released digest-bound row join makes the gate usable, not permanently false."""
    report = assess_saddle_modal_authority(_artifact(complete=True))

    assert report["status"] == "authority_verified"
    assert report["canonical_modal_bindings_admissible"] is True
    assert report["blockers"] == []


def test_missing_members_and_source_attributes_fail_closed() -> None:
    """Absent arrays or authority-bearing attributes are reported without inference."""
    base = _artifact(complete=True)
    arrays = {key: value for key, value in base.arrays.items() if key != GEOMETRY_KEYS[2]}
    metadata = {key: value for key, value in base.metadata.items() if key != GEOMETRY_KEYS[2]}
    missing = assess_saddle_modal_authority(_artifact(complete=True, arrays=arrays, metadata=metadata))
    assert missing["blockers"] == ["saddle_modal_source_members_absent"]

    metadata = dict(base.metadata)
    metadata[FIELD_KEY] = _metadata(
        dimensions=("b_field_tor_probe_saddle_field_channel", "time_saddle"),
        units="T",
        attributes=None,
    )
    metadata[GEOMETRY_KEYS[0]] = _metadata(
        dimensions=("b_field_tor_probe_saddle_l_geometry_channel", "coordinate"),
        units="degrees",
        attributes=None,
    )
    absent = assess_saddle_modal_authority(_artifact(complete=True, metadata=metadata))
    blockers = cast(list[str], absent["blockers"])
    assert "saddle_field_source_attributes_absent" in blockers
    assert "saddle_geometry_source_attributes_absent" in blockers


@pytest.mark.parametrize(
    ("replacement", "suffix"),
    [
        (
            _metadata(
                dimensions=("b_field_tor_probe_saddle_field_channel", "time_saddle"),
                units="T",
                attributes=_field_attributes(complete=True),
                metadata_status="values_only",
            ),
            "metadata_status",
        ),
        (
            _metadata(dimensions=("channel", "time"), units="T", attributes=_field_attributes(complete=True)),
            "dimensions",
        ),
        (
            _metadata(
                dimensions=("b_field_tor_probe_saddle_field_channel", "time_saddle"),
                units="mT",
                attributes=_field_attributes(complete=True),
            ),
            "units",
        ),
    ],
)
def test_source_metadata_drift_is_explicit(replacement: Mapping[str, Any], suffix: str) -> None:
    """Status, dimensions, and units cannot drift behind plausible values."""
    base = _artifact(complete=True)
    metadata = dict(base.metadata)
    metadata[FIELD_KEY] = replacement

    report = assess_saddle_modal_authority(_artifact(complete=True, metadata=metadata))

    mismatches = cast(dict[str, object], report["metadata_evidence"])["metadata_mismatches"]
    assert f"{FIELD_KEY}:{suffix}" in cast(list[str], mismatches)


def test_uncertainty_policies_and_geometry_identity_have_independent_guards() -> None:
    """Invalid uncertainty and selected-geometry identity cannot hide behind metadata."""
    base = _artifact(complete=True)
    metadata = dict(base.metadata)
    field_attributes = _field_attributes(complete=True)
    field_attributes["standard_uncertainty_t"] = -1.0
    metadata[FIELD_KEY] = _metadata(
        dimensions=("b_field_tor_probe_saddle_field_channel", "time_saddle"),
        units="T",
        attributes=field_attributes,
    )
    negative = assess_saddle_modal_authority(_artifact(complete=True, metadata=metadata))
    assert "saddle_field_standard_uncertainty_not_positive" in cast(list[str], negative["blockers"])

    metadata = dict(base.metadata)
    metadata[GEOMETRY_KEYS[2]] = _metadata(
        dimensions=("b_field_tor_probe_saddle_u_geometry_channel", "coordinate"),
        units="degrees",
        attributes=_geometry_attributes(complete=True),
        value_sha256="invalid",
    )
    mismatch = assess_saddle_modal_authority(_artifact(complete=True, metadata=metadata))
    assert "saddle_geometry_value_digest_absent" in cast(list[str], mismatch["blockers"])

    metadata = dict(base.metadata)
    field_attributes = _field_attributes(complete=True)
    field_attributes.update(
        {
            "geometry_value_sha256": "0" * 64,
            "row_join_evidence_sha256": "invalid",
            "calibration_evidence_sha256": None,
            "authority_citation": "https://example.com/not-primary",
        }
    )
    metadata[FIELD_KEY] = _metadata(
        dimensions=("b_field_tor_probe_saddle_field_channel", "time_saddle"),
        units="T",
        attributes=field_attributes,
    )
    unattested = assess_saddle_modal_authority(_artifact(complete=True, metadata=metadata))
    blockers = cast(list[str], unattested["blockers"])
    assert "saddle_field_geometry_value_identity_not_attested" in blockers
    assert "saddle_field_row_join_evidence_absent" in blockers
    assert "saddle_field_calibration_evidence_absent" in blockers
    assert "saddle_field_authority_citation_absent" in blockers


def test_distinct_unselected_vertical_geometry_does_not_block_selected_join() -> None:
    """A digest-bound selected set need not equal both unselected vertical sets."""
    base = _artifact(complete=True)
    arrays = dict(base.arrays)
    widened = arrays[GEOMETRY_KEYS[2]].copy()
    centres = np.asarray(EXPECTED_CENTRES_DEG, dtype=np.float64).reshape(-1, 1)
    widened = np.mod(centres + 1.1 * (widened - centres), 360.0)
    arrays[GEOMETRY_KEYS[2]] = widened
    metadata = dict(base.metadata)
    metadata[GEOMETRY_KEYS[2]] = _metadata(
        dimensions=("b_field_tor_probe_saddle_u_geometry_channel", "coordinate"),
        units="degrees",
        attributes=_geometry_attributes(complete=True),
        value_sha256="e" * 64,
    )

    report = assess_saddle_modal_authority(_artifact(complete=True, arrays=arrays, metadata=metadata))

    assert report["status"] == "authority_verified"
    evidence = cast(dict[str, object], report["metadata_evidence"])
    assert evidence["geometry_vertical_sets_share_value_digest"] is False
    array_evidence = cast(dict[str, object], report["array_evidence"])
    assert array_evidence["geometry_vertical_sets_have_identical_values"] is False


def test_authority_citation_allowlist_is_enforced_through_the_gate() -> None:
    """Only credential-free HTTPS primary surfaces can clear citation authority."""
    base = _artifact(complete=True)
    for citation, blocked in (
        ("http://github.com/ukaea/fair-mast-ingestion", True),
        ("https://user@github.com/ukaea/fair-mast-ingestion", True),
        ("https://mastapp.site/level2-data.html", False),
    ):
        metadata = dict(base.metadata)
        attributes = _field_attributes(complete=True)
        attributes["authority_citation"] = citation
        metadata[FIELD_KEY] = _metadata(
            dimensions=("b_field_tor_probe_saddle_field_channel", "time_saddle"),
            units="T",
            attributes=attributes,
        )
        report = assess_saddle_modal_authority(_artifact(complete=True, metadata=metadata))
        assert ("saddle_field_authority_citation_absent" in cast(list[str], report["blockers"])) is blocked


def test_shape_finite_time_and_geometry_branches_are_fail_closed() -> None:
    """Array structure and geometry calculations reject every invalid edge."""
    base = _artifact(complete=True)

    arrays = dict(base.arrays)
    arrays[FIELD_KEY] = np.ones((11, 4), dtype=np.float64)
    shape = assess_saddle_modal_authority(_artifact(complete=True, arrays=arrays))
    assert "saddle_field_or_timebase_shape_mismatch" in cast(list[str], shape["blockers"])

    arrays = dict(base.arrays)
    arrays[TIMEBASE_KEY] = np.asarray([[0.0, 0.1], [0.2, 0.3]], dtype=np.float64)
    time_shape = assess_saddle_modal_authority(_artifact(complete=True, arrays=arrays))
    assert "saddle_field_or_timebase_shape_mismatch" in cast(list[str], time_shape["blockers"])

    arrays = dict(base.arrays)
    arrays[GEOMETRY_KEYS[0]] = np.ones((12, 2), dtype=np.float64)
    geometry_shape = assess_saddle_modal_authority(_artifact(complete=True, arrays=arrays))
    assert "saddle_geometry_shape_mismatch" in cast(list[str], geometry_shape["blockers"])

    arrays = dict(base.arrays)
    arrays[TIMEBASE_KEY] = np.full(4, np.nan, dtype=np.float64)
    empty_time = assess_saddle_modal_authority(_artifact(complete=True, arrays=arrays))
    assert "saddle_timebase_has_no_finite_samples" in cast(list[str], empty_time["blockers"])

    arrays = dict(base.arrays)
    arrays[TIMEBASE_KEY] = np.asarray([0.0, np.nan, 0.2, 0.1], dtype=np.float64)
    bad_time = assess_saddle_modal_authority(_artifact(complete=True, arrays=arrays))
    assert "saddle_timebase_not_finite_and_strictly_increasing" in cast(list[str], bad_time["blockers"])

    arrays = dict(base.arrays)
    field = arrays[FIELD_KEY].copy()
    field[4] = np.nan
    arrays[FIELD_KEY] = field
    empty_row = assess_saddle_modal_authority(_artifact(complete=True, arrays=arrays))
    assert "saddle_field_row_has_no_finite_samples" in cast(list[str], empty_row["blockers"])
    assert cast(dict[str, object], empty_row["array_evidence"])["field_rows_without_finite_samples_one_based"] == [5]

    arrays = dict(base.arrays)
    undefined = arrays[GEOMETRY_KEYS[0]].copy()
    undefined[0] = np.tile(np.asarray([0.0, 90.0, 180.0, 270.0]), 7)
    arrays[GEOMETRY_KEYS[0]] = undefined
    no_centre = assess_saddle_modal_authority(_artifact(complete=True, arrays=arrays))
    assert "saddle_geometry_circular_centre_undefined" in cast(list[str], no_centre["blockers"])

    arrays = dict(base.arrays)
    shifted = arrays[GEOMETRY_KEYS[0]].copy()
    shifted[0] = np.mod(shifted[0] + 1.0, 360.0)
    arrays[GEOMETRY_KEYS[0]] = shifted
    different = assess_saddle_modal_authority(_artifact(complete=True, arrays=arrays))
    blockers = cast(list[str], different["blockers"])
    assert "saddle_geometry_centres_do_not_match_pinned_mapping_order" in blockers


def test_multi_shot_report_is_canonical_and_requires_unique_order() -> None:
    """Campaign evidence is deterministic and rejects empty or ambiguous identity."""
    report = build_saddle_modal_authority_report([_artifact(shot_id=30421), _artifact(shot_id=30424)])

    assert report["status"] == "blocked"
    assert report["shot_count"] == 2
    assert report["payload_sha256"] == _digest(report)
    with pytest.raises(SaddleModalAuthorityError, match="at least one"):
        build_saddle_modal_authority_report([])
    with pytest.raises(SaddleModalAuthorityError, match="unique, ascending"):
        build_saddle_modal_authority_report([_artifact(shot_id=2), _artifact(shot_id=1)])
    with pytest.raises(SaddleModalAuthorityError, match="unique, ascending"):
        build_saddle_modal_authority_report([_artifact(shot_id=1), _artifact(shot_id=1)])


def test_cli_sorts_writes_exclusively_and_rejects_duplicate_shots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command canonicalises shot order and never overwrites evidence."""
    seen: list[int] = []
    monkeypatch.setattr(modal_module, "load_verified_source_manifest", lambda *_args, **_kwargs: {})

    def _read(_manifest: Mapping[str, Any], *, artifact_root: Path, shot_id: int) -> VerifiedSourceArtifact:
        assert artifact_root == tmp_path
        seen.append(shot_id)
        return _artifact(shot_id=shot_id)

    monkeypatch.setattr(modal_module, "read_verified_npz_artifact", _read)
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
    with pytest.raises(SaddleModalAuthorityError, match="refusing to overwrite"):
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


def test_cli_removes_partial_report_after_serialisation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A handled writer failure cannot leave an apparently valid report."""
    monkeypatch.setattr(modal_module, "load_verified_source_manifest", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(modal_module, "read_verified_npz_artifact", lambda *_args, **_kwargs: _artifact())

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
