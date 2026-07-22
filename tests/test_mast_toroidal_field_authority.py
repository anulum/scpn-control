# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST toroidal-field authority tests
"""Real-contract tests for the fail-closed MAST toroidal-field gate."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pytest

import validation.mast_toroidal_field_authority as authority_module
from validation.mast_source_artifact_reader import VerifiedSourceArtifact
from validation.mast_source_object_manifest import canonical_json_sha256
from validation.mast_toroidal_field_authority import (
    DIRECT_FIELD_KEY,
    REFERENCE_RADIUS_KEY,
    TIMEBASE_KEY,
    TOROIDAL_FIELD_AUTHORITY_SCHEMA,
    VACUUM_CANDIDATE_KEY,
    DirectFieldAuthority,
    ToroidalFieldAuthorityError,
    assess_toroidal_field_authority,
    build_toroidal_field_authority_report,
    main,
    mast_toroidal_field_authority_spec,
)


def _metadata(
    *,
    units: str,
    name: str | None = None,
    uda_name: str | None = None,
    dimensions: tuple[str, ...] = ("time",),
    metadata_status: str = "source_xarray",
) -> Mapping[str, Any]:
    attributes = {} if name is None else {"name": name, "uda_name": uda_name, "units": units}
    return MappingProxyType(
        {
            "metadata_status": metadata_status,
            "dimensions": dimensions,
            "units": units,
            "source_attributes": MappingProxyType(attributes),
        }
    )


def _artifact(
    *,
    shot_id: int = 30421,
    arrays: Mapping[str, np.ndarray[Any, Any]] | None = None,
    metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> VerifiedSourceArtifact:
    default_arrays: dict[str, np.ndarray[Any, Any]] = {
        TIMEBASE_KEY: np.asarray([0.0, 0.1, 0.2], dtype=np.float64),
        DIRECT_FIELD_KEY: np.asarray([np.nan, -0.62, -0.60], dtype=np.float64),
        VACUUM_CANDIDATE_KEY: np.asarray([np.nan, -0.57, -0.55], dtype=np.float64),
        REFERENCE_RADIUS_KEY: np.asarray([np.nan, 0.82, 0.84], dtype=np.float64),
    }
    default_metadata: dict[str, Mapping[str, Any]] = {
        TIMEBASE_KEY: _metadata(units="s"),
        DIRECT_FIELD_KEY: _metadata(units="T", name="bphi_rmag", uda_name="EFM_BPHI_RMAG"),
        VACUUM_CANDIDATE_KEY: _metadata(units="T", name="bvac_rmag", uda_name="EFM_BVAC_RMAG"),
        REFERENCE_RADIUS_KEY: _metadata(
            units="m",
            name="magnetic_axis_r",
            uda_name="EFM_MAGNETIC_AXIS_R",
        ),
    }
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


def _constant_authority() -> DirectFieldAuthority:
    return DirectFieldAuthority(
        positive_direction="positive is anti-clockwise when viewed from above",
        positive_direction_citation="https://mastapp.site/mast-sign",
        positive_direction_evidence_sha256="a" * 64,
        quantity_citation="https://github.com/ukaea/fair-mast-ingestion/blob/pin/mappings/level2/mast.yml",
        quantity_evidence_sha256="b" * 64,
        uncertainty_kind="constant_standard_uncertainty",
        uncertainty_citation="https://www.ukaea.uk/mast-uncertainty",
        uncertainty_evidence_sha256="c" * 64,
        constant_standard_uncertainty_t=0.01,
    )


def _payload_digest(payload: Mapping[str, object]) -> str:
    digest_payload = dict(payload)
    digest_payload["payload_sha256"] = None
    return canonical_json_sha256(digest_payload)


def test_spec_is_self_digested_and_separates_total_from_vacuum_routes() -> None:
    """The contract names distinct direct, vacuum-candidate, and derivation routes."""
    spec = mast_toroidal_field_authority_spec()

    assert spec["schema_version"] == TOROIDAL_FIELD_AUTHORITY_SCHEMA
    assert spec["payload_sha256"] == _payload_digest(spec)
    direct = cast(dict[str, object], spec["direct_total_field_route"])
    vacuum = cast(dict[str, object], spec["vacuum_candidate"])
    derivation = cast(dict[str, object], spec["tf_current_derivation_route"])
    assert direct["field_key"] == DIRECT_FIELD_KEY
    assert direct["reference_radius_key"] == REFERENCE_RADIUS_KEY
    assert vacuum["field_key"] == VACUUM_CANDIDATE_KEY
    assert "inadmissible" in cast(str, vacuum["status"])
    assert "K_TF" in cast(str, derivation["formula"])
    assert "zero uncertainty" in cast(list[str], derivation["prohibited"])


def test_realistic_source_without_external_authority_stays_blocked() -> None:
    """Array presence and observed sign cannot replace physical authority."""
    report = assess_toroidal_field_authority(_artifact())

    assert report["status"] == "blocked"
    assert report["canonical_bt_binding_admissible"] is False
    assert report["blockers"] == [
        "total_versus_vacuum_quantity_authority_absent",
        "physical_positive_direction_authority_absent",
        "one_standard_deviation_uncertainty_authority_absent",
    ]
    route = cast(dict[str, object], report["direct_total_field_route"])
    comparison = cast(dict[str, object], route["vacuum_candidate_comparison"])
    assert comparison["byte_or_value_equal_with_nan"] is False
    assert comparison["maximum_absolute_difference_t"] == pytest.approx(0.05)
    assert route["observed_field_sign_counts_not_authority"] == {"negative": 2, "zero": 0, "positive": 0}
    assert report["payload_sha256"] == _payload_digest(report)
    assert all(value is False for value in cast(dict[str, bool], report["claim_boundary"]).values())


def test_complete_constant_uncertainty_authority_clears_direct_route() -> None:
    """A complete sourced declaration admits a constant-uncertainty direct route."""
    report = assess_toroidal_field_authority(_artifact(), authority=_constant_authority())

    assert report["status"] == "authority_verified"
    assert report["canonical_bt_binding_admissible"] is True
    assert report["blockers"] == []
    uncertainty = cast(dict[str, object], report["uncertainty_evidence"])
    assert uncertainty == {"kind": "constant_standard_uncertainty", "constant_standard_uncertainty_t": 0.01}


def test_complete_per_sample_uncertainty_authority_clears_direct_route() -> None:
    """A verified per-sample uncertainty array can complete the direct route."""
    key = "equilibrium.bphi_rmag_standard_uncertainty"
    base = _artifact()
    arrays = dict(base.arrays)
    metadata = dict(base.metadata)
    arrays[key] = np.asarray([0.02, 0.02, 0.03], dtype=np.float64)
    metadata[key] = _metadata(units="T", name="bphi_rmag_standard_uncertainty", uda_name="AUTH_BPHI_SIGMA")
    authority = replace(
        _constant_authority(),
        uncertainty_kind="per_sample_standard_uncertainty",
        constant_standard_uncertainty_t=None,
        uncertainty_key=key,
    )

    report = assess_toroidal_field_authority(_artifact(arrays=arrays, metadata=metadata), authority=authority)

    assert report["canonical_bt_binding_admissible"] is True
    assert cast(dict[str, object], report["uncertainty_evidence"])["source_key"] == key


def test_assessment_rejects_a_corrupted_frozen_authority_object() -> None:
    """A caller bypassing dataclass construction still cannot remove the uncertainty key."""
    authority = replace(
        _constant_authority(),
        uncertainty_kind="per_sample_standard_uncertainty",
        constant_standard_uncertainty_t=None,
        uncertainty_key="equilibrium.bphi_sigma",
    )
    object.__setattr__(authority, "uncertainty_key", None)

    with pytest.raises(ToroidalFieldAuthorityError, match="lost its validated uncertainty key"):
        assess_toroidal_field_authority(_artifact(), authority=authority)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"positive_direction": ""}, "must be non-empty"),
        ({"positive_direction_citation": "ftp://invalid"}, "trusted primary-source HTTPS"),
        ({"quantity_citation": "https://github.com/untrusted/project"}, "trusted primary-source HTTPS"),
        ({"uncertainty_evidence_sha256": "A" * 64}, "lowercase SHA-256"),
        ({"constant_standard_uncertainty_t": 0.0}, "finite and positive"),
        ({"constant_standard_uncertainty_t": float("nan")}, "finite and positive"),
        ({"uncertainty_key": "sigma"}, "cannot also name"),
        (
            {
                "uncertainty_kind": "per_sample_standard_uncertainty",
                "constant_standard_uncertainty_t": None,
                "uncertainty_key": None,
            },
            "requires an exact source key",
        ),
        (
            {
                "uncertainty_kind": "per_sample_standard_uncertainty",
                "constant_standard_uncertainty_t": 0.1,
                "uncertainty_key": "sigma",
            },
            "cannot also provide a constant",
        ),
        ({"uncertainty_kind": "unknown"}, "unsupported uncertainty kind"),
    ],
)
def test_authority_declaration_rejects_inconsistent_fields(
    overrides: Mapping[str, object],
    message: str,
) -> None:
    """Malformed authority fails before it can influence an assessment."""
    with pytest.raises(ToroidalFieldAuthorityError, match=message):
        replace(_constant_authority(), **cast(Any, overrides))


def test_missing_direct_members_are_exact_blockers() -> None:
    """Missing total-field or reference-radius members remain explicit."""
    base = _artifact()
    arrays = {key: value for key, value in base.arrays.items() if key not in {DIRECT_FIELD_KEY, REFERENCE_RADIUS_KEY}}
    metadata = {key: value for key, value in base.metadata.items() if key in arrays}

    report = assess_toroidal_field_authority(_artifact(arrays=arrays, metadata=metadata))

    assert cast(list[str], report["blockers"])[:1] == ["direct_route_source_members_absent"]
    route = cast(dict[str, object], report["direct_total_field_route"])
    assert route["missing_source_keys"] == [DIRECT_FIELD_KEY, REFERENCE_RADIUS_KEY]


def test_vacuum_candidate_is_optional_evidence_not_a_direct_route_requirement() -> None:
    """Omitting the vacuum comparator does not weaken direct-route source checks."""
    base = _artifact()
    arrays = {key: value for key, value in base.arrays.items() if key != VACUUM_CANDIDATE_KEY}
    metadata = {key: value for key, value in base.metadata.items() if key in arrays}

    report = assess_toroidal_field_authority(
        _artifact(arrays=arrays, metadata=metadata),
        authority=_constant_authority(),
    )

    route = cast(dict[str, object], report["direct_total_field_route"])
    assert "vacuum_candidate_comparison" not in route
    assert report["canonical_bt_binding_admissible"] is True


@pytest.mark.parametrize(
    ("key", "replacement", "suffix"),
    [
        (
            DIRECT_FIELD_KEY,
            _metadata(units="T", name="bphi_rmag", uda_name="EFM_BPHI_RMAG", metadata_status="values_only"),
            "metadata_status",
        ),
        (
            DIRECT_FIELD_KEY,
            _metadata(units="T", name="bphi_rmag", uda_name="EFM_BPHI_RMAG", dimensions=("sample",)),
            "dimensions",
        ),
        (DIRECT_FIELD_KEY, _metadata(units="G", name="bphi_rmag", uda_name="EFM_BPHI_RMAG"), "units"),
        (DIRECT_FIELD_KEY, _metadata(units="T", name="wrong", uda_name="EFM_BPHI_RMAG"), "name"),
        (DIRECT_FIELD_KEY, _metadata(units="T", name="bphi_rmag", uda_name="wrong"), "uda_name"),
        (
            REFERENCE_RADIUS_KEY,
            MappingProxyType(
                {"metadata_status": "source_xarray", "dimensions": ("time",), "units": "m", "source_attributes": None}
            ),
            "source_attributes",
        ),
    ],
)
def test_source_metadata_drift_never_clears(
    key: str,
    replacement: Mapping[str, Any],
    suffix: str,
) -> None:
    """Every required source-metadata field is checked exactly."""
    base = _artifact()
    metadata = dict(base.metadata)
    metadata[key] = replacement

    report = assess_toroidal_field_authority(_artifact(metadata=metadata), authority=_constant_authority())

    assert "direct_route_source_metadata_mismatch" in cast(list[str], report["blockers"])
    mismatches = cast(dict[str, object], report["direct_total_field_route"])["metadata_mismatches"]
    assert f"{key}:{suffix}" in cast(list[str], mismatches)


def test_shape_no_coverage_and_time_order_fail_closed() -> None:
    """Shape, finite coverage, and time ordering are independent blockers."""
    base = _artifact()
    arrays = dict(base.arrays)
    arrays[REFERENCE_RADIUS_KEY] = np.asarray([0.8, 0.9], dtype=np.float64)
    shape_report = assess_toroidal_field_authority(_artifact(arrays=arrays), authority=_constant_authority())
    assert "direct_route_source_shape_mismatch" in cast(list[str], shape_report["blockers"])

    arrays = dict(base.arrays)
    arrays[DIRECT_FIELD_KEY] = np.asarray([np.nan, np.nan, np.nan], dtype=np.float64)
    empty_report = assess_toroidal_field_authority(_artifact(arrays=arrays), authority=_constant_authority())
    assert "direct_route_has_no_joint_finite_positive_radius_samples" in cast(list[str], empty_report["blockers"])

    arrays = dict(base.arrays)
    arrays[TIMEBASE_KEY] = np.asarray([0.0, 0.2, 0.1], dtype=np.float64)
    order_report = assess_toroidal_field_authority(_artifact(arrays=arrays), authority=_constant_authority())
    assert "direct_route_timebase_not_strictly_increasing" in cast(list[str], order_report["blockers"])


@pytest.mark.parametrize(
    ("metadata", "values", "blocker"),
    [
        (None, None, "declared_uncertainty_source_member_absent"),
        (_metadata(units="G"), np.asarray([0.1, 0.1, 0.1]), "declared_uncertainty_source_metadata_mismatch"),
        (_metadata(units="T"), np.asarray([0.1, 0.1]), "declared_uncertainty_source_shape_mismatch"),
        (
            _metadata(units="T"),
            np.asarray([0.1, -0.1, np.nan]),
            "declared_uncertainty_is_missing_negative_or_non_finite",
        ),
    ],
)
def test_per_sample_uncertainty_failures_are_distinct(
    metadata: Mapping[str, Any] | None,
    values: np.ndarray[Any, Any] | None,
    blocker: str,
) -> None:
    """Each malformed uncertainty surface emits an actionable blocker."""
    key = "equilibrium.bphi_sigma"
    base = _artifact()
    arrays = dict(base.arrays)
    descriptions = dict(base.metadata)
    if values is not None:
        arrays[key] = values.astype(np.float64)
    if metadata is not None:
        descriptions[key] = metadata
    authority = replace(
        _constant_authority(),
        uncertainty_kind="per_sample_standard_uncertainty",
        constant_standard_uncertainty_t=None,
        uncertainty_key=key,
    )

    report = assess_toroidal_field_authority(
        _artifact(arrays=arrays, metadata=descriptions),
        authority=authority,
    )

    assert blocker in cast(list[str], report["blockers"])


def test_multi_shot_report_is_ordered_self_digested_and_rejects_bad_sets() -> None:
    """Campaign reports require a canonical, non-empty shot set."""
    report = build_toroidal_field_authority_report([_artifact(shot_id=30421), _artifact(shot_id=30424)])

    assert report["status"] == "blocked"
    assert report["shot_count"] == 2
    assert [shot["shot_id"] for shot in cast(list[dict[str, object]], report["shots"])] == [30421, 30424]
    assert report["payload_sha256"] == _payload_digest(report)
    with pytest.raises(ToroidalFieldAuthorityError, match="at least one"):
        build_toroidal_field_authority_report([])
    with pytest.raises(ToroidalFieldAuthorityError, match="unique, ascending"):
        build_toroidal_field_authority_report([_artifact(shot_id=2), _artifact(shot_id=1)])
    with pytest.raises(ToroidalFieldAuthorityError, match="unique, ascending"):
        build_toroidal_field_authority_report([_artifact(shot_id=1), _artifact(shot_id=1)])


def test_cli_sorts_shots_and_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI resolves unique shots canonically and writes the assessed payload."""
    seen: list[int] = []
    monkeypatch.setattr(authority_module, "load_verified_source_manifest", lambda *_args, **_kwargs: {})

    def _read(_manifest: Mapping[str, Any], *, artifact_root: Path, shot_id: int) -> VerifiedSourceArtifact:
        assert artifact_root == tmp_path
        seen.append(shot_id)
        return _artifact(shot_id=shot_id)

    monkeypatch.setattr(authority_module, "read_verified_npz_artifact", _read)
    output = tmp_path / "nested" / "report.json"

    result = main(
        [
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
    )

    assert result == 0
    assert seen == [30421, 30424]
    assert json.loads(output.read_text(encoding="utf-8"))["shot_count"] == 2
    with pytest.raises(ToroidalFieldAuthorityError, match="refusing to overwrite"):
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
                str(output),
            ]
        )
    assert exc_info.value.code == 2


def test_cli_removes_partial_report_after_serialisation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A handled writer failure cannot leave an apparently valid evidence file."""
    monkeypatch.setattr(authority_module, "load_verified_source_manifest", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        authority_module,
        "read_verified_npz_artifact",
        lambda *_args, **_kwargs: _artifact(),
    )

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
