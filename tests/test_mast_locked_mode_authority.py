# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST locked-mode authority tests
"""Contract tests for the fail-closed MAST locked-mode authority gate."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pytest

import validation.mast_locked_mode_authority as locked_module
from validation.mast_locked_mode_authority import (
    LEGACY_WINDOW_SAMPLES,
    LOCKED_MODE_AUTHORITY_SCHEMA,
    LockedModeAuthorityError,
    assess_locked_mode_authority,
    build_locked_mode_authority_report,
    main,
    mast_locked_mode_authority_spec,
)
from validation.mast_saddle_modal_authority import (
    EXPECTED_CENTRES_DEG,
    EXPECTED_FIELD_CHANNELS,
    FAIR_MAST_MAPPING_URL,
    FIELD_KEY,
    GEOMETRY_KEYS,
    TIMEBASE_KEY,
)
from validation.mast_source_artifact_reader import VerifiedSourceArtifact
from validation.mast_source_object_manifest import canonical_json_sha256


def _geometry() -> np.ndarray[Any, np.dtype[np.float64]]:
    """Build twelve regular toroidal polygons with the pinned centres."""
    offsets = np.linspace(-10.0, 10.0, 28, dtype=np.float64)
    return np.asarray([np.mod(centre + offsets, 360.0) for centre in EXPECTED_CENTRES_DEG], dtype=np.float64)


def _metadata(
    *,
    dimensions: tuple[str, ...],
    units: str,
    attributes: Mapping[str, object] | None,
    value_sha256: str,
) -> Mapping[str, Any]:
    """Build immutable source metadata matching the verified reader surface."""
    return MappingProxyType(
        {
            "metadata_status": "source_xarray",
            "dimensions": dimensions,
            "units": units,
            "source_attributes": None if attributes is None else MappingProxyType(dict(attributes)),
            "value_sha256": value_sha256,
        }
    )


def _field_attributes(*, saddle_complete: bool, locked_complete: bool) -> dict[str, object]:
    """Build field metadata with independently selectable authority layers."""
    attributes: dict[str, object] = {
        "name": "b_field_tor_probe_saddle_field",
        "uda_name": "ASM_SAD/M01",
        "label": "mT",
        "units": "T",
        "description": "",
    }
    if saddle_complete:
        attributes.update(
            {
                "source_channels": EXPECTED_FIELD_CHANNELS,
                "geometry_vertical_set": "middle",
                "geometry_value_sha256": "a" * 64,
                "row_join_evidence_sha256": "b" * 64,
                "calibration_evidence_sha256": "c" * 64,
                "authority_citation": FAIR_MAST_MAPPING_URL,
                "standard_uncertainty_t": 1.0e-6,
                "baseline_policy": "subtract the source-attested pre-plasma baseline per row",
                "bad_channel_policy": "mask every source-rejected row",
            }
        )
    if locked_complete:
        attributes.update(
            {
                "measured_field_component": "radial",
                "probe_location": "outer_midplane",
                "locked_mode_estimator": "stationary_n1_radial_field_perturbation",
                "locked_mode_toroidal_order": 1,
                "locked_mode_stationary_frame": "machine",
                "locked_mode_low_pass_cutoff_hz": 25.0,
                "locked_mode_filter_policy": "zero-phase source-approved low-pass in physical time",
                "locked_mode_edge_policy": "reflect source-valid samples without crossing finite gaps",
                "locked_mode_background_policy": "remove the source-attested no-plasma radial background",
                "locked_mode_pickup_correction_policy": "subtract the source-attested PF-coil pickup model",
                "locked_mode_vessel_response_policy": "apply the source-attested vessel transfer response",
                "locked_mode_standard_uncertainty_t": 2.0e-6,
                "locked_mode_estimator_evidence_sha256": "d" * 64,
                "locked_mode_authority_citation": "https://doi.org/10.1088/0741-3335/56/10/104003",
            }
        )
    return attributes


def _geometry_attributes(*, complete: bool) -> dict[str, object]:
    """Build released or current-development geometry attributes."""
    return {
        "status": "released" if complete else "development",
        "revision": 1 if complete else 0,
        "creatorCommitId": "e" * 40 if complete else "",
        "signedOffBy": "MAST Data Systems Team" if complete else "lkogan",
        "source": "https://git.ccfe.ac.uk/MAST-U/mast-geometry/sources/saddle.pdf",
    }


def _artifact(
    *,
    shot_id: int = 30421,
    saddle_complete: bool = False,
    locked_complete: bool = False,
    arrays: Mapping[str, np.ndarray[Any, Any]] | None = None,
    metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> VerifiedSourceArtifact:
    """Build one immutable verified source fixture."""
    geometry = _geometry()
    time = np.arange(300, dtype=np.float64) * 2.0e-5
    default_arrays: dict[str, np.ndarray[Any, Any]] = {
        FIELD_KEY: np.arange(3600, dtype=np.float64).reshape(12, 300),
        TIMEBASE_KEY: time,
        **{key: geometry.copy() for key in GEOMETRY_KEYS},
    }
    default_metadata: dict[str, Mapping[str, Any]] = {
        FIELD_KEY: _metadata(
            dimensions=("b_field_tor_probe_saddle_field_channel", "time_saddle"),
            units="T",
            attributes=_field_attributes(saddle_complete=saddle_complete, locked_complete=locked_complete),
            value_sha256="f" * 64,
        ),
        TIMEBASE_KEY: _metadata(
            dimensions=("time_saddle",),
            units="s",
            attributes={"units": "s"},
            value_sha256="1" * 64,
        ),
    }
    for key, prefix in zip(GEOMETRY_KEYS, ("l", "m", "u"), strict=True):
        default_metadata[key] = _metadata(
            dimensions=(f"b_field_tor_probe_saddle_{prefix}_geometry_channel", "coordinate"),
            units="degrees" if saddle_complete else "SI, degrees, m",
            attributes=_geometry_attributes(complete=saddle_complete),
            value_sha256="a" * 64,
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


def _digest(payload: Mapping[str, object]) -> str:
    """Recompute one report's canonical self-digest."""
    copy = dict(payload)
    copy["payload_sha256"] = None
    return canonical_json_sha256(copy)


def test_spec_pins_primary_observation_and_marks_legacy_window_candidate_only() -> None:
    """The contract distinguishes MAST evidence from the historical filter."""
    spec = mast_locked_mode_authority_spec()

    assert spec["schema_version"] == LOCKED_MODE_AUTHORITY_SCHEMA
    observation = cast(dict[str, object], spec["primary_source_observation"])
    assert observation["measured_component"] == "radial"
    assert observation["toroidal_order"] == 1
    candidate = cast(dict[str, object], spec["legacy_candidate"])
    assert candidate["window_samples"] == LEGACY_WINDOW_SAMPLES
    assert "not_source_authorised" in cast(str, candidate["status"])
    assert spec["payload_sha256"] == _digest(spec)


def test_current_source_is_measured_but_locked_mode_remains_blocked() -> None:
    """A regular saddle array cannot authorise the locked-mode estimator."""
    report = assess_locked_mode_authority(_artifact())

    assert report["status"] == "blocked"
    assert report["canonical_locked_mode_binding_admissible"] is False
    assert report["locked_mode_estimator_executed"] is False
    blockers = cast(list[str], report["blockers"])
    assert "saddle_field_row_identities_not_preserved" in blockers
    assert "locked_mode_measured_component_not_attested" in blockers
    assert "locked_mode_filter_policy_absent" in blockers
    assert "locked_mode_standard_uncertainty_absent" in blockers
    candidate = cast(dict[str, object], report["legacy_candidate_evidence"])
    assert candidate["sample_rate_hz"] == pytest.approx(50_000.0)
    assert candidate["legacy_window_endpoint_span_s"] == pytest.approx(0.004)
    assert candidate["legacy_window_support_s"] == pytest.approx(0.00402)
    assert candidate["legacy_boxcar_first_null_hz"] == pytest.approx(50_000.0 / 201.0)
    assert candidate["legacy_estimator_executed"] is False
    assert report["payload_sha256"] == _digest(report)
    assert all(value is False for value in cast(dict[str, bool], report["claim_boundary"]).values())


def test_complete_saddle_and_locked_metadata_can_admit_the_source_binding() -> None:
    """A fully attested stationary-n1 contract is not permanently blocked."""
    report = assess_locked_mode_authority(_artifact(saddle_complete=True, locked_complete=True))

    assert report["status"] == "authority_verified"
    assert report["canonical_locked_mode_binding_admissible"] is True
    assert report["saddle_modal_authority_admissible"] is True
    assert report["blockers"] == []


def test_missing_field_metadata_and_attributes_fail_closed() -> None:
    """Absent authority-bearing metadata is explicit rather than inferred."""
    base = _artifact(saddle_complete=True, locked_complete=True)
    metadata = {key: value for key, value in base.metadata.items() if key != FIELD_KEY}
    missing = assess_locked_mode_authority(
        _artifact(saddle_complete=True, locked_complete=True, arrays=base.arrays, metadata=metadata)
    )
    assert "locked_mode_field_metadata_absent" in cast(list[str], missing["blockers"])

    metadata = dict(base.metadata)
    metadata[FIELD_KEY] = _metadata(
        dimensions=("b_field_tor_probe_saddle_field_channel", "time_saddle"),
        units="T",
        attributes=None,
        value_sha256="f" * 64,
    )
    absent = assess_locked_mode_authority(
        _artifact(saddle_complete=True, locked_complete=True, arrays=base.arrays, metadata=metadata)
    )
    assert "locked_mode_field_source_attributes_absent" in cast(list[str], absent["blockers"])


@pytest.mark.parametrize(
    ("key", "replacement", "blocker"),
    [
        ("measured_field_component", "toroidal", "locked_mode_measured_component_not_attested"),
        ("probe_location", "unknown", "locked_mode_probe_location_not_attested"),
        ("locked_mode_estimator", "boxcar", "locked_mode_estimator_not_attested"),
        ("locked_mode_toroidal_order", 2, "locked_mode_toroidal_order_not_attested"),
        ("locked_mode_stationary_frame", "probe", "locked_mode_stationary_frame_not_attested"),
    ],
)
def test_locked_mode_identity_declarations_have_independent_guards(
    key: str,
    replacement: object,
    blocker: str,
) -> None:
    """Component, location, estimator, order, and frame cannot drift silently."""
    base = _artifact(saddle_complete=True, locked_complete=True)
    metadata = dict(base.metadata)
    attributes = _field_attributes(saddle_complete=True, locked_complete=True)
    attributes[key] = replacement
    metadata[FIELD_KEY] = _metadata(
        dimensions=("b_field_tor_probe_saddle_field_channel", "time_saddle"),
        units="T",
        attributes=attributes,
        value_sha256="f" * 64,
    )

    report = assess_locked_mode_authority(
        _artifact(saddle_complete=True, locked_complete=True, arrays=base.arrays, metadata=metadata)
    )

    assert blocker in cast(list[str], report["blockers"])


@pytest.mark.parametrize("bad_number", [None, True, 0.0, -1.0, float("inf")])
def test_cutoff_and_uncertainty_require_positive_finite_numbers(bad_number: object) -> None:
    """Boolean, absent, non-positive, and non-finite numeric claims fail closed."""
    base = _artifact(saddle_complete=True, locked_complete=True)
    metadata = dict(base.metadata)
    attributes = _field_attributes(saddle_complete=True, locked_complete=True)
    attributes["locked_mode_low_pass_cutoff_hz"] = bad_number
    attributes["locked_mode_standard_uncertainty_t"] = bad_number
    metadata[FIELD_KEY] = _metadata(
        dimensions=("b_field_tor_probe_saddle_field_channel", "time_saddle"),
        units="T",
        attributes=attributes,
        value_sha256="f" * 64,
    )

    report = assess_locked_mode_authority(
        _artifact(saddle_complete=True, locked_complete=True, arrays=base.arrays, metadata=metadata)
    )

    blockers = cast(list[str], report["blockers"])
    assert "locked_mode_low_pass_cutoff_not_attested" in blockers
    assert "locked_mode_standard_uncertainty_absent" in blockers


def test_policy_digest_and_primary_citation_require_independent_authority() -> None:
    """Text placeholders, invalid digests, and alternate citations cannot clear."""
    base = _artifact(saddle_complete=True, locked_complete=True)
    metadata = dict(base.metadata)
    attributes = _field_attributes(saddle_complete=True, locked_complete=True)
    for key in (
        "locked_mode_filter_policy",
        "locked_mode_edge_policy",
        "locked_mode_background_policy",
        "locked_mode_pickup_correction_policy",
        "locked_mode_vessel_response_policy",
    ):
        attributes[key] = "  "
    attributes["locked_mode_estimator_evidence_sha256"] = "invalid"
    attributes["locked_mode_authority_citation"] = "https://example.com/secondary"
    metadata[FIELD_KEY] = _metadata(
        dimensions=("b_field_tor_probe_saddle_field_channel", "time_saddle"),
        units="T",
        attributes=attributes,
        value_sha256="f" * 64,
    )

    report = assess_locked_mode_authority(
        _artifact(saddle_complete=True, locked_complete=True, arrays=base.arrays, metadata=metadata)
    )

    blockers = cast(list[str], report["blockers"])
    assert "locked_mode_filter_policy_absent" in blockers
    assert "locked_mode_edge_policy_absent" in blockers
    assert "locked_mode_background_policy_absent" in blockers
    assert "locked_mode_pickup_correction_policy_absent" in blockers
    assert "locked_mode_vessel_response_policy_absent" in blockers
    assert "locked_mode_estimator_evidence_absent" in blockers
    assert "locked_mode_primary_source_citation_absent" in blockers


def test_cutoff_must_remain_below_the_observed_source_nyquist() -> None:
    """A positive cutoff at or above Nyquist cannot authorise the estimator."""
    base = _artifact(saddle_complete=True, locked_complete=True)
    metadata = dict(base.metadata)
    attributes = _field_attributes(saddle_complete=True, locked_complete=True)
    attributes["locked_mode_low_pass_cutoff_hz"] = 25_000.0
    metadata[FIELD_KEY] = _metadata(
        dimensions=("b_field_tor_probe_saddle_field_channel", "time_saddle"),
        units="T",
        attributes=attributes,
        value_sha256="f" * 64,
    )

    report = assess_locked_mode_authority(
        _artifact(saddle_complete=True, locked_complete=True, arrays=base.arrays, metadata=metadata)
    )

    assert "locked_mode_low_pass_cutoff_not_below_nyquist" in cast(list[str], report["blockers"])


def test_timebase_absence_shape_order_window_and_uniformity_are_measured() -> None:
    """Every timebase edge either blocks or is explicitly recorded."""
    base = _artifact(saddle_complete=True, locked_complete=True)
    arrays = {key: value for key, value in base.arrays.items() if key != TIMEBASE_KEY}
    absent = assess_locked_mode_authority(
        _artifact(saddle_complete=True, locked_complete=True, arrays=arrays, metadata=base.metadata)
    )
    assert "locked_mode_timebase_absent" in cast(list[str], absent["blockers"])

    arrays = dict(base.arrays)
    arrays[TIMEBASE_KEY] = np.asarray([[0.0, 1.0]], dtype=np.float64)
    shape = assess_locked_mode_authority(
        _artifact(saddle_complete=True, locked_complete=True, arrays=arrays, metadata=base.metadata)
    )
    assert "locked_mode_timebase_shape_mismatch" in cast(list[str], shape["blockers"])

    arrays = dict(base.arrays)
    arrays[TIMEBASE_KEY] = np.asarray([0.0, np.nan, 0.1], dtype=np.float64)
    invalid = assess_locked_mode_authority(
        _artifact(saddle_complete=True, locked_complete=True, arrays=arrays, metadata=base.metadata)
    )
    assert "locked_mode_timebase_not_finite_and_strictly_increasing" in cast(list[str], invalid["blockers"])

    arrays = dict(base.arrays)
    arrays[TIMEBASE_KEY] = np.arange(100, dtype=np.float64) * 2.0e-5
    short = assess_locked_mode_authority(
        _artifact(saddle_complete=True, locked_complete=True, arrays=arrays, metadata=base.metadata)
    )
    assert "locked_mode_legacy_window_exceeds_source_samples" in cast(list[str], short["blockers"])

    arrays = dict(base.arrays)
    nonuniform = np.arange(300, dtype=np.float64) * 2.0e-5
    nonuniform[200:] += 1.0e-6
    arrays[TIMEBASE_KEY] = nonuniform
    measured = assess_locked_mode_authority(
        _artifact(saddle_complete=True, locked_complete=True, arrays=arrays, metadata=base.metadata)
    )
    evidence = cast(dict[str, object], measured["legacy_candidate_evidence"])
    assert evidence["timebase_uniform"] is False
    assert evidence["legacy_boxcar_first_null_hz"] is None


def test_multi_shot_report_is_canonical_and_requires_unique_order() -> None:
    """Campaign evidence is deterministic and rejects empty or ambiguous identity."""
    report = build_locked_mode_authority_report([_artifact(shot_id=30421), _artifact(shot_id=30424)])

    assert report["status"] == "blocked"
    assert report["shot_count"] == 2
    assert report["payload_sha256"] == _digest(report)
    with pytest.raises(LockedModeAuthorityError, match="at least one"):
        build_locked_mode_authority_report([])
    with pytest.raises(LockedModeAuthorityError, match="unique, ascending"):
        build_locked_mode_authority_report([_artifact(shot_id=2), _artifact(shot_id=1)])
    with pytest.raises(LockedModeAuthorityError, match="unique, ascending"):
        build_locked_mode_authority_report([_artifact(shot_id=1), _artifact(shot_id=1)])


def test_cli_sorts_writes_exclusively_and_rejects_duplicate_shots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command canonicalises shot order and never overwrites evidence."""
    seen: list[int] = []
    monkeypatch.setattr(locked_module, "load_verified_source_manifest", lambda *_args, **_kwargs: {})

    def _read(_manifest: Mapping[str, Any], *, artifact_root: Path, shot_id: int) -> VerifiedSourceArtifact:
        assert artifact_root == tmp_path
        seen.append(shot_id)
        return _artifact(shot_id=shot_id)

    monkeypatch.setattr(locked_module, "read_verified_npz_artifact", _read)
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
    with pytest.raises(LockedModeAuthorityError, match="refusing to overwrite"):
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
    monkeypatch.setattr(locked_module, "load_verified_source_manifest", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(locked_module, "read_verified_npz_artifact", lambda *_args, **_kwargs: _artifact())

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
