# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST normalised-beta authority tests
"""Contract tests for the fail-closed MAST normalised-beta gate."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pytest

import validation.mast_normalised_beta_authority as beta_module
from validation.mast_normalised_beta_authority import (
    BETA_N_KEY,
    BETA_TOR_KEY,
    FITTED_PLASMA_CURRENT_KEY,
    MINOR_RADIUS_KEY,
    NORMALISED_BETA_AUTHORITY_SCHEMA,
    TIMEBASE_KEY,
    VACUUM_FIELD_GEOMETRIC_AXIS_KEY,
    NormalisedBetaAuthorityError,
    assess_normalised_beta_authority,
    build_normalised_beta_authority_report,
    main,
    mast_normalised_beta_authority_spec,
)
from validation.mast_source_artifact_reader import VerifiedSourceArtifact
from validation.mast_source_object_manifest import canonical_json_sha256

_DESCRIPTION = "Normalized toroidal beta, defined as 100 * beta_tor * a[m] * B0 [T] / ip [MA]"
_IMAS_TARGET = "equilibrium.time_slice[:].global_quantities.beta_tor_normal"


def _metadata(
    *,
    units: str,
    name: str | None = None,
    uda_name: str | None = None,
    description: str | None = None,
    imas: str | None = None,
    dimensions: tuple[str, ...] = ("time",),
    metadata_status: str = "source_xarray",
) -> Mapping[str, Any]:
    attributes: dict[str, object] = {"units": units}
    if name is not None:
        attributes.update({"name": name, "uda_name": uda_name, "description": description, "imas": imas})
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
        TIMEBASE_KEY: np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float64),
        BETA_N_KEY: np.asarray([np.nan, -1.0, 2.8, 3.1], dtype=np.float64),
        BETA_TOR_KEY: np.asarray([np.nan, -2.0, 7.0, 7.2], dtype=np.float64),
        MINOR_RADIUS_KEY: np.asarray([np.nan, 0.55, 0.56, 0.57], dtype=np.float64),
    }
    default_metadata: dict[str, Mapping[str, Any]] = {
        TIMEBASE_KEY: _metadata(units="s"),
        BETA_N_KEY: _metadata(
            units="T",
            name="beta_tor_normal",
            uda_name="EFM_BETAN",
            description=_DESCRIPTION,
            imas=_IMAS_TARGET,
        ),
        BETA_TOR_KEY: _metadata(units="", name="beta_tor", uda_name="EFM_BETAT"),
        MINOR_RADIUS_KEY: _metadata(units="m", name="minor_radius", uda_name="EFM_MINOR_RADIUS"),
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


def _digest(payload: Mapping[str, object]) -> str:
    copy = dict(payload)
    copy["payload_sha256"] = None
    return canonical_json_sha256(copy)


def test_spec_pins_the_unit_conflict_and_forbids_numeric_unit_conversion() -> None:
    """The contract treats T-to-1 as metadata repair, not arithmetic."""
    spec = mast_normalised_beta_authority_spec()

    assert spec["schema_version"] == NORMALISED_BETA_AUTHORITY_SCHEMA
    assert spec["payload_sha256"] == _digest(spec)
    direct = cast(dict[str, object], spec["direct_source"])
    reproduction = cast(dict[str, object], spec["independent_reproduction_route"])
    assert direct["observed_source_units"] == "T"
    assert direct["current_imas_leaf"] == "equilibrium.time_slice[:].global_quantities.beta_tor_norm"
    assert direct["numeric_unit_conversion"] == "forbidden"
    assert VACUUM_FIELD_GEOMETRIC_AXIS_KEY in cast(list[str], reproduction["required_source_keys"])
    assert FITTED_PLASMA_CURRENT_KEY in cast(list[str], reproduction["required_source_keys"])


def test_realistic_source_records_exact_conflicts_and_negative_values() -> None:
    """Plausible beta_N values cannot conceal metadata, validity, or lineage gaps."""
    report = assess_normalised_beta_authority(_artifact())

    assert report["status"] == "blocked"
    assert report["canonical_beta_n_binding_admissible"] is False
    blockers = cast(list[str], report["blockers"])
    assert blockers == [
        "normalised_beta_negative_value_validity_policy_absent",
        "normalised_beta_formula_inputs_not_lineage_bound",
        "fair_mast_beta_n_source_unit_conflicts_with_imas",
        "fair_mast_beta_n_imas_target_name_is_not_current",
        "normalised_beta_formula_sign_and_scale_authority_absent",
        "normalised_beta_reconstruction_quality_authority_absent",
        "one_standard_deviation_uncertainty_authority_absent",
    ]
    direct = cast(dict[str, object], report["direct_source_evidence"])
    assert direct["observed_value_summary_not_scale_authority"] == {
        "minimum": -1.0,
        "maximum": 3.1,
        "negative": 1,
        "zero": 0,
        "positive": 2,
    }
    formula = cast(dict[str, object], report["formula_input_evidence"])
    assert formula["missing_source_keys"] == [VACUUM_FIELD_GEOMETRIC_AXIS_KEY, FITTED_PLASMA_CURRENT_KEY]
    assert report["metadata_repair_candidate"] == {
        "from_units": "T",
        "to_units": "1",
        "numeric_transform": None,
        "admissible": False,
    }
    assert report["payload_sha256"] == _digest(report)
    assert all(value is False for value in cast(dict[str, bool], report["claim_boundary"]).values())


def test_missing_direct_members_fail_before_value_inspection() -> None:
    """The gate reports an absent beta array or timebase without guessing."""
    base = _artifact()
    arrays = {BETA_TOR_KEY: base.arrays[BETA_TOR_KEY], MINOR_RADIUS_KEY: base.arrays[MINOR_RADIUS_KEY]}
    metadata = {key: base.metadata[key] for key in arrays}

    report = assess_normalised_beta_authority(_artifact(arrays=arrays, metadata=metadata))

    assert cast(list[str], report["blockers"])[0] == "normalised_beta_source_members_absent"
    direct = cast(dict[str, object], report["direct_source_evidence"])
    assert direct["missing_source_keys"] == [BETA_N_KEY, TIMEBASE_KEY]


@pytest.mark.parametrize(
    ("replacement", "suffix"),
    [
        (
            _metadata(
                units="T",
                name="beta_tor_normal",
                uda_name="EFM_BETAN",
                description=_DESCRIPTION,
                imas=_IMAS_TARGET,
                metadata_status="values_only",
            ),
            "metadata_status",
        ),
        (
            _metadata(
                units="T",
                name="beta_tor_normal",
                uda_name="EFM_BETAN",
                description=_DESCRIPTION,
                imas=_IMAS_TARGET,
                dimensions=("sample",),
            ),
            "dimensions",
        ),
        (
            _metadata(
                units="1", name="beta_tor_normal", uda_name="EFM_BETAN", description=_DESCRIPTION, imas=_IMAS_TARGET
            ),
            "observed_units",
        ),
        (
            MappingProxyType(
                {"metadata_status": "source_xarray", "dimensions": ("time",), "units": "T", "source_attributes": None}
            ),
            "source_attributes",
        ),
        (
            MappingProxyType(
                {
                    "metadata_status": "source_xarray",
                    "dimensions": ("time",),
                    "units": "T",
                    "source_attributes": MappingProxyType(
                        {
                            "name": "beta_tor_normal",
                            "uda_name": "EFM_BETAN",
                            "description": _DESCRIPTION,
                            "imas": _IMAS_TARGET,
                            "units": "1",
                        }
                    ),
                }
            ),
            "units",
        ),
        (_metadata(units="T", name="wrong", uda_name="EFM_BETAN", description=_DESCRIPTION, imas=_IMAS_TARGET), "name"),
        (
            _metadata(units="T", name="beta_tor_normal", uda_name="wrong", description=_DESCRIPTION, imas=_IMAS_TARGET),
            "uda_name",
        ),
        (
            _metadata(units="T", name="beta_tor_normal", uda_name="EFM_BETAN", description="wrong", imas=_IMAS_TARGET),
            "description",
        ),
        (
            _metadata(units="T", name="beta_tor_normal", uda_name="EFM_BETAN", description=_DESCRIPTION, imas="wrong"),
            "imas",
        ),
    ],
)
def test_beta_source_metadata_drift_is_explicit(replacement: Mapping[str, Any], suffix: str) -> None:
    """Every direct-source identity field is matched exactly."""
    base = _artifact()
    metadata = dict(base.metadata)
    metadata[BETA_N_KEY] = replacement

    report = assess_normalised_beta_authority(_artifact(metadata=metadata))

    assert "normalised_beta_source_metadata_mismatch" in cast(list[str], report["blockers"])
    mismatches = cast(dict[str, object], report["direct_source_evidence"])["metadata_mismatches"]
    assert f"{BETA_N_KEY}:{suffix}" in cast(list[str], mismatches)


@pytest.mark.parametrize(
    ("replacement", "suffix"),
    [
        (_metadata(units="s", metadata_status="values_only"), "metadata_status"),
        (_metadata(units="s", dimensions=("sample",)), "dimensions"),
        (_metadata(units="ms"), "units"),
    ],
)
def test_timebase_metadata_drift_is_explicit(replacement: Mapping[str, Any], suffix: str) -> None:
    """Time dimensions and seconds cannot drift silently."""
    base = _artifact()
    metadata = dict(base.metadata)
    metadata[TIMEBASE_KEY] = replacement

    report = assess_normalised_beta_authority(_artifact(metadata=metadata))

    mismatches = cast(dict[str, object], report["direct_source_evidence"])["metadata_mismatches"]
    assert f"{TIMEBASE_KEY}:{suffix}" in cast(list[str], mismatches)


def test_shape_finite_coverage_time_order_and_nonnegative_branch_are_guarded() -> None:
    """Array structure, finite coverage, ordering, and sign each have a branch."""
    base = _artifact()
    arrays = dict(base.arrays)
    arrays[BETA_N_KEY] = np.asarray([1.0, 2.0], dtype=np.float64)
    shape = assess_normalised_beta_authority(_artifact(arrays=arrays))
    assert "normalised_beta_source_shape_mismatch" in cast(list[str], shape["blockers"])

    arrays = dict(base.arrays)
    arrays[BETA_N_KEY] = np.full(4, np.nan, dtype=np.float64)
    empty = assess_normalised_beta_authority(_artifact(arrays=arrays))
    assert "normalised_beta_has_no_finite_time_aligned_samples" in cast(list[str], empty["blockers"])

    arrays = dict(base.arrays)
    arrays[TIMEBASE_KEY] = np.asarray([0.0, 0.2, 0.1, 0.3], dtype=np.float64)
    order = assess_normalised_beta_authority(_artifact(arrays=arrays))
    assert "normalised_beta_timebase_not_strictly_increasing" in cast(list[str], order["blockers"])

    arrays = dict(base.arrays)
    arrays[BETA_N_KEY] = np.asarray([np.nan, 0.0, 2.8, 3.1], dtype=np.float64)
    nonnegative = assess_normalised_beta_authority(_artifact(arrays=arrays))
    assert "normalised_beta_negative_value_validity_policy_absent" not in cast(list[str], nonnegative["blockers"])
    summary = cast(dict[str, object], nonnegative["direct_source_evidence"])[
        "observed_value_summary_not_scale_authority"
    ]
    assert cast(dict[str, int | float], summary)["zero"] == 1


def test_formula_input_inventory_checks_complete_and_misaligned_routes() -> None:
    """A complete input name set still requires exact common shapes."""
    base = _artifact()
    arrays = dict(base.arrays)
    metadata = dict(base.metadata)
    arrays[VACUUM_FIELD_GEOMETRIC_AXIS_KEY] = np.asarray([np.nan, -0.58, -0.57, -0.56], dtype=np.float64)
    arrays[FITTED_PLASMA_CURRENT_KEY] = np.asarray([np.nan, -0.6, -0.61, -0.62], dtype=np.float64)
    metadata[VACUUM_FIELD_GEOMETRIC_AXIS_KEY] = _metadata(units="T")
    metadata[FITTED_PLASMA_CURRENT_KEY] = _metadata(units="MA")
    complete = assess_normalised_beta_authority(_artifact(arrays=arrays, metadata=metadata))
    assert "normalised_beta_formula_inputs_not_lineage_bound" not in cast(list[str], complete["blockers"])
    assert "normalised_beta_formula_input_shape_mismatch" not in cast(list[str], complete["blockers"])

    arrays[FITTED_PLASMA_CURRENT_KEY] = np.asarray([1.0, 2.0], dtype=np.float64)
    mismatch = assess_normalised_beta_authority(_artifact(arrays=arrays, metadata=metadata))
    assert "normalised_beta_formula_input_shape_mismatch" in cast(list[str], mismatch["blockers"])


def test_multi_shot_report_is_canonical_and_self_digested() -> None:
    """Campaign evidence requires unique ascending shot identities."""
    report = build_normalised_beta_authority_report([_artifact(shot_id=30421), _artifact(shot_id=30424)])

    assert report["status"] == "blocked"
    assert report["shot_count"] == 2
    assert report["payload_sha256"] == _digest(report)
    with pytest.raises(NormalisedBetaAuthorityError, match="at least one"):
        build_normalised_beta_authority_report([])
    with pytest.raises(NormalisedBetaAuthorityError, match="unique, ascending"):
        build_normalised_beta_authority_report([_artifact(shot_id=2), _artifact(shot_id=1)])
    with pytest.raises(NormalisedBetaAuthorityError, match="unique, ascending"):
        build_normalised_beta_authority_report([_artifact(shot_id=1), _artifact(shot_id=1)])


def test_cli_sorts_writes_exclusively_and_rejects_duplicate_shots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI canonicalises shot order and never overwrites evidence."""
    seen: list[int] = []
    monkeypatch.setattr(beta_module, "load_verified_source_manifest", lambda *_args, **_kwargs: {})

    def _read(_manifest: Mapping[str, Any], *, artifact_root: Path, shot_id: int) -> VerifiedSourceArtifact:
        assert artifact_root == tmp_path
        seen.append(shot_id)
        return _artifact(shot_id=shot_id)

    monkeypatch.setattr(beta_module, "read_verified_npz_artifact", _read)
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
    with pytest.raises(NormalisedBetaAuthorityError, match="refusing to overwrite"):
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


def test_cli_removes_partial_report_on_serialisation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A writer failure cannot leave a partial authority report."""
    monkeypatch.setattr(beta_module, "load_verified_source_manifest", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(beta_module, "read_verified_npz_artifact", lambda *_args, **_kwargs: _artifact())

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
