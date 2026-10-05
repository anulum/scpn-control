# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Blob transport declaration domain tests

"""Exercise original SOL blob geometry, coordinates, ordered pairs and bounds through persisted public APIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from test_blob_transport_reference_validation import _valid_blob_reference_artifact

from validation.validate_blob_transport_reference import canonical_artifact_sha256, validate_blob_transport_reference

ERROR_FIELDS = (
    "radial_velocity_rmse_m_s",
    "density_profile_relative_l2",
    "wall_flux_relative_error",
    "event_duration_relative_error",
    "event_size_relative_error",
)
GEOMETRY_FIELDS = ("R0_m", "B0_T", "L_parallel_m", "Te_eV", "density_m3")


def declaration() -> dict[str, Any]:
    """Supply original engineering blob metadata with a consistent body hash, without producer authentication."""
    payload = cast(dict[str, Any], _valid_blob_reference_artifact())
    payload["metrics"] = dict.fromkeys(ERROR_FIELDS, 0.0)
    payload["tolerances"] = dict.fromkeys(ERROR_FIELDS, 1.0)
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    return payload


def inspect(tmp_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Persist an engineering declaration with its recalculated body consistency hash and call the public reader."""
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    return validate_blob_transport_reference(path, require_reference_artifacts=True)


@pytest.mark.parametrize("field", GEOMETRY_FIELDS)
@pytest.mark.parametrize(
    ("value", "accepted"),
    [
        (None, False),
        ([], False),
        ("1", False),
        (True, False),
        (0, False),
        (-1, False),
        (float("nan"), False),
        (float("inf"), False),
        (10**400, False),
        (5e-324, True),
    ],
)
def test_original_positive_geometry(tmp_path: Path, field: str, value: object, accepted: bool) -> None:
    """Five original positive geometry inputs retain representable subnormals and refuse overflowing integers."""
    payload = declaration()
    payload["magnetic_geometry"][field] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == "magnetic_geometry." + field for error in report["errors"])


@pytest.mark.parametrize(
    ("value", "accepted"),
    [
        (None, False),
        ({}, False),
        ("0,1", False),
        ([], False),
        ([0], False),
        ([True, 1], False),
        (["0", 1], False),
        ([0, float("nan")], False),
        ([0, float("inf")], False),
        ([0, 10**400], False),
        ([-1, 0], False),
        ([0, 0], False),
        ([1, 0], False),
        ([0, 5e-324], True),
        ([0, 1, 2], True),
        ([5e-324, 1e308], True),
    ],
)
def test_original_coordinate_domain(tmp_path: Path, value: object, accepted: bool) -> None:
    """SOL coordinates require at least two finite nonnegative strictly increasing samples without conversion."""
    payload = declaration()
    payload["separatrix_to_wall_coordinates_m"] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == "separatrix_to_wall_coordinates_m" for error in report["errors"])


@pytest.mark.parametrize("field", ["detector_time_domain_s", "blob_size_range_m"])
@pytest.mark.parametrize(
    ("value", "accepted"),
    [
        (None, False),
        ({}, False),
        ("0,1", False),
        ([], False),
        ([0], False),
        ([0, 1, 2], False),
        ([True, 1], False),
        ([0, True], False),
        ([-1, 1], False),
        ([0, 0], False),
        ([1, 1], False),
        ([1, 0], False),
        ([10**400, 1], False),
        ([0, 10**400], False),
        ([0, float("nan")], False),
        ([0, 5e-324], True),
        ([0, 1], True),
        ([5e-324, 1e308], True),
    ],
)
def test_zero_lower_ordered_pairs(tmp_path: Path, field: str, value: object, accepted: bool) -> None:
    """Both original detector time AND blob-size pairs admit lower zero despite the old positive-size error wording."""
    payload = declaration()
    payload[field] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize("field", ERROR_FIELDS)
@pytest.mark.parametrize("block", ["metrics", "tolerances"])
@pytest.mark.parametrize("value", [None, [], "1", True, -1, float("nan"), float("inf"), 10**400, 0, 1.001])
def test_error_and_bound_domains(tmp_path: Path, field: str, block: str, value: object) -> None:
    """Five declared errors retain finite nonnegative values with positive finite inclusive bounds."""
    payload = declaration()
    payload[block][field] = value
    accepted = (block == "metrics" and value == 0) or (block == "tolerances" and value == 1.001)
    report = inspect(tmp_path, payload)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == field for error in report["errors"])


def test_inclusive_bounds_and_unfetched_references(tmp_path: Path) -> None:
    """Equal bounds, lower-zero pairs and format-only nonexistent reference hashes pass without physical authenticity."""
    payload = declaration()
    payload["metrics"] = dict(payload["tolerances"])
    payload["detector_time_domain_s"] = [0, 5e-324]
    payload["blob_size_range_m"] = [0, 5e-324]
    payload["reference_artifact_sha256"] = "B" * 64
    payload["profile_artifact_uri"] = "s3://"
    payload["reference_case_count"] = 10**400
    payload["model_id"] = []
    payload["units"]["extra"] = "unknown"
    report = inspect(tmp_path, payload)
    assert report["status"] == "pass" and report["errors"] == []
    assert report["entries"][0]["payload_sha256"] == canonical_artifact_sha256(payload)
    assert "reference_case_count" not in report["entries"][0]
