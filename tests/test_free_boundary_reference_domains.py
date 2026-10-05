# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Free-boundary identity and provenance tests

"""Check original declaration identity and presence-only provenance at the public reader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from validation.validate_free_boundary_reference import validate_free_boundary_reference

ERROR_FIELDS = (
    "shape_rms_abs_error",
    "x_point_position_abs_error_m",
    "x_point_flux_abs_error",
    "divertor_rms_abs_error",
    "coil_current_relative_error",
)
COUNT_FIELDS = ("coil_count", "boundary_point_count", "divertor_point_count")
POSITIVE_FIELDS = ("control_dt_s", "coil_slew_limit_MA_s")


def declaration() -> dict[str, Any]:
    """Supply an engineering free-boundary declaration without authenticating physical comparison bytes."""
    return {
        "schema_version": "1.0",
        "source": "documented_public_reference",
        "model_id": "engineering-test-model",
        "model_version": "1",
        "reference_dataset_id": "engineering-free-boundary-declaration-only",
        "reference_artifact_sha256": "a" * 64,
        "executed_at": "declared-time",
        "reference_doi": "presence-only",
        "units": {"position": "m", "flux": "Wb/rad", "current": "MA", "time": "s", "tracking_error": "1"},
        "equilibrium_metadata": {
            "coil_count": 1,
            "boundary_point_count": 1,
            "divertor_point_count": 1,
            "control_dt_s": 1.0,
            "coil_slew_limit_MA_s": 1.0,
        },
        "reference_case_count": 1,
        "metrics": dict.fromkeys(ERROR_FIELDS, 0.0),
        "tolerances": dict.fromkeys(ERROR_FIELDS, 1.0),
    }


@pytest.mark.parametrize("field", COUNT_FIELDS)
@pytest.mark.parametrize(
    ("value", "accepted"),
    [
        (None, False),
        ([], False),
        ("1", False),
        (True, False),
        (-1, False),
        (0, False),
        (1.0, False),
        (1, True),
        (10**400, True),
    ],
)
def test_uncapped_positive_metadata_counts(tmp_path: Path, field: str, value: object, accepted: bool) -> None:
    """Three original equilibrium counts remain positive nonboolean integers without a cap or cross-count relation."""
    payload = declaration()
    payload["equilibrium_metadata"][field] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_free_boundary_reference(path)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert report["errors"][0]["field"] == "equilibrium_metadata"


@pytest.mark.parametrize("field", POSITIVE_FIELDS)
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
def test_positive_timing_and_slew(tmp_path: Path, field: str, value: object, accepted: bool) -> None:
    """Original control interval and coil slew declarations require finite positive binary64 values, admitting subnormals."""
    payload = declaration()
    payload["equilibrium_metadata"][field] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    assert validate_free_boundary_reference(path)["status"] == ("pass" if accepted else "fail")


@pytest.mark.parametrize("field", [*COUNT_FIELDS, *POSITIVE_FIELDS])
def test_metadata_keys_required(tmp_path: Path, field: str) -> None:
    """All original equilibrium metadata keys remain mandatory at the persisted reader."""
    payload = declaration()
    payload["equilibrium_metadata"].pop(field)
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    assert validate_free_boundary_reference(path)["errors"][0]["field"] == "equilibrium_metadata"


@pytest.mark.parametrize("field", ERROR_FIELDS)
@pytest.mark.parametrize("block", ["metrics", "tolerances"])
@pytest.mark.parametrize("value", [None, [], "1", True, -1, float("nan"), float("inf"), 10**400, 0, 1.001])
def test_error_and_bound_domains(tmp_path: Path, field: str, block: str, value: object) -> None:
    """Five original tracking errors remain finite nonnegative with positive finite inclusive bounds, refusing overflow."""
    payload = declaration()
    payload[block][field] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    accepted = (block == "metrics" and value == 0) or (block == "tolerances" and value == 1.001)
    report = validate_free_boundary_reference(path)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize(
    ("count", "accepted"),
    [
        (None, False),
        ([], False),
        ("1", False),
        (True, False),
        (0, False),
        (-1, False),
        (1.0, False),
        (1, True),
        (10**400, True),
    ],
)
def test_positive_uncapped_case_count(tmp_path: Path, count: object, accepted: bool) -> None:
    """Reference case count remains a positive nonboolean integer without an artificial cap."""
    payload = declaration()
    payload["reference_case_count"] = count
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_free_boundary_reference(path)
    assert report["status"] == ("pass" if accepted else "fail")
    if accepted:
        assert report["entries"][0]["reference_case_count"] == count


def test_inclusive_bounds_and_independent_metadata(tmp_path: Path) -> None:
    """Equal errors, independent counts/timing and presence-only citations pass without physical consistency or reference authentication."""
    payload = declaration()
    payload["metrics"] = dict(payload["tolerances"])
    payload["equilibrium_metadata"].update(
        coil_count=10**400,
        boundary_point_count=1,
        divertor_point_count=2,
        control_dt_s=5e-324,
        coil_slew_limit_MA_s=1e308,
    )
    payload["reference_artifact_sha256"] = "B" * 64
    payload["reference_doi"] = "../unresolved\x00presence"
    payload["units"]["extra"] = "unknown"
    payload["payload_sha256"] = "ignored-extra"
    path = tmp_path / "input.data"
    path.write_text(json.dumps(payload))
    report = validate_free_boundary_reference(path, require_reference_artifacts=True)
    assert report["status"] == "pass" and report["errors"] == []
    assert report["entries"] == [
        {
            "path": str(path),
            "source": "documented_public_reference",
            "model_id": payload["model_id"],
            "model_version": payload["model_version"],
            "reference_dataset_id": payload["reference_dataset_id"],
            "reference_case_count": 1,
        }
    ]
