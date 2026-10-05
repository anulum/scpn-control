# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — RZIP declared numeric domain tests

"""Exercise original RZIP numeric domains through persisted public declarations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from test_rzip_reference_validation import _valid_rzip_reference_artifact

from validation.validate_rzip_reference import validate_rzip_reference

ERROR_FIELDS = ("growth_rate_relative_error", "vertical_displacement_rmse_m", "closed_loop_pole_real_abs_error")
POSITIVE_FIELDS = (
    "major_radius_m",
    "minor_radius_m",
    "elongation",
    "plasma_current_A",
    "toroidal_field_T",
    "wall_time_constant_s",
)


def declaration() -> dict[str, Any]:
    """Supply original engineering metadata with exact-zero errors; no physical evidence is authenticated."""
    payload = cast(dict[str, Any], _valid_rzip_reference_artifact())
    payload["metrics"] = dict.fromkeys(ERROR_FIELDS, 0.0)
    payload["tolerances"] = dict.fromkeys(ERROR_FIELDS, 1.0)
    return payload


@pytest.mark.parametrize("field", POSITIVE_FIELDS)
@pytest.mark.parametrize("value", [None, [], "1", True, 0, -1, float("nan"), float("inf"), 10**400, 5e-324])
def test_positive_parameters(tmp_path: Path, field: str, value: object) -> None:
    """All six original positive inputs reject invalid/overflow values and retain positive subnormals."""
    payload = declaration()
    payload["physical_parameters"][field] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_rzip_reference(path)
    assert report["status"] == ("pass" if value == 5e-324 else "fail")
    if report["status"] == "fail":
        assert any(error["field"] == "physical_parameters" for error in report["errors"])


@pytest.mark.parametrize("value", [None, [], True, "0", float("nan"), float("inf"), 10**400, -1, 0, 1, -5e-324])
def test_signed_vertical_index(tmp_path: Path, value: object) -> None:
    """Vertical field index remains signed finite, including zero and positive values without physical inference."""
    payload = declaration()
    payload["physical_parameters"]["vertical_field_index"] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    accepted = isinstance(value, int | float) and not isinstance(value, bool) and value in [-1, 0, 1, -5e-324]
    assert validate_rzip_reference(path)["status"] == ("pass" if accepted else "fail")


@pytest.mark.parametrize("field", ERROR_FIELDS)
@pytest.mark.parametrize("block", ["metrics", "tolerances"])
@pytest.mark.parametrize("value", [None, [], "1", True, -1, float("nan"), float("inf"), 10**400, 0, 1.001])
def test_error_and_bound_domains(tmp_path: Path, field: str, block: str, value: object) -> None:
    """Three original errors require finite nonnegative numbers, positive bounds and inclusive comparison."""
    payload = declaration()
    payload[block][field] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_rzip_reference(path)
    accepted = (block == "metrics" and value == 0) or (block == "tolerances" and value == 1.001)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize("count", [None, [], "1", True, 0, -1, 1.0, 1, 10**400])
def test_positive_uncapped_case_count(tmp_path: Path, count: object) -> None:
    """Positive nonboolean integer counts remain uncapped and are returned unchanged in identity metadata."""
    payload = declaration()
    payload["reference_case_count"] = count
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_rzip_reference(path)
    accepted = isinstance(count, int) and not isinstance(count, bool) and count > 0
    assert report["status"] == ("pass" if accepted else "fail")
    if accepted:
        assert report["entries"][0]["reference_case_count"] == count


def test_equal_bounds_and_metadata_only(tmp_path: Path) -> None:
    """Equality and independently positive geometry pass without new ordering, conversions, hashing or dynamics."""
    payload = declaration()
    payload["metrics"] = dict(payload["tolerances"])
    payload["physical_parameters"].update(minor_radius_m=10.0, elongation=0.5, wall_time_constant_s=5e-324)
    payload["reference_artifact_sha256"] = "B" * 64
    payload["reference_doi"] = "../unresolved\x00presence"
    payload["units"]["extra"] = "unknown"
    payload["payload_sha256"] = "ignored-extra"
    path = tmp_path / "input.data"
    path.write_text(json.dumps(payload))
    report = validate_rzip_reference(path, require_reference_artifacts=True)
    assert report["status"] == "pass" and report["errors"] == []
    assert report["entries"] == [
        {
            "path": str(path),
            "source": "documented_public_reference",
            "model_id": payload["model_id"],
            "model_version": payload["model_version"],
            "reference_dataset_id": payload["reference_dataset_id"],
            "reference_case_count": 5,
        }
    ]
