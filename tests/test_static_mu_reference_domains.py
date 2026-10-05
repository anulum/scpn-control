# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — SOC reference validation tests

"""Exercise uncapped plant dimensions and original static error bounds through public persisted declarations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from validation.validate_static_mu_analysis_reference import validate_static_mu_analysis_reference

ERROR_FIELDS = (
    "mu_upper_bound_relative_error",
    "robustness_margin_abs_error",
    "controller_gain_relative_error",
    "d_scaling_relative_error",
    "closed_loop_spectral_abscissa_abs_error",
)


def declaration() -> dict[str, Any]:
    """Supply engineering static zero-frequency metadata without designing a controller or authenticating a reference."""
    return {
        "schema_version": "1.0",
        "source": "documented_public_reference",
        "model_id": "engineering-static",
        "model_version": "original",
        "reference_dataset_id": "engineering-only",
        "reference_artifact_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "executed_at": "2026-10-02T00:00:00Z",
        "reference_doi": "unfetched-declaration",
        "units": {
            "mu": "1",
            "robustness_margin": "1",
            "controller_gain": "1",
            "d_scaling": "1",
            "spectral_abscissa": "s^-1",
        },
        "plant_metadata": {
            "state_dimension": 2,
            "control_dimension": 2,
            "output_dimension": 2,
            "uncertainty_total_size": 2,
        },
        "reference_case_count": 2,
        "metrics": {
            "mu_upper_bound_relative_error": 0.0,
            "robustness_margin_abs_error": 0.0,
            "controller_gain_relative_error": 0.0,
            "d_scaling_relative_error": 0.0,
            "closed_loop_spectral_abscissa_abs_error": 0.0,
        },
        "tolerances": {
            "mu_upper_bound_relative_error": 1.0,
            "robustness_margin_abs_error": 1.0,
            "controller_gain_relative_error": 1.0,
            "d_scaling_relative_error": 1.0,
            "closed_loop_spectral_abscissa_abs_error": 1.0,
        },
    }


def inspect(tmp_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Persist a caller declaration and invoke the public static reference reader."""
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    return validate_static_mu_analysis_reference(path, require_reference_artifacts=True)


@pytest.mark.parametrize("field", ERROR_FIELDS)
@pytest.mark.parametrize("block", ["metrics", "tolerances"])
@pytest.mark.parametrize("value", [None, [], "1", True, -1, float("nan"), float("inf"), 10**400, 0, 1.001])
def test_error_and_bound_domains(tmp_path: Path, field: str, block: str, value: object) -> None:
    """Five errors remain nonnegative and within positive finite bounds, including equality."""
    payload = declaration()
    payload[block][field] = value
    accepted = (block == "metrics" and value == 0) or (block == "tolerances" and value == 1.001)
    report = inspect(tmp_path, payload)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize(
    "field", ["state_dimension", "control_dimension", "output_dimension", "uncertainty_total_size"]
)
@pytest.mark.parametrize("value", [None, [], "1", True, 0, -1, 1.0, 1, 10**400])
def test_uncapped_plant_dimensions(tmp_path: Path, field: str, value: object) -> None:
    """Four declared plant dimensions remain positive uncapped nonboolean integers without matrix allocation."""
    payload = declaration()
    payload["plant_metadata"][field] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == ("pass" if type(value) is int and value in [1, 10**400] else "fail")
    if report["status"] == "fail":
        assert any(error["field"] == "plant_metadata" for error in report["errors"])


@pytest.mark.parametrize("value", [None, [], "1", True, 0, -1, 1.0, 1, 10**400])
def test_uncapped_case_count(tmp_path: Path, value: object) -> None:
    """Case count retains positive uncapped nonboolean integers and admitted values in report metadata."""
    payload = declaration()
    payload["reference_case_count"] = value
    report = inspect(tmp_path, payload)
    accepted = type(value) is int and value in [1, 10**400]
    assert report["status"] == ("pass" if accepted else "fail")
    if accepted:
        assert report["entries"][0]["reference_case_count"] == value
    else:
        assert any(error["field"] == "reference_case_count" for error in report["errors"])


def test_inclusive_declared_errors_and_subnormal_bounds(tmp_path: Path) -> None:
    """Equality at each declared bound and representable positive subnormal tolerances remain accepted."""
    payload = declaration()
    for field in ERROR_FIELDS:
        payload["metrics"][field] = payload["tolerances"][field]
    assert inspect(tmp_path, payload)["status"] == "pass"
    payload["metrics"][ERROR_FIELDS[0]] = 0
    payload["tolerances"][ERROR_FIELDS[0]] = 5e-324
    assert inspect(tmp_path, payload)["status"] == "pass"
