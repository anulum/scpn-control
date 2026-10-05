# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural turbulence reference validation tests

"""Exercise original transport error/score domains and uncapped sample counts through public persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from test_neural_turbulence_reference_validation import _valid_turbulence_reference_artifact

from validation.validate_neural_turbulence_reference import (
    validate_neural_turbulence_reference,
)

ERROR_FIELDS = ("Q_i_rmse_gB", "Q_e_rmse_gB", "Gamma_e_rmse_gB", "flux_relative_mae")


def declaration() -> dict[str, Any]:
    """Supply original engineering transport metadata, zero errors and a unit declared critical-gradient score."""
    payload = cast(dict[str, Any], _valid_turbulence_reference_artifact())
    payload["metrics"] = dict.fromkeys(ERROR_FIELDS, 0.0)
    payload["metrics"]["critical_gradient_accuracy"] = 1.0
    payload["tolerances"] = dict.fromkeys(ERROR_FIELDS, 1.0)
    payload["tolerances"]["critical_gradient_accuracy_min"] = 0.0
    return payload


def inspect(tmp_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Persist an original engineering declaration and call the public reader."""
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    return validate_neural_turbulence_reference(path, require_reference_artifacts=True)


@pytest.mark.parametrize("field", ERROR_FIELDS)
@pytest.mark.parametrize("block", ["metrics", "tolerances"])
@pytest.mark.parametrize("value", [None, [], "1", True, -1, float("nan"), float("inf"), 10**400, 0, 1.001])
def test_error_and_bound_domains(tmp_path: Path, field: str, block: str, value: object) -> None:
    """Four errors retain finite nonnegative values and positive finite bounds with equality accepted."""
    payload = declaration()
    payload[block][field] = value
    accepted = (block == "metrics" and value == 0) or (block == "tolerances" and value == 1.001)
    report = inspect(tmp_path, payload)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize("block", ["metrics", "tolerances"])
@pytest.mark.parametrize("value", [None, [], "1", True, -1, 1.001, float("nan"), float("inf"), 10**400, 0, 1, 5e-324])
def test_inclusive_score_domains(tmp_path: Path, block: str, value: object) -> None:
    """Both branch accuracy and its minimum accept zero/one/subnormals and refuse nonnumbers, overflow and out-of-range scores."""
    payload = declaration()
    key = "critical_gradient_accuracy" + ("_min" if block == "tolerances" else "")
    payload[block][key] = value
    accepted = type(value) in {int, float} and value in [0, 1, 5e-324]
    report = inspect(tmp_path, payload)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == "critical_gradient_accuracy" for error in report["errors"])


@pytest.mark.parametrize("value", [None, [], "1", True, 0, -1, 1.0, 1, 10**400])
def test_uncapped_sample_count(tmp_path: Path, value: object) -> None:
    """Sample count remains an uncapped positive nonboolean integer without numeric conversion."""
    payload = declaration()
    payload["reference_sample_count"] = value
    report = inspect(tmp_path, payload)
    accepted = type(value) is int and value in [1, 10**400]
    assert report["status"] == ("pass" if accepted else "fail")
    if accepted:
        assert report["entries"][0]["reference_sample_count"] == value
    else:
        assert any(error["field"] == "reference_sample_count" for error in report["errors"])


def test_inclusive_errors_and_minimum_score(tmp_path: Path) -> None:
    """Equal maximum bounds and minimum score pass, while scores below their declared minimum refuse."""
    payload = declaration()
    for field in ERROR_FIELDS:
        payload["metrics"][field] = payload["tolerances"][field]
    payload["metrics"]["critical_gradient_accuracy"] = 0.5
    payload["tolerances"]["critical_gradient_accuracy_min"] = 0.5
    assert inspect(tmp_path, payload)["status"] == "pass"
    payload["metrics"]["critical_gradient_accuracy"] = 0.49
    assert inspect(tmp_path, payload)["errors"][0]["field"] == "critical_gradient_accuracy"
    payload["metrics"]["critical_gradient_accuracy"] = 1.0
    payload["metrics"]["Gamma_e_rmse_gB"] = 0
    payload["tolerances"]["Gamma_e_rmse_gB"] = 5e-324
    assert inspect(tmp_path, payload)["status"] == "pass"


@pytest.mark.parametrize("source", ["documented_public_reference", "real_gk_campaign"])
@pytest.mark.parametrize("value", [None, [], "", " ", "../unparsed\x00citation", "s3://", "file:///unfetched"])
def test_campaign_and_public_presence(tmp_path: Path, source: str, value: object) -> None:
    """Campaign and public provenance retain nonblank presence-only policy without fetching or URI restrictions."""
    payload = declaration()
    payload.pop("reference_doi")
    payload["source"] = source
    field = "campaign_artifact_uri" if source == "real_gk_campaign" else "reference_url"
    payload[field] = value
    accepted = isinstance(value, str) and bool(value.strip())
    report = inspect(tmp_path, payload)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        expected = "campaign_artifact_uri" if source == "real_gk_campaign" else "reference"
        assert any(error["field"] == expected for error in report["errors"])
