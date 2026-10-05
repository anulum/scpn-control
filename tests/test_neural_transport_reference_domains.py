# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural transport reference validation tests

"""Exercise original transport error/score domains and uncapped sample counts through public persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from test_neural_transport_reference_validation import _valid_qualikiz_reference_artifact

from validation.validate_neural_transport_reference import (
    canonical_artifact_sha256,
    validate_neural_transport_reference,
)

ERROR_FIELDS = ("chi_i_rmse_m2_s", "chi_e_rmse_m2_s", "D_e_rmse_m2_s", "chi_i_relative_mae")


def declaration() -> dict[str, Any]:
    """Supply original engineering transport metadata, zero errors and a fully accurate declared branch score."""
    payload = cast(dict[str, Any], _valid_qualikiz_reference_artifact())
    payload["metrics"] = dict.fromkeys(ERROR_FIELDS, 0.0)
    payload["metrics"]["unstable_branch_accuracy"] = 1.0
    payload["tolerances"] = dict.fromkeys(ERROR_FIELDS, 1.0)
    payload["tolerances"]["unstable_branch_accuracy_min"] = 0.0
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    return payload


def inspect(tmp_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Persist an engineering declaration with recomputed body consistency and call the public reader."""
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    return validate_neural_transport_reference(path, require_reference_artifacts=True)


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
    key = "unstable_branch_accuracy" + ("_min" if block == "tolerances" else "")
    payload[block][key] = value
    accepted = type(value) in {int, float} and value in [0, 1, 5e-324]
    report = inspect(tmp_path, payload)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == "unstable_branch_accuracy" for error in report["errors"])


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
    payload["metrics"]["unstable_branch_accuracy"] = 0.5
    payload["tolerances"]["unstable_branch_accuracy_min"] = 0.5
    assert inspect(tmp_path, payload)["status"] == "pass"
    payload["metrics"]["unstable_branch_accuracy"] = 0.49
    assert inspect(tmp_path, payload)["errors"][0]["field"] == "unstable_branch_accuracy"
    payload["metrics"]["unstable_branch_accuracy"] = 1.0
    payload["metrics"]["D_e_rmse_m2_s"] = 0
    payload["tolerances"]["D_e_rmse_m2_s"] = 5e-324
    assert inspect(tmp_path, payload)["status"] == "pass"


@pytest.mark.parametrize(
    "binary",
    [
        None,
        [],
        "",
        "bin/QuaLiKiz",
        "file:///opt/QuaLiKiz",
        "/opt/x/../QuaLiKiz",
        "/opt/QuaLiKiz\x00",
        "/tmp/QuaLiKiz",
        "/etc/QuaLiKiz",
        "/opt/QuaLiKiz/",
    ],
)
def test_actual_executable_policy_findings(tmp_path: Path, binary: object) -> None:
    """Real-QuaLiKiz provenance uses existing lexical executable policy, refusing malformed or out-of-policy declarations."""
    payload = declaration()
    payload["source"] = "real_qualikiz"
    payload["binary_path"] = binary
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and any(error["field"] == "binary_path" for error in report["errors"])


def test_unfetched_executable_and_public_citation(tmp_path: Path) -> None:
    """Admitted nonexistent executable paths are declaration-only; public citations require presence without URI parsing."""
    payload = declaration()
    payload.pop("reference_doi")
    payload["source"] = "real_qualikiz"
    payload["binary_path"] = "/opt/not-present/QuaLiKiz"
    assert inspect(tmp_path, payload)["status"] == "pass"
    payload["source"] = "documented_public_reference"
    assert inspect(tmp_path, payload)["errors"][0]["field"] == "reference"
    for field in ["reference_url", "reference_doi"]:
        payload[field] = "../unparsed\x00citation"
        assert inspect(tmp_path, payload)["status"] == "pass"
        payload.pop(field)
