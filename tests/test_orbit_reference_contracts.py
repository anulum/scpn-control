# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Persisted orbit-reference declaration contract tests.

"""Exercise declaration findings through the real persisted public reader, without orbit qualification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from test_orbit_reference_validation import _valid_orbit_reference_artifact

from validation.validate_orbit_reference import validate_orbit_reference


@pytest.fixture
def declaration(tmp_path: Path) -> Path:
    """Persist the original metadata carrier; its citation, bytes and orbit execution are unauthenticated."""
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(_valid_orbit_reference_artifact()), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 1),
        ("source", []),
        ("source", {}),
        ("source", None),
        ("model_id", " "),
        ("model_version", None),
        ("reference_dataset_id", ""),
        ("executed_at", False),
        ("reference_artifact_sha256", None),
        ("reference_artifact_sha256", "1" * 64 + "\n"),
        ("reference_artifact_sha256", "é" * 64),
        ("units", []),
        ("reference_case_count", True),
        ("reference_case_count", 0),
        ("reference_case_count", "12"),
        ("metrics", []),
        ("tolerances", []),
    ],
)
def test_declared_field_refusals(declaration: Path, field: str, value: object) -> None:
    """Malformed persisted field types/values produce authored field findings without interpreter exceptions."""
    payload: dict[str, object] = json.loads(declaration.read_text())
    payload[field] = value
    declaration.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_orbit_reference(declaration, require_reference_artifacts=True)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    assert any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize("block", ["metrics", "tolerances"])
@pytest.mark.parametrize("number", [True, "1", None, -1, 10**400, float("inf"), float("nan")])
def test_declared_error_number_refusals(declaration: Path, block: str, number: object) -> None:
    """Persisted errors and bounds reject booleans, wrong types, negative/nonfinite and overflowing numbers."""
    payload: dict[str, object] = json.loads(declaration.read_text())
    cast(dict[str, object], payload[block])["banana_width_relative_error"] = number
    declaration.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_orbit_reference(declaration)
    assert report["status"] == "fail"
    assert any(error["field"] == "banana_width_relative_error" for error in report["errors"])


@pytest.mark.parametrize("block", ["metrics", "tolerances"])
@pytest.mark.parametrize("number", [True, "1", None, -1, 2, 10**400, float("inf")])
def test_classification_score_refusals(declaration: Path, block: str, number: object) -> None:
    """Classification observations and minima must be binary64-representable within inclusive [0,1]."""
    payload: dict[str, object] = json.loads(declaration.read_text())
    field = "passing_trapped_classification_accuracy" + ("_min" if block == "tolerances" else "")
    cast(dict[str, object], payload[block])[field] = number
    declaration.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_orbit_reference(declaration)
    assert report["status"] == "fail"
    assert any(error["field"] == "passing_trapped_classification_accuracy" for error in report["errors"])


@pytest.mark.parametrize("mode", ["equal", "zero-error", "zero-bound", "below-minimum"])
def test_declared_comparison_boundaries(declaration: Path, mode: str) -> None:
    """Exact error/score equality and zero errors pass; zero error bounds or scores below minima fail."""
    payload: dict[str, object] = json.loads(declaration.read_text())
    metrics = cast(dict[str, object], payload["metrics"])
    tolerances = cast(dict[str, object], payload["tolerances"])
    metrics["banana_width_relative_error"] = tolerances["banana_width_relative_error"]
    metrics["passing_trapped_classification_accuracy"] = tolerances["passing_trapped_classification_accuracy_min"]
    if mode == "zero-error":
        metrics["banana_width_relative_error"] = 0
        metrics["passing_trapped_classification_accuracy"] = 0
        tolerances["passing_trapped_classification_accuracy_min"] = 0
    elif mode == "zero-bound":
        tolerances["banana_width_relative_error"] = 0
    elif mode == "below-minimum":
        metrics["passing_trapped_classification_accuracy"] = 0.89
    declaration.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_orbit_reference(declaration)
    assert report["status"] == ("pass" if mode in {"equal", "zero-error"} else "fail")


@pytest.mark.parametrize("citation", [None, "", " ", "arbitrary nonblank citation"])
def test_public_citation_presence_limits(declaration: Path, citation: str | None) -> None:
    """Citation presence alone is accepted; no DOI resolution, referenced checksum or metric recomputation occurs."""
    payload: dict[str, object] = json.loads(declaration.read_text())
    payload["reference_doi"] = citation
    payload["reference_artifact_sha256"] = "A" * 64
    declaration.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_orbit_reference(declaration)
    assert report["status"] == ("pass" if citation and citation.strip() else "fail")


@pytest.mark.parametrize("uri", ["https://[", "https://example.invalid/a\0b", None])
def test_campaign_uri_findings(declaration: Path, uri: str | None) -> None:
    """The actual campaign branch retains shared lexical URI refusals at its public field."""
    payload: dict[str, object] = json.loads(declaration.read_text())
    payload["source"] = "real_orbit_campaign"
    payload["campaign_artifact_uri"] = uri
    declaration.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_orbit_reference(declaration)
    assert report["status"] == "fail"
    assert any(error["field"] == "campaign_artifact_uri" for error in report["errors"])


@pytest.mark.parametrize("token", ["1e-400", "-1e-400", "0.0", "-0.0", "0e-400", "5e-324"])
def test_decimal_underflow_cannot_fabricate_zero(tmp_path: Path, token: str) -> None:
    """Actual JSON decoding rejects nonzero tokens rounded to zero, while retaining exact zeros and subnormals."""
    path = tmp_path / "input.json"
    encoded = json.dumps(_valid_orbit_reference_artifact())
    assert '"banana_width_relative_error": 0.015' in encoded
    path.write_text(encoded.replace('"banana_width_relative_error": 0.015', '"banana_width_relative_error": ' + token))
    report = validate_orbit_reference(path)
    if token in {"1e-400", "-1e-400"}:
        assert report["status"] == "fail" and report["errors"][0]["field"] == "json"
    else:
        assert report["status"] == "pass" and report["reference_artifacts"] == 1
