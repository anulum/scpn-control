# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Persisted uncertainty-reference declaration contract tests.

"""Exercise declaration findings through the real persisted public reader, without UQ propagation qualification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from test_uncertainty_reference_validation import _valid_uncertainty_reference_artifact

from validation.validate_uncertainty_reference import validate_uncertainty_reference


@pytest.fixture
def declaration(tmp_path: Path) -> Path:
    """Persist the original metadata carrier; its citation, bytes and UQ execution are unauthenticated."""
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(_valid_uncertainty_reference_artifact()), encoding="utf-8")
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
    report = validate_uncertainty_reference(declaration, require_reference_artifacts=True)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    assert any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize("block", ["metrics", "tolerances"])
@pytest.mark.parametrize("number", [True, "1", None, -1, 10**400, float("inf"), float("nan")])
def test_declared_error_number_refusals(declaration: Path, block: str, number: object) -> None:
    """Persisted errors and bounds reject booleans, wrong types, negative/nonfinite and overflowing numbers."""
    payload: dict[str, object] = json.loads(declaration.read_text())
    cast(dict[str, object], payload[block])["tau_E_relative_error"] = number
    declaration.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_uncertainty_reference(declaration)
    assert report["status"] == "fail"
    assert any(error["field"] == "tau_E_relative_error" for error in report["errors"])


@pytest.mark.parametrize("block", ["metrics", "tolerances"])
@pytest.mark.parametrize("number", [True, "1", None, -1, 2, 10**400, float("inf")])
def test_monotonicity_score_refusals(declaration: Path, block: str, number: object) -> None:
    """Monotonicity observations and minima must be binary64-representable within inclusive [0,1]."""
    payload: dict[str, object] = json.loads(declaration.read_text())
    field = "percentile_monotonicity_fraction" + ("_min" if block == "tolerances" else "")
    cast(dict[str, object], payload[block])[field] = number
    declaration.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_uncertainty_reference(declaration)
    assert report["status"] == "fail"
    assert any(error["field"] == "percentile_monotonicity_fraction" for error in report["errors"])


@pytest.mark.parametrize("mode", ["equal", "zero-error", "zero-bound", "below-minimum"])
def test_declared_comparison_boundaries(declaration: Path, mode: str) -> None:
    """Exact error/score equality and zero errors pass; zero error bounds or scores below minima fail."""
    payload: dict[str, object] = json.loads(declaration.read_text())
    metrics = cast(dict[str, object], payload["metrics"])
    tolerances = cast(dict[str, object], payload["tolerances"])
    metrics["tau_E_relative_error"] = tolerances["tau_E_relative_error"]
    metrics["percentile_monotonicity_fraction"] = tolerances["percentile_monotonicity_fraction_min"]
    if mode == "zero-error":
        metrics["tau_E_relative_error"] = 0
        metrics["percentile_monotonicity_fraction"] = 0
        tolerances["percentile_monotonicity_fraction_min"] = 0
    elif mode == "zero-bound":
        tolerances["tau_E_relative_error"] = 0
    elif mode == "below-minimum":
        metrics["percentile_monotonicity_fraction"] = 0.89
    declaration.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_uncertainty_reference(declaration)
    assert report["status"] == ("pass" if mode in {"equal", "zero-error"} else "fail")


@pytest.mark.parametrize("citation", [None, "", " ", "arbitrary nonblank citation"])
def test_public_citation_presence_limits(declaration: Path, citation: str | None) -> None:
    """Citation presence alone is accepted; no DOI resolution, referenced checksum or metric recomputation occurs."""
    payload: dict[str, object] = json.loads(declaration.read_text())
    payload["reference_doi"] = citation
    payload["reference_artifact_sha256"] = "A" * 64
    declaration.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_uncertainty_reference(declaration)
    assert report["status"] == ("pass" if citation and citation.strip() else "fail")


@pytest.mark.parametrize("uri", [None, "", " ", [], "https://[", "arbitrary nonblank location"])
def test_campaign_reference_presence_contract(declaration: Path, uri: object) -> None:
    """Retain the campaign nonblank-string contract explicitly without granting URI or byte authenticity."""
    payload: dict[str, object] = json.loads(declaration.read_text())
    payload["source"] = "real_uq_campaign"
    payload["campaign_artifact_uri"] = uri
    declaration.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_uncertainty_reference(declaration)
    accepted = isinstance(uri, str) and bool(uri.strip())
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == "campaign_artifact_uri" for error in report["errors"])


@pytest.mark.parametrize("token", ["1e-400", "-1e-400", "0.0", "-0.0", "0e-400", "5e-324"])
def test_decimal_underflow_cannot_fabricate_zero(tmp_path: Path, token: str) -> None:
    """Actual JSON decoding rejects nonzero tokens rounded to zero, while retaining exact zeros and subnormals."""
    path = tmp_path / "input.json"
    encoded = json.dumps(_valid_uncertainty_reference_artifact())
    assert '"tau_E_relative_error": 0.025' in encoded
    path.write_text(encoded.replace('"tau_E_relative_error": 0.025', '"tau_E_relative_error": ' + token))
    report = validate_uncertainty_reference(path)
    if token in {"1e-400", "-1e-400"}:
        assert report["status"] == "fail" and report["errors"][0]["field"] == "json"
    else:
        assert report["status"] == "pass" and report["reference_artifacts"] == 1


@pytest.mark.parametrize("field", ["tau_E_relative_error", "P_fusion_relative_error", "Q_relative_error"])
def test_each_declared_output_error_bound(declaration: Path, field: str) -> None:
    """Each of the three declared output errors must respect its own positive bound."""
    payload: dict[str, object] = json.loads(declaration.read_text())
    cast(dict[str, object], payload["metrics"])[field] = 2
    declaration.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_uncertainty_reference(declaration)
    assert report["status"] == "fail" and report["errors"][0]["field"] == field
