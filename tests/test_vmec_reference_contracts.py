# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Persisted VMEC-reference declaration contract tests.

"""Exercise declaration findings through the real persisted public reader, without equilibrium qualification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from test_vmec_reference_validation import _valid_vmec_reference_artifact

from validation.validate_vmec_reference import validate_vmec_reference


@pytest.fixture
def declaration(tmp_path: Path) -> Path:
    """Persist the original metadata carrier; its citation, bytes and VMEC execution are unauthenticated."""
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(_valid_vmec_reference_artifact()), encoding="utf-8")
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
    report = validate_vmec_reference(declaration, require_reference_artifacts=True)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    assert any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize("block", ["metrics", "tolerances"])
@pytest.mark.parametrize("number", [True, "1", None, -1, 10**400, float("inf"), float("nan")])
def test_declared_error_number_refusals(declaration: Path, block: str, number: object) -> None:
    """Persisted errors and bounds reject booleans, wrong types, negative/nonfinite and overflowing numbers."""
    payload: dict[str, object] = json.loads(declaration.read_text())
    cast(dict[str, object], payload[block])["surface_R_rmse_m"] = number
    declaration.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_vmec_reference(declaration)
    assert report["status"] == "fail"
    assert any(error["field"] == "surface_R_rmse_m" for error in report["errors"])


@pytest.mark.parametrize("mode", ["equal", "zero-error", "zero-bound"])
def test_declared_comparison_boundaries(declaration: Path, mode: str) -> None:
    """Exact declared error equality and zero errors pass; zero error bounds fail."""
    payload: dict[str, object] = json.loads(declaration.read_text())
    metrics = cast(dict[str, object], payload["metrics"])
    tolerances = cast(dict[str, object], payload["tolerances"])
    metrics["surface_R_rmse_m"] = tolerances["surface_R_rmse_m"]
    if mode == "zero-error":
        metrics["surface_R_rmse_m"] = 0
    elif mode == "zero-bound":
        tolerances["surface_R_rmse_m"] = 0
    declaration.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_vmec_reference(declaration)
    assert report["status"] == ("pass" if mode in {"equal", "zero-error"} else "fail")


@pytest.mark.parametrize("citation", [None, "", " ", "arbitrary nonblank citation"])
def test_public_citation_presence_limits(declaration: Path, citation: str | None) -> None:
    """Citation presence alone is accepted; no DOI resolution, referenced checksum or metric recomputation occurs."""
    payload: dict[str, object] = json.loads(declaration.read_text())
    payload["reference_doi"] = citation
    payload["reference_artifact_sha256"] = "A" * 64
    declaration.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_vmec_reference(declaration)
    assert report["status"] == ("pass" if citation and citation.strip() else "fail")


@pytest.mark.parametrize("uri", ["https://[", "https://example.invalid/a\0b", None])
def test_campaign_uri_findings(declaration: Path, uri: str | None) -> None:
    """The actual campaign branch retains shared lexical URI refusals at its public field."""
    payload: dict[str, object] = json.loads(declaration.read_text())
    payload["source"] = "real_vmec_run"
    payload["vmec_artifact_uri"] = uri
    declaration.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_vmec_reference(declaration)
    assert report["status"] == "fail"
    assert any(error["field"] == "vmec_artifact_uri" for error in report["errors"])


@pytest.mark.parametrize("token", ["1e-400", "-1e-400", "0.0", "-0.0", "0e-400", "5e-324"])
def test_decimal_underflow_cannot_fabricate_zero(tmp_path: Path, token: str) -> None:
    """Actual JSON decoding rejects nonzero tokens rounded to zero, while retaining exact zeros and subnormals."""
    path = tmp_path / "input.json"
    encoded = json.dumps(_valid_vmec_reference_artifact())
    assert '"surface_R_rmse_m": 0.008' in encoded
    path.write_text(encoded.replace('"surface_R_rmse_m": 0.008', '"surface_R_rmse_m": ' + token))
    report = validate_vmec_reference(path)
    if token in {"1e-400", "-1e-400"}:
        assert report["status"] == "fail" and report["errors"][0]["field"] == "json"
    else:
        assert report["status"] == "pass" and report["reference_artifacts"] == 1


@pytest.mark.parametrize("field", ["surface_R_rmse_m", "surface_Z_rmse_m", "iota_rmse", "force_residual_relative"])
def test_each_declared_equilibrium_error(declaration: Path, field: str) -> None:
    """All four declared geometry/iota/residual errors must respect their own bound without numerical recomputation."""
    payload: dict[str, object] = json.loads(declaration.read_text())
    cast(dict[str, object], payload["metrics"])[field] = 2
    declaration.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_vmec_reference(declaration)
    assert report["status"] == "fail" and report["errors"][0]["field"] == field


@pytest.mark.parametrize(
    "value",
    [
        [],
        {},
        {"m_pol": True, "n_tor": 2, "n_fp": 5},
        {"m_pol": 0, "n_tor": 2, "n_fp": 5},
        {"m_pol": 3, "n_tor": True, "n_fp": 5},
        {"m_pol": 3, "n_tor": -1, "n_fp": 5},
        {"m_pol": 3, "n_tor": 2, "n_fp": False},
        {"m_pol": 3, "n_tor": 2, "n_fp": 0},
    ],
)
def test_declared_fourier_refusals(declaration: Path, value: object) -> None:
    """Persisted Fourier truncation requires exact nonboolean integer domains for each named field."""
    payload: dict[str, object] = json.loads(declaration.read_text())
    payload["fourier_truncation"] = value
    declaration.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_vmec_reference(declaration)
    assert report["status"] == "fail" and report["errors"][0]["field"] == "fourier_truncation"


def test_zero_toroidal_mode_is_admitted(declaration: Path) -> None:
    """A declared axisymmetric truncation with n_tor=0 remains accepted as metadata, with no geometry solved."""
    payload: dict[str, object] = json.loads(declaration.read_text())
    cast(dict[str, object], payload["fourier_truncation"])["n_tor"] = 0
    declaration.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_vmec_reference(declaration)["status"] == "pass"
