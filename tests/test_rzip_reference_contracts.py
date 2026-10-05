# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — RZIP identity and provenance tests

"""Check original identity and distinct source URI policies through the public RZIP reader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_rzip_reference_domains import declaration

from validation.validate_rzip_reference import validate_rzip_reference


@pytest.mark.parametrize(
    "field", ["source", "model_id", "model_version", "reference_dataset_id", "reference_artifact_sha256", "executed_at"]
)
@pytest.mark.parametrize("value", [None, [], {}, False, "", " "])
def test_required_strings(tmp_path: Path, field: str, value: object) -> None:
    """Required identity strings reject missing/nonstring/blank values without unhashable-source exceptions."""
    payload = declaration()
    payload[field] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_rzip_reference(path)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    assert any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 1),
        ("source", "synthetic"),
        ("reference_artifact_sha256", "a" * 64 + "\n"),
        ("reference_artifact_sha256", "é" * 64),
        ("reference_artifact_sha256", "z" * 64),
        ("units", []),
        ("physical_parameters", []),
        ("metrics", []),
        ("tolerances", []),
    ],
)
def test_contract_findings(tmp_path: Path, field: str, value: object) -> None:
    """Schema, complete digest, required units and input/metric blocks yield authored findings."""
    payload = declaration()
    payload[field] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_rzip_reference(path)
    assert report["status"] == "fail" and any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize("field", ["vertical_displacement", "growth_rate", "growth_time", "coil_current", "time"])
def test_original_unit_labels(tmp_path: Path, field: str) -> None:
    """Original m, s^-1, ms, A and s labels are required independently of numerical conversion."""
    payload = declaration()
    payload["units"][field] = "wrong"
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    assert validate_rzip_reference(path)["errors"][0]["field"] == "units"


@pytest.mark.parametrize("source", ["documented_public_reference", "measured_discharge"])
@pytest.mark.parametrize("value", [None, [], " ", "../presence-only\x00"])
def test_source_presence(tmp_path: Path, source: str, value: object) -> None:
    """Public citations and measured diagnostics retain nonblank presence checks, distinct from external URI policy."""
    payload = declaration()
    payload.update(source=source, shot_id="declared-shot")
    payload.pop("reference_doi")
    field = "reference_url" if source == "documented_public_reference" else "diagnostic_uri"
    payload[field] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    assert validate_rzip_reference(path)["status"] == ("pass" if isinstance(value, str) and value.strip() else "fail")


@pytest.mark.parametrize("value", [None, [], {}, " ", "UNKNOWN", "CREATE-L", "CREATE-NL", "TSC", "reference_rzip"])
def test_external_code_membership(tmp_path: Path, value: object) -> None:
    """Original named external string codes are admitted; collection values yield findings."""
    payload = declaration()
    payload.update(
        source="external_code_benchmark",
        external_code=value,
        reference_artifact_uri="https://declared.invalid/reference",
    )
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    accepted = isinstance(value, str) and value in {"CREATE-L", "CREATE-NL", "TSC", "reference_rzip"}
    assert validate_rzip_reference(path)["status"] == ("pass" if accepted else "fail")


@pytest.mark.parametrize(
    ("uri", "accepted"),
    [
        (None, False),
        ([], False),
        ("../presence-only", False),
        ("https://[", False),
        ("https://host/../ref", False),
        ("https://host/ref\x00", False),
        ("file://host/validation/reports/ref", False),
        ("file:///etc/ref", False),
        ("file:///validation/reports/../ref", False),
        ("ftp://host/ref", False),
        ("https://host", False),
        ("file:///validation/reports/ref", True),
        ("file:///validation/reference_data/ref", True),
        ("https://host/ref", True),
        ("s3://bucket/ref", True),
        ("gs://bucket/ref", True),
        ("https://host/%2e%2e/ref?token=declared#fragment", True),
    ],
)
def test_external_uri_policy(tmp_path: Path, uri: object, accepted: bool) -> None:
    """Actual persisted external declarations enforce the original shared lexical policy without fetching bytes."""
    payload = declaration()
    payload.update(source="external_code_benchmark", external_code="CREATE-L", reference_artifact_uri=uri)
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_rzip_reference(path)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == "reference_artifact_uri" for error in report["errors"])


def test_measured_shot_and_public_alternative(tmp_path: Path) -> None:
    """Measured references require shot identity; public references allow either nonblank URL or DOI."""
    payload = declaration()
    payload.update(source="measured_discharge", diagnostic_uri="mdsplus://declared/shot", shot_id=[])
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    assert validate_rzip_reference(path)["errors"][0]["field"] == "shot_id"
    payload["shot_id"] = "declared-shot"
    path.write_text(json.dumps(payload))
    assert validate_rzip_reference(path)["status"] == "pass"
    payload["source"] = "documented_public_reference"
    payload.pop("reference_doi")
    payload["reference_url"] = "presence-only"
    path.write_text(json.dumps(payload))
    assert validate_rzip_reference(path)["status"] == "pass"


@pytest.mark.parametrize("token", ["1e-400", "-1e-400", "0.0", "-0.0", "0e-400", "5e-324"])
def test_decimal_underflow_refusal(tmp_path: Path, token: str) -> None:
    """Nonzero decimal errors collapsing to binary64 zero refuse; zero and representable subnormals pass."""
    encoded = json.dumps(declaration())
    needle = '"growth_rate_relative_error": 0.0'
    assert needle in encoded
    path = tmp_path / "input.json"
    path.write_text(encoded.replace(needle, '"growth_rate_relative_error": ' + token, 1))
    report = validate_rzip_reference(path)
    assert report["status"] == ("fail" if token in {"1e-400", "-1e-400"} else "pass")
    if report["status"] == "fail":
        assert report["errors"][0]["field"] == "json"
