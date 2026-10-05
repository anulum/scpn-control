# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MARFE declaration identity and provenance tests

"""Exercise original MARFE declaration contracts through persisted public inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_marfe_reference_validation import _valid_marfe_reference_artifact

from validation.validate_marfe_reference import canonical_artifact_sha256, validate_marfe_reference


@pytest.fixture
def declaration(tmp_path: Path) -> Path:
    """Persist original metadata only; no referenced MARFE experiment is supplied."""
    path = tmp_path / "input.json"
    path.write_text(json.dumps(_valid_marfe_reference_artifact()), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 1),
        ("source", []),
        ("source", {}),
        ("source", None),
        ("reference_dataset_id", " "),
        ("executed_at", False),
        ("units", []),
        ("units", {}),
        ("impurity", None),
        ("impurity", " "),
        ("impurity", []),
        ("temperature_profile_sha256", None),
        ("temperature_profile_sha256", "1" * 64 + "\n"),
        ("density_limit_sha256", "é" * 64),
        ("payload_sha256", "é" * 64),
        ("payload_sha256", None),
        ("payload_sha256", "1" * 64 + "\n"),
        ("radiation_curve_sha256", "no"),
        ("power_balance_sha256", ""),
        ("source", "synthetic"),
    ],
)
def test_identity_findings(declaration: Path, field: str, value: object) -> None:
    """Malformed field types, full digest strings and sources return findings without exceptions."""
    payload: dict[str, object] = json.loads(declaration.read_text())
    payload[field] = value
    if field != "payload_sha256":
        payload["payload_sha256"] = canonical_artifact_sha256(payload)
    declaration.write_text(json.dumps(payload))
    report = validate_marfe_reference(declaration, require_reference_artifacts=True)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    assert any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize(
    "field", ["temperature_profile_uri", "density_limit_uri", "radiation_curve_uri", "power_balance_uri"]
)
@pytest.mark.parametrize(
    "uri",
    [
        None,
        " ",
        "a\x00b",
        "/absolute/file",
        "a/../b",
        "https://[",
        "doi:any",
        "s3://",
        "gs://bucket",
        "./local/reference",
    ],
)
def test_lexical_uri_contract(declaration: Path, field: str, uri: str | None) -> None:
    """Retain admitted prefix and relative-path rules without claiming URI parsing or retrieval."""
    payload: dict[str, object] = json.loads(declaration.read_text())
    payload[field] = uri
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    declaration.write_text(json.dumps(payload))
    report = validate_marfe_reference(declaration)
    accepted = uri in {"https://[", "doi:any", "s3://", "gs://bucket", "./local/reference"}
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize("citation", [None, "", " ", "arbitrary nonblank reference"])
def test_public_citation_presence(declaration: Path, citation: str | None) -> None:
    """Public citation admission is presence only; no authenticity is inferred from its checksum."""
    payload: dict[str, object] = json.loads(declaration.read_text())
    payload["source"] = "documented_public_reference"
    payload["reference_url"] = citation
    payload.pop("machine")
    payload.pop("shot_id")
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    declaration.write_text(json.dumps(payload))
    report = validate_marfe_reference(declaration)
    assert report["status"] == ("pass" if citation and citation.strip() else "fail")


@pytest.mark.parametrize("machine", [None, " ", "JET"])
@pytest.mark.parametrize("identity", [None, " ", "campaign-label"])
def test_measured_campaign_presence(declaration: Path, machine: str | None, identity: str | None) -> None:
    """Machine and campaign alternative stay nonblank string declarations, without facility validation."""
    payload: dict[str, object] = json.loads(declaration.read_text())
    payload["machine"] = machine
    payload.pop("shot_id")
    payload["campaign_id"] = identity
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    declaration.write_text(json.dumps(payload))
    assert validate_marfe_reference(declaration)["status"] == (
        "pass" if machine == "JET" and identity == "campaign-label" else "fail"
    )


def test_canonical_serialization_and_uppercase_digest(declaration: Path) -> None:
    """Original sorted ASCII compact serialization and case-insensitive body checks remain exact."""
    payload: dict[str, object] = json.loads(declaration.read_text())
    payload["extra"] = "é"
    canonical = dict(payload)
    canonical.pop("payload_sha256")
    expected = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    assert canonical_artifact_sha256(payload) == expected
    payload["payload_sha256"] = expected.upper()
    declaration.write_text(json.dumps(payload))
    report = validate_marfe_reference(declaration)
    assert report["status"] == "pass" and report["entries"][0]["payload_sha256"] == expected
    assert canonical_artifact_sha256({"extra": "é"}) == canonical_artifact_sha256(
        {"extra": "é", "payload_sha256": "ignored"}
    )


@pytest.mark.parametrize("token", ["1e-400", "-1e-400", "0.0", "-0.0", "0e-400", "5e-324"])
def test_decimal_underflow_and_body_hash(tmp_path: Path, token: str) -> None:
    """Actual JSON decoding refuses nonzero tokens collapsed to zero before canonical checksum admission."""
    payload = _valid_marfe_reference_artifact()
    metrics = payload["metrics"]
    assert isinstance(metrics, dict)
    metrics["onset_temperature_relative_error"] = float(token)
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    encoded = json.dumps(payload)
    needle = '"onset_temperature_relative_error": ' + json.dumps(float(token))
    assert needle in encoded
    path = tmp_path / "input.json"
    path.write_text(encoded.replace(needle, '"onset_temperature_relative_error": ' + token))
    report = validate_marfe_reference(path)
    if token in {"1e-400", "-1e-400"}:
        assert report["status"] == "fail" and report["errors"][0]["field"] == "json"
    else:
        assert report["status"] == "pass"
