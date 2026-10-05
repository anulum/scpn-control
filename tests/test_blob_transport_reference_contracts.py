# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Blob transport identity and body integrity tests

"""Check original blob identity, lexical URIs, campaign presence and canonical body consistency through public APIs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_blob_transport_reference_domains import declaration, inspect

from validation.validate_blob_transport_reference import canonical_artifact_sha256, validate_blob_transport_reference

IDENTITY_FIELDS = (
    "source",
    "reference_dataset_id",
    "executed_at",
    "reference_artifact_uri",
    "profile_artifact_uri",
    "detector_artifact_uri",
    "reference_artifact_sha256",
    "profile_artifact_sha256",
    "detector_artifact_sha256",
    "payload_sha256",
)
BYTE_SHA_FIELDS = ("reference_artifact_sha256", "profile_artifact_sha256", "detector_artifact_sha256")
URI_FIELDS = ("reference_artifact_uri", "profile_artifact_uri", "detector_artifact_uri")


@pytest.mark.parametrize("field", IDENTITY_FIELDS)
@pytest.mark.parametrize("value", [None, [], {}, False, "", " "])
def test_required_strings(tmp_path: Path, field: str, value: object) -> None:
    """All ten original nonblank identity fields refuse malformed values without set/digest exceptions."""
    payload = declaration()
    payload[field] = value
    if field != "payload_sha256":
        report = inspect(tmp_path, payload)
    else:
        path = tmp_path / "input.json"
        path.write_text(json.dumps(payload))
        report = validate_blob_transport_reference(path)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    assert any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "1.0"),
        ("source", "synthetic"),
        ("units", []),
        ("magnetic_geometry", []),
        ("metrics", []),
        ("tolerances", []),
    ],
)
def test_contract_findings(tmp_path: Path, field: str, value: object) -> None:
    """Original named schema, source membership, units and geometry/metric block errors remain findings."""
    payload = declaration()
    payload[field] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize("field", BYTE_SHA_FIELDS)
@pytest.mark.parametrize(
    ("value", "accepted"),
    [
        ("a" * 64 + "\n", False),
        ("é" * 64, False),
        ("z" * 64, False),
        ("a" * 63, False),
        ("A" * 64, True),
    ],
)
def test_format_only_byte_hashes(tmp_path: Path, field: str, value: str, accepted: bool) -> None:
    """Three reference-byte hashes require exact hex format, without fetching or authenticating referenced bytes."""
    payload = declaration()
    payload[field] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize("value", ["é" * 64, "\ud800" * 64, "a" * 64 + "\n", "z" * 64, "0" * 64])
def test_malformed_and_mismatched_body_hashes(tmp_path: Path, value: str) -> None:
    """Non-ASCII/malformed body digests refuse before comparison; well-formed wrong hashes remain consistency findings."""
    payload = declaration()
    payload["payload_sha256"] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_blob_transport_reference(path)
    assert report["status"] == "fail"
    assert any(error["field"] == "payload_sha256" for error in report["errors"])
    assert all("é" not in error["error"] and "\ud800" not in error["error"] for error in report["errors"])


@pytest.mark.parametrize("field", URI_FIELDS)
@pytest.mark.parametrize(
    ("value", "accepted"),
    [
        ("relative/data.npz", True),
        (".", True),
        ("file:///unfetched/data", True),
        ("https://", True),
        ("http://", True),
        ("doi:", True),
        ("s3://", True),
        ("gs://", True),
        ("/absolute/data", False),
        ("../data", False),
        ("relative/../data", False),
        ("relative\x00data", False),
    ],
)
def test_original_lexical_uri_policy(tmp_path: Path, field: str, value: str, accepted: bool) -> None:
    """Each original URI field retains lexical prefix/path rules including bare prefixes and relative-looking file URI text."""
    payload = declaration()
    payload[field] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize(
    "field", ["radius", "time", "velocity", "density", "temperature", "magnetic_field", "wall_flux"]
)
def test_original_unit_labels(tmp_path: Path, field: str) -> None:
    """All seven original SOL unit labels remain mandatory without numerical conversion."""
    payload = declaration()
    payload["units"][field] = "wrong"
    assert inspect(tmp_path, payload)["errors"][0]["field"] == "units"


@pytest.mark.parametrize("field", ["machine", "shot_id"])
@pytest.mark.parametrize("value", [None, [], " "])
def test_measured_campaign_identity(tmp_path: Path, field: str, value: object) -> None:
    """Measured provenance needs nonblank machine and at least one shot/campaign identity."""
    payload = declaration()
    payload[field] = value
    assert inspect(tmp_path, payload)["errors"][0]["field"] == "campaign"


def test_campaign_alternative_and_public_presence(tmp_path: Path) -> None:
    """Campaign can replace shot; public URL or DOI presence remains unparsed and unauthenticated."""
    payload = declaration()
    payload.pop("shot_id")
    payload["campaign_id"] = "declared-campaign"
    assert inspect(tmp_path, payload)["status"] == "pass"
    payload["source"] = "documented_public_reference"
    assert inspect(tmp_path, payload)["errors"][0]["field"] == "reference"
    payload["reference_url"] = "../unparsed\x00presence"
    assert inspect(tmp_path, payload)["status"] == "pass"
    payload.pop("reference_url")
    payload["reference_doi"] = "presence-only"
    assert inspect(tmp_path, payload)["status"] == "pass"


def test_canonical_body_algorithm_and_uppercase_digest(tmp_path: Path) -> None:
    """Public canonical hashing preserves sorted compact ASCII JSON, excludes its own field and normalizes admitted uppercase digests."""
    expected_vector = "d21e2672702dd4740fcf206124917de268edf17bdc68715ae0e71e3558c5c746"
    assert canonical_artifact_sha256({"value": 1, "extra": "ž", "payload_sha256": "ignored"}) == expected_vector
    payload = declaration()
    expected = payload["payload_sha256"]
    payload["payload_sha256"] = expected.upper()
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_blob_transport_reference(path)
    assert report["status"] == "pass" and report["entries"][0]["payload_sha256"] == expected


@pytest.mark.parametrize("token", ["1e-400", "-1e-400", "0.0", "-0.0", "0e-400", "5e-324"])
def test_decimal_underflow_refusal(tmp_path: Path, token: str) -> None:
    """Nonzero decimal error tokens collapsed to zero refuse before canonical hashing; exact zero and subnormal remain admitted."""
    payload = declaration()
    payload["metrics"]["wall_flux_relative_error"] = float(token)
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    needle = '"wall_flux_relative_error": ' + str(float(token))
    encoded = json.dumps(payload)
    assert needle in encoded
    path = tmp_path / "input.json"
    path.write_text(encoded.replace(needle, '"wall_flux_relative_error": ' + token, 1))
    report = validate_blob_transport_reference(path)
    assert report["status"] == ("fail" if token in {"1e-400", "-1e-400"} else "pass")
    if report["status"] == "fail":
        assert report["errors"][0]["field"] == "json"
