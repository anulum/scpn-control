# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK species reference validation tests

"""Check actual species reference shape, exact-byte provenance and bounded consistency digest."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from test_gk_species_reference_domains import REFERENCE, declaration, inspect

from validation.validate_gk_species_reference import validate_gk_species_reference, verify_payload_digest


@pytest.mark.parametrize(
    "field",
    ("spdx_license_id", "commercial_license", "concepts_copyright", "code_copyright", "orcid", "contact", "file"),
)
@pytest.mark.parametrize("value", [None, [], "", " "])
def test_header_declarations(tmp_path: Path, field: str, value: object) -> None:
    """Original header fields require nonblank text without authenticating authorship."""
    payload = declaration()
    payload[field] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize("value", [None, [], {}, [None], [{"case": []}], [{"case": " "}]])
def test_case_shapes(tmp_path: Path, value: object) -> None:
    """Case arrays and object/name declarations produce deterministic findings."""
    payload = declaration()
    payload["cases"] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and report["cases"] == 0


@pytest.mark.parametrize("block", ["species", "collision", "drive", "expected"])
def test_case_block_shapes(tmp_path: Path, block: str) -> None:
    """Each numerical input block is an object before its fields are compared."""
    payload = declaration()
    payload["cases"][0][block] = []
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and any(error["field"] == block for error in report["errors"])


def test_input_schema_and_unit_correction(tmp_path: Path) -> None:
    """Schema1.0 remains separate from reportv3; fresh larmor coefficient has m*T units and unchanged numbers."""
    report = inspect(tmp_path, declaration())
    assert report["status"] == "pass" and report["schema_version"] == "scpn-control.gk-species-reference.v3"
    assert all(entry["units"]["larmor_radius_per_tesla_m"] == "m*T" for entry in report["entries"])
    assert verify_payload_digest(report)
    payload = declaration()
    payload["schema_version"] = "wrong"
    assert inspect(tmp_path, payload)["status"] == "fail"


@pytest.mark.parametrize(
    "raw", [b"[]", b"null", b"{broken", b"\xff", b'{"private_input_key":1,"private_input_key":2}', b'{"x":1e-400}']
)
def test_decode_findings(tmp_path: Path, raw: bytes) -> None:
    """Decode/root failures produce digest-valid authored findings without parser/key leakage."""
    path = tmp_path / "input.json"
    path.write_bytes(raw)
    report = validate_gk_species_reference(path)
    assert report["status"] == "fail" and report["cases"] == 0 and verify_payload_digest(report)
    assert "private_input_key" not in str(report["errors"])


def test_CRLF_input_identity(tmp_path: Path) -> None:
    """The input digest binds precisely the bytes inspected including CRLF line endings."""
    path = tmp_path / "input.json"
    raw = REFERENCE.read_bytes().replace(b"\n", b"\r\n")
    path.write_bytes(raw)
    report = validate_gk_species_reference(path)
    assert report["status"] == "pass"
    assert report["reference_sha256"] == hashlib.sha256(raw).hexdigest() and verify_payload_digest(report)


@pytest.mark.parametrize("digest", [None, [], "short", "f" * 64])
def test_invalid_digest_claims(tmp_path: Path, digest: object) -> None:
    """Missing/malformed/wrong digest declarations fail through the public report verifier."""
    report = inspect(tmp_path, declaration())
    report["payload_sha256"] = digest
    assert verify_payload_digest(report) is False


@pytest.mark.parametrize("value", [float("nan"), object(), pytest.param(10**5000, id="unencodable-integer")])
def test_unserializable_report_digest(tmp_path: Path, value: object) -> None:
    """A claimed digest over unserializable/nonfinite report values refuses rather than leaking encoder exceptions."""
    report = inspect(tmp_path, declaration())
    report["extra"] = value
    assert verify_payload_digest(report) is False
