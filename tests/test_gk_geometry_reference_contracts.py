# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK geometry reference validation tests

"""Check actual reference header/case structure, exact-byte custody and bounded report metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_gk_geometry_reference_domains import REFERENCE, declaration, inspect

from validation.validate_gk_geometry_independent import validate_gk_geometry_independent
from validation.validate_gk_geometry_reference import validate_gk_geometry_reference

HEADERS = ("spdx_license_id", "commercial_license", "concepts_copyright", "code_copyright", "orcid", "contact", "file")


@pytest.mark.parametrize("field", HEADERS)
@pytest.mark.parametrize("value", [None, [], "", " "])
def test_header_presence(tmp_path: Path, field: str, value: object) -> None:
    """Seven original canonical-header declarations require nonblank text without identity authentication."""
    payload = declaration()
    payload[field] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize("value", [None, [], {}, [None], [{"case": []}], [{"case": " "}]])
def test_case_structure(tmp_path: Path, value: object) -> None:
    """Empty/nonarray cases and malformed case objects/names are fixed findings."""
    payload = declaration()
    payload["cases"] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and report["cases"] == 0
    assert report["public_claims"]["full_equilibrium_reconstruction"] is False


@pytest.mark.parametrize(
    ("field", "value"), [("parameters", []), ("sample_points", []), ("sample_points", {}), ("sample_points", [None])]
)
def test_case_blocks(tmp_path: Path, field: str, value: object) -> None:
    """Case parameter/sample blocks preserve their original object/nonempty-array contracts."""
    payload = declaration()
    payload["cases"][0][field] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and any(error["field"] == field for error in report["errors"])


def test_original_header_schema_and_required_cases(tmp_path: Path) -> None:
    """Input schema1.0 and each required named geometry case remain separate from output reportv2 metadata."""
    payload = declaration()
    payload["schema_version"] = "wrong"
    assert inspect(tmp_path, payload)["errors"][0]["field"] == "schema_version"
    payload = declaration()
    payload["cases"].pop()
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and report["cases"] == 2
    assert any("high_shear_local_equilibrium" in error["error"] for error in report["errors"])


def test_exact_CRLF_byte_custody(tmp_path: Path) -> None:
    """Reference SHA binds actual read bytes including CRLF; original numeric cases and fixed metadata still pass."""
    path = tmp_path / "reference.json"
    raw = REFERENCE.read_bytes().replace(b"\n", b"\r\n")
    path.write_bytes(raw)
    report = validate_gk_geometry_reference(path)
    assert report["status"] == "pass" and report["cases"] == 3
    assert report["reference_file_sha256"] == hashlib.sha256(raw).hexdigest()
    assert report["reference_file_sha256"] != hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    assert report["tolerances"] == {"absolute": 1e-11, "relative": 1e-10}
    assert report["schema_version"] == "scpn-control.gk-geometry-reference.v2"
    assert len(report["payload_sha256"]) == 64
    assert report["public_claims"]["bounded_local_miller_geometry_reference"] is True
    assert report["public_claims"]["full_equilibrium_reconstruction"] is False


@pytest.mark.parametrize("raw", [b"[]", b"null", b"{broken", b"\xff", b'{"secret":1,"secret":2}', b'{"x":1e-400}'])
def test_actual_decode_findings(tmp_path: Path, raw: bytes) -> None:
    """Root/JSON/encoding/duplicate/nonrepresentable decimal failures keep fixed findings without key/parser leakage."""
    path = tmp_path / "input.json"
    path.write_bytes(raw)
    report = validate_gk_geometry_reference(path)
    assert report["status"] == "fail" and report["cases"] == 0
    error = report["errors"][0]
    assert error["field"] == ("root" if raw in {b"[]", b"null"} else "json")
    assert "secret" not in error["error"] and "decode" not in error["error"]


@pytest.mark.parametrize("token", ["0.0", "-0.0", "0e-400", "5e-324"])
def test_zero_and_subnormal_decode_domains(tmp_path: Path, token: str) -> None:
    """Exact zero and representable subnormal tokens remain decodable without constraining ignored reference metadata."""
    payload = declaration()
    encoded = json.dumps(payload)[:-1] + ',"unused":' + token + "}"
    path = tmp_path / "input.json"
    path.write_text(encoded)
    assert validate_gk_geometry_reference(path)["status"] == "pass"


def test_metric_units_match_both_actual_public_comparisons(tmp_path: Path) -> None:
    """Both real Miller comparisons report reciprocal metric dimensions and a metre 2-D Jacobian."""
    stored = inspect(tmp_path, declaration())
    independent = validate_gk_geometry_independent()
    for report, count in [(stored, 3), (independent, 5)]:
        assert report["status"] == "pass" and report["cases"] == count
        assert report["units"]["jacobian"] == "m"
        assert report["units"]["g_rt"] == "m-1"
        assert report["units"]["g_tt"] == "m-2"
        assert report["public_claims"]["full_equilibrium_reconstruction"] is False
    assert all(entry["samples"] == 128 for entry in independent["entries"])
    assert all(field["agrees"] for entry in independent["entries"] for field in entry["fields"].values())
