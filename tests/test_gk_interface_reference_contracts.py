# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK interface declaration contract behaviour

"""Exercise stored author declarations through the public reader without authenticating external runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from test_gk_interface_artifact_validation import _valid_real_executable_artifact

from validation.validate_gk_interface_artifacts import canonical_artifact_sha256, validate_gk_interface_artifacts


def declaration(**changes: Any) -> dict[str, Any]:
    """Seal a test declaration using an independent compact sorted ASCII JSON oracle."""
    payload = _valid_real_executable_artifact()
    payload.update(changes)
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    payload["payload_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return payload


def inspect(tmp_path: Path, payload: object) -> dict[str, Any]:
    """Persist exactly the supplied declaration and invoke the public required reader."""
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return validate_gk_interface_artifacts(path, require_interface_artifacts=True)


@pytest.mark.parametrize("payload", [None, [], 17, "declaration"])
def test_object_root_required(tmp_path: Path, payload: object) -> None:
    """Nonobject decoded roots produce a field finding instead of a declaration count."""
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and report["interface_artifacts"] == 0
    assert report["errors"][0]["field"] == "root"


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "interface_code",
        "source",
        "binary_path",
        "code_version",
        "run_id",
        "executed_at",
        "units",
        "input_deck_uri",
        "output_artifact_uri",
        "parsed_output_uri",
        "parser_version",
        "input_deck_sha256",
        "output_artifact_sha256",
        "parsed_output_sha256",
        "payload_sha256",
    ],
)
@pytest.mark.parametrize("value", [None, [], {}, True, "", "   "])
def test_required_string_fields(tmp_path: Path, field: str, value: object) -> None:
    """Malformed identities, including unhashable code names, return their field findings without crashing."""
    payload = declaration(**{field: value})
    if field == "payload_sha256":
        payload[field] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and report["interface_artifacts"] == 0
    assert field in {error["field"] for error in report["errors"]}


@pytest.mark.parametrize(
    "field",
    [
        "gamma_max_cs_over_a",
        "omega_r_cs_over_a",
        "k_y_rho_s_at_max",
        "chi_i_m2_s",
        "chi_e_m2_s",
        "D_e_m2_s",
    ],
)
@pytest.mark.parametrize("value", [True, False, None, [], "0.1", 10**400])
def test_numeric_type_and_representability(tmp_path: Path, field: str, value: object) -> None:
    """Every scalar refuses booleans, wrong types and conversion overflow through the public reader."""
    report = inspect(tmp_path, declaration(**{field: value}))
    assert report["status"] == "fail" and report["interface_artifacts"] == 0
    assert any(
        error["field"] == field and error["error"] == "field must be finite numeric" for error in report["errors"]
    )


@pytest.mark.parametrize(
    "field", ["input_deck_sha256", "output_artifact_sha256", "parsed_output_sha256", "payload_sha256"]
)
@pytest.mark.parametrize(
    "digest", ["+" + "a" * 63, " " + "a" * 63, "٠" * 64, "a" * 63, "a" * 65, "g" * 64, "a" * 64 + "\n", "é" * 64]
)
def test_ascii_digest_contract(tmp_path: Path, field: str, digest: str) -> None:
    """Signs, spaces, Unicode digits, length errors and nonhex characters are refused in all four digest fields."""
    payload = declaration(**{field: digest})
    if field == "payload_sha256":
        payload[field] = digest
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and report["interface_artifacts"] == 0
    assert any(
        error["field"] == field and error["error"] == "field must be a SHA-256 hex digest" for error in report["errors"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chi_i_m2_s", -1),
        ("chi_e_m2_s", -1),
        ("D_e_m2_s", -1),
        ("gamma_max_cs_over_a", -1),
        ("k_y_rho_s_at_max", 0),
        ("k_y_rho_s_at_max", -1),
        ("source", "mock_subprocess"),
        ("units", "Hz"),
        ("interface_code", "unsupported"),
        ("schema_version", "v2"),
        ("binary_path", "gene"),
        ("binary_path", "/tmp/gene"),
    ],
)
def test_original_scalar_source_unit_domains(tmp_path: Path, field: str, value: object) -> None:
    """Keep original nonnegative transport/growth, positive wavenumber and lexical source contracts."""
    report = inspect(tmp_path, declaration(**{field: value}))
    assert report["status"] == "fail" and report["interface_artifacts"] == 0
    assert any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize("code", ["TGLF", "GENE", "GS2", "CGYRO", "QuaLiKiz"])
def test_declared_codes_and_zero_signed_numeric_domains(tmp_path: Path, code: str) -> None:
    """Original code labels, nonnegative endpoint zero and signed frequencies remain metadata-admissible."""
    payload = declaration(
        interface_code=code,
        chi_i_m2_s=0,
        chi_e_m2_s=0,
        D_e_m2_s=0,
        gamma_max_cs_over_a=0,
        omega_r_cs_over_a=-1e100,
        input_deck_sha256="A" * 64,
        unused="ž",
    )
    payload["payload_sha256"] = payload["payload_sha256"].upper()
    report = inspect(tmp_path, payload)
    assert report["status"] == "pass" and report["interface_artifacts"] == 1
    assert report["entries"][0]["payload_sha256"] == payload["payload_sha256"].lower()
    assert report["public_claims"]["external_interface_artifacts_admitted"] is True
    assert report["public_claims"]["full_gk_cross_code_claim_admitted"] is False


@pytest.mark.parametrize("field", ["input_deck_uri", "output_artifact_uri", "parsed_output_uri"])
@pytest.mark.parametrize("uri", [None, [], "", " ", "private\0name", "/absolute", "a/../b"])
def test_original_uri_refusals(tmp_path: Path, field: str, uri: object) -> None:
    """Each provenance URI preserves original nonblank, NUL, absolute and traversal refusals."""
    report = inspect(tmp_path, declaration(**{field: uri}))
    assert report["status"] == "fail" and report["interface_artifacts"] == 0
    assert any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize("uri", ["a/b", "http://", "https://reference", "doi:10.1/x", "s3://x", "gs://x"])
def test_original_uri_prefix_admission(tmp_path: Path, uri: str) -> None:
    """Keep original lexical URI-prefix semantics, including bare prefix, without fetching resources."""
    assert inspect(tmp_path, declaration(input_deck_uri=uri))["status"] == "pass"


@pytest.mark.parametrize(
    ("reference", "value", "accepted"),
    [
        ("reference_url", "https://author", True),
        ("reference_doi", "10.1/author", True),
        ("reference_url", "", False),
        ("reference_url", [], False),
    ],
)
def test_public_reference_metadata(tmp_path: Path, reference: str, value: object, accepted: bool) -> None:
    """Public reference source requires original nonblank URL or DOI metadata rather than source download."""
    report = inspect(tmp_path, declaration(source="documented_public_reference", **{reference: value}))
    assert (report["status"] == "pass") is accepted


def test_original_canonical_hash_and_actual_report_hash(tmp_path: Path) -> None:
    """Independent compact JSON oracle checks public body/report digest semantics including actual UTC timestamp."""
    payload = {"unused": "ž", "value": -0.0}
    original = dict(payload)
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    assert canonical_artifact_sha256(payload) == expected and payload == original
    assert canonical_artifact_sha256(dict(payload, payload_sha256="ignored")) == expected
    report = inspect(tmp_path, declaration())
    body = dict(report, payload_sha256=None)
    assert (
        report["payload_sha256"]
        == hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()
    )
    assert report["generated_at_utc"].endswith("Z")


@pytest.mark.parametrize("number", [0.0, -0.0, 1.25, 5e-324])
def test_ignored_representable_values_preserved(tmp_path: Path, number: float) -> None:
    """Ignored finite values, signed zero and representable subnormals keep original body consistency semantics."""
    assert inspect(tmp_path, declaration(unused=number))["status"] == "pass"
