# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK declaration contract behaviour

"""Exercise stored author declarations through the public reader without authenticating external runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from test_gk_crosscode_validation import _valid_gene_report

from validation.validate_gk_crosscode import validate_gk_crosscode_evidence


def declaration(**changes: Any) -> dict[str, Any]:
    """Seal a test declaration using an independent compact sorted ASCII JSON oracle."""
    payload = _valid_gene_report()
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
    return validate_gk_crosscode_evidence(path, require_external_runs=True)


@pytest.mark.parametrize("payload", [None, [], 17, "declaration"])
def test_object_root_required(tmp_path: Path, payload: object) -> None:
    """Nonobject decoded roots produce a field finding instead of a declaration count."""
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and report["external_runs"] == 0
    assert report["errors"][0]["field"] == "root"


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "case",
        "external_code",
        "source",
        "binary_path",
        "code_version",
        "run_id",
        "executed_at",
        "units",
        "input_deck_sha256",
        "external_output_sha256",
        "native_input_sha256",
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
    assert report["status"] == "fail" and report["external_runs"] == 0
    assert field in {error["field"] for error in report["errors"]}


@pytest.mark.parametrize(
    "field",
    [
        "gamma_max_cs_over_a",
        "omega_r_cs_over_a",
        "k_y_rho_s_at_max",
        "native_gamma_max_cs_over_a",
        "native_omega_r_cs_over_a",
        "native_k_y_rho_s_at_max",
    ],
)
@pytest.mark.parametrize("value", [True, False, None, [], "0.1", 10**400])
def test_numeric_type_and_representability(tmp_path: Path, field: str, value: object) -> None:
    """Every scalar refuses booleans, wrong types and conversion overflow through the public reader."""
    report = inspect(tmp_path, declaration(**{field: value}))
    assert report["status"] == "fail" and report["external_runs"] == 0
    assert any(
        error["field"] == field and error["error"] == "field must be finite numeric" for error in report["errors"]
    )


@pytest.mark.parametrize(
    "field", ["input_deck_sha256", "external_output_sha256", "native_input_sha256", "payload_sha256"]
)
@pytest.mark.parametrize("digest", ["+" + "a" * 63, " " + "a" * 63, "٠" * 64, "a" * 63, "a" * 65, "g" * 64])
def test_ascii_digest_contract(tmp_path: Path, field: str, digest: str) -> None:
    """Signs, spaces, Unicode digits, length errors and nonhex characters are refused in all four digest fields."""
    payload = declaration(**{field: digest})
    if field == "payload_sha256":
        payload[field] = digest
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and report["external_runs"] == 0
    assert any(error["field"] == field and error["error"] == "field must be SHA-256 hex" for error in report["errors"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gamma_max_cs_over_a", -0.1),
        ("native_gamma_max_cs_over_a", -0.1),
        ("k_y_rho_s_at_max", 0),
        ("native_k_y_rho_s_at_max", 0),
        ("k_y_rho_s_at_max", -0.1),
        ("native_k_y_rho_s_at_max", -0.1),
        ("native_gamma_max_cs_over_a", 1.0),
        ("native_omega_r_cs_over_a", 1.0),
        ("native_k_y_rho_s_at_max", 1.0),
        ("source", "reference"),
        ("units", "Hz"),
        ("external_code", "unsupported"),
        ("schema_version", "v2"),
    ],
)
def test_original_domains_and_discrepancy_limits(tmp_path: Path, field: str, value: object) -> None:
    """Original growth/wavenumber domains, literal identities and three discrepancy bounds remain enforced."""
    report = inspect(tmp_path, declaration(**{field: value}))
    assert report["status"] == "fail" and report["external_runs"] == 0
    assert report["errors"]


@pytest.mark.parametrize("code", ["TGLF", "GENE", "GS2", "CGYRO", "GYRO", "QuaLiKiz"])
def test_supported_self_declared_codes(tmp_path: Path, code: str) -> None:
    """All six original code labels pass metadata checks without proving the named binary was run."""
    payload = declaration(external_code=code, input_deck_sha256="A" * 64, unused="ž")
    payload["payload_sha256"] = payload["payload_sha256"].upper()
    report = inspect(tmp_path, payload)
    assert report["status"] == "pass" and report["external_runs"] == 1
    entry = report["entries"][0]
    assert entry["external_code"] == code and entry["payload_sha256"] == payload["payload_sha256"].lower()
    assert entry["gamma_relative_error"] == abs(0.19 - 0.18) / 0.18
    assert entry["omega_relative_error"] == abs(-0.40 + 0.42) / 0.42
    assert entry["k_y_absolute_error"] == abs(0.31 - 0.30)


def test_inclusive_limits_and_denominator_floors(tmp_path: Path) -> None:
    """Equality at all original discrepancy bounds passes, including zero growth and frequency denominator floors."""
    report = inspect(
        tmp_path,
        declaration(
            gamma_max_cs_over_a=0.0,
            native_gamma_max_cs_over_a=2e-13,
            omega_r_cs_over_a=0.0,
            native_omega_r_cs_over_a=-3e-13,
            k_y_rho_s_at_max=0.1,
            native_k_y_rho_s_at_max=0.2,
        ),
    )
    assert report["status"] == "pass" and report["external_runs"] == 1
    assert report["entries"][0]["gamma_relative_error"] == 0.2
    assert report["entries"][0]["omega_relative_error"] == 0.3
    assert report["entries"][0]["k_y_absolute_error"] == 0.1


@pytest.mark.parametrize("number", [0.0, -0.0, 1.25, 5e-324])
def test_ignored_representable_values_preserved(tmp_path: Path, number: float) -> None:
    """Ignored finite values, signed zero and representable subnormals keep original body consistency semantics."""
    assert inspect(tmp_path, declaration(unused=number))["status"] == "pass"
