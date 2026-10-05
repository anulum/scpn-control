# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Burn-control identity and provenance tests

"""Check original declaration identity and presence-only provenance at the public reader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_burn_reference_validation import valid_burn_declaration

from validation.validate_burn_reference import validate_burn_reference


@pytest.mark.parametrize(
    "field", ["source", "model_id", "model_version", "reference_dataset_id", "reference_artifact_sha256", "executed_at"]
)
@pytest.mark.parametrize("value", [None, [], {}, False, "", " "])
def test_required_strings(tmp_path: Path, field: str, value: object) -> None:
    """Each required identity string rejects missing/nonstring/blank values through authored findings."""
    payload = valid_burn_declaration()
    payload[field] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_burn_reference(path)
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
        ("plasma_metadata", []),
        ("metrics", []),
        ("tolerances", []),
    ],
)
def test_contract_findings(tmp_path: Path, field: str, value: object) -> None:
    """Schema, exact full digest, unit and block failures remain findings rather than exceptions."""
    payload = valid_burn_declaration()
    payload[field] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_burn_reference(path)
    assert report["status"] == "fail" and any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize(
    "field", ["density", "temperature", "power", "time", "reactivity", "triple_product", "dimensionless"]
)
def test_original_unit_labels(tmp_path: Path, field: str) -> None:
    """Every original required unit label is enforced without numerical conversion."""
    payload = valid_burn_declaration()
    payload["units"][field] = "wrong"
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    assert validate_burn_reference(path)["errors"][0]["field"] == "units"


@pytest.mark.parametrize(
    "source", ["documented_public_reference", "measured_burn_replay", "integrated_transport_benchmark"]
)
@pytest.mark.parametrize("value", [None, [], " ", "../presence-only\x00"])
def test_source_presence(tmp_path: Path, source: str, value: object) -> None:
    """Source-specific URI/citation fields are original nonblank presence checks, including unparsed relative/NUL text."""
    payload = valid_burn_declaration()
    payload.update(source=source, external_code="TSC", shot_id="declared-shot")
    payload.pop("reference_url")
    field = {
        "documented_public_reference": "reference_doi",
        "measured_burn_replay": "diagnostic_uri",
        "integrated_transport_benchmark": "reference_artifact_uri",
    }[source]
    payload[field] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_burn_reference(path)
    assert report["status"] == ("pass" if isinstance(value, str) and value.strip() else "fail")


@pytest.mark.parametrize("value", [None, [], {}, " ", "UNKNOWN", "TRANSP", "TSC", "ASTRA", "JINTRAC"])
def test_external_code_membership(tmp_path: Path, value: object) -> None:
    """Only named string external codes are admitted; list/dict values cannot raise set-membership TypeError."""
    payload = valid_burn_declaration()
    payload.update(
        source="integrated_transport_benchmark", external_code=value, reference_artifact_uri="../presence-only"
    )
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_burn_reference(path)
    assert report["status"] == (
        "pass" if isinstance(value, str) and value in {"TRANSP", "TSC", "ASTRA", "JINTRAC"} else "fail"
    )


def test_measured_shot_required_and_public_url_alternative(tmp_path: Path) -> None:
    """Measured replay requires shot identity, while public provenance allows either declared URL or DOI."""
    payload = valid_burn_declaration()
    payload.update(source="measured_burn_replay", diagnostic_uri="presence-only", shot_id=[])
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    assert validate_burn_reference(path)["errors"][0]["field"] == "shot_id"
    payload["shot_id"] = "declared-shot"
    path.write_text(json.dumps(payload))
    assert validate_burn_reference(path)["status"] == "pass"
    payload["source"] = "documented_public_reference"
    path.write_text(json.dumps(payload))
    assert validate_burn_reference(path)["status"] == "pass"


@pytest.mark.parametrize("token", ["1e-400", "-1e-400", "0.0", "-0.0", "0e-400", "5e-324"])
def test_decimal_underflow_refusal(tmp_path: Path, token: str) -> None:
    """Nonzero decimal errors that collapse to zero refuse; exact zero and representable subnormal retain admission."""
    payload = valid_burn_declaration()
    needle = '"P_alpha_relative_error": 0.0'
    encoded = json.dumps(payload)
    assert needle in encoded
    path = tmp_path / "input.json"
    path.write_text(encoded.replace(needle, '"P_alpha_relative_error": ' + token, 1))
    report = validate_burn_reference(path)
    assert report["status"] == ("fail" if token in {"1e-400", "-1e-400"} else "pass")
    if report["status"] == "fail":
        assert report["errors"][0]["field"] == "json"
