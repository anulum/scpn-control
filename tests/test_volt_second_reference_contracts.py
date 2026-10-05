# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Volt-second identity and provenance tests

"""Check original declaration identity and presence-only provenance at the public reader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_volt_second_reference_validation import valid_volt_second_declaration

from validation.validate_volt_second_reference import validate_volt_second_reference


@pytest.mark.parametrize(
    "field", ["source", "model_id", "model_version", "reference_dataset_id", "reference_artifact_sha256", "executed_at"]
)
@pytest.mark.parametrize("value", [None, [], {}, False, "", " "])
def test_required_strings(tmp_path: Path, field: str, value: object) -> None:
    """Each required identity string rejects missing/nonstring/blank values through authored findings."""
    payload = valid_volt_second_declaration()
    payload[field] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_volt_second_reference(path)
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
        ("machine_metadata", []),
        ("metrics", []),
        ("tolerances", []),
    ],
)
def test_contract_findings(tmp_path: Path, field: str, value: object) -> None:
    """Schema, exact full digest, unit and block failures remain findings rather than exceptions."""
    payload = valid_volt_second_declaration()
    payload[field] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_volt_second_reference(path)
    assert report["status"] == "fail" and any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize(
    "field", ["flux", "voltage", "current", "current_MA", "time", "resistance", "inductance", "radius", "dimensionless"]
)
def test_original_unit_labels(tmp_path: Path, field: str) -> None:
    """Every original required unit label is enforced without numerical conversion."""
    payload = valid_volt_second_declaration()
    payload["units"][field] = "wrong"
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    assert validate_volt_second_reference(path)["errors"][0]["field"] == "units"


@pytest.mark.parametrize(
    "source", ["documented_public_reference", "measured_loop_voltage_replay", "external_scenario_benchmark"]
)
@pytest.mark.parametrize("value", [None, [], " ", "../presence-only\x00"])
def test_source_presence(tmp_path: Path, source: str, value: object) -> None:
    """Source-specific URI/citation fields are original nonblank presence checks, including unparsed relative/NUL text."""
    payload = valid_volt_second_declaration()
    payload.update(source=source, external_code="TSC", shot_id="declared-shot")
    payload.pop("reference_url")
    field = {
        "documented_public_reference": "reference_doi",
        "measured_loop_voltage_replay": "diagnostic_uri",
        "external_scenario_benchmark": "reference_artifact_uri",
    }[source]
    payload[field] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_volt_second_reference(path)
    assert report["status"] == ("pass" if isinstance(value, str) and value.strip() else "fail")


@pytest.mark.parametrize("value", [None, [], {}, " ", "UNKNOWN", "TRANSP", "TSC", "ASTRA", "JINTRAC", "PROCESS"])
def test_external_code_membership(tmp_path: Path, value: object) -> None:
    """Only named string external codes are admitted; list/dict values cannot raise set-membership TypeError."""
    payload = valid_volt_second_declaration()
    payload.update(source="external_scenario_benchmark", external_code=value, reference_artifact_uri="../presence-only")
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_volt_second_reference(path)
    assert report["status"] == (
        "pass" if isinstance(value, str) and value in {"TRANSP", "TSC", "ASTRA", "JINTRAC", "PROCESS"} else "fail"
    )


def test_measured_shot_required_and_public_url_alternative(tmp_path: Path) -> None:
    """Measured replay requires shot identity, while public provenance allows either declared URL or DOI."""
    payload = valid_volt_second_declaration()
    payload.update(source="measured_loop_voltage_replay", diagnostic_uri="presence-only", shot_id=[])
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    assert validate_volt_second_reference(path)["errors"][0]["field"] == "shot_id"
    payload["shot_id"] = "declared-shot"
    path.write_text(json.dumps(payload))
    assert validate_volt_second_reference(path)["status"] == "pass"
    payload["source"] = "documented_public_reference"
    path.write_text(json.dumps(payload))
    assert validate_volt_second_reference(path)["status"] == "pass"


@pytest.mark.parametrize("token", ["1e-400", "-1e-400", "0.0", "-0.0", "0e-400", "5e-324"])
def test_decimal_underflow_refusal(tmp_path: Path, token: str) -> None:
    """Nonzero decimal errors that collapse to zero refuse; exact zero and representable subnormal retain admission."""
    payload = valid_volt_second_declaration()
    needle = '"total_flux_relative_error": 0.0'
    encoded = json.dumps(payload)
    assert needle in encoded
    path = tmp_path / "input.json"
    path.write_text(encoded.replace(needle, '"total_flux_relative_error": ' + token, 1))
    report = validate_volt_second_reference(path)
    assert report["status"] == ("fail" if token in {"1e-400", "-1e-400"} else "pass")
    if report["status"] == "fail":
        assert report["errors"][0]["field"] == "json"
