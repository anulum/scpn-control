# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Digital twin identity and provenance tests

"""Check original declaration identity and presence-only provenance at the public reader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_digital_twin_reference_domains import declaration

from validation.validate_digital_twin_reference import validate_digital_twin_reference


@pytest.mark.parametrize(
    "field", ["source", "model_id", "model_version", "reference_dataset_id", "reference_artifact_sha256", "executed_at"]
)
@pytest.mark.parametrize("value", [None, [], {}, False, "", " "])
def test_required_strings(tmp_path: Path, field: str, value: object) -> None:
    """Each required identity string rejects missing/nonstring/blank values through authored findings."""
    payload = declaration()
    payload[field] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_digital_twin_reference(path)
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
        ("actuator_metadata", []),
        ("grid_metadata", []),
        ("metrics", []),
        ("tolerances", []),
    ],
)
def test_contract_findings(tmp_path: Path, field: str, value: object) -> None:
    """Schema, exact full digest, unit and block failures remain findings rather than exceptions."""
    payload = declaration()
    payload[field] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_digital_twin_reference(path)
    assert report["status"] == "fail" and any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize("field", ["temperature", "density", "q_profile", "actuator_action", "time", "ids_pulse"])
def test_original_unit_labels(tmp_path: Path, field: str) -> None:
    """Every original required unit label is enforced without numerical conversion."""
    payload = declaration()
    payload["units"][field] = "wrong"
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    assert validate_digital_twin_reference(path)["errors"][0]["field"] == "units"


@pytest.mark.parametrize(
    "source", ["documented_public_reference", "measured_discharge_replay", "external_integrated_modelling"]
)
@pytest.mark.parametrize("value", [None, [], " ", "../presence-only\x00"])
def test_source_presence(tmp_path: Path, source: str, value: object) -> None:
    """Source-specific URI/citation fields are original nonblank presence checks, including unparsed relative/NUL text."""
    payload = declaration()
    payload.update(source=source, external_code="TSC", shot_id="declared-shot")
    payload.pop("reference_doi")
    field = {
        "documented_public_reference": "reference_doi",
        "measured_discharge_replay": "diagnostic_uri",
        "external_integrated_modelling": "reference_artifact_uri",
    }[source]
    payload[field] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_digital_twin_reference(path)
    assert report["status"] == ("pass" if isinstance(value, str) and value.strip() else "fail")


@pytest.mark.parametrize("value", [None, [], {}, " ", "UNKNOWN", "ASTRA", "IMAS", "JINTRAC", "TRANSP", "TSC"])
def test_external_code_membership(tmp_path: Path, value: object) -> None:
    """Only named string external codes are admitted; list/dict values cannot raise set-membership TypeError."""
    payload = declaration()
    payload.update(
        source="external_integrated_modelling", external_code=value, reference_artifact_uri="../presence-only"
    )
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_digital_twin_reference(path)
    assert report["status"] == (
        "pass" if isinstance(value, str) and value in {"ASTRA", "IMAS", "JINTRAC", "TRANSP", "TSC"} else "fail"
    )


def test_measured_shot_required_and_public_url_alternative(tmp_path: Path) -> None:
    """Measured replay requires shot identity, while public provenance allows either declared URL or DOI."""
    payload = declaration()
    payload.update(source="measured_discharge_replay", diagnostic_uri="presence-only", shot_id=[])
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    assert validate_digital_twin_reference(path)["errors"][0]["field"] == "shot_id"
    payload["shot_id"] = "declared-shot"
    path.write_text(json.dumps(payload))
    assert validate_digital_twin_reference(path)["status"] == "pass"
    payload["source"] = "documented_public_reference"
    path.write_text(json.dumps(payload))
    assert validate_digital_twin_reference(path)["status"] == "pass"


@pytest.mark.parametrize("token", ["1e-400", "-1e-400", "0.0", "-0.0", "0e-400", "5e-324"])
def test_decimal_underflow_refusal(tmp_path: Path, token: str) -> None:
    """Nonzero decimal errors that collapse to zero refuse; exact zero and representable subnormal retain admission."""
    payload = declaration()
    needle = '"final_avg_temp_relative_error": 0.0'
    encoded = json.dumps(payload)
    assert needle in encoded
    path = tmp_path / "input.json"
    path.write_text(encoded.replace(needle, '"final_avg_temp_relative_error": ' + token, 1))
    report = validate_digital_twin_reference(path)
    assert report["status"] == ("fail" if token in {"1e-400", "-1e-400"} else "pass")
    if report["status"] == "fail":
        assert report["errors"][0]["field"] == "json"
