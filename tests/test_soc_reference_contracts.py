# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — SOC identity and unit declaration tests

"""Check original SOC identity, source and declared unit/error domains through public APIs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_soc_reference_domains import declaration, inspect

from validation.validate_soc_reference import (
    validate_soc_reference,
)

IDENTITY_FIELDS = (
    "source",
    "model_id",
    "model_version",
    "reference_dataset_id",
    "reference_artifact_sha256",
    "executed_at",
)
BYTE_SHA_FIELDS = ("reference_artifact_sha256",)


@pytest.mark.parametrize("field", IDENTITY_FIELDS)
@pytest.mark.parametrize("value", [None, [], {}, False, "", " "])
def test_required_strings(tmp_path: Path, field: str, value: object) -> None:
    """All six original nonblank identity fields refuse malformed values without set/digest exceptions."""
    payload = declaration()
    payload[field] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    assert any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "2.0"),
        ("source", "synthetic"),
        ("units", []),
        ("lattice_metadata", []),
        ("learning_metadata", []),
        ("metrics", []),
        ("tolerances", []),
    ],
)
def test_contract_findings(tmp_path: Path, field: str, value: object) -> None:
    """Original schema1.0, source membership, units and feature/metric block errors remain findings."""
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
    """Reference-byte hash require exact hex format, without fetching or authenticating referenced bytes."""
    payload = declaration()
    payload[field] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize("field", ["lattice_gradient", "flow", "shear", "topple_count", "q_value", "reward", "time"])
def test_original_unit_labels(tmp_path: Path, field: str) -> None:
    """All seven original SOC unit labels remain mandatory without numerical conversion."""
    payload = declaration()
    payload["units"][field] = "wrong"
    assert inspect(tmp_path, payload)["errors"][0]["field"] == "units"


@pytest.mark.parametrize("token", ["1e-400", "-1e-400", "0.0", "-0.0", "0e-400", "5e-324"])
def test_decimal_underflow_refusal(tmp_path: Path, token: str) -> None:
    """Nonzero decimal error tokens collapsed to zero refuse during decoding; exact zero and subnormal remain admitted."""
    payload = declaration()
    payload["metrics"]["flow_mean_abs_error"] = float(token)
    needle = '"flow_mean_abs_error": ' + str(float(token))
    encoded = json.dumps(payload)
    assert needle in encoded
    path = tmp_path / "input.json"
    path.write_text(encoded.replace(needle, '"flow_mean_abs_error": ' + token, 1))
    report = validate_soc_reference(path)
    assert report["status"] == ("fail" if token in {"1e-400", "-1e-400"} else "pass")
    if report["status"] == "fail":
        assert report["errors"][0]["field"] == "json"


@pytest.mark.parametrize(
    "source", ["documented_public_reference", "measured_turbulence_replay", "external_gyrokinetic_reference"]
)
@pytest.mark.parametrize("value", [None, [], "", " ", "../unparsed\x00citation", "s3://", "file:///unfetched"])
def test_source_reference_presence(tmp_path: Path, source: str, value: object) -> None:
    """Three SOC source families retain presence-only citation/diagnostic/artifact text."""
    payload = declaration()
    payload.pop("reference_doi")
    payload["source"] = source
    field = (
        "reference_url"
        if source == "documented_public_reference"
        else ("diagnostic_uri" if source == "measured_turbulence_replay" else "reference_artifact_uri")
    )
    payload.update(shot_id="observed-shot", external_code="GENE")
    payload[field] = value
    accepted = isinstance(value, str) and bool(value.strip())
    report = inspect(tmp_path, payload)
    assert report["status"] == ("pass" if accepted else "fail")


@pytest.mark.parametrize("field", ["shot_id", "diagnostic_uri"])
@pytest.mark.parametrize("value", [None, [], False, "", " ", "nonblank"])
def test_measured_identity_presence(tmp_path: Path, field: str, value: object) -> None:
    """Measured declarations require both shot identity and diagnostic text without reading measurements."""
    payload = declaration()
    payload.update(source="measured_turbulence_replay", shot_id="shot", diagnostic_uri="diagnostic")
    payload[field] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == ("pass" if value == "nonblank" else "fail")
    if report["status"] == "fail":
        assert any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize("code", [None, [], {}, False, "", "gene", "GENE", "CGYRO", "GS2", "TGLF", "QuaLiKiz"])
def test_external_code_identity(tmp_path: Path, code: object) -> None:
    """Exact admitted code names remain case-sensitive and malformed nonhashable values become findings."""
    payload = declaration()
    payload.update(source="external_gyrokinetic_reference", external_code=code, reference_artifact_uri="unfetched")
    accepted = isinstance(code, str) and code in {"GENE", "CGYRO", "GS2", "TGLF", "QuaLiKiz"}
    report = inspect(tmp_path, payload)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == "external_code" for error in report["errors"])


def test_public_doi_presence_alternative(tmp_path: Path) -> None:
    """A nonblank DOI satisfies original public provenance independently of reference_url."""
    payload = declaration()
    assert inspect(tmp_path, payload)["status"] == "pass"
    payload["reference_doi"] = " "
    assert inspect(tmp_path, payload)["errors"][0]["field"] == "reference"
