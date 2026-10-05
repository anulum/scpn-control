# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural turbulence identity and unit declaration tests

"""Check original neural turbulence identity, feature ordering and declared unit/error domains through public APIs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_neural_turbulence_reference_domains import declaration, inspect

from validation.validate_neural_turbulence_reference import (
    validate_neural_turbulence_reference,
)

IDENTITY_FIELDS = (
    "source",
    "model_id",
    "model_version",
    "trained_weights_sha256",
    "reference_dataset_id",
    "reference_artifact_sha256",
    "executed_at",
)
BYTE_SHA_FIELDS = ("trained_weights_sha256", "reference_artifact_sha256")


@pytest.mark.parametrize("field", IDENTITY_FIELDS)
@pytest.mark.parametrize("value", [None, [], {}, False, "", " "])
def test_required_strings(tmp_path: Path, field: str, value: object) -> None:
    """All seven original nonblank identity fields refuse malformed values without set/digest exceptions."""
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
        ("feature_schema", []),
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
    """Weight and reference-byte hashes require exact hex format, without fetching or authenticating referenced bytes."""
    payload = declaration()
    payload[field] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize("field", ["Q_i", "Q_e", "Gamma_e", "input_gradients"])
def test_original_unit_labels(tmp_path: Path, field: str) -> None:
    """All four original gyroBohm unit labels remain mandatory without numerical conversion."""
    payload = declaration()
    payload["units"][field] = "wrong"
    assert inspect(tmp_path, payload)["errors"][0]["field"] == "units"


@pytest.mark.parametrize("token", ["1e-400", "-1e-400", "0.0", "-0.0", "0e-400", "5e-324"])
def test_decimal_underflow_refusal(tmp_path: Path, token: str) -> None:
    """Nonzero decimal error tokens collapsed to zero refuse during decoding; exact zero and subnormal remain admitted."""
    payload = declaration()
    payload["metrics"]["Gamma_e_rmse_gB"] = float(token)
    needle = '"Gamma_e_rmse_gB": ' + str(float(token))
    encoded = json.dumps(payload)
    assert needle in encoded
    path = tmp_path / "input.json"
    path.write_text(encoded.replace(needle, '"Gamma_e_rmse_gB": ' + token, 1))
    report = validate_neural_turbulence_reference(path)
    assert report["status"] == ("fail" if token in {"1e-400", "-1e-400"} else "pass")
    if report["status"] == "fail":
        assert report["errors"][0]["field"] == "json"
