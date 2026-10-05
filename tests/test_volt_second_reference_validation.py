# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Volt-second declared numeric domain tests

"""Exercise original volt-second domains through persisted public declarations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from validation.validate_volt_second_reference import validate_volt_second_reference

ERROR_FIELDS = (
    "total_flux_relative_error",
    "flat_top_duration_relative_error",
    "ejima_flux_relative_error",
    "bootstrap_current_abs_error_MA",
    "margin_abs_error_Vs",
)
MACHINE_FIELDS = ("Phi_CS_Vs", "L_plasma_H", "R_plasma_Ohm", "Ip_MA", "R0_m")


def valid_volt_second_declaration() -> dict[str, Any]:
    """Supply engineering metadata only; no authenticated physical or external benchmark is represented."""
    return {
        "schema_version": "1.0",
        "source": "documented_public_reference",
        "model_id": "engineering-test-model",
        "model_version": "1",
        "reference_dataset_id": "engineering-declaration-only",
        "reference_artifact_sha256": "a" * 64,
        "executed_at": "declared-time",
        "reference_url": "presence-only",
        "units": {
            "flux": "V s",
            "voltage": "V",
            "current": "A",
            "current_MA": "MA",
            "time": "s",
            "resistance": "ohm",
            "inductance": "H",
            "radius": "m",
            "dimensionless": "1",
        },
        "machine_metadata": {field: 1.0 for field in MACHINE_FIELDS},
        "reference_case_count": 1,
        "metrics": {field: 0.0 for field in ERROR_FIELDS},
        "tolerances": {field: 1.0 for field in ERROR_FIELDS},
    }


@pytest.mark.parametrize("field", MACHINE_FIELDS)
@pytest.mark.parametrize("value", [None, [], "1", True, 0, -1, float("nan"), float("inf"), 10**400, 5e-324])
def test_machine_domains(tmp_path: Path, field: str, value: object) -> None:
    """Each machine value must be positive finite, without throwing on huge integers or imposing physical consistency."""
    payload = valid_volt_second_declaration()
    payload["machine_metadata"][field] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_volt_second_reference(path)
    assert report["status"] == ("pass" if value == 5e-324 else "fail")
    if report["status"] == "fail":
        assert any(error["field"] == "machine_metadata" for error in report["errors"])


@pytest.mark.parametrize("field", ERROR_FIELDS)
@pytest.mark.parametrize("block", ["metrics", "tolerances"])
@pytest.mark.parametrize("value", [None, [], "1", True, -1, float("nan"), float("inf"), 10**400, 0, 1.001])
def test_error_and_bound_domains(tmp_path: Path, field: str, block: str, value: object) -> None:
    """All five declared comparisons refuse invalid numbers and accept only original nonnegative/positive domains."""
    payload = valid_volt_second_declaration()
    payload[block][field] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_volt_second_reference(path)
    accepted = (block == "metrics" and value == 0) or (block == "tolerances" and value == 1.001)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize("count", [None, [], "1", True, 0, -1, 1.0, 1, 10**400])
def test_positive_uncapped_case_count(tmp_path: Path, count: object) -> None:
    """Positive nonboolean integer counts retain their uncapped declaration domain."""
    payload = valid_volt_second_declaration()
    payload["reference_case_count"] = count
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_volt_second_reference(path)
    accepted = isinstance(count, int) and not isinstance(count, bool) and count > 0
    assert report["status"] == ("pass" if accepted else "fail")
    if accepted:
        assert report["entries"][0]["reference_case_count"] == count


def test_equal_bounds_and_metadata_only(tmp_path: Path) -> None:
    """Equality, independent units, subnormal positives and unknown extras pass without hashing or recomputing physics."""
    payload = valid_volt_second_declaration()
    payload["metrics"] = dict(payload["tolerances"])
    payload["machine_metadata"].update(Ip_MA=5e-324, Phi_CS_Vs=1e308)
    payload["reference_artifact_sha256"] = "B" * 64
    payload["reference_url"] = "../unresolved\x00presence"
    payload["units"]["extra"] = "unknown"
    payload["payload_sha256"] = "ignored-extra"
    path = tmp_path / "input.data"
    path.write_text(json.dumps(payload))
    report = validate_volt_second_reference(path, require_reference_artifacts=True)
    assert report["status"] == "pass" and report["errors"] == []
    assert report["entries"] == [
        {
            "path": str(path),
            "source": "documented_public_reference",
            "model_id": payload["model_id"],
            "model_version": payload["model_version"],
            "reference_dataset_id": payload["reference_dataset_id"],
            "reference_case_count": 1,
        }
    ]
