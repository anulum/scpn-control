# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disruption declared numeric domain tests

"""Exercise original Disruption numeric domains through persisted public declarations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from test_disruption_reference_validation import _valid_disruption_reference_artifact

from validation.validate_disruption_reference import validate_disruption_reference

ERROR_FIELDS = (
    "risk_after_abs_error",
    "detection_lead_time_abs_error_ms",
    "halo_current_relative_error",
    "runaway_beam_relative_error",
    "tbr_abs_error",
)
SIGNAL_FIELDS = (
    "sample_period_s",
    "pre_disruption_duration_s",
    "current_quench_duration_ms",
    "thermal_quench_duration_ms",
)
MITIGATION_FIELDS = (
    "neon_quantity_mol",
    "argon_quantity_mol",
    "xenon_quantity_mol",
    "total_impurity_mol",
    "mitigation_strength",
    "tbr_reference",
)


def declaration() -> dict[str, Any]:
    """Supply original engineering metadata with zero errors, without authenticated mitigation evidence."""
    payload = cast(dict[str, Any], _valid_disruption_reference_artifact())
    payload["metrics"] = dict.fromkeys(ERROR_FIELDS, 0.0)
    payload["tolerances"] = dict.fromkeys(ERROR_FIELDS, 1.0)
    return payload


@pytest.mark.parametrize("field", SIGNAL_FIELDS)
@pytest.mark.parametrize("value", [None, [], "1", True, 0, -1, float("nan"), float("inf"), 10**400, 5e-324])
def test_positive_signal_parameters(tmp_path: Path, field: str, value: object) -> None:
    """Four original timing inputs require finite positive values and retain representable subnormals."""
    payload = declaration()
    payload["signal_window"][field] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_disruption_reference(path)
    assert report["status"] == ("pass" if value == 5e-324 else "fail")
    if report["status"] == "fail":
        assert any(error["field"] == "signal_window" for error in report["errors"])


@pytest.mark.parametrize("count", [None, [], "8", True, -1, 0, 7, 8.0, 8, 10**400])
def test_signal_sample_count(tmp_path: Path, count: object) -> None:
    """Signal sample count remains nonboolean integer at least eight, without an upper cap."""
    payload = declaration()
    payload["signal_window"]["sample_count"] = count
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    accepted = isinstance(count, int) and not isinstance(count, bool) and count >= 8
    assert validate_disruption_reference(path)["status"] == ("pass" if accepted else "fail")


@pytest.mark.parametrize("field", MITIGATION_FIELDS)
@pytest.mark.parametrize("value", [None, [], "1", True, -1, float("nan"), float("inf"), 10**400, 0, 5e-324])
def test_nonnegative_mitigation_parameters(tmp_path: Path, field: str, value: object) -> None:
    """Six original inventories/strength/TBR values admit zero, refuse overflow and impose no inventory sum."""
    payload = declaration()
    payload["mitigation_metadata"][field] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    accepted = isinstance(value, int | float) and not isinstance(value, bool) and value in [0, 5e-324]
    report = validate_disruption_reference(path)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == "mitigation_metadata" for error in report["errors"])


@pytest.mark.parametrize("strength", [0, 1, 1.001])
def test_original_strength_bound(tmp_path: Path, strength: float) -> None:
    """Original mitigation strength includes both zero and one, and refuses values above one."""
    payload = declaration()
    payload["mitigation_metadata"]["mitigation_strength"] = strength
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    assert validate_disruption_reference(path)["status"] == ("pass" if strength <= 1 else "fail")


@pytest.mark.parametrize("field", ERROR_FIELDS)
@pytest.mark.parametrize("block", ["metrics", "tolerances"])
@pytest.mark.parametrize("value", [None, [], "1", True, -1, float("nan"), float("inf"), 10**400, 0, 1.001])
def test_error_and_bound_domains(tmp_path: Path, field: str, block: str, value: object) -> None:
    """All five errors retain finite nonnegative values with positive bounds and inclusive equality."""
    payload = declaration()
    payload[block][field] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    accepted = (block == "metrics" and value == 0) or (block == "tolerances" and value == 1.001)
    report = validate_disruption_reference(path)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize("count", [None, [], "1", True, 0, -1, 1.0, 1, 10**400])
def test_positive_uncapped_case_count(tmp_path: Path, count: object) -> None:
    """Reference case count remains positive nonboolean integer without an artificial cap."""
    payload = declaration()
    payload["reference_case_count"] = count
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    accepted = isinstance(count, int) and not isinstance(count, bool) and count > 0
    report = validate_disruption_reference(path)
    assert report["status"] == ("pass" if accepted else "fail")
    if accepted:
        assert report["entries"][0]["reference_case_count"] == count


def test_equal_bounds_and_metadata_only(tmp_path: Path) -> None:
    """Equal errors, independent duration/inventory metadata and zero TBR pass without new physical consistency rules."""
    payload = declaration()
    payload["metrics"] = dict(payload["tolerances"])
    payload["signal_window"].update(pre_disruption_duration_s=5e-324, thermal_quench_duration_ms=1e308)
    payload["mitigation_metadata"].update(total_impurity_mol=0, tbr_reference=0, neon_quantity_mol=1e308)
    payload["reference_artifact_sha256"] = "B" * 64
    payload["reference_doi"] = "../unresolved\x00presence"
    payload["units"]["extra"] = "unknown"
    payload["payload_sha256"] = "ignored-extra"
    path = tmp_path / "input.data"
    path.write_text(json.dumps(payload))
    report = validate_disruption_reference(path, require_reference_artifacts=True)
    assert report["status"] == "pass" and report["errors"] == []
    assert report["entries"] == [
        {
            "path": str(path),
            "source": "documented_public_reference",
            "model_id": payload["model_id"],
            "model_version": payload["model_version"],
            "reference_dataset_id": payload["reference_dataset_id"],
            "reference_case_count": 7,
        }
    ]
