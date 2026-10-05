# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — ELM reference validation tests

"""Check original ELM rho-grid, time-window, Type-I energy and declared error domains through public persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from test_elm_reference_validation import _valid_elm_reference_artifact

from validation.validate_elm_reference import canonical_artifact_sha256, validate_elm_reference

ERROR_FIELDS = (
    "elm_frequency_relative_error",
    "crash_energy_fraction_error",
    "pedestal_temperature_drop_relative_error",
    "pedestal_density_drop_relative_error",
    "rmp_suppression_window_error_s",
    "peak_heat_flux_relative_error",
)


def declaration() -> dict[str, Any]:
    """Supply original engineering ELM metadata with zero errors and unit bounds, without measured bytes."""
    payload = cast(dict[str, Any], _valid_elm_reference_artifact())
    payload["metrics"] = dict.fromkeys(ERROR_FIELDS, 0.0)
    payload["tolerances"] = dict.fromkeys(ERROR_FIELDS, 1.0)
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    return payload


def inspect(tmp_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Persist a declaration with original body consistency and inspect it through the public reader."""
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    return validate_elm_reference(path, require_reference_artifacts=True)


@pytest.mark.parametrize("field", ERROR_FIELDS)
@pytest.mark.parametrize("block", ["metrics", "tolerances"])
@pytest.mark.parametrize("value", [None, [], "1", True, -1, float("nan"), float("inf"), 10**400, 0, 1.001])
def test_error_and_bound_domains(tmp_path: Path, field: str, block: str, value: object) -> None:
    """Six errors retain finite nonnegative values and positive finite inclusive bounds, refusing huge conversion."""
    payload = declaration()
    payload[block][field] = value
    accepted = (block == "metrics" and value == 0) or (block == "tolerances" and value == 1.001)
    report = inspect(tmp_path, payload)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize(
    ("grid", "accepted"),
    [
        (None, False),
        ([], False),
        ([0], False),
        ([0, 1], True),
        ([0, 0.5, 1], True),
        ([True, 1], False),
        (["0", 1], False),
        ([0, 10**400], False),
        ([0, float("nan")], False),
        ([0, float("inf")], False),
        ([-0.01, 1], False),
        ([0, 1.01], False),
        ([0.5, 0.5], False),
        ([1, 0], False),
    ],
)
def test_original_rho_grid_domain(tmp_path: Path, grid: object, accepted: bool) -> None:
    """At least two finite nonboolean rho coordinates must strictly increase within inclusive zero/one endpoints."""
    payload = declaration()
    payload["pedestal_rho_grid"] = grid
    report = inspect(tmp_path, payload)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == "pedestal_rho_grid" for error in report["errors"])


@pytest.mark.parametrize("field", ["event_time_window_s", "rmp_suppression_window_s"])
@pytest.mark.parametrize(
    ("window", "accepted"),
    [
        (None, False),
        ([], False),
        ([0], False),
        ([0, 1, 2], False),
        ([0, 1], True),
        ([0, 5e-324], True),
        ([50, 60], True),
        ([1, 1], False),
        ([1, 0], False),
        ([-1, 1], False),
        ([True, 1], False),
        ([0, False], False),
        ([0, 10**400], False),
        ([10**400, 1], False),
        ([0, float("inf")], False),
    ],
)
def test_independent_time_window_domains(tmp_path: Path, field: str, window: object, accepted: bool) -> None:
    """Each original window starts at nonnegative time and increases strictly, independently of the other window."""
    payload = declaration()
    payload[field] = window
    report = inspect(tmp_path, payload)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize(
    ("fractions", "accepted"),
    [
        (None, False),
        ([], False),
        ([0.04], False),
        ([0.04, 0.15, 0.15], False),
        ([0.04, 0.15], True),
        ([0.04, 0.04], True),
        ([0.15, 0.15], True),
        ([0.039, 0.15], False),
        ([0.04, 0.151], False),
        ([0.15, 0.04], False),
        ([0, 0.15], False),
        ([True, 0.15], False),
        ([0.04, "0.15"], False),
        ([0.04, 10**400], False),
        ([float("nan"), 0.15], False),
    ],
)
def test_original_type_i_fraction_domain(tmp_path: Path, fractions: object, accepted: bool) -> None:
    """Original energy endpoints remain inclusive 0.04/0.15 and equal endpoints are accepted without changing physics."""
    payload = declaration()
    payload["elm_energy_fraction_range"] = fractions
    report = inspect(tmp_path, payload)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == "elm_energy_fraction_range" for error in report["errors"])


def test_inclusive_errors_and_representable_subnormal(tmp_path: Path) -> None:
    """Errors equal to each bound pass; representable positive subnormal bounds remain accepted."""
    payload = declaration()
    for field in ERROR_FIELDS:
        payload["metrics"][field] = payload["tolerances"][field]
    assert inspect(tmp_path, payload)["status"] == "pass"
    payload["metrics"][ERROR_FIELDS[0]] = 0
    payload["tolerances"][ERROR_FIELDS[0]] = 5e-324
    assert inspect(tmp_path, payload)["status"] == "pass"


@pytest.mark.parametrize("field", ["machine", "shot_id", "campaign_id"])
@pytest.mark.parametrize("value", [None, [], "", " ", "../unparsed\x00identity"])
def test_measured_identity_presence(tmp_path: Path, field: str, value: object) -> None:
    """Measured declarations require machine and either nonblank shot or campaign text; identity syntax remains uninterpreted."""
    payload = declaration()
    payload.pop("shot_id")
    payload["shot_id" if field != "campaign_id" else "campaign_id"] = "declared"
    payload[field] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == ("pass" if isinstance(value, str) and value.strip() else "fail")


def test_public_citation_presence(tmp_path: Path) -> None:
    """Public citation needs nonblank URL/DOI only and does not authenticate or parse supplied reference text."""
    payload = declaration()
    payload["source"] = "documented_public_reference"
    assert inspect(tmp_path, payload)["errors"][0]["field"] == "reference"
    for field in ["reference_url", "reference_doi"]:
        payload[field] = "../unparsed\x00citation"
        assert inspect(tmp_path, payload)["status"] == "pass"
        payload.pop(field)
