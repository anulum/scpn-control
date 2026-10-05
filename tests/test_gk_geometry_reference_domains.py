# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK geometry reference validation tests

"""Exercise real Miller comparisons with original fixed tolerances and malformed physical/numeric domains."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from validation.validate_gk_geometry_reference import validate_gk_geometry_reference

REFERENCE = Path(__file__).resolve().parents[1] / "validation/reference_data/gk_geometry/miller_reference_cases.json"
PARAMETERS = ("R0", "a", "rho", "kappa", "delta", "dR_dr", "q", "B0")
SAMPLES = ("theta", "R", "Z", "jacobian", "g_rr", "g_rt", "g_tt", "B_toroidal", "b_dot_grad_theta")


def declaration() -> dict[str, Any]:
    """Read actual unchanged circular/shaped/high-shear repository reference cases."""
    return cast(dict[str, Any], json.loads(REFERENCE.read_text()))


def inspect(tmp_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Persist a copy of actual reference cases and exercise the public numerical validator."""
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(payload))
    return validate_gk_geometry_reference(path)


@pytest.mark.parametrize("field", PARAMETERS)
@pytest.mark.parametrize("value", [None, [], "1", True, float("nan"), float("inf"), 10**400])
def test_required_parameter_refusals(tmp_path: Path, field: str, value: object) -> None:
    """Original required fields refuse nonnumbers/booleans and unrepresentable/nonfinite values as findings before numerical evaluation."""
    payload = declaration()
    payload["cases"][0]["parameters"][field] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and any(error["field"] == field for error in report["errors"])
    assert report["public_claims"]["full_equilibrium_reconstruction"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("R0", 0),
        ("a", 0),
        ("rho", 0),
        ("rho", 1.1),
        ("kappa", -1),
        ("delta", 1),
        ("delta", -1),
        ("q", 0),
        ("B0", 0),
        ("dR_dr", -10),
    ],
)
def test_actual_physical_domain_findings(tmp_path: Path, field: str, value: float) -> None:
    """Actual Miller physical-domain exceptions become authored parameter findings without changing the core equations."""
    payload = declaration()
    payload["cases"][0]["parameters"][field] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail"
    assert any(
        error["field"] == "parameters"
        and error["error"] == "parameters must satisfy the local Miller equilibrium domain"
        for error in report["errors"]
    )


@pytest.mark.parametrize("field", ["s_kappa", "s_delta", "s_hat", "alpha_MHD"])
@pytest.mark.parametrize("value", [None, [], "not-a-number", float("nan"), float("inf"), 10**400])
def test_optional_conversion_findings(tmp_path: Path, field: str, value: object) -> None:
    """Recognized optional values refuse failed/nonfinite/overflow conversions rather than leaking exceptions."""
    payload = declaration()
    payload["cases"][0]["parameters"][field] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and any(error["field"] == field for error in report["errors"])


def test_optional_coercion_and_unused_parameters(tmp_path: Path) -> None:
    """Original optional boolean/numeric-string coercion and ignored extras retain the actual admitted comparison domain."""
    payload = declaration()
    payload["cases"][0]["parameters"].update(s_kappa=False, s_delta="0", s_hat=True, alpha_MHD="0.0", unused=[])
    report = inspect(tmp_path, payload)
    assert report["status"] == "pass" and report["cases"] == 3


@pytest.mark.parametrize("field", SAMPLES)
@pytest.mark.parametrize("value", [None, [], "1", True, float("nan"), float("inf"), 10**400])
def test_sample_refusals(tmp_path: Path, field: str, value: object) -> None:
    """All nine sample fields require finite nonboolean numeric values, including theta before grid selection."""
    payload = declaration()
    payload["cases"][0]["sample_points"][0][field] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and any(error["field"] == field for error in report["errors"])


def test_nonfinite_actual_comparison(tmp_path: Path) -> None:
    """Real finite extreme inputs that overflow computed magnetic fields cannot pass a bounded comparison."""
    payload = declaration()
    payload["cases"][0]["parameters"]["B0"] = 1e308
    with pytest.warns(RuntimeWarning):
        report = inspect(tmp_path, payload)
    assert report["status"] == "fail"
    assert any(error["field"] in {"B_toroidal", "b_dot_grad_theta"} for error in report["errors"])
    assert report["public_claims"]["bounded_local_miller_geometry_reference"] is False
