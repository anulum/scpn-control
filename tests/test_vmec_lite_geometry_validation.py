# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — VMEC-lite geometry validation tests
"""Tests for VMEC-lite spectral-geometry exact validation."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from validation.validate_vmec_lite_geometry import (
    VMEC_LITE_GEOMETRY_SCHEMA_VERSION,
    VMECLiteGeometryValidationResult,
    axisymmetric_boundary_errors,
    basis_mode_count_error,
    build_evidence,
    bfield_coefficient_error,
    default_case,
    fixed_boundary_radial_scaling_error,
    q_iota_reciprocal_error,
    spectral_evaluation_error,
    validate_evidence_payload,
    validate_vmec_lite_geometry,
)


@pytest.fixture(scope="module")
def result() -> VMECLiteGeometryValidationResult:
    """Run the VMEC-lite geometry validator once for module tests."""
    return validate_vmec_lite_geometry()


def test_basis_mode_count_matches_declared_truncation() -> None:
    assert basis_mode_count_error(default_case()) == 0


def test_spectral_evaluation_matches_manual_fourier_series() -> None:
    errors = spectral_evaluation_error(default_case())
    assert errors["cosine_max_abs_error"] < 1e-12
    assert errors["sine_max_abs_error"] < 1e-12


def test_axisymmetric_boundary_coefficients_match_closed_form() -> None:
    errors = axisymmetric_boundary_errors(default_case())
    assert errors["R00_abs_error"] < 1e-12
    assert errors["R10_abs_error"] < 1e-12
    assert errors["R20_abs_error"] < 1e-12
    assert errors["Z10_abs_error"] < 1e-12


def test_fixed_boundary_radial_scaling_matches_s_power_law() -> None:
    assert fixed_boundary_radial_scaling_error(default_case()) < 1e-12


def test_q_iota_reciprocal_contract_is_exact() -> None:
    assert q_iota_reciprocal_error(default_case()) < 1e-12


def test_bfield_coefficient_construction_matches_declared_formula() -> None:
    assert bfield_coefficient_error(default_case()) < 1e-12


def test_vmec_lite_geometry_validation_passes(result: VMECLiteGeometryValidationResult) -> None:
    assert result.passed is True
    assert result.basis_passed is True
    assert result.boundary_passed is True
    assert result.radial_scaling_passed is True
    assert result.q_iota_passed is True
    assert result.bfield_passed is True


def test_vmec_lite_geometry_evidence_roundtrip_is_sealed(result: VMECLiteGeometryValidationResult) -> None:
    evidence = build_evidence(result, target_id="vmec-lite-test")
    assert evidence["schema_version"] == VMEC_LITE_GEOMETRY_SCHEMA_VERSION
    assert evidence["public_claim_allowed"] is False
    assert validate_evidence_payload(evidence) is True


def test_vmec_lite_geometry_evidence_tamper_is_rejected(result: VMECLiteGeometryValidationResult) -> None:
    evidence = build_evidence(result, target_id="vmec-lite-test")
    evidence["basis_mode_count_error"] = 1
    with pytest.raises(ValueError, match="payload_sha256 does not match"):
        validate_evidence_payload(evidence)


def test_vmec_lite_geometry_evidence_rejects_bad_schema(result: VMECLiteGeometryValidationResult) -> None:
    evidence = build_evidence(result, target_id="vmec-lite-test")
    evidence["schema_version"] = "scpn-control.vmec-lite-geometry.v0"
    with pytest.raises(ValueError, match="unsupported"):
        validate_evidence_payload(evidence)


def test_vmec_lite_geometry_evidence_rejects_empty_target(result: VMECLiteGeometryValidationResult) -> None:
    with pytest.raises(ValueError, match="target_id"):
        build_evidence(result, target_id=" ")


def test_vmec_lite_geometry_cli_text_output_passes(capsys) -> None:
    import validation.validate_vmec_lite_geometry as mod

    assert mod.main([]) == 0
    out = capsys.readouterr().out
    assert "Status: pass" in out
    assert "VMEC-lite spectral-geometry validation" in out


def test_vmec_lite_geometry_cli_json_and_report(capsys, tmp_path) -> None:
    import validation.validate_vmec_lite_geometry as mod

    report = tmp_path / "vmec_lite_geometry.json"
    assert mod.main(["--json-out", "--report", str(report)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == VMEC_LITE_GEOMETRY_SCHEMA_VERSION
    assert report.exists() and report.with_suffix(".md").exists()
    assert validate_evidence_payload(json.loads(report.read_text(encoding="utf-8"))) is True
    assert "VMEC-lite Spectral-Geometry Validation" in report.with_suffix(".md").read_text(encoding="utf-8")


def test_vmec_lite_geometry_cli_returns_one_on_failure(monkeypatch, capsys) -> None:
    import validation.validate_vmec_lite_geometry as mod

    real = mod.validate_vmec_lite_geometry
    monkeypatch.setattr(mod, "validate_vmec_lite_geometry", lambda: replace(real(), passed=False))
    assert mod.main([]) == 1
    assert "Status: fail" in capsys.readouterr().out
