# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Current-drive validation tests
"""Tests for the auxiliary current-drive analytic validation."""

from __future__ import annotations

import json

import pytest

from validation.validate_current_drive import (
    CURRENT_DRIVE_SCHEMA_VERSION,
    CurrentDriveValidationResult,
    build_evidence,
    critical_energy_rel_error,
    critical_energy_scaling_check,
    deposition_centroid_rel_error,
    deposition_conservation_rel_errors,
    eccd_efficiency_rel_error,
    jcd_proportionality_rel_error,
    launch_factor_is_maximised_at_unity,
    nbi_current_chain_rel_error,
    slowing_down_scaling_checks,
    validate_current_drive,
    validate_evidence_payload,
    _rho_grid,
)


@pytest.fixture(scope="module")
def result() -> CurrentDriveValidationResult:
    """Run the current-drive validation once for the module."""
    return validate_current_drive()


# ── Exact closed-form references ─────────────────────────────────────


def test_deposition_power_is_conserved() -> None:
    errors = deposition_conservation_rel_errors(_rho_grid())
    assert set(errors) == {"eccd", "lhcd", "nbi"}
    assert max(errors.values()) < 1e-12


def test_deposition_centroid_matches_midpoint() -> None:
    assert deposition_centroid_rel_error(_rho_grid()) < 1e-12


def test_stix_critical_energy_matches_closed_form() -> None:
    assert critical_energy_rel_error() < 1e-12


def test_critical_energy_mass_ratio_scaling() -> None:
    check = critical_energy_scaling_check()
    assert check.measured_ratio == pytest.approx(2.0 ** (2.0 / 3.0), rel=1e-12)


def test_slowing_down_scaling_is_exact() -> None:
    checks = {c.name: c for c in slowing_down_scaling_checks()}
    assert checks["temperature_1.5"].measured_ratio == pytest.approx(4.0**1.5, rel=1e-12)
    assert checks["density_inverse"].measured_ratio == pytest.approx(0.5, rel=1e-12)
    assert checks["z_eff_inverse"].measured_ratio == pytest.approx(0.5, rel=1e-12)


def test_prater_eccd_efficiency_matches_closed_form() -> None:
    assert eccd_efficiency_rel_error() < 1e-12


def test_launch_factor_peaks_at_unity_parallel_index() -> None:
    assert launch_factor_is_maximised_at_unity() is True


def test_jcd_proportionality_is_exact() -> None:
    assert jcd_proportionality_rel_error(_rho_grid()) < 1e-12


def test_nbi_current_chain_is_exact() -> None:
    assert nbi_current_chain_rel_error(_rho_grid()) < 1e-12


# ── Aggregate result ─────────────────────────────────────────────────


def test_overall_validation_passes(result: CurrentDriveValidationResult) -> None:
    assert result.passed is True
    assert result.deposition_passed
    assert result.centroid_passed
    assert result.critical_energy_passed
    assert result.slowing_down_passed
    assert result.eccd_efficiency_passed
    assert result.jcd_passed
    assert result.nbi_chain_passed
    assert len(result.slowing_down_scaling) == 3
    assert result.grid_points == 401


def test_validation_is_deterministic() -> None:
    a = validate_current_drive()
    b = validate_current_drive()
    assert a.max_deposition_conservation_rel_error == b.max_deposition_conservation_rel_error
    assert a.nbi_current_chain_rel_error == b.nbi_current_chain_rel_error


# ── Grid contract ────────────────────────────────────────────────────


def test_validation_rejects_coarse_grid() -> None:
    with pytest.raises(ValueError, match="at least 51"):
        validate_current_drive(nr=10)


def test_validation_rejects_nonpositive_grid() -> None:
    with pytest.raises(ValueError, match="nr must be a positive integer"):
        validate_current_drive(nr=0)


# ── Evidence seal ────────────────────────────────────────────────────


def test_evidence_roundtrip_is_sealed_and_passing(result: CurrentDriveValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    assert evidence["schema_version"] == CURRENT_DRIVE_SCHEMA_VERSION
    assert validate_evidence_payload(evidence) is True
    assert evidence["launch_factor_maximised"] is True
    assert len(evidence["slowing_down_scaling"]) == 3


def test_evidence_tamper_is_rejected(result: CurrentDriveValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["max_deposition_conservation_rel_error"] = 1.0
    with pytest.raises(ValueError, match="payload_sha256 does not match"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_empty_target_id(result: CurrentDriveValidationResult) -> None:
    with pytest.raises(ValueError, match="target_id"):
        build_evidence(result, target_id="   ")


def test_evidence_rejects_unknown_schema(result: CurrentDriveValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["schema_version"] = "scpn-control.unknown.v9"
    with pytest.raises(ValueError, match="unsupported"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_non_hex_seal(result: CurrentDriveValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["payload_sha256"] = "notadigest"
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        validate_evidence_payload(evidence)


# ── CLI / report writer ──────────────────────────────────────────────


def test_main_text_output_passes(capsys) -> None:
    import validation.validate_current_drive as mod

    assert mod.main([]) == 0
    out = capsys.readouterr().out
    assert "Status: pass" in out
    assert "NBI chain" in out


def test_main_json_output_and_report(capsys, tmp_path) -> None:
    import validation.validate_current_drive as mod

    report = tmp_path / "cd.json"
    assert mod.main(["--json-out", "--report", str(report)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == CURRENT_DRIVE_SCHEMA_VERSION
    assert report.exists() and report.with_suffix(".md").exists()
    assert validate_evidence_payload(json.loads(report.read_text())) is True
    assert "Auxiliary Current-Drive" in report.with_suffix(".md").read_text()


def test_main_returns_one_on_failure(monkeypatch, capsys) -> None:
    import validation.validate_current_drive as mod

    real = mod.validate_current_drive
    monkeypatch.setattr(mod, "validate_current_drive", lambda: real(exact_tol=1e-30))
    assert mod.main([]) == 1
    assert "Status: fail" in capsys.readouterr().out
