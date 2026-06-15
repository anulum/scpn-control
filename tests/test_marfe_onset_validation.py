# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MARFE onset and density-limit validation tests
"""Tests for MARFE radiation-condensation closed-form validation."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from validation.validate_marfe_onset import (
    MARFE_ONSET_SCHEMA_VERSION,
    MARFEConfig,
    MARFEValidationResult,
    build_evidence,
    connection_length_rel_error,
    critical_density_boundary,
    default_config,
    front_detection_thresholds,
    greenwald_limit_rel_error,
    greenwald_scaling_checks,
    marfe_limit_rel_error,
    marfe_limit_scaling_checks,
    onset_temperature_membership,
    scan_boundary_classification,
    validate_evidence_payload,
    validate_marfe_onset,
)


@pytest.fixture(scope="module")
def config() -> MARFEConfig:
    """Default ITER-like geometry and tungsten impurity setup."""
    return default_config()


@pytest.fixture(scope="module")
def result() -> MARFEValidationResult:
    """Run the MARFE validation once for the module."""
    return validate_marfe_onset()


def test_greenwald_limit_matches_closed_form(config: MARFEConfig) -> None:
    assert greenwald_limit_rel_error(config) < 1e-12


def test_greenwald_scaling_laws_are_exact(config: MARFEConfig) -> None:
    checks = {check.name: check for check in greenwald_scaling_checks(config)}
    assert checks["current_linear"].measured_ratio == pytest.approx(2.0, rel=1e-12)
    assert checks["minor_radius_inverse_square"].measured_ratio == pytest.approx(0.25, rel=1e-12)
    assert all(check.rel_error < 1e-12 for check in checks.values())


def test_marfe_limit_matches_declared_scaling(config: MARFEConfig) -> None:
    assert marfe_limit_rel_error(config) < 1e-12


def test_marfe_limit_scaling_laws_are_exact(config: MARFEConfig) -> None:
    checks = {check.name: check for check in marfe_limit_scaling_checks(config)}
    assert checks["power_sqrt"].measured_ratio == pytest.approx(2.0**0.5, rel=1e-12)
    assert checks["impurity_inverse_sqrt"].measured_ratio == pytest.approx(2.0**-0.5, rel=1e-12)
    assert all(check.rel_error < 1e-12 for check in checks.values())


def test_connection_length_matches_pi_q95_r0(config: MARFEConfig) -> None:
    assert connection_length_rel_error(config) < 1e-12


def test_critical_density_brackets_growth_rate(config: MARFEConfig) -> None:
    boundary = critical_density_boundary(config)
    assert boundary.critical_density_20 > 0.0
    assert boundary.low_density_growth_rate_s < 0.0
    assert boundary.high_density_growth_rate_s > 0.0


def test_onset_temperature_lies_in_negative_cooling_slope(config: MARFEConfig) -> None:
    onset = onset_temperature_membership(config)
    assert onset.onset_temperature_eV > config.temperature_scan_eV[0]
    assert onset.slope_at_onset < 0.0
    assert onset.is_scan_member is True


def test_scan_boundary_classifies_below_and_above_limit(config: MARFEConfig) -> None:
    boundary = scan_boundary_classification(config)
    assert boundary.critical_density_20 > 0.0
    assert boundary.below_limit_state == 1
    assert boundary.above_limit_state == -1


def test_front_detection_thresholds_are_exact() -> None:
    thresholds = front_detection_thresholds()
    assert thresholds.below_threshold_is_marfe is True
    assert thresholds.at_temperature_threshold_is_marfe is False
    assert thresholds.hot_profile_is_marfe is False


def test_overall_validation_passes(result: MARFEValidationResult) -> None:
    assert result.passed is True
    assert result.greenwald_passed is True
    assert result.marfe_limit_passed is True
    assert result.condensation_passed is True
    assert result.scan_passed is True
    assert result.front_detection_passed is True


def test_validation_is_deterministic() -> None:
    a = validate_marfe_onset()
    b = validate_marfe_onset()
    assert a.greenwald_limit_rel_error == b.greenwald_limit_rel_error
    assert a.critical_density.critical_density_20 == b.critical_density.critical_density_20


def test_config_rejects_minor_radius_above_major() -> None:
    with pytest.raises(ValueError, match="minor_radius_m must be smaller"):
        MARFEConfig(
            major_radius_m=2.0,
            minor_radius_m=2.0,
            q95=3.0,
            plasma_current_ma=15.0,
            p_sol_mw=40.0,
            impurity="W",
            impurity_fraction=1e-4,
            electron_density_20=2.0,
            temperature_eV=2000.0,
            k_parallel_m_inv=0.01,
            kappa_parallel=2000.0,
            temperature_scan_eV=(50.0, 500.0, 2000.0),
        )


def test_config_rejects_nonmonotone_temperature_scan() -> None:
    with pytest.raises(ValueError, match="temperature_scan_eV"):
        MARFEConfig(
            major_radius_m=6.2,
            minor_radius_m=2.0,
            q95=3.0,
            plasma_current_ma=15.0,
            p_sol_mw=40.0,
            impurity="W",
            impurity_fraction=1e-4,
            electron_density_20=2.0,
            temperature_eV=2000.0,
            k_parallel_m_inv=0.01,
            kappa_parallel=2000.0,
            temperature_scan_eV=(50.0, 2000.0, 500.0),
        )


def test_evidence_roundtrip_is_sealed_and_passing(result: MARFEValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    assert evidence["schema_version"] == MARFE_ONSET_SCHEMA_VERSION
    assert validate_evidence_payload(evidence) is True
    assert len(evidence["greenwald_scaling"]) == 2
    assert len(evidence["marfe_limit_scaling"]) == 2


def test_evidence_tamper_is_rejected(result: MARFEValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["greenwald_limit_rel_error"] = 1.0
    with pytest.raises(ValueError, match="payload_sha256 does not match"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_empty_target_id(result: MARFEValidationResult) -> None:
    with pytest.raises(ValueError, match="target_id"):
        build_evidence(result, target_id="   ")


def test_evidence_rejects_unknown_schema(result: MARFEValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["schema_version"] = "scpn-control.unknown.v9"
    with pytest.raises(ValueError, match="unsupported"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_non_hex_seal(result: MARFEValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["payload_sha256"] = "notadigest"
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        validate_evidence_payload(evidence)


def test_main_text_output_passes(capsys) -> None:
    import validation.validate_marfe_onset as mod

    assert mod.main([]) == 0
    out = capsys.readouterr().out
    assert "Status: pass" in out
    assert "critical density" in out


def test_main_json_output_and_report(capsys, tmp_path) -> None:
    import validation.validate_marfe_onset as mod

    report = tmp_path / "marfe.json"
    assert mod.main(["--json-out", "--report", str(report)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == MARFE_ONSET_SCHEMA_VERSION
    assert report.exists() and report.with_suffix(".md").exists()
    assert validate_evidence_payload(json.loads(report.read_text())) is True
    assert "MARFE Radiation-Condensation" in report.with_suffix(".md").read_text()


def test_main_returns_one_on_failure(monkeypatch, capsys) -> None:
    import validation.validate_marfe_onset as mod

    real = mod.validate_marfe_onset
    monkeypatch.setattr(mod, "validate_marfe_onset", lambda: replace(real(), passed=False))
    assert mod.main([]) == 1
    assert "Status: fail" in capsys.readouterr().out
