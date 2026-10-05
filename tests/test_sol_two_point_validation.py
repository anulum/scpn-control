# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Two-point SOL model validation tests
"""Tests for the two-point scrape-off-layer model analytic validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from validation.validate_sol_two_point import (
    SOL_TWO_POINT_SCHEMA_VERSION,
    SOLConfig,
    SOLValidationResult,
    build_evidence,
    conduction_integral_rel_error,
    connection_length_rel_error,
    default_config,
    detachment_boundary,
    eich_scaling_checks,
    flux_mapping_rel_error,
    peak_heat_flux_rel_error,
    pressure_balance_rel_error,
    validate_evidence_payload,
    validate_sol_two_point,
)


@pytest.fixture(scope="module")
def result() -> SOLValidationResult:
    """Run the two-point SOL validation once for the module."""
    return validate_sol_two_point()


# ── Exact closed-form references ─────────────────────────────────────


def test_connection_length_matches_pi_q95_r0() -> None:
    """Connection length matches pi q95 r0."""
    assert connection_length_rel_error(default_config()) < 1e-12


def test_flux_mapping_matches_closed_form() -> None:
    """Flux mapping matches closed form."""
    config = default_config()
    for p, n in ((10.0, 3.0), (20.0, 5.0)):
        assert flux_mapping_rel_error(config, p, n) < 1e-12


def test_upstream_conduction_integral_holds() -> None:
    """Upstream conduction integral holds."""
    config = default_config()
    for p, n in ((10.0, 3.0), (5.0, 1.5)):
        assert conduction_integral_rel_error(config, p, n) < 1e-12


def test_pressure_balance_holds() -> None:
    """Pressure balance holds."""
    config = default_config()
    for p, n in ((10.0, 3.0), (20.0, 5.0)):
        assert pressure_balance_rel_error(config, p, n) < 1e-12


def test_eich_scaling_exponents_are_exact() -> None:
    """Eich scaling exponents are exact."""
    checks = {c.name: c for c in eich_scaling_checks(default_config(), 10.0)}
    assert checks["b_pol_-0.92"].measured_ratio == pytest.approx(2.0**-0.92, rel=1e-12)
    assert checks["epsilon_0.42"].measured_ratio == pytest.approx(2.0**0.42, rel=1e-12)
    assert checks["power_-0.02"].measured_ratio == pytest.approx(2.0**-0.02, rel=1e-12)
    assert checks["major_radius_0.04"].measured_ratio == pytest.approx(2.0**0.04, rel=1e-12)


def test_peak_heat_flux_matches_closed_form() -> None:
    """Peak heat flux matches closed form."""
    assert peak_heat_flux_rel_error(default_config(), 10.0) < 1e-12


# ── Detachment boundary ──────────────────────────────────────────────


def test_detachment_boundary_brackets_critical_density() -> None:
    """Detachment boundary brackets critical density."""
    boundary = detachment_boundary(100.0, 20.0)
    assert boundary.critical_density_19 > 0.0
    assert boundary.detached_below_critical is False
    assert boundary.detached_above_critical is True


def test_detachment_boundary_rejects_nonpositive_flux() -> None:
    """Detachment boundary rejects nonpositive flux."""
    with pytest.raises(ValueError, match="q_par_mw_m2 must be positive"):
        detachment_boundary(0.0, 20.0)


# ── Aggregate result ─────────────────────────────────────────────────


def test_overall_validation_passes(result: SOLValidationResult) -> None:
    """Overall validation passes."""
    assert result.passed is True
    assert result.connection_passed
    assert result.flux_mapping_passed
    assert result.conduction_passed
    assert result.pressure_passed
    assert result.scaling_passed
    assert result.peak_flux_passed
    assert result.detachment_passed
    assert len(result.scaling) == 4
    assert len(result.operating_points) == 3


def test_validation_is_deterministic() -> None:
    """Validation is deterministic."""
    a = validate_sol_two_point()
    b = validate_sol_two_point()
    assert a.max_conduction_rel_error == b.max_conduction_rel_error
    assert a.detachment.critical_density_19 == b.detachment.critical_density_19


def test_validation_rejects_empty_operating_points() -> None:
    """Validation rejects empty operating points."""
    with pytest.raises(ValueError, match="at least one operating point"):
        validate_sol_two_point(operating_points=())


# ── Config contract ──────────────────────────────────────────────────


def test_config_rejects_minor_radius_above_major() -> None:
    """Config rejects minor radius above major."""
    with pytest.raises(ValueError, match="a must be smaller than r0"):
        SOLConfig(r0=0.5, a=1.7, q95=3.5, b_pol=0.4)


def test_config_rejects_nonpositive_b_pol() -> None:
    """Config rejects nonpositive b pol."""
    with pytest.raises(ValueError, match="b_pol must be positive"):
        SOLConfig(r0=1.7, a=0.5, q95=3.5, b_pol=0.0)


def test_config_rejects_nonfinite_q95() -> None:
    """Config rejects nonfinite q95."""
    with pytest.raises(ValueError, match="q95 must be finite"):
        SOLConfig(r0=1.7, a=0.5, q95=float("inf"), b_pol=0.4)


def test_config_rejects_nonnumeric_radius() -> None:
    """Config rejects nonnumeric radius."""
    with pytest.raises(ValueError, match="r0 must be a finite number"):
        SOLConfig(r0=cast(float, "big"), a=0.5, q95=3.5, b_pol=0.4)


def test_config_epsilon() -> None:
    """Config epsilon."""
    assert default_config().epsilon == pytest.approx(0.5 / 1.7)


# ── Evidence seal ────────────────────────────────────────────────────


def test_evidence_roundtrip_is_sealed_and_passing(result: SOLValidationResult) -> None:
    """Evidence roundtrip is sealed and passing."""
    evidence = build_evidence(result, target_id="test-target")
    assert evidence["schema_version"] == SOL_TWO_POINT_SCHEMA_VERSION
    assert validate_evidence_payload(evidence) is True
    assert len(evidence["scaling"]) == 4


def test_evidence_tamper_is_rejected(result: SOLValidationResult) -> None:
    """Evidence tamper is rejected."""
    evidence = build_evidence(result, target_id="test-target")
    evidence["max_conduction_rel_error"] = 1.0
    with pytest.raises(ValueError, match="payload_sha256 does not match"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_empty_target_id(result: SOLValidationResult) -> None:
    """Evidence rejects empty target id."""
    with pytest.raises(ValueError, match="target_id"):
        build_evidence(result, target_id="   ")


def test_evidence_rejects_unknown_schema(result: SOLValidationResult) -> None:
    """Evidence rejects unknown schema."""
    evidence = build_evidence(result, target_id="test-target")
    evidence["schema_version"] = "scpn-control.unknown.v9"
    with pytest.raises(ValueError, match="unsupported"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_non_hex_seal(result: SOLValidationResult) -> None:
    """Evidence rejects non hex seal."""
    evidence = build_evidence(result, target_id="test-target")
    evidence["payload_sha256"] = "notadigest"
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        validate_evidence_payload(evidence)


# ── CLI / report writer ──────────────────────────────────────────────


def test_main_text_output_passes(capsys: pytest.CaptureFixture[str]) -> None:
    """Main text output passes."""
    import validation.validate_sol_two_point as mod

    assert mod.main([]) == 0
    out = capsys.readouterr().out
    assert "Status: pass" in out
    assert "detachment boundary" in out


def test_main_json_output_and_report(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Main json output and report."""
    import validation.validate_sol_two_point as mod

    report = tmp_path / "sol.json"
    assert mod.main(["--json-out", "--report", str(report)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == SOL_TWO_POINT_SCHEMA_VERSION
    assert report.exists() and report.with_suffix(".md").exists()
    assert validate_evidence_payload(json.loads(report.read_text())) is True
    assert "Scrape-Off-Layer" in report.with_suffix(".md").read_text()


def test_main_returns_one_on_failure(capsys: pytest.CaptureFixture[str]) -> None:
    """Main returns one on failure."""
    import validation.validate_sol_two_point as mod

    assert mod.main(["--exact-tol", "1e-30"]) == 1
    assert "Status: fail" in capsys.readouterr().out
