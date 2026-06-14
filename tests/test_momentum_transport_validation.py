# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Momentum-transport validation tests
"""Tests for the toroidal momentum-transport analytic validation."""

from __future__ import annotations

import json

import pytest

from validation.validate_momentum_transport import (
    MOMENTUM_TRANSPORT_SCHEMA_VERSION,
    MomentumConfig,
    MomentumValidationResult,
    build_evidence,
    default_config,
    efield_force_balance_rel_error,
    efield_rotation_term_rel_error,
    exb_shearing_rel_error,
    mach_number_rel_error,
    nbi_torque_rel_error,
    nbi_zero_for_nonpositive_beam,
    rice_rel_error,
    rice_scaling_checks,
    turbulence_suppression_rel_error,
    validate_evidence_payload,
    validate_momentum_transport,
)


@pytest.fixture(scope="module")
def result() -> MomentumValidationResult:
    """Run the momentum-transport validation once for the module."""
    return validate_momentum_transport()


# ── Exact closed-form references ─────────────────────────────────────


def test_nbi_torque_matches_closed_form() -> None:
    config = default_config()
    assert nbi_torque_rel_error(config) < 1e-12
    assert nbi_zero_for_nonpositive_beam(config) is True


def test_radial_efield_rotation_term() -> None:
    assert efield_rotation_term_rel_error(default_config()) < 1e-9


def test_radial_efield_force_balance_linear_pressure() -> None:
    assert efield_force_balance_rel_error(default_config()) < 1e-12


def test_exb_shearing_rate_linear_rotation() -> None:
    assert exb_shearing_rel_error(default_config()) < 1e-12


def test_turbulence_suppression_factor() -> None:
    assert turbulence_suppression_rel_error() < 1e-12


def test_rice_intrinsic_velocity_and_scaling() -> None:
    assert rice_rel_error() < 1e-12
    checks = {c.name: c for c in rice_scaling_checks()}
    assert checks["stored_energy_linear"].measured_ratio == pytest.approx(2.0, rel=1e-12)
    assert checks["current_inverse"].measured_ratio == pytest.approx(0.5, rel=1e-12)


def test_mach_number_matches_closed_form() -> None:
    assert mach_number_rel_error(default_config()) < 1e-12


# ── Aggregate result ─────────────────────────────────────────────────


def test_overall_validation_passes(result: MomentumValidationResult) -> None:
    assert result.passed is True
    assert result.nbi_passed
    assert result.efield_passed
    assert result.exb_passed
    assert result.suppression_passed
    assert result.rice_passed
    assert result.mach_passed
    assert len(result.rice_scaling) == 2


def test_validation_is_deterministic() -> None:
    a = validate_momentum_transport()
    b = validate_momentum_transport()
    assert a.efield_force_balance_rel_error == b.efield_force_balance_rel_error
    assert a.mach_rel_error == b.mach_rel_error


# ── Config contract ──────────────────────────────────────────────────


def test_config_rejects_minor_radius_above_major() -> None:
    with pytest.raises(ValueError, match="a must be smaller than r0"):
        MomentumConfig(r0=0.5, a=1.7, b0=2.0, nr=51)


def test_config_rejects_nonpositive_field() -> None:
    with pytest.raises(ValueError, match="b0 must be positive"):
        MomentumConfig(r0=1.7, a=0.5, b0=0.0, nr=51)


def test_config_rejects_coarse_grid() -> None:
    with pytest.raises(ValueError, match="at least 5"):
        MomentumConfig(r0=1.7, a=0.5, b0=2.0, nr=3)


def test_config_rejects_nonnumeric_radius() -> None:
    with pytest.raises(ValueError, match="r0 must be a finite number"):
        MomentumConfig(r0="big", a=0.5, b0=2.0, nr=51)  # type: ignore[arg-type]


def test_config_rejects_nonfinite_radius() -> None:
    with pytest.raises(ValueError, match="r0 must be finite"):
        MomentumConfig(r0=float("inf"), a=0.5, b0=2.0, nr=51)


def test_config_rejects_nonpositive_grid() -> None:
    with pytest.raises(ValueError, match="nr must be a positive integer"):
        MomentumConfig(r0=1.7, a=0.5, b0=2.0, nr=0)


# ── Evidence seal ────────────────────────────────────────────────────


def test_evidence_roundtrip_is_sealed_and_passing(result: MomentumValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    assert evidence["schema_version"] == MOMENTUM_TRANSPORT_SCHEMA_VERSION
    assert validate_evidence_payload(evidence) is True
    assert evidence["nbi_zero_for_nonpositive_beam"] is True
    assert len(evidence["rice_scaling"]) == 2


def test_evidence_tamper_is_rejected(result: MomentumValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["mach_rel_error"] = 1.0
    with pytest.raises(ValueError, match="payload_sha256 does not match"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_empty_target_id(result: MomentumValidationResult) -> None:
    with pytest.raises(ValueError, match="target_id"):
        build_evidence(result, target_id="   ")


def test_evidence_rejects_unknown_schema(result: MomentumValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["schema_version"] = "scpn-control.unknown.v9"
    with pytest.raises(ValueError, match="unsupported"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_non_hex_seal(result: MomentumValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["payload_sha256"] = "notadigest"
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        validate_evidence_payload(evidence)


# ── CLI / report writer ──────────────────────────────────────────────


def test_main_text_output_passes(capsys) -> None:
    import validation.validate_momentum_transport as mod

    assert mod.main([]) == 0
    out = capsys.readouterr().out
    assert "Status: pass" in out
    assert "radial E_r" in out


def test_main_json_output_and_report(capsys, tmp_path) -> None:
    import validation.validate_momentum_transport as mod

    report = tmp_path / "mom.json"
    assert mod.main(["--json-out", "--report", str(report)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == MOMENTUM_TRANSPORT_SCHEMA_VERSION
    assert report.exists() and report.with_suffix(".md").exists()
    assert validate_evidence_payload(json.loads(report.read_text())) is True
    assert "Momentum-Transport" in report.with_suffix(".md").read_text()


def test_main_returns_one_on_failure(monkeypatch, capsys) -> None:
    import validation.validate_momentum_transport as mod

    real = mod.validate_momentum_transport
    # The E_r rotation term carries a ~1e-14 residual, so an impossibly tight
    # tolerance forces the field gate to fail.
    monkeypatch.setattr(mod, "validate_momentum_transport", lambda: real(exact_tol=1e-30))
    assert mod.main([]) == 1
    assert "Status: fail" in capsys.readouterr().out
