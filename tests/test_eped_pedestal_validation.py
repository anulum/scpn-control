# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — EPED pedestal validation tests
"""Tests for the EPED pedestal model analytic validation."""

from __future__ import annotations

import json

import pytest

from validation.validate_eped_pedestal import (
    EPED_PEDESTAL_SCHEMA_VERSION,
    EPEDValidationResult,
    alpha_inversion_rel_error,
    beta_p_rel_error,
    build_evidence,
    collisionality_rel_error,
    collisionless_identity_holds,
    default_config,
    kbm_fixed_point_rel_error,
    q95_rel_error,
    shaping_checks,
    temperature_rel_error,
    validate_eped_pedestal,
    validate_evidence_payload,
)


@pytest.fixture(scope="module")
def result() -> EPEDValidationResult:
    """Run the EPED pedestal validation once for the module."""
    return validate_eped_pedestal()


# ── Exact construction relations ─────────────────────────────────────


def test_q95_matches_closed_form() -> None:
    assert q95_rel_error(default_config()) < 1e-12


def test_alpha_inversion_pressure_relation() -> None:
    assert alpha_inversion_rel_error(default_config()) < 1e-12


def test_poloidal_beta_definition() -> None:
    assert beta_p_rel_error(default_config()) < 1e-12


def test_pedestal_temperature_ideal_gas() -> None:
    assert temperature_rel_error(default_config()) < 1e-12


def test_collisionality_correction() -> None:
    assert collisionality_rel_error(default_config()) < 1e-12


def test_collisionless_identity_at_zero_nu_star() -> None:
    assert collisionless_identity_holds(default_config()) is True


def test_shaping_factor_reference_and_monotonicity() -> None:
    check = shaping_checks()
    assert check.reference_value == pytest.approx(1.0)
    assert check.reference_rel_error < 1e-12
    assert check.monotonic_in_triangularity is True


def test_kbm_fixed_point_within_iteration_tolerance() -> None:
    assert kbm_fixed_point_rel_error(default_config()) < 3e-2


# ── Aggregate result ─────────────────────────────────────────────────


def test_overall_validation_passes(result: EPEDValidationResult) -> None:
    assert result.passed is True
    assert result.construction_passed
    assert result.collisionality_passed
    assert result.shaping_passed
    assert result.kbm_passed


def test_validation_is_deterministic() -> None:
    a = validate_eped_pedestal()
    b = validate_eped_pedestal()
    assert a.alpha_inversion_rel_error == b.alpha_inversion_rel_error
    assert a.kbm_fixed_point_rel_error == b.kbm_fixed_point_rel_error


# ── Evidence seal ────────────────────────────────────────────────────


def test_evidence_roundtrip_is_sealed_and_passing(result: EPEDValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    assert evidence["schema_version"] == EPED_PEDESTAL_SCHEMA_VERSION
    assert validate_evidence_payload(evidence) is True
    assert evidence["collisionless_identity"] is True


def test_evidence_tamper_is_rejected(result: EPEDValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["kbm_fixed_point_rel_error"] = 1.0
    with pytest.raises(ValueError, match="payload_sha256 does not match"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_empty_target_id(result: EPEDValidationResult) -> None:
    with pytest.raises(ValueError, match="target_id"):
        build_evidence(result, target_id="   ")


def test_evidence_rejects_unknown_schema(result: EPEDValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["schema_version"] = "scpn-control.unknown.v9"
    with pytest.raises(ValueError, match="unsupported"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_non_hex_seal(result: EPEDValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["payload_sha256"] = "notadigest"
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        validate_evidence_payload(evidence)


# ── CLI / report writer ──────────────────────────────────────────────


def test_main_text_output_passes(capsys) -> None:
    import validation.validate_eped_pedestal as mod

    assert mod.main([]) == 0
    out = capsys.readouterr().out
    assert "Status: pass" in out
    assert "KBM constraint" in out


def test_main_json_output_and_report(capsys, tmp_path) -> None:
    import validation.validate_eped_pedestal as mod

    report = tmp_path / "eped.json"
    assert mod.main(["--json-out", "--report", str(report)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == EPED_PEDESTAL_SCHEMA_VERSION
    assert report.exists() and report.with_suffix(".md").exists()
    assert validate_evidence_payload(json.loads(report.read_text())) is True
    assert "EPED Pedestal" in report.with_suffix(".md").read_text()


def test_main_returns_one_on_failure(monkeypatch, capsys) -> None:
    import validation.validate_eped_pedestal as mod

    real = mod.validate_eped_pedestal
    # Construction errors are exactly zero, so tighten the KBM tolerance to force failure.
    monkeypatch.setattr(mod, "validate_eped_pedestal", lambda: real(kbm_tol=1e-30))
    assert mod.main([]) == 1
    assert "Status: fail" in capsys.readouterr().out
