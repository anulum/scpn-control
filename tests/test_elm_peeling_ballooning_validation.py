# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — ELM peeling-ballooning validation tests
"""Tests for the ELM peeling-ballooning and crash analytic validation."""

from __future__ import annotations

import json

import pytest

from validation.validate_elm_peeling_ballooning import (
    ELM_PEELING_BALLOONING_SCHEMA_VERSION,
    ELMConfig,
    ELMValidationResult,
    ballooning_limit_rel_error,
    boundary_checks,
    build_evidence,
    crash_checks,
    default_config,
    peeling_limit_rel_error,
    peeling_scaling_checks,
    validate_elm_peeling_ballooning,
    validate_evidence_payload,
)


@pytest.fixture(scope="module")
def result() -> ELMValidationResult:
    """Run the ELM validation once for the module."""
    return validate_elm_peeling_ballooning()


# ── Exact closed-form references ─────────────────────────────────────


def test_ballooning_limit_matches_closed_form() -> None:
    assert ballooning_limit_rel_error(default_config()) < 1e-12


def test_peeling_limit_matches_closed_form() -> None:
    assert peeling_limit_rel_error(default_config()) < 1e-12


def test_peeling_scaling_laws_are_exact() -> None:
    checks = {c.name: c for c in peeling_scaling_checks(default_config())}
    assert checks["q95_inverse"].measured_ratio == pytest.approx(0.5, rel=1e-12)
    assert checks["n_mode_inverse_sqrt"].measured_ratio == pytest.approx(0.5, rel=1e-12)
    assert checks["aspect_ratio_linear"].measured_ratio == pytest.approx(2.0, rel=1e-12)


def test_elliptical_boundary_is_consistent() -> None:
    boundary = boundary_checks(default_config())
    assert boundary.on_boundary_margin_abs < 1e-12
    assert boundary.margin_formula_rel_error < 1e-12
    assert boundary.interior_stable is True
    assert boundary.exterior_unstable is True
    assert boundary.flag_margin_consistent is True


def test_elm_crash_conserves_energy_fraction() -> None:
    crash = crash_checks()
    assert crash.energy_loss_rel_error < 1e-12
    assert crash.stored_energy_ratio_rel_error < 1e-12
    assert crash.peak_heat_flux_rel_error < 1e-12
    assert crash.profile_pedestal_drop_rel_error < 1e-12
    assert crash.profile_core_unchanged is True


# ── Aggregate result ─────────────────────────────────────────────────


def test_overall_validation_passes(result: ELMValidationResult) -> None:
    assert result.passed is True
    assert result.ballooning_passed
    assert result.peeling_passed
    assert result.boundary_passed
    assert result.crash_passed
    assert len(result.peeling_scaling) == 3


def test_validation_is_deterministic() -> None:
    a = validate_elm_peeling_ballooning()
    b = validate_elm_peeling_ballooning()
    assert a.peeling_rel_error == b.peeling_rel_error
    assert a.crash.stored_energy_ratio_rel_error == b.crash.stored_energy_ratio_rel_error


# ── Config contract ──────────────────────────────────────────────────


def test_config_rejects_minor_radius_above_major() -> None:
    with pytest.raises(ValueError, match="a must be smaller than r0"):
        ELMConfig(q95=3.5, kappa=1.7, delta=0.33, a=2.0, r0=1.7)


def test_config_rejects_nonpositive_q95() -> None:
    with pytest.raises(ValueError, match="q95 must be positive"):
        ELMConfig(q95=0.0, kappa=1.7, delta=0.33, a=0.5, r0=1.7)


def test_config_rejects_nonfinite_delta() -> None:
    with pytest.raises(ValueError, match="delta must be finite"):
        ELMConfig(q95=3.5, kappa=1.7, delta=float("inf"), a=0.5, r0=1.7)


def test_config_rejects_nonnumeric_kappa() -> None:
    with pytest.raises(ValueError, match="kappa must be a finite number"):
        ELMConfig(q95=3.5, kappa="tall", delta=0.33, a=0.5, r0=1.7)  # type: ignore[arg-type]


# ── Evidence seal ────────────────────────────────────────────────────


def test_evidence_roundtrip_is_sealed_and_passing(result: ELMValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    assert evidence["schema_version"] == ELM_PEELING_BALLOONING_SCHEMA_VERSION
    assert validate_evidence_payload(evidence) is True
    assert evidence["crash"]["profile_core_unchanged"] is True
    assert len(evidence["peeling_scaling"]) == 3


def test_evidence_tamper_is_rejected(result: ELMValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["ballooning_rel_error"] = 1.0
    with pytest.raises(ValueError, match="payload_sha256 does not match"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_empty_target_id(result: ELMValidationResult) -> None:
    with pytest.raises(ValueError, match="target_id"):
        build_evidence(result, target_id="   ")


def test_evidence_rejects_unknown_schema(result: ELMValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["schema_version"] = "scpn-control.unknown.v9"
    with pytest.raises(ValueError, match="unsupported"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_non_hex_seal(result: ELMValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["payload_sha256"] = "notadigest"
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        validate_evidence_payload(evidence)


# ── CLI / report writer ──────────────────────────────────────────────


def test_main_text_output_passes(capsys) -> None:
    import validation.validate_elm_peeling_ballooning as mod

    assert mod.main([]) == 0
    out = capsys.readouterr().out
    assert "Status: pass" in out
    assert "crash:" in out


def test_main_json_output_and_report(capsys, tmp_path) -> None:
    import validation.validate_elm_peeling_ballooning as mod

    report = tmp_path / "elm.json"
    assert mod.main(["--json-out", "--report", str(report)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == ELM_PEELING_BALLOONING_SCHEMA_VERSION
    assert report.exists() and report.with_suffix(".md").exists()
    assert validate_evidence_payload(json.loads(report.read_text())) is True
    assert "Peeling-Ballooning" in report.with_suffix(".md").read_text()


def test_main_returns_one_on_failure(monkeypatch, capsys) -> None:
    import validation.validate_elm_peeling_ballooning as mod

    real = mod.validate_elm_peeling_ballooning
    # The stored-energy ratio carries a ~1e-16 residual, so an impossibly tight
    # tolerance forces the crash gate to fail.
    monkeypatch.setattr(mod, "validate_elm_peeling_ballooning", lambda: real(exact_tol=1e-30))
    assert mod.main([]) == 1
    assert "Status: fail" in capsys.readouterr().out
