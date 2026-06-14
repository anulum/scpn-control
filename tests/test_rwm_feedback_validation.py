# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — RWM feedback validation tests
"""Tests for the resistive-wall-mode feedback analytic validation."""

from __future__ import annotations

import json
import math

import pytest

from validation.validate_rwm_feedback import (
    RWM_FEEDBACK_SCHEMA_VERSION,
    RWMConfig,
    RwmValidationResult,
    analytic_gamma_wall,
    build_evidence,
    critical_rotation_residual,
    default_config,
    feedback_check,
    gamma_wall_rel_error,
    rotation_rel_error,
    tau_eff_rel_error,
    tau_scaling_rel_error,
    validate_evidence_payload,
    validate_rwm_feedback,
)


@pytest.fixture(scope="module")
def result() -> RwmValidationResult:
    """Run the RWM validation once for the module."""
    return validate_rwm_feedback()


# ── Exact closed-form references ─────────────────────────────────────


def test_growth_rate_matches_bondeson_ward() -> None:
    config = default_config()
    for beta_n in (2.5, 3.0, 3.5):
        assert gamma_wall_rel_error(config, beta_n) < 1e-12


def test_analytic_gamma_wall_formula_and_tau_override() -> None:
    config = default_config()
    # Explicit tau_eff override exercises the non-default branch.
    assert analytic_gamma_wall(config, 3.0, tau_eff=1.0) == pytest.approx((3.0 - 2.0) / (4.0 - 3.0))


def test_analytic_gamma_wall_rejects_outside_window() -> None:
    with pytest.raises(ValueError, match="strictly inside the unstable window"):
        analytic_gamma_wall(default_config(), 5.0)


def test_tau_eff_wall_gap_correction() -> None:
    assert tau_eff_rel_error(default_config(), wall_radius=1.2, plasma_radius=1.0) < 1e-12


def test_rotation_stabilisation_matches_closed_form() -> None:
    config = default_config()
    for beta_n in (2.5, 3.0, 3.5):
        assert rotation_rel_error(config, beta_n, 300.0) < 1e-12


def test_critical_rotation_marginalises_growth() -> None:
    config = default_config()
    for beta_n in (2.3, 2.5, 2.8):
        assert critical_rotation_residual(config, beta_n) < 1e-12


def test_critical_rotation_requires_subcritical_window() -> None:
    # beta_N at/above the midpoint has A >= 1; passive rotation cannot stabilise.
    with pytest.raises(ValueError, match="window midpoint"):
        critical_rotation_residual(default_config(), 3.5)


def test_required_gain_marginalises_closed_loop() -> None:
    config = default_config()
    for beta_n in (2.5, 3.0, 3.5):
        check = feedback_check(config, beta_n, tau_controller=1e-4, m_coil=1.3)
        assert check.required_gain_rel_error < 1e-12
        assert check.closed_loop_residual < 1e-12


def test_wall_time_scaling_is_exact() -> None:
    assert tau_scaling_rel_error(default_config(), 3.0) < 1e-12


# ── Aggregate result ─────────────────────────────────────────────────


def test_overall_validation_passes(result: RwmValidationResult) -> None:
    assert result.passed is True
    assert result.gamma_wall_passed
    assert result.tau_eff_passed
    assert result.rotation_passed
    assert result.critical_rotation_passed
    assert result.feedback_passed
    assert result.scaling_passed
    assert result.boundary_passed
    assert len(result.feedback_checks) == 3


def test_stability_window_boundaries(result: RwmValidationResult) -> None:
    assert result.stable_growth_rate == 0.0
    assert not math.isfinite(result.ideal_growth_rate)


def test_validation_is_deterministic() -> None:
    a = validate_rwm_feedback()
    b = validate_rwm_feedback()
    assert a.max_gamma_wall_rel_error == b.max_gamma_wall_rel_error
    assert a.max_feedback_residual == b.max_feedback_residual


# ── Config contract ──────────────────────────────────────────────────


def test_config_rejects_inverted_beta_limits() -> None:
    with pytest.raises(ValueError, match="beta_n_nowall must be less than beta_n_wall"):
        RWMConfig(beta_n_nowall=4.0, beta_n_wall=2.0, tau_wall_s=5e-3)


def test_config_rejects_nonpositive_wall_time() -> None:
    with pytest.raises(ValueError, match="tau_wall_s must be positive"):
        RWMConfig(beta_n_nowall=2.0, beta_n_wall=4.0, tau_wall_s=0.0)


def test_config_rejects_nonfinite_limit() -> None:
    with pytest.raises(ValueError, match="beta_n_wall must be finite"):
        RWMConfig(beta_n_nowall=2.0, beta_n_wall=float("inf"), tau_wall_s=5e-3)


def test_config_rejects_nonnumeric_limit() -> None:
    with pytest.raises(ValueError, match="beta_n_nowall must be a finite number"):
        RWMConfig(beta_n_nowall="low", beta_n_wall=4.0, tau_wall_s=5e-3)  # type: ignore[arg-type]


def test_window_midpoint() -> None:
    assert default_config().window_midpoint == pytest.approx(3.0)


def test_validation_rejects_empty_index_sets() -> None:
    with pytest.raises(ValueError, match="at least one window index and one rotation index"):
        validate_rwm_feedback(window_indices=(), rotation_indices=())


# ── Evidence seal ────────────────────────────────────────────────────


def test_evidence_roundtrip_is_sealed_and_passing(result: RwmValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    assert evidence["schema_version"] == RWM_FEEDBACK_SCHEMA_VERSION
    assert validate_evidence_payload(evidence) is True
    assert evidence["ideal_growth_rate_is_infinite"] is True
    assert len(evidence["feedback_checks"]) == 3


def test_evidence_tamper_is_rejected(result: RwmValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["max_gamma_wall_rel_error"] = 1.0
    with pytest.raises(ValueError, match="payload_sha256 does not match"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_empty_target_id(result: RwmValidationResult) -> None:
    with pytest.raises(ValueError, match="target_id"):
        build_evidence(result, target_id="   ")


def test_evidence_rejects_unknown_schema(result: RwmValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["schema_version"] = "scpn-control.unknown.v9"
    with pytest.raises(ValueError, match="unsupported"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_non_hex_seal(result: RwmValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["payload_sha256"] = "notadigest"
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        validate_evidence_payload(evidence)


# ── CLI / report writer ──────────────────────────────────────────────


def test_main_text_output_passes(capsys) -> None:
    import validation.validate_rwm_feedback as mod

    assert mod.main([]) == 0
    out = capsys.readouterr().out
    assert "Status: pass" in out
    assert "Bondeson-Ward growth" in out


def test_main_json_output_and_report(capsys, tmp_path) -> None:
    import validation.validate_rwm_feedback as mod

    report = tmp_path / "rwm.json"
    assert mod.main(["--json-out", "--report", str(report)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == RWM_FEEDBACK_SCHEMA_VERSION
    assert report.exists() and report.with_suffix(".md").exists()
    assert validate_evidence_payload(json.loads(report.read_text())) is True
    assert "Resistive-Wall-Mode Feedback" in report.with_suffix(".md").read_text()


def test_main_returns_one_on_failure(monkeypatch, capsys) -> None:
    import validation.validate_rwm_feedback as mod

    real = mod.validate_rwm_feedback
    monkeypatch.setattr(mod, "validate_rwm_feedback", lambda: real(exact_tol=1e-30))
    assert mod.main([]) == 1
    assert "Status: fail" in capsys.readouterr().out
