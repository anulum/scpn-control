# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Volt-second flux-budget validation tests
"""Tests for the volt-second flux-budget analytic validation."""

from __future__ import annotations

import json

import pytest

from validation.validate_volt_second import (
    VOLT_SECOND_SCHEMA_VERSION,
    VoltSecondConfig,
    VoltSecondValidationResult,
    build_evidence,
    default_config,
    ejima_flux_rel_error,
    flat_top_closure_rel_error,
    flux_scaling_checks,
    inductive_flux_rel_error,
    monitor_integration_check,
    ramp_optimizer_check,
    resistive_ramp_rel_error,
    scenario_decomposition_check,
    validate_evidence_payload,
    validate_volt_second,
)


@pytest.fixture(scope="module")
def config() -> VoltSecondConfig:
    """The ITER-like default budget."""
    return default_config()


@pytest.fixture(scope="module")
def result() -> VoltSecondValidationResult:
    """Run the volt-second validation once for the module."""
    return validate_volt_second()


# ── Exact closed-form flux references ────────────────────────────────


def test_inductive_flux_matches_closed_form(config: VoltSecondConfig) -> None:
    assert inductive_flux_rel_error(config) < 1e-12


def test_ejima_flux_matches_closed_form(config: VoltSecondConfig) -> None:
    assert ejima_flux_rel_error(config) < 1e-12


def test_resistive_ramp_matches_riemann_sum(config: VoltSecondConfig) -> None:
    assert resistive_ramp_rel_error(config) < 1e-12


def test_flat_top_budget_closure(config: VoltSecondConfig) -> None:
    assert flat_top_closure_rel_error(config) < 1e-12


def test_scenario_decomposition_matches_closed_form(config: VoltSecondConfig) -> None:
    decomposition = scenario_decomposition_check(config)
    assert decomposition.ramp_rel_error < 1e-12
    assert decomposition.flat_top_rel_error < 1e-12
    assert decomposition.ramp_down_rel_error < 1e-12
    assert decomposition.sum_rel_error < 1e-12
    assert decomposition.margin_abs_error < 1e-9
    assert decomposition.max_rel_error < 1e-12


def test_monitor_integration_is_exact(config: VoltSecondConfig) -> None:
    monitor = monitor_integration_check(config)
    assert monitor.consumed_rel_error < 1e-12
    assert monitor.remaining_rel_error < 1e-12
    assert monitor.fraction_rel_error < 1e-12
    assert monitor.max_rel_error < 1e-12


def test_ramp_optimizer_is_linear(config: VoltSecondConfig) -> None:
    optimiser = ramp_optimizer_check(config)
    assert optimiser.is_linear is True
    assert optimiser.start_abs_error < 1e-12
    assert optimiser.end_rel_error < 1e-12
    assert optimiser.spacing_max_rel_error < 1e-12


def test_flux_scaling_laws_are_exact(config: VoltSecondConfig) -> None:
    checks = {c.name: c for c in flux_scaling_checks(config)}
    assert checks["inductive_current_linear"].measured_ratio == pytest.approx(2.0, rel=1e-12)
    assert checks["inductive_inductance_linear"].measured_ratio == pytest.approx(2.0, rel=1e-12)
    assert checks["ejima_major_radius_linear"].measured_ratio == pytest.approx(2.0, rel=1e-12)
    assert checks["ejima_current_linear"].measured_ratio == pytest.approx(2.0, rel=1e-12)
    assert all(c.rel_error < 1e-12 for c in checks.values())


# ── Aggregate result ─────────────────────────────────────────────────


def test_overall_validation_passes(result: VoltSecondValidationResult) -> None:
    assert result.passed is True
    assert result.fluxes_passed is True
    assert result.decomposition_passed is True
    assert result.monitor_passed is True
    assert result.optimizer_passed is True
    assert result.scaling_passed is True
    assert len(result.scaling) == 4


def test_validation_is_deterministic() -> None:
    a = validate_volt_second()
    b = validate_volt_second()
    assert a.inductive_rel_error == b.inductive_rel_error
    assert a.flat_top_closure_rel_error == b.flat_top_closure_rel_error


# ── Configuration guards ─────────────────────────────────────────────


def _kwargs() -> dict[str, float]:
    return {
        "flux_budget_vs": 300.0,
        "plasma_inductance_uh": 10.0,
        "plasma_resistance_uohm": 5.0,
        "major_radius_m": 6.2,
        "plasma_current_ma": 15.0,
        "bootstrap_current_ma": 2.0,
        "ramp_duration_s": 10.0,
        "flat_duration_s": 100.0,
        "ramp_down_duration_s": 10.0,
        "standalone_ramp_flux_vs": 5.0,
    }


def test_config_rejects_non_positive_budget() -> None:
    kwargs = _kwargs()
    kwargs["flux_budget_vs"] = 0.0
    with pytest.raises(ValueError, match="flux_budget_vs must be positive"):
        VoltSecondConfig(**kwargs)


def test_config_rejects_negative_duration() -> None:
    kwargs = _kwargs()
    kwargs["ramp_duration_s"] = -1.0
    with pytest.raises(ValueError, match="ramp_duration_s must be nonnegative"):
        VoltSecondConfig(**kwargs)


def test_config_rejects_non_finite_value() -> None:
    kwargs = _kwargs()
    kwargs["plasma_resistance_uohm"] = float("inf")
    with pytest.raises(ValueError, match="plasma_resistance_uohm must be finite"):
        VoltSecondConfig(**kwargs)


def test_config_rejects_bool_value() -> None:
    kwargs = _kwargs()
    kwargs["major_radius_m"] = True  # type: ignore[assignment]
    with pytest.raises(ValueError, match="major_radius_m must be a finite number"):
        VoltSecondConfig(**kwargs)


def test_config_rejects_bootstrap_not_smaller_than_plasma() -> None:
    kwargs = _kwargs()
    kwargs["bootstrap_current_ma"] = 15.0
    with pytest.raises(ValueError, match="bootstrap current must be smaller"):
        VoltSecondConfig(**kwargs)


# ── Evidence seal ────────────────────────────────────────────────────


def test_evidence_roundtrip_is_sealed_and_passing(result: VoltSecondValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    assert evidence["schema_version"] == VOLT_SECOND_SCHEMA_VERSION
    assert validate_evidence_payload(evidence) is True
    assert evidence["ramp_optimizer"]["is_linear"] is True
    assert len(evidence["scaling"]) == 4


def test_evidence_tamper_is_rejected(result: VoltSecondValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["inductive_rel_error"] = 1.0
    with pytest.raises(ValueError, match="payload_sha256 does not match"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_empty_target_id(result: VoltSecondValidationResult) -> None:
    with pytest.raises(ValueError, match="target_id"):
        build_evidence(result, target_id="   ")


def test_evidence_rejects_unknown_schema(result: VoltSecondValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["schema_version"] = "scpn-control.unknown.v9"
    with pytest.raises(ValueError, match="unsupported"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_non_hex_seal(result: VoltSecondValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["payload_sha256"] = "notadigest"
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_wrong_length_hex_lookalike(result: VoltSecondValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["payload_sha256"] = "z" * 64
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        validate_evidence_payload(evidence)


# ── CLI / report writer ──────────────────────────────────────────────


def test_main_text_output_passes(capsys) -> None:
    import validation.validate_volt_second as mod

    assert mod.main([]) == 0
    out = capsys.readouterr().out
    assert "Status: pass" in out
    assert "optimiser:" in out


def test_main_json_output_and_report(capsys, tmp_path) -> None:
    import validation.validate_volt_second as mod

    report = tmp_path / "vs.json"
    assert mod.main(["--json-out", "--report", str(report)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == VOLT_SECOND_SCHEMA_VERSION
    assert report.exists() and report.with_suffix(".md").exists()
    assert validate_evidence_payload(json.loads(report.read_text())) is True
    assert "Volt-Second" in report.with_suffix(".md").read_text()


def test_main_returns_one_on_failure(monkeypatch, capsys) -> None:
    import validation.validate_volt_second as mod

    # The closed-form flux errors are exactly zero, so force an inductive mismatch.
    monkeypatch.setattr(mod, "inductive_flux_rel_error", lambda config: 1.0)
    assert mod.main([]) == 1
    assert "Status: fail" in capsys.readouterr().out
