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
from pathlib import Path
from typing import Callable

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
    """Build the default bounded analytic circuit and pulse parameters."""
    return default_config()


@pytest.fixture(scope="module")
def result() -> VoltSecondValidationResult:
    """Run the volt-second validation once for the module."""
    return validate_volt_second()


# ── Exact closed-form flux references ────────────────────────────────


def test_inductive_flux_matches_closed_form(config: VoltSecondConfig) -> None:
    """Inductive flux matches closed form."""
    assert inductive_flux_rel_error(config) < 1e-12


def test_ejima_flux_matches_closed_form(config: VoltSecondConfig) -> None:
    """Ejima flux matches closed form."""
    assert ejima_flux_rel_error(config) < 1e-12


def test_resistive_ramp_matches_riemann_sum(config: VoltSecondConfig) -> None:
    """Resistive ramp matches riemann sum."""
    assert resistive_ramp_rel_error(config) < 1e-12


def test_flat_top_budget_closure(config: VoltSecondConfig) -> None:
    """Flat top budget closure."""
    assert flat_top_closure_rel_error(config) < 1e-12


def test_scenario_decomposition_matches_closed_form(config: VoltSecondConfig) -> None:
    """Scenario decomposition matches closed form."""
    decomposition = scenario_decomposition_check(config)
    assert decomposition.ramp_rel_error < 1e-12
    assert decomposition.flat_top_rel_error < 1e-12
    assert decomposition.ramp_down_rel_error < 1e-12
    assert decomposition.sum_rel_error < 1e-12
    assert decomposition.margin_abs_error < 1e-9
    assert decomposition.max_rel_error < 1e-12


def test_monitor_integration_is_exact(config: VoltSecondConfig) -> None:
    """Monitor integration is exact."""
    monitor = monitor_integration_check(config)
    assert monitor.consumed_rel_error < 1e-12
    assert monitor.remaining_rel_error < 1e-12
    assert monitor.fraction_rel_error < 1e-12
    assert monitor.max_rel_error < 1e-12


def test_ramp_optimizer_is_linear(config: VoltSecondConfig) -> None:
    """Ramp optimizer is linear."""
    optimiser = ramp_optimizer_check(config)
    assert optimiser.is_linear is True
    assert optimiser.start_abs_error < 1e-12
    assert optimiser.end_rel_error < 1e-12
    assert optimiser.spacing_max_rel_error < 1e-12


def test_flux_scaling_laws_are_exact(config: VoltSecondConfig) -> None:
    """Flux scaling laws are exact."""
    checks = {c.name: c for c in flux_scaling_checks(config)}
    assert checks["inductive_current_linear"].measured_ratio == pytest.approx(2.0, rel=1e-12)
    assert checks["inductive_inductance_linear"].measured_ratio == pytest.approx(2.0, rel=1e-12)
    assert checks["ejima_major_radius_linear"].measured_ratio == pytest.approx(2.0, rel=1e-12)
    assert checks["ejima_current_linear"].measured_ratio == pytest.approx(2.0, rel=1e-12)
    assert all(c.rel_error < 1e-12 for c in checks.values())


# ── Aggregate result ─────────────────────────────────────────────────


def test_overall_validation_passes(result: VoltSecondValidationResult) -> None:
    """Overall validation passes."""
    assert result.passed is True
    assert result.fluxes_passed is True
    assert result.decomposition_passed is True
    assert result.monitor_passed is True
    assert result.optimizer_passed is True
    assert result.scaling_passed is True
    assert len(result.scaling) == 4


def test_validation_is_deterministic() -> None:
    """Validation is deterministic."""
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
    """Config rejects non positive budget."""
    kwargs = _kwargs()
    kwargs["flux_budget_vs"] = 0.0
    with pytest.raises(ValueError, match="flux_budget_vs must be positive"):
        VoltSecondConfig(**kwargs)


def test_config_rejects_negative_duration() -> None:
    """Config rejects negative duration."""
    kwargs = _kwargs()
    kwargs["ramp_duration_s"] = -1.0
    with pytest.raises(ValueError, match="ramp_duration_s must be nonnegative"):
        VoltSecondConfig(**kwargs)


def test_config_rejects_non_finite_value() -> None:
    """Config rejects non finite value."""
    kwargs = _kwargs()
    kwargs["plasma_resistance_uohm"] = float("inf")
    with pytest.raises(ValueError, match="plasma_resistance_uohm must be finite"):
        VoltSecondConfig(**kwargs)


def test_config_rejects_bool_value() -> None:
    """Config rejects bool value."""
    kwargs = _kwargs()
    kwargs["major_radius_m"] = True
    with pytest.raises(ValueError, match="major_radius_m must be a finite number"):
        VoltSecondConfig(**kwargs)


def test_config_rejects_bootstrap_not_smaller_than_plasma() -> None:
    """Config rejects bootstrap not smaller than plasma."""
    kwargs = _kwargs()
    kwargs["bootstrap_current_ma"] = 15.0
    with pytest.raises(ValueError, match="bootstrap current must be smaller"):
        VoltSecondConfig(**kwargs)


# ── Evidence seal ────────────────────────────────────────────────────


def test_evidence_roundtrip_is_sealed_and_passing(result: VoltSecondValidationResult) -> None:
    """Evidence roundtrip is sealed and passing."""
    evidence = build_evidence(result, target_id="test-target")
    assert evidence["schema_version"] == VOLT_SECOND_SCHEMA_VERSION
    assert validate_evidence_payload(evidence) is True
    assert evidence["ramp_optimizer"]["is_linear"] is True
    assert len(evidence["scaling"]) == 4


def test_evidence_tamper_is_rejected(result: VoltSecondValidationResult) -> None:
    """Evidence tamper is rejected."""
    evidence = build_evidence(result, target_id="test-target")
    evidence["inductive_rel_error"] = 1.0
    with pytest.raises(ValueError, match="payload_sha256 does not match"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_empty_target_id(result: VoltSecondValidationResult) -> None:
    """Evidence rejects empty target id."""
    with pytest.raises(ValueError, match="target_id"):
        build_evidence(result, target_id="   ")


def test_evidence_rejects_unknown_schema(result: VoltSecondValidationResult) -> None:
    """Evidence rejects unknown schema."""
    evidence = build_evidence(result, target_id="test-target")
    evidence["schema_version"] = "scpn-control.unknown.v9"
    with pytest.raises(ValueError, match="unsupported"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_non_hex_seal(result: VoltSecondValidationResult) -> None:
    """Evidence rejects non hex seal."""
    evidence = build_evidence(result, target_id="test-target")
    evidence["payload_sha256"] = "notadigest"
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_wrong_length_hex_lookalike(result: VoltSecondValidationResult) -> None:
    """Evidence rejects wrong length hex lookalike."""
    evidence = build_evidence(result, target_id="test-target")
    evidence["payload_sha256"] = "z" * 64
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        validate_evidence_payload(evidence)


# ── CLI / report writer ──────────────────────────────────────────────


def test_main_text_output_passes(capsys: pytest.CaptureFixture[str]) -> None:
    """Main text output passes."""
    import validation.validate_volt_second as mod

    assert mod.main([]) == 0
    out = capsys.readouterr().out
    assert "Status: pass" in out
    assert "optimiser:" in out


def test_main_json_output_and_report(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Main json output and report."""
    import validation.validate_volt_second as mod

    report = tmp_path / "vs.json"
    assert mod.main(["--json-out", "--report", str(report)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == VOLT_SECOND_SCHEMA_VERSION
    assert report.exists() and report.with_suffix(".md").exists()
    assert validate_evidence_payload(json.loads(report.read_text())) is True
    assert "Volt-Second" in report.with_suffix(".md").read_text()


def test_main_returns_one_on_failure(capsys: pytest.CaptureFixture[str]) -> None:
    """Fail the real accumulation-error tolerance through the public CLI API."""
    import validation.validate_volt_second as mod

    assert mod.main(["--exact-tol", "1e-30"]) == 1
    assert "Status: fail" in capsys.readouterr().out


@pytest.mark.parametrize("function", [resistive_ramp_rel_error, monitor_integration_check])
@pytest.mark.parametrize("count", [0, -1, True])
def test_iteration_counts_are_checked(function: Callable[..., object], count: int) -> None:
    """Refuse empty or invalid iteration counts before integration."""
    with pytest.raises(ValueError, match="n_steps"):
        function(default_config(), n_steps=count)


@pytest.mark.parametrize("count", [0, 1, True])
def test_ramp_requires_two_segments(count: int) -> None:
    """Refuse ramp grids without a defined step and endpoint."""
    with pytest.raises(ValueError, match="n_segments"):
        ramp_optimizer_check(default_config(), n_segments=count)


@pytest.mark.parametrize("changes", [{"flat_duration_s": 0.0}, {"ramp_down_duration_s": 2.0}])
def test_zero_phase_reference_has_finite_error(changes: dict[str, float]) -> None:
    """Calculate legitimate zero phase fluxes without dividing by zero."""
    from dataclasses import replace

    config = replace(default_config(), **changes)
    observed = validate_volt_second(config=config)
    assert observed.passed is True
    assert observed.decomposition.max_rel_error == 0.0
    assert validate_evidence_payload(build_evidence(observed, target_id="zero-phase")) is True


def test_zero_voltage_has_finite_monitor_errors() -> None:
    """Compare a zero-consumption monitor against its unchanged full budget."""
    observed = monitor_integration_check(default_config(), v_loop=0.0)
    assert observed.consumed_rel_error == observed.fraction_rel_error == 0.0
    assert observed.remaining_rel_error == 0.0


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1", "invalid"])
def test_cli_tolerance_refusals_are_authored(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    """Reject malformed tolerance arguments without native conversion text."""
    import validation.validate_volt_second as mod

    with pytest.raises(SystemExit) as exit_info:
        mod.main(["--exact-tol", value])
    assert exit_info.value.code == 2
    assert "Tolerance must be a finite positive number" in capsys.readouterr().err
