# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Runaway-electron validation tests
"""Tests for the Rosenbluth-Putvinski runaway-electron analytic validation."""

from __future__ import annotations

import json

import pytest

from validation.validate_runaway_electron import (
    RUNAWAY_ELECTRON_SCHEMA_VERSION,
    RunawayValidationResult,
    avalanche_behaviour,
    avalanche_rate_rel_error,
    avalanche_time_rel_error,
    build_evidence,
    collision_time_rel_error,
    critical_field_includes_impurity_electrons,
    critical_field_rel_error,
    dreicer_field_rel_error,
    scaling_checks,
    validate_evidence_payload,
    validate_runaway_electron,
)


@pytest.fixture(scope="module")
def result() -> RunawayValidationResult:
    """Run the runaway-electron validation once for the module."""
    return validate_runaway_electron()


# ── Exact closed-form references ─────────────────────────────────────


def test_critical_field_matches_closed_form() -> None:
    assert critical_field_rel_error() < 1e-12


def test_dreicer_field_matches_closed_form() -> None:
    assert dreicer_field_rel_error() < 1e-12


def test_collision_time_matches_closed_form() -> None:
    assert collision_time_rel_error() < 1e-12


def test_avalanche_time_matches_closed_form() -> None:
    assert avalanche_time_rel_error() < 1e-12


def test_impurity_critical_field() -> None:
    assert critical_field_includes_impurity_electrons() < 1e-12


def test_avalanche_rate_matches_rosenbluth_putvinski() -> None:
    assert avalanche_rate_rel_error() < 1e-12


def test_avalanche_behaviour() -> None:
    behaviour = avalanche_behaviour()
    assert behaviour.zero_below_critical is True
    assert behaviour.density_linearity_rel_error < 1e-12
    assert behaviour.drive_linearity_rel_error < 1e-12
    assert behaviour.deconfinement_rel_error < 1e-12


def test_scaling_laws_are_exact() -> None:
    checks = {c.name: c for c in scaling_checks()}
    assert checks["critical_field_density_linear"].measured_ratio == pytest.approx(2.0, rel=1e-12)
    assert checks["dreicer_temperature_inverse"].measured_ratio == pytest.approx(0.5, rel=1e-12)
    assert checks["avalanche_time_z_eff"].measured_ratio == pytest.approx(4.0, rel=1e-12)


# ── Aggregate result ─────────────────────────────────────────────────


def test_overall_validation_passes(result: RunawayValidationResult) -> None:
    assert result.passed is True
    assert result.fields_passed
    assert result.avalanche_passed
    assert result.scaling_passed
    assert len(result.scaling) == 4


def test_validation_is_deterministic() -> None:
    a = validate_runaway_electron()
    b = validate_runaway_electron()
    assert a.critical_field_rel_error == b.critical_field_rel_error
    assert a.avalanche_rate_rel_error == b.avalanche_rate_rel_error


# ── Evidence seal ────────────────────────────────────────────────────


def test_evidence_roundtrip_is_sealed_and_passing(result: RunawayValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    assert evidence["schema_version"] == RUNAWAY_ELECTRON_SCHEMA_VERSION
    assert validate_evidence_payload(evidence) is True
    assert evidence["avalanche"]["zero_below_critical"] is True
    assert len(evidence["scaling"]) == 4


def test_evidence_tamper_is_rejected(result: RunawayValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["critical_field_rel_error"] = 1.0
    with pytest.raises(ValueError, match="payload_sha256 does not match"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_empty_target_id(result: RunawayValidationResult) -> None:
    with pytest.raises(ValueError, match="target_id"):
        build_evidence(result, target_id="   ")


def test_evidence_rejects_unknown_schema(result: RunawayValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["schema_version"] = "scpn-control.unknown.v9"
    with pytest.raises(ValueError, match="unsupported"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_non_hex_seal(result: RunawayValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["payload_sha256"] = "notadigest"
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        validate_evidence_payload(evidence)


# ── CLI / report writer ──────────────────────────────────────────────


def test_main_text_output_passes(capsys) -> None:
    import validation.validate_runaway_electron as mod

    assert mod.main([]) == 0
    out = capsys.readouterr().out
    assert "Status: pass" in out
    assert "avalanche:" in out


def test_main_json_output_and_report(capsys, tmp_path) -> None:
    import validation.validate_runaway_electron as mod

    report = tmp_path / "re.json"
    assert mod.main(["--json-out", "--report", str(report)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == RUNAWAY_ELECTRON_SCHEMA_VERSION
    assert report.exists() and report.with_suffix(".md").exists()
    assert validate_evidence_payload(json.loads(report.read_text())) is True
    assert "Runaway-Electron" in report.with_suffix(".md").read_text()


def test_main_returns_one_on_failure(monkeypatch, capsys) -> None:
    import validation.validate_runaway_electron as mod

    # The closed-form errors are exactly zero, so force a critical-field mismatch.
    monkeypatch.setattr(mod, "critical_field_rel_error", lambda: 1.0)
    assert mod.main([]) == 1
    assert "Status: fail" in capsys.readouterr().out
