# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — RZIP vertical stability validation tests
"""Tests for the RZIP rigid vertical stability analytic validation."""

from __future__ import annotations

import json
import math

import pytest

from validation.validate_rzip_vertical_stability import (
    RZIP_VERTICAL_STABILITY_SCHEMA_VERSION,
    RzipValidationResult,
    VerticalConfig,
    analytic_no_wall_frequency,
    analytic_no_wall_growth_rate,
    build_evidence,
    default_config,
    no_wall_frequency_rel_error,
    no_wall_growth_rel_error,
    no_wall_growth_time_consistency,
    scaling_checks,
    validate_evidence_payload,
    validate_rzip_vertical_stability,
    wall_stabilisation,
)


@pytest.fixture(scope="module")
def result() -> RzipValidationResult:
    """Run the RZIP validation once for the module."""
    return validate_rzip_vertical_stability()


# ── Exact no-wall references ─────────────────────────────────────────


def test_growth_rate_matches_closed_form() -> None:
    config = default_config()
    for n in (-2.5, -1.2, -0.6):
        assert no_wall_growth_rel_error(config, n) < 1e-12


def test_growth_rate_formula_matches_hand_calculation() -> None:
    config = default_config()
    mu0 = 4.0e-7 * math.pi
    k = -1.3 * mu0 * (config.ip_ma * 1e6) ** 2 / (4.0 * math.pi * config.r0)
    expected = math.sqrt(-k / config.m_eff_kg)
    assert analytic_no_wall_growth_rate(config, -1.3) == pytest.approx(expected)


def test_oscillation_frequency_matches_closed_form() -> None:
    config = default_config()
    for n in (0.8, 1.5):
        assert no_wall_frequency_rel_error(config, n) < 1e-12


def test_growth_time_identity_holds() -> None:
    config = default_config()
    assert no_wall_growth_time_consistency(config, -1.2) < 1e-12


def test_growth_rate_closed_form_rejects_stable_index() -> None:
    with pytest.raises(ValueError, match="destabilising index"):
        analytic_no_wall_growth_rate(default_config(), 0.5)


def test_frequency_closed_form_rejects_unstable_index() -> None:
    with pytest.raises(ValueError, match="stabilising index"):
        analytic_no_wall_frequency(default_config(), -0.5)


# ── Scaling laws and wall stabilisation ──────────────────────────────


def test_scaling_laws_are_exact() -> None:
    checks = {c.name: c for c in scaling_checks(default_config())}
    assert checks["current_linear"].measured_ratio == pytest.approx(2.0, abs=1e-12)
    assert checks["index_sqrt"].measured_ratio == pytest.approx(2.0, abs=1e-12)
    assert checks["inertia_inverse_sqrt"].measured_ratio == pytest.approx(0.5, abs=1e-12)


def test_resistive_wall_slows_growth() -> None:
    wall = wall_stabilisation(default_config(), -1.0)
    assert wall.wall_slows_growth is True
    assert wall.with_wall_finite is True
    assert wall.with_wall_growth_rate < wall.no_wall_growth_rate


# ── Aggregate result ─────────────────────────────────────────────────


def test_overall_validation_passes(result: RzipValidationResult) -> None:
    assert result.passed is True
    assert result.growth_passed
    assert result.frequency_passed
    assert result.growth_time_passed
    assert result.marginal_passed
    assert result.scaling_passed
    assert result.wall_passed


def test_marginal_index_is_neutrally_stable(result: RzipValidationResult) -> None:
    assert abs(result.marginal_growth_rate) < result.marginal_tol


def test_validation_is_deterministic() -> None:
    a = validate_rzip_vertical_stability()
    b = validate_rzip_vertical_stability()
    assert a.max_growth_rel_error == b.max_growth_rel_error
    assert a.wall.with_wall_growth_rate == b.wall.with_wall_growth_rate


# ── Config contract ──────────────────────────────────────────────────


def test_config_rejects_minor_radius_above_major() -> None:
    with pytest.raises(ValueError, match="a must be smaller than r0"):
        VerticalConfig(r0=1.0, a=1.5, kappa=1.8, ip_ma=1.0, b0=2.0, m_eff_kg=2.0)


def test_config_rejects_nonpositive_inertia() -> None:
    with pytest.raises(ValueError, match="m_eff_kg must be positive"):
        VerticalConfig(r0=1.7, a=0.5, kappa=1.8, ip_ma=1.0, b0=2.0, m_eff_kg=0.0)


def test_config_rejects_nonfinite_current() -> None:
    with pytest.raises(ValueError, match="ip_ma must be finite"):
        VerticalConfig(r0=1.7, a=0.5, kappa=1.8, ip_ma=float("inf"), b0=2.0, m_eff_kg=2.0)


def test_config_rejects_nonnumeric_radius() -> None:
    with pytest.raises(ValueError, match="r0 must be a finite number"):
        VerticalConfig(r0="big", a=0.5, kappa=1.8, ip_ma=1.0, b0=2.0, m_eff_kg=2.0)  # type: ignore[arg-type]


# ── Index-sign guards ────────────────────────────────────────────────


def test_validation_rejects_positive_unstable_index() -> None:
    with pytest.raises(ValueError, match="unstable index must be negative"):
        validate_rzip_vertical_stability(unstable_indices=(1.0,))


def test_validation_rejects_negative_stable_index() -> None:
    with pytest.raises(ValueError, match="stable index must be positive"):
        validate_rzip_vertical_stability(stable_indices=(-1.0,))


def test_validation_rejects_empty_index_sets() -> None:
    with pytest.raises(ValueError, match="at least one unstable and one stable"):
        validate_rzip_vertical_stability(unstable_indices=(), stable_indices=())


# ── Evidence seal ────────────────────────────────────────────────────


def test_evidence_roundtrip_is_sealed_and_passing(result: RzipValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    assert evidence["schema_version"] == RZIP_VERTICAL_STABILITY_SCHEMA_VERSION
    assert validate_evidence_payload(evidence) is True
    assert len(evidence["scaling"]) == 3


def test_evidence_tamper_is_rejected(result: RzipValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["max_growth_rel_error"] = 1.0
    with pytest.raises(ValueError, match="payload_sha256 does not match"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_empty_target_id(result: RzipValidationResult) -> None:
    with pytest.raises(ValueError, match="target_id"):
        build_evidence(result, target_id="   ")


def test_evidence_rejects_unknown_schema(result: RzipValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["schema_version"] = "scpn-control.unknown.v9"
    with pytest.raises(ValueError, match="unsupported"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_non_hex_seal(result: RzipValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["payload_sha256"] = "notadigest"
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        validate_evidence_payload(evidence)


# ── CLI / report writer ──────────────────────────────────────────────


def test_main_text_output_passes(capsys) -> None:
    import validation.validate_rzip_vertical_stability as mod

    assert mod.main([]) == 0
    out = capsys.readouterr().out
    assert "Status: pass" in out
    assert "wall stabilisation" in out


def test_main_json_output_and_report(capsys, tmp_path) -> None:
    import validation.validate_rzip_vertical_stability as mod

    report = tmp_path / "rzip.json"
    assert mod.main(["--json-out", "--report", str(report)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == RZIP_VERTICAL_STABILITY_SCHEMA_VERSION
    assert report.exists() and report.with_suffix(".md").exists()
    assert validate_evidence_payload(json.loads(report.read_text())) is True
    assert "Rigid Vertical Stability" in report.with_suffix(".md").read_text()


def test_main_returns_one_on_failure(monkeypatch, capsys) -> None:
    import validation.validate_rzip_vertical_stability as mod

    real = mod.validate_rzip_vertical_stability
    monkeypatch.setattr(mod, "validate_rzip_vertical_stability", lambda: real(exact_tol=1e-30))
    assert mod.main([]) == 1
    assert "Status: fail" in capsys.readouterr().out
