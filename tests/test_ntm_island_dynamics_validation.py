# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — NTM island dynamics validation tests
"""Tests for the Modified Rutherford Equation analytic validation."""

from __future__ import annotations

import json

import pytest

from scpn_control.core.ntm_dynamics import _A1
from validation.validate_ntm_island_dynamics import (
    NTM_ISLAND_DYNAMICS_SCHEMA_VERSION,
    NtmValidationResult,
    RationalSurfaceConfig,
    analytic_saturated_width,
    build_evidence,
    classical_dw_dt_rel_error,
    classical_trajectory_max_rel_error,
    default_surface,
    saturated_width_residual,
    saturation_convergence,
    validate_evidence_payload,
    validate_ntm_island_dynamics,
)


@pytest.fixture(scope="module")
def result() -> NtmValidationResult:
    """Run the NTM validation once for the module."""
    return validate_ntm_island_dynamics()


# ── Classical-only exact references ──────────────────────────────────


def test_classical_dw_dt_matches_closed_form() -> None:
    surface = default_surface()
    err = classical_dw_dt_rel_error(surface, -3.0, w=0.04, eta=5e-8, j_phi=1e6)
    assert err < 1e-12


def test_classical_trajectory_matches_separable_solution() -> None:
    surface = default_surface()
    # Decaying and growing branches both follow the exact sqrt law.
    decay = classical_trajectory_max_rel_error(surface, -3.0, w0=0.05, eta=5e-8, t_end=0.02, n_steps=4000, j_phi=1e6)
    grow = classical_trajectory_max_rel_error(surface, 2.0, w0=0.01, eta=5e-8, t_end=0.02, n_steps=4000, j_phi=1e6)
    assert decay < 1e-9
    assert grow < 1e-9


def test_trajectory_validation_passes(result: NtmValidationResult) -> None:
    assert result.trajectory_passed is True
    assert result.max_dw_dt_rel_error < result.dw_dt_tol
    assert result.max_trajectory_rel_error < result.trajectory_tol
    assert len(result.trajectory_cases) == 3


# ── Saturated width (closed-form fixed point + attractor) ────────────


def test_analytic_saturated_width_matches_hand_formula() -> None:
    surface = default_surface()
    j_bs, j_phi, dp0 = 2.0e4, 1.0e6, -6.0
    expected = -_A1 * (j_bs / j_phi) * surface.r_s / (dp0 * surface.r_s + 0.5 * _A1 * (j_bs / j_phi))
    assert analytic_saturated_width(surface, dp0, j_bs=j_bs, j_phi=j_phi) == pytest.approx(expected)


def test_saturated_width_rejects_classically_unstable_surface() -> None:
    with pytest.raises(ValueError, match="classically stable"):
        analytic_saturated_width(default_surface(), 1.0, j_bs=2e4, j_phi=1e6)


def test_saturated_width_rejects_non_positive_root() -> None:
    # Weak |Delta'_0| with strong drive makes the denominator positive -> w_sat < 0.
    with pytest.raises(ValueError, match="positive saturated width"):
        analytic_saturated_width(default_surface(), -0.001, j_bs=2e4, j_phi=1e6)


def test_saturated_width_residual_is_machine_zero() -> None:
    residual = saturated_width_residual(default_surface(), -6.0, j_bs=2e4, j_phi=1e6, eta=5e-8)
    assert residual < 1e-9


def test_saturation_is_a_stable_attractor() -> None:
    approach = saturation_convergence(default_surface(), -6.0, j_bs=2e4, j_phi=1e6, eta=5e-8, t_end=0.4, n_steps=20000)
    assert approach.from_below_monotonic is True
    assert approach.from_above_monotonic is True
    assert approach.from_below_rel_error < 5e-3
    assert approach.from_above_rel_error < 5e-3


def test_saturation_validation_passes(result: NtmValidationResult) -> None:
    assert result.saturation_passed is True
    assert result.saturated_residual < result.residual_tol
    assert result.saturation.saturated_width > 0.0


def test_overall_validation_passes(result: NtmValidationResult) -> None:
    assert result.passed is True


def test_validation_is_deterministic() -> None:
    a = validate_ntm_island_dynamics(n_steps=2000)
    b = validate_ntm_island_dynamics(n_steps=2000)
    assert a.max_trajectory_rel_error == b.max_trajectory_rel_error
    assert a.saturation.from_below_rel_error == b.saturation.from_below_rel_error


# ── Surface contract ─────────────────────────────────────────────────


def test_surface_rejects_r_s_exceeding_minor_radius() -> None:
    with pytest.raises(ValueError, match="r_s must not exceed"):
        RationalSurfaceConfig(r_s=0.6, m=2, n=1, a=0.5, R0=1.7, B0=2.0)


def test_surface_rejects_minor_radius_above_major() -> None:
    with pytest.raises(ValueError, match="a must be smaller than R0"):
        RationalSurfaceConfig(r_s=0.3, m=2, n=1, a=2.0, R0=1.7, B0=2.0)


def test_surface_rejects_nonpositive_mode_number() -> None:
    with pytest.raises(ValueError, match="m must be a positive integer"):
        RationalSurfaceConfig(r_s=0.3, m=0, n=1, a=0.5, R0=1.7, B0=2.0)


def test_surface_rejects_boolean_mode_number() -> None:
    with pytest.raises(ValueError, match="n must be a positive integer"):
        RationalSurfaceConfig(r_s=0.3, m=2, n=True, a=0.5, R0=1.7, B0=2.0)  # type: ignore[arg-type]


# ── Numeric guards via public surface ────────────────────────────────


def test_validation_rejects_nonpositive_eta() -> None:
    with pytest.raises(ValueError, match="eta must be positive"):
        validate_ntm_island_dynamics(eta=0.0)


def test_validation_rejects_nonfinite_eta() -> None:
    with pytest.raises(ValueError, match="eta must be finite"):
        validate_ntm_island_dynamics(eta=float("inf"))


def test_validation_rejects_nonnumeric_eta() -> None:
    with pytest.raises(ValueError, match="eta must be a finite number"):
        validate_ntm_island_dynamics(eta="cold")  # type: ignore[arg-type]


# ── Evidence seal ────────────────────────────────────────────────────


def test_evidence_roundtrip_is_sealed_and_passing(result: NtmValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    assert evidence["schema_version"] == NTM_ISLAND_DYNAMICS_SCHEMA_VERSION
    assert validate_evidence_payload(evidence) is True
    assert len(evidence["trajectory_cases"]) == 3


def test_evidence_tamper_is_rejected(result: NtmValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["saturated_width"] = 99.0
    with pytest.raises(ValueError, match="payload_sha256 does not match"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_empty_target_id(result: NtmValidationResult) -> None:
    with pytest.raises(ValueError, match="target_id"):
        build_evidence(result, target_id="   ")


def test_evidence_rejects_unknown_schema(result: NtmValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["schema_version"] = "scpn-control.unknown.v9"
    with pytest.raises(ValueError, match="unsupported"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_non_hex_seal(result: NtmValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["payload_sha256"] = "notadigest"
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        validate_evidence_payload(evidence)


# ── CLI / report writer ──────────────────────────────────────────────


def test_main_text_output_passes(capsys) -> None:
    import validation.validate_ntm_island_dynamics as mod

    assert mod.main([]) == 0
    out = capsys.readouterr().out
    assert "Status: pass" in out
    assert "saturated width" in out


def test_main_json_output_and_report(capsys, tmp_path) -> None:
    import validation.validate_ntm_island_dynamics as mod

    report = tmp_path / "ntm.json"
    assert mod.main(["--json-out", "--report", str(report)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == NTM_ISLAND_DYNAMICS_SCHEMA_VERSION
    assert report.exists() and report.with_suffix(".md").exists()
    assert validate_evidence_payload(json.loads(report.read_text())) is True
    assert "Modified Rutherford Equation" in report.with_suffix(".md").read_text()


def test_main_returns_one_on_failure(monkeypatch, capsys) -> None:
    import validation.validate_ntm_island_dynamics as mod

    real = mod.validate_ntm_island_dynamics
    monkeypatch.setattr(mod, "validate_ntm_island_dynamics", lambda: real(trajectory_tol=1e-30))
    assert mod.main([]) == 1
    assert "Status: fail" in capsys.readouterr().out
