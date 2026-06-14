# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Ideal-MHD stability validation tests
"""Tests for the ideal-MHD stability metric analytic validation."""

from __future__ import annotations

import json

import pytest

from validation.validate_mhd_stability import (
    MHD_STABILITY_SCHEMA_VERSION,
    MHDStabilityValidationResult,
    ballooning_branches_consistent,
    ballooning_rel_error,
    build_evidence,
    kruskal_shafranov_consistent,
    mercier_marginal_cases_consistent,
    mercier_rel_error,
    troyon_boundary_is_consistent,
    troyon_rel_error,
    troyon_scaling_checks,
    validate_evidence_payload,
    validate_mhd_stability,
)


@pytest.fixture(scope="module")
def result() -> MHDStabilityValidationResult:
    """Run the ideal-MHD stability validation once for the module."""
    return validate_mhd_stability()


# ── Exact closed-form references ─────────────────────────────────────


def test_troyon_beta_limit_matches_closed_form() -> None:
    assert troyon_rel_error() < 1e-12


def test_troyon_scaling_laws_are_exact() -> None:
    checks = {c.name: c for c in troyon_scaling_checks()}
    assert checks["beta_t_linear"].measured_ratio == pytest.approx(2.0, rel=1e-12)
    assert checks["minor_radius_linear"].measured_ratio == pytest.approx(2.0, rel=1e-12)
    assert checks["field_linear"].measured_ratio == pytest.approx(2.0, rel=1e-12)
    assert checks["current_inverse"].measured_ratio == pytest.approx(0.5, rel=1e-12)


def test_troyon_boundary_flags_consistent() -> None:
    assert troyon_boundary_is_consistent() is True


def test_mercier_index_matches_freidberg_form() -> None:
    assert mercier_rel_error() < 1e-12


def test_mercier_marginal_cases() -> None:
    assert mercier_marginal_cases_consistent() is True


def test_ballooning_alpha_crit_matches_connor_hastie_taylor() -> None:
    assert ballooning_rel_error() < 1e-12


def test_ballooning_branches() -> None:
    assert ballooning_branches_consistent() is True


def test_kruskal_shafranov_boundary() -> None:
    assert kruskal_shafranov_consistent() is True


# ── Aggregate result ─────────────────────────────────────────────────


def test_overall_validation_passes(result: MHDStabilityValidationResult) -> None:
    assert result.passed is True
    assert result.troyon_passed
    assert result.mercier_passed
    assert result.ballooning_passed
    assert result.kruskal_shafranov_passed
    assert len(result.troyon_scaling) == 4


def test_validation_is_deterministic() -> None:
    a = validate_mhd_stability()
    b = validate_mhd_stability()
    assert a.troyon_rel_error == b.troyon_rel_error
    assert a.mercier_rel_error == b.mercier_rel_error


# ── Evidence seal ────────────────────────────────────────────────────


def test_evidence_roundtrip_is_sealed_and_passing(result: MHDStabilityValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    assert evidence["schema_version"] == MHD_STABILITY_SCHEMA_VERSION
    assert validate_evidence_payload(evidence) is True
    assert len(evidence["troyon_scaling"]) == 4
    assert evidence["kruskal_shafranov_consistent"] is True


def test_evidence_tamper_is_rejected(result: MHDStabilityValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["troyon_rel_error"] = 1.0
    with pytest.raises(ValueError, match="payload_sha256 does not match"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_empty_target_id(result: MHDStabilityValidationResult) -> None:
    with pytest.raises(ValueError, match="target_id"):
        build_evidence(result, target_id="   ")


def test_evidence_rejects_unknown_schema(result: MHDStabilityValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["schema_version"] = "scpn-control.unknown.v9"
    with pytest.raises(ValueError, match="unsupported"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_non_hex_seal(result: MHDStabilityValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["payload_sha256"] = "notadigest"
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        validate_evidence_payload(evidence)


# ── CLI / report writer ──────────────────────────────────────────────


def test_main_text_output_passes(capsys) -> None:
    import validation.validate_mhd_stability as mod

    assert mod.main([]) == 0
    out = capsys.readouterr().out
    assert "Status: pass" in out
    assert "Kruskal-Shafranov" in out


def test_main_json_output_and_report(capsys, tmp_path) -> None:
    import validation.validate_mhd_stability as mod

    report = tmp_path / "mhd.json"
    assert mod.main(["--json-out", "--report", str(report)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == MHD_STABILITY_SCHEMA_VERSION
    assert report.exists() and report.with_suffix(".md").exists()
    assert validate_evidence_payload(json.loads(report.read_text())) is True
    assert "Ideal-MHD Stability" in report.with_suffix(".md").read_text()


def test_main_returns_one_on_failure(monkeypatch, capsys) -> None:
    import validation.validate_mhd_stability as mod

    # The closed-form errors are exactly zero, so force a Troyon mismatch instead.
    monkeypatch.setattr(mod, "troyon_rel_error", lambda: 1.0)
    assert mod.main([]) == 1
    assert "Status: fail" in capsys.readouterr().out
