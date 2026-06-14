# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Sawtooth Kadomtsev crash validation tests
"""Tests for the Kadomtsev sawtooth crash analytic validation."""

from __future__ import annotations

import json
import math

import pytest

from validation.validate_sawtooth_kadomtsev import (
    SAWTOOTH_KADOMTSEV_SCHEMA_VERSION,
    SawtoothConfig,
    SawtoothValidationResult,
    analytic_q1_radius,
    build_evidence,
    crash_conservation,
    default_config,
    no_crash_leaves_profiles_unchanged,
    q1_radius_rel_error,
    validate_evidence_payload,
    validate_sawtooth_kadomtsev,
)


@pytest.fixture(scope="module")
def result() -> SawtoothValidationResult:
    """Run the Kadomtsev validation once for the module."""
    return validate_sawtooth_kadomtsev()


# ── Exact conservation and structure ─────────────────────────────────


def test_volume_integrals_are_conserved() -> None:
    cons = crash_conservation(default_config())
    assert cons.temperature_integral_rel_error < 1e-12
    assert cons.density_integral_rel_error < 1e-12


def test_helical_flux_vanishes_at_mixing_radius() -> None:
    cons = crash_conservation(default_config())
    assert cons.helical_flux_residual < 1e-12


def test_profiles_flatten_inside_and_reset_q() -> None:
    cons = crash_conservation(default_config())
    assert cons.inner_temperature_flatness < 1e-12
    assert cons.inner_density_flatness < 1e-12
    assert cons.inner_q_value == pytest.approx(1.01)


def test_profiles_unchanged_outside_mixing_radius() -> None:
    cons = crash_conservation(default_config())
    assert cons.outside_max_abs_change == 0.0
    assert 0.0 < cons.rho_1 < cons.rho_mix


# ── q=1 radius ───────────────────────────────────────────────────────


def test_q1_radius_matches_analytic_and_converges() -> None:
    config = default_config()
    coarse = q1_radius_rel_error(config)
    fine = q1_radius_rel_error(
        SawtoothConfig(nr=2 * config.nr - 1, q0=config.q0, qa=config.qa, r0=config.r0, a=config.a)
    )
    assert fine < coarse
    assert fine < 5e-3


def test_analytic_q1_radius_formula() -> None:
    config = SawtoothConfig(nr=201, q0=0.5, qa=2.5, r0=1.7, a=0.5)
    assert analytic_q1_radius(config) == pytest.approx(0.5)


def test_q1_radius_exact_on_grid_node_yields_infinite_order() -> None:
    # rho_1 = 0.5 lands on a node for both coarse and fine grids -> exact match.
    config = SawtoothConfig(nr=201, q0=0.5, qa=2.5, r0=1.7, a=0.5)
    res = validate_sawtooth_kadomtsev(config=config)
    assert res.q1_rel_error_fine == 0.0
    assert math.isinf(res.q1_order)
    assert res.q1_passed is True


def test_q1_radius_none_raises(monkeypatch) -> None:
    import validation.validate_sawtooth_kadomtsev as mod

    monkeypatch.setattr(mod.SawtoothMonitor, "find_q1_radius", lambda self, q: None)
    with pytest.raises(ValueError, match="no q=1 surface"):
        q1_radius_rel_error(default_config())


# ── No-crash guard ───────────────────────────────────────────────────


def test_no_crash_when_q_above_one_everywhere() -> None:
    assert no_crash_leaves_profiles_unchanged(default_config()) is True


def test_crash_conservation_rejects_degenerate_mixing(monkeypatch) -> None:
    import validation.validate_sawtooth_kadomtsev as mod

    rho = default_config().rho()
    monkeypatch.setattr(
        mod,
        "kadomtsev_crash",
        lambda rho_, t, n, q, r0, a: (t.copy(), n.copy(), q.copy(), 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="mixing region is too small"):
        crash_conservation(default_config())


# ── Aggregate result ─────────────────────────────────────────────────


def test_overall_validation_passes(result: SawtoothValidationResult) -> None:
    assert result.passed is True
    assert result.conservation_passed
    assert result.helical_passed
    assert result.structure_passed
    assert result.outside_passed
    assert result.q1_passed
    assert result.no_crash_passed
    assert result.q1_order == pytest.approx(2.0, abs=0.4)


def test_validation_is_deterministic() -> None:
    a = validate_sawtooth_kadomtsev()
    b = validate_sawtooth_kadomtsev()
    assert a.conservation.temperature_integral_rel_error == b.conservation.temperature_integral_rel_error
    assert a.q1_rel_error_fine == b.q1_rel_error_fine


# ── Config contract ──────────────────────────────────────────────────


def test_config_requires_q1_surface() -> None:
    with pytest.raises(ValueError, match="q0 < 1 < qa"):
        SawtoothConfig(nr=201, q0=1.2, qa=3.0, r0=1.7, a=0.5)


def test_config_rejects_minor_radius_above_major() -> None:
    with pytest.raises(ValueError, match="a must be smaller than r0"):
        SawtoothConfig(nr=201, q0=0.8, qa=3.0, r0=0.5, a=1.7)


def test_config_rejects_coarse_grid() -> None:
    with pytest.raises(ValueError, match="at least 11"):
        SawtoothConfig(nr=5, q0=0.8, qa=3.0, r0=1.7, a=0.5)


def test_config_rejects_nonfinite_q() -> None:
    with pytest.raises(ValueError, match="qa must be finite"):
        SawtoothConfig(nr=201, q0=0.8, qa=float("inf"), r0=1.7, a=0.5)


def test_config_rejects_nonnumeric_q() -> None:
    with pytest.raises(ValueError, match="q0 must be a finite number"):
        SawtoothConfig(nr=201, q0="low", qa=3.0, r0=1.7, a=0.5)  # type: ignore[arg-type]


def test_config_rejects_nonpositive_major_radius() -> None:
    with pytest.raises(ValueError, match="r0 must be positive"):
        SawtoothConfig(nr=201, q0=0.8, qa=3.0, r0=0.0, a=0.5)


def test_config_rejects_nonpositive_grid() -> None:
    with pytest.raises(ValueError, match="nr must be a positive integer"):
        SawtoothConfig(nr=0, q0=0.8, qa=3.0, r0=1.7, a=0.5)


# ── Evidence seal ────────────────────────────────────────────────────


def test_evidence_roundtrip_is_sealed_and_passing(result: SawtoothValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    assert evidence["schema_version"] == SAWTOOTH_KADOMTSEV_SCHEMA_VERSION
    assert validate_evidence_payload(evidence) is True
    assert evidence["no_crash_ok"] is True


def test_evidence_tamper_is_rejected(result: SawtoothValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["rho_mix"] = 0.999
    with pytest.raises(ValueError, match="payload_sha256 does not match"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_empty_target_id(result: SawtoothValidationResult) -> None:
    with pytest.raises(ValueError, match="target_id"):
        build_evidence(result, target_id="   ")


def test_evidence_rejects_unknown_schema(result: SawtoothValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["schema_version"] = "scpn-control.unknown.v9"
    with pytest.raises(ValueError, match="unsupported"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_non_hex_seal(result: SawtoothValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["payload_sha256"] = "notadigest"
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        validate_evidence_payload(evidence)


# ── CLI / report writer ──────────────────────────────────────────────


def test_main_text_output_passes(capsys) -> None:
    import validation.validate_sawtooth_kadomtsev as mod

    assert mod.main([]) == 0
    out = capsys.readouterr().out
    assert "Status: pass" in out
    assert "helical flux" in out


def test_main_json_output_and_report(capsys, tmp_path) -> None:
    import validation.validate_sawtooth_kadomtsev as mod

    report = tmp_path / "saw.json"
    assert mod.main(["--json-out", "--report", str(report)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == SAWTOOTH_KADOMTSEV_SCHEMA_VERSION
    assert report.exists() and report.with_suffix(".md").exists()
    assert validate_evidence_payload(json.loads(report.read_text())) is True
    assert "Kadomtsev Crash" in report.with_suffix(".md").read_text()


def test_main_returns_one_on_failure(monkeypatch, capsys) -> None:
    import validation.validate_sawtooth_kadomtsev as mod

    real = mod.validate_sawtooth_kadomtsev
    monkeypatch.setattr(mod, "validate_sawtooth_kadomtsev", lambda: real(conservation_tol=1e-30))
    assert mod.main([]) == 1
    assert "Status: fail" in capsys.readouterr().out
