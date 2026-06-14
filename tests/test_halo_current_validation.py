# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Halo-current validation tests
"""Tests for the Fitzpatrick halo-current L/R circuit analytic validation."""

from __future__ import annotations

import json

import pytest

from validation.validate_halo_current import (
    HALO_CURRENT_SCHEMA_VERSION,
    HaloConfig,
    HaloValidationResult,
    build_evidence,
    default_config,
    inductance_rel_error,
    mutual_inductance_rel_error,
    quasi_static_tracking,
    resistance_rel_error,
    resistance_scaling_checks,
    time_constant_rel_error,
    tpf_product_rel_error,
    validate_evidence_payload,
    validate_halo_current,
    wall_force_rel_error,
)


@pytest.fixture(scope="module")
def config() -> HaloConfig:
    """The ITER-like default geometry."""
    return default_config()


@pytest.fixture(scope="module")
def result() -> HaloValidationResult:
    """Run the halo-current validation once for the module."""
    return validate_halo_current()


# ── Exact closed-form circuit references ─────────────────────────────


def test_resistance_matches_closed_form(config: HaloConfig) -> None:
    assert resistance_rel_error(config) < 1e-12


def test_inductance_matches_closed_form(config: HaloConfig) -> None:
    assert inductance_rel_error(config) < 1e-12


def test_mutual_inductance_matches_closed_form(config: HaloConfig) -> None:
    assert mutual_inductance_rel_error(config) < 1e-12


def test_time_constant_matches_closed_form(config: HaloConfig) -> None:
    assert time_constant_rel_error(config) < 1e-12


def test_wall_force_matches_closed_form(config: HaloConfig) -> None:
    assert wall_force_rel_error(config) < 1e-12


def test_tpf_product_matches_closed_form(config: HaloConfig) -> None:
    assert tpf_product_rel_error(config) < 1e-12


def test_resistance_scaling_laws_are_exact(config: HaloConfig) -> None:
    checks = {c.name: c for c in resistance_scaling_checks(config)}
    assert checks["resistivity_linear"].measured_ratio == pytest.approx(2.0, rel=1e-12)
    assert checks["contact_inverse"].measured_ratio == pytest.approx(0.5, rel=1e-12)
    assert checks["major_radius_linear"].measured_ratio == pytest.approx(2.0, rel=1e-12)
    assert checks["thickness_inverse"].measured_ratio == pytest.approx(0.5, rel=1e-12)
    assert all(c.rel_error < 1e-12 for c in checks.values())


# ── Quasi-static L/R tracking ────────────────────────────────────────


def test_quasi_static_tracking_converges(config: HaloConfig) -> None:
    quasi = quasi_static_tracking(config)
    assert quasi.tau_cq_values == (0.5, 1.0, 2.0)
    assert len(quasi.tracking_errors) == 3
    assert quasi.monotonic_decrease is True
    assert quasi.finest_error < 1e-2
    assert quasi.finest_error == quasi.tracking_errors[-1]


# ── Aggregate result ─────────────────────────────────────────────────


def test_overall_validation_passes(result: HaloValidationResult) -> None:
    assert result.passed is True
    assert result.circuit_passed is True
    assert result.scaling_passed is True
    assert result.loads_passed is True
    assert result.quasi_static_passed is True
    assert len(result.resistance_scaling) == 4


def test_validation_is_deterministic() -> None:
    a = validate_halo_current()
    b = validate_halo_current()
    assert a.resistance_rel_error == b.resistance_rel_error
    assert a.quasi_static.finest_error == b.quasi_static.finest_error


# ── Configuration guards ─────────────────────────────────────────────


def test_config_rejects_non_positive_current() -> None:
    with pytest.raises(ValueError, match="plasma_current_ma must be positive"):
        HaloConfig(
            plasma_current_ma=0.0,
            minor_radius_m=2.0,
            major_radius_m=6.2,
            wall_resistivity_ohm_m=7e-7,
            wall_thickness_m=0.06,
            tpf=2.0,
            contact_fraction=0.3,
        )


def test_config_rejects_non_finite_value() -> None:
    with pytest.raises(ValueError, match="major_radius_m must be finite"):
        HaloConfig(
            plasma_current_ma=15.0,
            minor_radius_m=2.0,
            major_radius_m=float("inf"),
            wall_resistivity_ohm_m=7e-7,
            wall_thickness_m=0.06,
            tpf=2.0,
            contact_fraction=0.3,
        )


def test_config_rejects_bool_value() -> None:
    with pytest.raises(ValueError, match="tpf must be a finite number"):
        HaloConfig(
            plasma_current_ma=15.0,
            minor_radius_m=2.0,
            major_radius_m=6.2,
            wall_resistivity_ohm_m=7e-7,
            wall_thickness_m=0.06,
            tpf=True,  # type: ignore[arg-type]
            contact_fraction=0.3,
        )


def test_config_rejects_contact_fraction_out_of_range() -> None:
    with pytest.raises(ValueError, match="contact_fraction must lie in"):
        HaloConfig(
            plasma_current_ma=15.0,
            minor_radius_m=2.0,
            major_radius_m=6.2,
            wall_resistivity_ohm_m=7e-7,
            wall_thickness_m=0.06,
            tpf=2.0,
            contact_fraction=1.5,
        )


def test_config_rejects_minor_not_smaller_than_major() -> None:
    with pytest.raises(ValueError, match="minor radius must be smaller"):
        HaloConfig(
            plasma_current_ma=15.0,
            minor_radius_m=6.2,
            major_radius_m=6.2,
            wall_resistivity_ohm_m=7e-7,
            wall_thickness_m=0.06,
            tpf=2.0,
            contact_fraction=0.3,
        )


# ── Evidence seal ────────────────────────────────────────────────────


def test_evidence_roundtrip_is_sealed_and_passing(result: HaloValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    assert evidence["schema_version"] == HALO_CURRENT_SCHEMA_VERSION
    assert validate_evidence_payload(evidence) is True
    assert evidence["quasi_static"]["monotonic_decrease"] is True
    assert len(evidence["resistance_scaling"]) == 4


def test_evidence_tamper_is_rejected(result: HaloValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["resistance_rel_error"] = 1.0
    with pytest.raises(ValueError, match="payload_sha256 does not match"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_empty_target_id(result: HaloValidationResult) -> None:
    with pytest.raises(ValueError, match="target_id"):
        build_evidence(result, target_id="   ")


def test_evidence_rejects_unknown_schema(result: HaloValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["schema_version"] = "scpn-control.unknown.v9"
    with pytest.raises(ValueError, match="unsupported"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_non_hex_seal(result: HaloValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["payload_sha256"] = "notadigest"
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_wrong_length_hex_lookalike(result: HaloValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["payload_sha256"] = "z" * 64
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        validate_evidence_payload(evidence)


# ── CLI / report writer ──────────────────────────────────────────────


def test_main_text_output_passes(capsys) -> None:
    import validation.validate_halo_current as mod

    assert mod.main([]) == 0
    out = capsys.readouterr().out
    assert "Status: pass" in out
    assert "quasi-static:" in out


def test_main_json_output_and_report(capsys, tmp_path) -> None:
    import validation.validate_halo_current as mod

    report = tmp_path / "halo.json"
    assert mod.main(["--json-out", "--report", str(report)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == HALO_CURRENT_SCHEMA_VERSION
    assert report.exists() and report.with_suffix(".md").exists()
    assert validate_evidence_payload(json.loads(report.read_text())) is True
    assert "Halo-Current" in report.with_suffix(".md").read_text()


def test_main_returns_one_on_failure(monkeypatch, capsys) -> None:
    import validation.validate_halo_current as mod

    # The circuit errors are exactly zero, so force a resistance mismatch.
    monkeypatch.setattr(mod, "resistance_rel_error", lambda config: 1.0)
    assert mod.main([]) == 1
    assert "Status: fail" in capsys.readouterr().out
