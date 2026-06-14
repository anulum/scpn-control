# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — DT burn-control alpha-heating validation tests
"""Tests for the DT burn-control alpha-heating analytic validation."""

from __future__ import annotations

import json

import pytest

from validation.validate_burn_control import (
    BURN_CONTROL_SCHEMA_VERSION,
    BurnConfig,
    BurnValidationResult,
    build_evidence,
    burn_fraction_rel_error,
    burn_scaling_checks,
    default_config,
    energy_gain_rel_error,
    energy_partition_rel_error,
    ignition_limit_check,
    lawson_margin_rel_error,
    lawson_rel_error,
    power_density_rel_error,
    reactivity_exponent_low_temperature_guard,
    reactivity_exponent_rel_error,
    validate_burn_control,
    validate_evidence_payload,
    volume_integral_rel_error,
)


@pytest.fixture(scope="module")
def config() -> BurnConfig:
    """The ITER-like default operating point."""
    return default_config()


@pytest.fixture(scope="module")
def result() -> BurnValidationResult:
    """Run the burn-control validation once for the module."""
    return validate_burn_control()


# ── Exact closed-form references ─────────────────────────────────────


def test_energy_partition_is_five() -> None:
    assert energy_partition_rel_error() < 1e-12


def test_power_density_matches_closed_form(config: BurnConfig) -> None:
    assert power_density_rel_error(config) < 1e-12


def test_volume_integral_matches_closed_form(config: BurnConfig) -> None:
    assert volume_integral_rel_error(config) < 1e-12


def test_energy_gain_matches_closed_form(config: BurnConfig) -> None:
    assert energy_gain_rel_error(config) < 1e-12


def test_ignition_limits(config: BurnConfig) -> None:
    limits = ignition_limit_check(config)
    assert limits.infinite_when_burning is True
    assert limits.zero_when_dark is True


def test_lawson_triple_product_matches_closed_form(config: BurnConfig) -> None:
    assert lawson_rel_error(config) < 1e-12


def test_lawson_margin_matches_closed_form(config: BurnConfig) -> None:
    assert lawson_margin_rel_error(config) < 1e-12


def test_burn_fraction_matches_closed_form(config: BurnConfig) -> None:
    assert burn_fraction_rel_error(config) < 1e-12


def test_reactivity_exponent_matches_central_difference(config: BurnConfig) -> None:
    assert reactivity_exponent_rel_error(config) < 1e-12


def test_reactivity_exponent_low_temperature_guard(config: BurnConfig) -> None:
    assert reactivity_exponent_low_temperature_guard(config) is True


def test_burn_scaling_laws_are_exact(config: BurnConfig) -> None:
    checks = {c.name: c for c in burn_scaling_checks(config)}
    assert checks["energy_gain_alpha_linear"].measured_ratio == pytest.approx(2.0, rel=1e-12)
    assert checks["energy_gain_aux_inverse"].measured_ratio == pytest.approx(0.5, rel=1e-12)
    assert checks["lawson_density_linear"].measured_ratio == pytest.approx(2.0, rel=1e-12)
    assert checks["burn_fraction_minor_radius_square"].measured_ratio == pytest.approx(4.0, rel=1e-12)
    assert all(c.rel_error < 1e-12 for c in checks.values())


# ── Aggregate result ─────────────────────────────────────────────────


def test_overall_validation_passes(result: BurnValidationResult) -> None:
    assert result.passed is True
    assert result.energetics_passed is True
    assert result.lawson_passed is True
    assert result.stability_passed is True
    assert result.scaling_passed is True
    assert len(result.scaling) == 4


def test_validation_is_deterministic() -> None:
    a = validate_burn_control()
    b = validate_burn_control()
    assert a.energy_gain_rel_error == b.energy_gain_rel_error
    assert a.reactivity_exponent_rel_error == b.reactivity_exponent_rel_error


# ── Configuration guards ─────────────────────────────────────────────


def _kwargs() -> dict[str, object]:
    return {
        "n_rho": 32,
        "major_radius_m": 6.2,
        "minor_radius_m": 2.0,
        "elongation": 1.7,
        "electron_density_1e20": 1.0,
        "electron_temperature_kev": 20.0,
        "ion_temperature_kev": 20.0,
        "alpha_power_mw": 100.0,
        "aux_power_mw": 50.0,
        "density_m3": 1e20,
        "confinement_time_s": 3.0,
        "thermal_speed_ms": 1e6,
    }


def test_config_rejects_small_grid() -> None:
    kwargs = _kwargs()
    kwargs["n_rho"] = 2
    with pytest.raises(ValueError, match="n_rho must be at least 4"):
        BurnConfig(**kwargs)  # type: ignore[arg-type]


def test_config_rejects_non_integer_grid() -> None:
    kwargs = _kwargs()
    kwargs["n_rho"] = 32.0
    with pytest.raises(ValueError, match="n_rho must be an integer"):
        BurnConfig(**kwargs)  # type: ignore[arg-type]


def test_config_rejects_non_positive_density() -> None:
    kwargs = _kwargs()
    kwargs["electron_density_1e20"] = 0.0
    with pytest.raises(ValueError, match="electron_density_1e20 must be positive"):
        BurnConfig(**kwargs)  # type: ignore[arg-type]


def test_config_rejects_non_finite_value() -> None:
    kwargs = _kwargs()
    kwargs["aux_power_mw"] = float("inf")
    with pytest.raises(ValueError, match="aux_power_mw must be finite"):
        BurnConfig(**kwargs)  # type: ignore[arg-type]


def test_config_rejects_bool_value() -> None:
    kwargs = _kwargs()
    kwargs["elongation"] = True
    with pytest.raises(ValueError, match="elongation must be a finite number"):
        BurnConfig(**kwargs)  # type: ignore[arg-type]


def test_config_rejects_minor_not_smaller_than_major() -> None:
    kwargs = _kwargs()
    kwargs["minor_radius_m"] = 6.2
    with pytest.raises(ValueError, match="minor radius must be smaller"):
        BurnConfig(**kwargs)  # type: ignore[arg-type]


# ── Evidence seal ────────────────────────────────────────────────────


def test_evidence_roundtrip_is_sealed_and_passing(result: BurnValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    assert evidence["schema_version"] == BURN_CONTROL_SCHEMA_VERSION
    assert validate_evidence_payload(evidence) is True
    assert evidence["ignition_limits"]["infinite_when_burning"] is True
    assert len(evidence["scaling"]) == 4


def test_evidence_tamper_is_rejected(result: BurnValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["energy_gain_rel_error"] = 1.0
    with pytest.raises(ValueError, match="payload_sha256 does not match"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_empty_target_id(result: BurnValidationResult) -> None:
    with pytest.raises(ValueError, match="target_id"):
        build_evidence(result, target_id="   ")


def test_evidence_rejects_unknown_schema(result: BurnValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["schema_version"] = "scpn-control.unknown.v9"
    with pytest.raises(ValueError, match="unsupported"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_non_hex_seal(result: BurnValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["payload_sha256"] = "notadigest"
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_wrong_length_hex_lookalike(result: BurnValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["payload_sha256"] = "z" * 64
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        validate_evidence_payload(evidence)


# ── CLI / report writer ──────────────────────────────────────────────


def test_main_text_output_passes(capsys) -> None:
    import validation.validate_burn_control as mod

    assert mod.main([]) == 0
    out = capsys.readouterr().out
    assert "Status: pass" in out
    assert "stability:" in out


def test_main_json_output_and_report(capsys, tmp_path) -> None:
    import validation.validate_burn_control as mod

    report = tmp_path / "bc.json"
    assert mod.main(["--json-out", "--report", str(report)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == BURN_CONTROL_SCHEMA_VERSION
    assert report.exists() and report.with_suffix(".md").exists()
    assert validate_evidence_payload(json.loads(report.read_text())) is True
    assert "Burn-Control" in report.with_suffix(".md").read_text()


def test_main_returns_one_on_failure(monkeypatch, capsys) -> None:
    import validation.validate_burn_control as mod

    # The closed-form errors are exactly zero, so force an energy-gain mismatch.
    monkeypatch.setattr(mod, "energy_gain_rel_error", lambda config: 1.0)
    assert mod.main([]) == 1
    assert "Status: fail" in capsys.readouterr().out
