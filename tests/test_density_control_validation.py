# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Density-control particle-balance validation tests
"""Tests for the density-control particle-balance analytic validation."""

from __future__ import annotations

import json

import pytest

from validation.validate_density_control import (
    DENSITY_CONTROL_SCHEMA_VERSION,
    DensityConfig,
    DensityValidationResult,
    build_evidence,
    cryopump_sink_rel_error,
    default_config,
    diffusion_uniform_invariance_abs_error,
    gas_puff_conservation_rel_error,
    greenwald_fraction_rel_error,
    greenwald_limit_rel_error,
    greenwald_scaling_checks,
    nbi_conservation_rel_error,
    recycling_conservation_rel_error,
    validate_density_control,
    validate_evidence_payload,
    volume_element_rel_error,
)


@pytest.fixture(scope="module")
def config() -> DensityConfig:
    """The ITER-like default geometry."""
    return default_config()


@pytest.fixture(scope="module")
def result() -> DensityValidationResult:
    """Run the density-control validation once for the module."""
    return validate_density_control()


# ── Exact closed-form references ─────────────────────────────────────


def test_greenwald_limit_matches_closed_form(config: DensityConfig) -> None:
    assert greenwald_limit_rel_error(config) < 1e-12


def test_greenwald_fraction_matches_volume_average(config: DensityConfig) -> None:
    assert greenwald_fraction_rel_error(config) < 1e-12


def test_volume_elements_match_closed_form(config: DensityConfig) -> None:
    assert volume_element_rel_error(config) < 1e-12


def test_gas_puff_source_conserves_particles(config: DensityConfig) -> None:
    assert gas_puff_conservation_rel_error(config) < 1e-9


def test_nbi_source_conserves_particles(config: DensityConfig) -> None:
    assert nbi_conservation_rel_error(config) < 1e-9


def test_recycling_source_conserves_particles(config: DensityConfig) -> None:
    assert recycling_conservation_rel_error(config) < 1e-9


def test_cryopump_sink_matches_closed_form(config: DensityConfig) -> None:
    assert cryopump_sink_rel_error(config) < 1e-12


def test_diffusion_leaves_uniform_interior_unchanged(config: DensityConfig) -> None:
    assert diffusion_uniform_invariance_abs_error(config) == 0.0


def test_greenwald_scaling_laws_are_exact(config: DensityConfig) -> None:
    checks = {c.name: c for c in greenwald_scaling_checks(config)}
    assert checks["current_linear"].measured_ratio == pytest.approx(2.0, rel=1e-12)
    assert checks["minor_radius_inverse_square"].measured_ratio == pytest.approx(0.25, rel=1e-12)
    assert all(c.rel_error < 1e-12 for c in checks.values())


# ── Aggregate result ─────────────────────────────────────────────────


def test_overall_validation_passes(result: DensityValidationResult) -> None:
    assert result.passed is True
    assert result.greenwald_passed is True
    assert result.sources_passed is True
    assert result.diffusion_passed is True
    assert result.scaling_passed is True
    assert len(result.scaling) == 2


def test_validation_is_deterministic() -> None:
    a = validate_density_control()
    b = validate_density_control()
    assert a.greenwald_limit_rel_error == b.greenwald_limit_rel_error
    assert a.gas_puff_conservation_rel_error == b.gas_puff_conservation_rel_error


# ── Configuration guards ─────────────────────────────────────────────


def _kwargs() -> dict[str, object]:
    return {
        "n_rho": 64,
        "major_radius_m": 6.2,
        "minor_radius_m": 2.0,
        "plasma_current_ma": 15.0,
        "gas_puff_rate_per_s": 1e21,
        "nbi_energy_kev": 100.0,
        "nbi_power_mw": 33.0,
        "recycling_outflux_per_s": 1e21,
        "recycling_coeff": 0.97,
        "pump_speed_m3_s": 50.0,
        "edge_density_m3": 1e19,
        "uniform_density_m3": 1e20,
    }


def test_config_rejects_small_grid() -> None:
    kwargs = _kwargs()
    kwargs["n_rho"] = 2
    with pytest.raises(ValueError, match="n_rho must be at least 4"):
        DensityConfig(**kwargs)  # type: ignore[arg-type]


def test_config_rejects_non_integer_grid() -> None:
    kwargs = _kwargs()
    kwargs["n_rho"] = 64.0
    with pytest.raises(ValueError, match="n_rho must be an integer"):
        DensityConfig(**kwargs)  # type: ignore[arg-type]


def test_config_rejects_non_positive_current() -> None:
    kwargs = _kwargs()
    kwargs["plasma_current_ma"] = 0.0
    with pytest.raises(ValueError, match="plasma_current_ma must be positive"):
        DensityConfig(**kwargs)  # type: ignore[arg-type]


def test_config_rejects_non_finite_value() -> None:
    kwargs = _kwargs()
    kwargs["nbi_power_mw"] = float("inf")
    with pytest.raises(ValueError, match="nbi_power_mw must be finite"):
        DensityConfig(**kwargs)  # type: ignore[arg-type]


def test_config_rejects_bool_value() -> None:
    kwargs = _kwargs()
    kwargs["edge_density_m3"] = True
    with pytest.raises(ValueError, match="edge_density_m3 must be a finite number"):
        DensityConfig(**kwargs)  # type: ignore[arg-type]


def test_config_rejects_recycling_coeff_out_of_range() -> None:
    kwargs = _kwargs()
    kwargs["recycling_coeff"] = 1.5
    with pytest.raises(ValueError, match="recycling_coeff must lie in"):
        DensityConfig(**kwargs)  # type: ignore[arg-type]


def test_config_rejects_minor_not_smaller_than_major() -> None:
    kwargs = _kwargs()
    kwargs["minor_radius_m"] = 6.2
    with pytest.raises(ValueError, match="minor radius must be smaller"):
        DensityConfig(**kwargs)  # type: ignore[arg-type]


# ── Evidence seal ────────────────────────────────────────────────────


def test_evidence_roundtrip_is_sealed_and_passing(result: DensityValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    assert evidence["schema_version"] == DENSITY_CONTROL_SCHEMA_VERSION
    assert validate_evidence_payload(evidence) is True
    assert evidence["diffusion_passed"] is True
    assert len(evidence["scaling"]) == 2


def test_evidence_tamper_is_rejected(result: DensityValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["greenwald_limit_rel_error"] = 1.0
    with pytest.raises(ValueError, match="payload_sha256 does not match"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_empty_target_id(result: DensityValidationResult) -> None:
    with pytest.raises(ValueError, match="target_id"):
        build_evidence(result, target_id="   ")


def test_evidence_rejects_unknown_schema(result: DensityValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["schema_version"] = "scpn-control.unknown.v9"
    with pytest.raises(ValueError, match="unsupported"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_non_hex_seal(result: DensityValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["payload_sha256"] = "notadigest"
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_wrong_length_hex_lookalike(result: DensityValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["payload_sha256"] = "z" * 64
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        validate_evidence_payload(evidence)


# ── CLI / report writer ──────────────────────────────────────────────


def test_main_text_output_passes(capsys) -> None:
    import validation.validate_density_control as mod

    assert mod.main([]) == 0
    out = capsys.readouterr().out
    assert "Status: pass" in out
    assert "diffusion:" in out


def test_main_json_output_and_report(capsys, tmp_path) -> None:
    import validation.validate_density_control as mod

    report = tmp_path / "dc.json"
    assert mod.main(["--json-out", "--report", str(report)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == DENSITY_CONTROL_SCHEMA_VERSION
    assert report.exists() and report.with_suffix(".md").exists()
    assert validate_evidence_payload(json.loads(report.read_text())) is True
    assert "Density-Control" in report.with_suffix(".md").read_text()


def test_main_returns_one_on_failure(monkeypatch, capsys) -> None:
    import validation.validate_density_control as mod

    # The closed-form errors are exactly zero, so force a Greenwald-limit mismatch.
    monkeypatch.setattr(mod, "greenwald_limit_rel_error", lambda config: 1.0)
    assert mod.main([]) == 1
    assert "Status: fail" in capsys.readouterr().out
