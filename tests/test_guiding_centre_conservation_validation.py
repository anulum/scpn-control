# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Guiding-centre conservation validation tests
"""Tests for the guiding-centre orbit conservation-law validation."""

from __future__ import annotations

import json
import math

import pytest

from validation.validate_guiding_centre_conservation import (
    GUIDING_CENTRE_CONSERVATION_SCHEMA_VERSION,
    AxisymmetricField,
    GuidingCentreValidationResult,
    OrbitCase,
    build_evidence,
    default_cases,
    default_field,
    run_orbit_conservation,
    validate_evidence_payload,
    validate_guiding_centre,
)


@pytest.fixture(scope="module")
def result() -> GuidingCentreValidationResult:
    """Run the default conservation validation once for the module."""
    return validate_guiding_centre()


# ── Analytic field ───────────────────────────────────────────────────


def test_field_is_divergence_free() -> None:
    """The analytic field must satisfy ∇·B = 0 in cylindrical coordinates.

    For axisymmetric B(R,Z): div B = (1/R) d(R B_R)/dR + dB_Z/dZ. With
    B_R = -2cZ/R and B_Z = 2c(R-R0)/R, R B_R = -2cZ so d(R B_R)/dR = 0, and
    dB_Z/dZ = 0, so the divergence is exactly zero.
    """
    field = AxisymmetricField(r0=1.7, a=0.5, b0=2.0, b_pol_coeff=0.5)
    h = 1e-6
    for r, z in [(1.6, 0.1), (1.9, -0.2), (1.7, 0.3)]:
        br_rp = field.components(r + h, z)[0]
        br_rm = field.components(r - h, z)[0]
        d_rbr_dr = ((r + h) * br_rp - (r - h) * br_rm) / (2.0 * h)
        bz_zp = field.components(r, z + h)[1]
        bz_zm = field.components(r, z - h)[1]
        dbz_dz = (bz_zp - bz_zm) / (2.0 * h)
        div = d_rbr_dr / r + dbz_dz
        assert abs(div) < 1e-6


def test_field_poloidal_field_vanishes_on_axis() -> None:
    field = default_field()
    b_r, b_z, b_phi = field.components(field.r0, 0.0)
    assert b_r == pytest.approx(0.0)
    assert b_z == pytest.approx(0.0)
    assert b_phi == pytest.approx(field.b0)


def test_field_toroidal_field_scales_as_one_over_r() -> None:
    field = default_field()
    assert field.components(2.0, 0.0)[2] == pytest.approx(field.b0 * field.r0 / 2.0)
    assert field.magnitude(1.7, 0.0) == pytest.approx(field.b0)
    assert field.poloidal_flux(field.r0, 0.0) == pytest.approx(0.0)
    assert field(1.7, 0.0) == field.components(1.7, 0.0)


def test_field_rejects_nonpositive_parameters() -> None:
    with pytest.raises(ValueError, match="b0 must be positive"):
        AxisymmetricField(r0=1.7, a=0.5, b0=0.0, b_pol_coeff=0.5)


def test_field_rejects_nonpositive_radius_query() -> None:
    with pytest.raises(ValueError, match="major radius R must be positive"):
        default_field().components(0.0, 0.0)


def test_field_rejects_nonfinite_parameter() -> None:
    with pytest.raises(ValueError, match="b_pol_coeff must be finite"):
        AxisymmetricField(r0=1.7, a=0.5, b0=2.0, b_pol_coeff=math.inf)


def test_case_rejects_nonnumeric_pitch() -> None:
    with pytest.raises(ValueError, match="pitch_angle must be a finite number"):
        OrbitCase("x", 2.0, 1, 100.0, "wide", 0.2, 1e-9, 1e-8)  # type: ignore[arg-type]


# ── Conservation result ──────────────────────────────────────────────


def test_validation_passes(result: GuidingCentreValidationResult) -> None:
    assert result.passed is True
    assert result.energy_passed is True
    assert result.momentum_passed is True
    assert result.parallel_passed is True
    assert result.covers_passing_and_trapped is True
    assert result.max_energy_drift < result.energy_tol
    assert result.max_momentum_drift < result.momentum_tol


def test_records_cover_both_orbit_classes(result: GuidingCentreValidationResult) -> None:
    by_label = {record.label: record for record in result.records}
    assert by_label["passing_deuteron"].trapped is False
    assert by_label["trapped_deuteron"].trapped is True
    assert by_label["passing_alpha"].trapped is False
    assert by_label["trapped_alpha"].trapped is True
    for record in result.records:
        assert record.energy_passed and record.momentum_passed and record.parallel_passed
        assert record.max_parallel_speed_ratio <= 1.0 + result.parallel_tol


def test_single_orbit_conservation_is_tight() -> None:
    field = default_field()
    case = OrbitCase("probe", 2.0, 1, 100.0, 1.2, 0.25, 1e-9, 4e-6)
    record = run_orbit_conservation(field, case, energy_tol=1e-3, momentum_tol=1e-3, parallel_tol=1e-6)
    assert record.energy_drift_rel < 1e-4
    assert record.momentum_drift_rel < 1e-4
    assert record.steps == 4000


def test_validation_is_deterministic() -> None:
    a = validate_guiding_centre()
    b = validate_guiding_centre()
    assert [r.energy_drift_rel for r in a.records] == [r.energy_drift_rel for r in b.records]
    assert [r.momentum_drift_rel for r in a.records] == [r.momentum_drift_rel for r in b.records]


def test_run_rejects_nonpositive_initial_radius() -> None:
    field = default_field()
    case = OrbitCase("bad", 2.0, 1, 100.0, 1.0, -2.0, 1e-9, 1e-8)
    with pytest.raises(ValueError, match="initial major radius must be positive"):
        run_orbit_conservation(field, case, energy_tol=1e-3, momentum_tol=1e-3, parallel_tol=1e-6)


def test_run_rejects_degenerate_baseline_momentum() -> None:
    """A purely perpendicular orbit on the magnetic axis has zero toroidal momentum."""
    field = default_field()
    case = OrbitCase("perp", 2.0, 1, 100.0, math.pi / 2.0, 0.0, 1e-9, 1e-8)
    with pytest.raises(ValueError, match="degenerate orbit"):
        run_orbit_conservation(field, case, energy_tol=1e-3, momentum_tol=1e-3, parallel_tol=1e-6)


# ── Case contract ────────────────────────────────────────────────────


def test_case_rejects_zero_charge() -> None:
    with pytest.raises(ValueError, match="charge must be a non-zero integer"):
        OrbitCase("x", 2.0, 0, 100.0, 1.0, 0.2, 1e-9, 1e-8)


def test_case_rejects_pitch_outside_open_interval() -> None:
    with pytest.raises(ValueError, match="pitch_angle"):
        OrbitCase("x", 2.0, 1, 100.0, 0.0, 0.2, 1e-9, 1e-8)
    with pytest.raises(ValueError, match="pitch_angle"):
        OrbitCase("x", 2.0, 1, 100.0, math.pi, 0.2, 1e-9, 1e-8)


def test_case_rejects_total_time_below_dt() -> None:
    with pytest.raises(ValueError, match="total_time_s must be at least one dt_s"):
        OrbitCase("x", 2.0, 1, 100.0, 1.0, 0.2, 1e-8, 1e-9)


def test_case_rejects_empty_label() -> None:
    with pytest.raises(ValueError, match="label must be a non-empty string"):
        OrbitCase("  ", 2.0, 1, 100.0, 1.0, 0.2, 1e-9, 1e-8)


def test_case_rejects_nonpositive_energy() -> None:
    with pytest.raises(ValueError, match="energy_kev must be positive"):
        OrbitCase("x", 2.0, 1, 0.0, 1.0, 0.2, 1e-9, 1e-8)


# ── validate_guiding_centre input contract ───────────────────────────


def test_validate_rejects_empty_cases() -> None:
    with pytest.raises(ValueError, match="at least one orbit case"):
        validate_guiding_centre(cases=())


def test_validate_rejects_nonpositive_tolerance() -> None:
    with pytest.raises(ValueError, match="energy_tol must be positive"):
        validate_guiding_centre(energy_tol=0.0)


def test_validate_accepts_custom_single_case() -> None:
    field = default_field()
    res = validate_guiding_centre(
        field=field,
        cases=(
            OrbitCase("passing", 2.0, 1, 100.0, 0.5, 0.2, 1e-9, 6e-6),
            OrbitCase("trapped", 2.0, 1, 100.0, 1.45, 0.2, 1e-9, 6e-6),
        ),
    )
    assert res.passed is True
    assert len(default_cases()) == 4


# ── Evidence seal ────────────────────────────────────────────────────


def test_evidence_roundtrip_is_sealed_and_passing(result: GuidingCentreValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    assert evidence["schema_version"] == GUIDING_CENTRE_CONSERVATION_SCHEMA_VERSION
    assert validate_evidence_payload(evidence) is True
    assert len(evidence["records"]) == 4


def test_evidence_tamper_is_rejected(result: GuidingCentreValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["max_energy_drift"] = 1.0
    with pytest.raises(ValueError, match="payload_sha256 does not match"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_empty_target_id(result: GuidingCentreValidationResult) -> None:
    with pytest.raises(ValueError, match="target_id"):
        build_evidence(result, target_id="   ")


def test_evidence_rejects_unknown_schema(result: GuidingCentreValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["schema_version"] = "scpn-control.unknown.v9"
    with pytest.raises(ValueError, match="unsupported"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_non_hex_seal(result: GuidingCentreValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["payload_sha256"] = "nothex"
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        validate_evidence_payload(evidence)


# ── CLI / report writer ──────────────────────────────────────────────


def test_main_text_output_passes(capsys) -> None:
    import validation.validate_guiding_centre_conservation as mod

    assert mod.main([]) == 0
    out = capsys.readouterr().out
    assert "Status: pass" in out
    assert "trapped_deuteron" in out


def test_main_json_output_and_report(capsys, tmp_path) -> None:
    import validation.validate_guiding_centre_conservation as mod

    report = tmp_path / "gc.json"
    assert mod.main(["--json-out", "--report", str(report)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == GUIDING_CENTRE_CONSERVATION_SCHEMA_VERSION
    assert report.exists() and report.with_suffix(".md").exists()
    assert validate_evidence_payload(json.loads(report.read_text())) is True
    assert "Guiding-Centre" in report.with_suffix(".md").read_text()


def test_main_returns_one_on_failure(monkeypatch, capsys) -> None:
    import validation.validate_guiding_centre_conservation as mod

    real = mod.validate_guiding_centre
    # An impossibly tight energy tolerance forces the gate to fail.
    monkeypatch.setattr(mod, "validate_guiding_centre", lambda: real(energy_tol=1e-18))
    assert mod.main([]) == 1
    assert "Status: fail" in capsys.readouterr().out
