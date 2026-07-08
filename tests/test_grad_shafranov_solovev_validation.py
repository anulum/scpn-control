# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Grad-Shafranov Solov'ev validation tests
"""Tests for the Solov'ev analytic-equilibrium Grad-Shafranov validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from validation.validate_grad_shafranov_solovev import (
    GRAD_SHAFRANOV_SOLOVEV_SCHEMA_VERSION,
    ConvergenceRecord,
    GradShafranovValidationResult,
    MultigridReconstruction,
    SolovevGeometry,
    build_evidence,
    multigrid_reconstruction,
    operator_truncation_error,
    solovev_psi,
    solovev_source,
    sor_reconstruction,
    validate_evidence_payload,
    validate_grad_shafranov,
)

# A light three-level study keeps the module fixture fast while still spanning a
# 4x mesh refinement, which is enough to resolve the second-order slope.
_TEST_RESOLUTIONS = (33, 49, 65)


@pytest.fixture(scope="module")
def result() -> GradShafranovValidationResult:
    """Run the Solov'ev validation once for the whole module."""
    return validate_grad_shafranov(resolutions=_TEST_RESOLUTIONS)


# ── Analytic identities ──────────────────────────────────────────────


def test_solovev_field_satisfies_grad_shafranov_analytically() -> None:
    """Δ*(c1 R⁴/8 + c2 Z²) must equal c1 R² + 2 c2 to machine precision.

    Verified against the exact toroidal operator
    Δ*ψ = ψ_RR − ψ_R/R + ψ_ZZ evaluated symbolically by hand.
    """
    geometry = SolovevGeometry.from_aspect(c1=1.3, c2=0.7)
    r = np.linspace(geometry.r_min, geometry.r_max, 11)
    z = np.linspace(geometry.z_min, geometry.z_max, 9)
    rr, zz = np.meshgrid(r, z)

    c1, c2 = geometry.c1, geometry.c2
    psi_rr = 3.0 * c1 * rr**2 / 2.0  # ∂²/∂R²(c1 R⁴/8) = 3 c1 R²/2
    psi_r_over_r = (c1 * rr**3 / 2.0) / rr  # (1/R) ∂/∂R(c1 R⁴/8) = c1 R²/2
    psi_zz = np.full_like(zz, 2.0 * c2)  # ∂²/∂Z²(c2 Z²) = 2 c2
    delta_star = psi_rr - psi_r_over_r + psi_zz

    np.testing.assert_allclose(delta_star, solovev_source(rr, geometry), rtol=0, atol=1e-12)
    # And the field sampler matches the closed form.
    np.testing.assert_allclose(solovev_psi(rr, zz, geometry), c1 * rr**4 / 8.0 + c2 * zz**2, atol=1e-12)


# ── Operator convergence ─────────────────────────────────────────────


def test_operator_second_order_convergence(result: GradShafranovValidationResult) -> None:
    assert result.operator_passed is True
    assert result.operator_order == pytest.approx(2.0, abs=0.15)
    assert result.operator_error_finest < result.operator_error_gate


def test_operator_truncation_error_quarters_on_refinement() -> None:
    """Halving h must quarter the operator truncation error (second order)."""
    geometry = SolovevGeometry.from_aspect()
    err_coarse = operator_truncation_error(geometry, 33)
    err_fine = operator_truncation_error(geometry, 65)
    # h halves from 33→65, so the error ratio should be close to 4.
    assert 3.5 < err_coarse / err_fine < 4.5


def test_operator_records_have_decreasing_error(result: GradShafranovValidationResult) -> None:
    errors = [record.error for record in result.operator_records]
    assert errors == sorted(errors, reverse=True)
    assert len(result.operator_records) == len(_TEST_RESOLUTIONS)


# ── SOR reconstruction convergence ───────────────────────────────────


def test_sor_reconstruction_second_order(result: GradShafranovValidationResult) -> None:
    assert result.reconstruction_passed is True
    assert result.reconstruction_order == pytest.approx(2.0, abs=0.2)
    assert result.reconstruction_nrmse_finest < result.reconstruction_nrmse_gate


def test_sor_reconstruction_converges_to_residual_floor() -> None:
    """Each SOR solve must reach the residual stopping criterion."""
    geometry = SolovevGeometry.from_aspect()
    reconstruction = sor_reconstruction(geometry, 49, residual_tol=1e-9, max_sweeps=20000)
    assert reconstruction.converged is True
    assert reconstruction.residual_inf < 1e-9
    assert reconstruction.nrmse < 1e-4


def test_sor_reconstruction_reports_unconverged_when_capped() -> None:
    """A too-small sweep budget must surface as non-convergence, not silent pass."""
    geometry = SolovevGeometry.from_aspect()
    reconstruction = sor_reconstruction(geometry, 49, residual_tol=1e-12, max_sweeps=50, check_every=50)
    assert reconstruction.converged is False
    assert reconstruction.iterations == 50


# ── Multigrid reconstruction ─────────────────────────────────────────


def test_multigrid_reconstructs_solovev_field() -> None:
    """The production V-cycle must reconstruct the exact ψ to the residual floor."""
    geometry = SolovevGeometry.from_aspect()
    reconstruction = multigrid_reconstruction(geometry, 65, residual_tol=1e-6, max_cycles=40)
    assert reconstruction.converged is True
    assert reconstruction.residual_inf < 1e-6
    # NRMSE settles at the second-order discretisation floor, well under the SOR gate.
    assert reconstruction.nrmse < 1e-4
    # A handful of V-cycles is enough; this is not a slow single-grid relaxation.
    assert reconstruction.cycles <= 20


def test_multigrid_reduces_residual_geometrically() -> None:
    """One V-cycle must collapse the residual by orders of magnitude, not a constant.

    A single-grid smoother removes only high-frequency error and stalls on the
    smooth component, so its per-sweep residual reduction is close to one. The
    corrected V-cycle damps the whole spectrum, so the residual after a few
    cycles is a tiny fraction of the initial Grad-Shafranov residual.
    """
    geometry = SolovevGeometry.from_aspect()
    one_cycle = multigrid_reconstruction(geometry, 65, max_cycles=1)
    assert one_cycle.residual_inf < 0.2 * one_cycle.initial_residual_inf
    converged = multigrid_reconstruction(geometry, 65, residual_tol=1e-6, max_cycles=40)
    assert converged.residual_inf < 1e-9 * converged.initial_residual_inf


def test_multigrid_reconstruction_order_matches_discretisation() -> None:
    """Halving h must quarter the converged reconstruction NRMSE (second order)."""
    geometry = SolovevGeometry.from_aspect()
    coarse = multigrid_reconstruction(geometry, 33, residual_tol=1e-7, max_cycles=60)
    fine = multigrid_reconstruction(geometry, 65, residual_tol=1e-7, max_cycles=60)
    assert coarse.converged and fine.converged
    assert 3.0 < coarse.nrmse / fine.nrmse < 5.0


def test_multigrid_reconstruction_is_frozen() -> None:
    record = MultigridReconstruction(nrmse=1e-6, cycles=8, converged=True, residual_inf=1e-7, initial_residual_inf=1e4)
    with pytest.raises(AttributeError):
        record.nrmse = 2e-6  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"omega": 0.0}, "omega"),
        ({"residual_tol": -1.0}, "residual_tol"),
        ({"max_cycles": 0}, "max_cycles"),
        ({"pre_smooth": 0}, "pre_smooth"),
    ],
)
def test_multigrid_reconstruction_rejects_invalid_arguments(kwargs: dict[str, Any], match: str) -> None:
    geometry = SolovevGeometry.from_aspect()
    with pytest.raises(ValueError, match=match):
        multigrid_reconstruction(geometry, 33, **kwargs)


# ── Rust backend record ──────────────────────────────────────────────


def test_rust_backend_recorded_or_absent(result: GradShafranovValidationResult) -> None:
    """The Rust record is present iff the extension is, and never spoofs a pass."""
    assert result.multigrid_passed is True
    if result.rust_available:
        assert result.rust_record is not None
        # With the matched -Δ*ψ sign convention the Rust multigrid reconstructs
        # the analytic field and meets the gate.
        assert result.rust_record.meets_analytic_tolerance is True
        assert result.rust_record.resolution == result.resolutions[-1]
    else:
        assert result.rust_record is None
    # Rust never gates the Python validation outcome.
    assert result.passed == (result.operator_passed and result.reconstruction_passed and result.multigrid_passed)


def test_excluding_rust_yields_no_record() -> None:
    res = validate_grad_shafranov(resolutions=_TEST_RESOLUTIONS, include_rust=False)
    assert res.rust_available is False
    assert res.rust_record is None
    assert res.passed is True


# ── Geometry contract ────────────────────────────────────────────────


def test_geometry_rejects_axis_touching_box() -> None:
    with pytest.raises(ValueError, match="r_min must be positive"):
        SolovevGeometry(r0=1.0, a=1.5, r_min=0.0, r_max=2.0, z_min=-1.0, z_max=1.0, c1=1.0, c2=0.5)


def test_geometry_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="r_max must exceed r_min"):
        SolovevGeometry(r0=1.7, a=0.5, r_min=2.0, r_max=1.0, z_min=-1.0, z_max=1.0, c1=1.0, c2=0.5)


def test_validate_requires_two_distinct_resolutions() -> None:
    with pytest.raises(ValueError, match="at least two distinct resolutions"):
        validate_grad_shafranov(resolutions=(33, 33))


def test_validate_rejects_degenerate_resolution() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        validate_grad_shafranov(resolutions=(2, 33))


# ── Evidence seal ────────────────────────────────────────────────────


def test_evidence_roundtrip_is_sealed_and_passing(result: GradShafranovValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    assert evidence["schema_version"] == GRAD_SHAFRANOV_SOLOVEV_SCHEMA_VERSION
    assert validate_evidence_payload(evidence) is True


def test_evidence_tamper_is_rejected(result: GradShafranovValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["operator_order"] = 99.0
    with pytest.raises(ValueError, match="payload_sha256 does not match"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_empty_target_id(result: GradShafranovValidationResult) -> None:
    with pytest.raises(ValueError, match="target_id"):
        build_evidence(result, target_id="   ")


def test_evidence_rejects_unknown_schema(result: GradShafranovValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["schema_version"] = "scpn-control.unknown.v9"
    with pytest.raises(ValueError, match="unsupported"):
        validate_evidence_payload(evidence)


def test_evidence_rust_record_serialised_faithfully(result: GradShafranovValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    assert evidence["multigrid_passed"] is True
    assert evidence["multigrid_nrmse_finest"] == result.multigrid_nrmse_finest
    multigrid_records = cast(list[dict[str, Any]], evidence["multigrid_records"])
    assert multigrid_records[-1]["cycles"] == result.multigrid_details[-1].cycles
    if result.rust_available:
        rust_record = cast(dict[str, Any], evidence["rust_record"])
        assert rust_record["meets_analytic_tolerance"] is True
        assert rust_record["resolution"] == result.resolutions[-1]
    else:
        assert evidence["rust_record"] is None


# ── Convergence helper guardrails ────────────────────────────────────


def test_convergence_record_is_frozen() -> None:
    record = ConvergenceRecord(resolution=33, mesh_spacing=0.05, error=1e-4)
    with pytest.raises(AttributeError):
        record.error = 2e-4  # type: ignore[misc]


# ── Numeric guardrails ───────────────────────────────────────────────


def test_geometry_rejects_inverted_z_bounds() -> None:
    with pytest.raises(ValueError, match="z_max must exceed z_min"):
        SolovevGeometry(r0=1.7, a=0.5, r_min=1.0, r_max=2.0, z_min=1.0, z_max=-1.0, c1=1.0, c2=0.5)


def test_geometry_rejects_nonpositive_coefficient() -> None:
    with pytest.raises(ValueError, match="c1 must be positive"):
        SolovevGeometry(r0=1.7, a=0.5, r_min=1.0, r_max=2.0, z_min=-1.0, z_max=1.0, c1=0.0, c2=0.5)


def test_geometry_rejects_nonfinite_coefficient() -> None:
    with pytest.raises(ValueError, match="c2 must be finite"):
        SolovevGeometry(r0=1.7, a=0.5, r_min=1.0, r_max=2.0, z_min=-1.0, z_max=1.0, c1=1.0, c2=float("inf"))


def test_geometry_rejects_nonnumeric_coefficient() -> None:
    with pytest.raises(ValueError, match="must be a finite number"):
        SolovevGeometry(r0=1.7, a=0.5, r_min=1.0, r_max=2.0, z_min=-1.0, z_max=1.0, c1="x", c2=0.5)  # type: ignore[arg-type]


def test_validate_rejects_noninteger_resolution() -> None:
    with pytest.raises(ValueError, match="must be a positive integer"):
        validate_grad_shafranov(resolutions=(33, 49.5))  # type: ignore[arg-type]


def test_log_log_slope_requires_two_points() -> None:
    from validation.validate_grad_shafranov_solovev import _log_log_slope

    with pytest.raises(ValueError, match="at least two resolutions"):
        _log_log_slope([ConvergenceRecord(resolution=33, mesh_spacing=0.05, error=1e-4)])


def test_evidence_rejects_non_hex_seal(result: GradShafranovValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["payload_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        validate_evidence_payload(evidence)


# ── Rust import fallbacks ────────────────────────────────────────────


def test_rust_record_none_when_extension_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    import validation.validate_grad_shafranov_solovev as mod

    monkeypatch.setitem(sys.modules, "scpn_control_rs", None)  # forces ImportError
    geometry = SolovevGeometry.from_aspect()
    assert mod.rust_multigrid_reconstruction(geometry, 33, analytic_tolerance=1e-4) is None


def test_rust_record_none_when_binding_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    import validation.validate_grad_shafranov_solovev as mod

    stub = types.ModuleType("scpn_control_rs")  # present but without py_multigrid_solve
    monkeypatch.setitem(sys.modules, "scpn_control_rs", stub)
    geometry = SolovevGeometry.from_aspect()
    assert mod.rust_multigrid_reconstruction(geometry, 33, analytic_tolerance=1e-4) is None


# ── CLI / report writer ──────────────────────────────────────────────


def test_main_text_output_passes(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import validation.validate_grad_shafranov_solovev as mod

    real = mod.validate_grad_shafranov
    monkeypatch.setattr(mod, "validate_grad_shafranov", lambda **_: real(resolutions=(33, 49)))
    assert mod.main([]) == 0
    assert "Status: pass" in capsys.readouterr().out


def test_main_json_output_and_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    import json

    import validation.validate_grad_shafranov_solovev as mod

    real = mod.validate_grad_shafranov
    monkeypatch.setattr(mod, "validate_grad_shafranov", lambda **_: real(resolutions=(33, 49)))
    report = tmp_path / "gs.json"
    assert mod.main(["--json-out", "--report", str(report)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == GRAD_SHAFRANOV_SOLOVEV_SCHEMA_VERSION
    assert report.exists() and report.with_suffix(".md").exists()
    # The sealed on-disk report must validate.
    assert validate_evidence_payload(json.loads(report.read_text())) is True


def test_main_report_without_rust_writes_absent_note(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import validation.validate_grad_shafranov_solovev as mod

    real = mod.validate_grad_shafranov
    monkeypatch.setattr(mod, "validate_grad_shafranov", lambda **_: real(resolutions=(33, 49), include_rust=False))
    report = tmp_path / "gs_norust.json"
    assert mod.main(["--no-rust", "--report", str(report)]) == 0
    assert "extension not present" in report.with_suffix(".md").read_text()


def test_report_writer_describes_rust_tolerance_miss(
    result: GradShafranovValidationResult,
    tmp_path: Path,
) -> None:
    import validation.validate_grad_shafranov_solovev as mod

    evidence = build_evidence(result, target_id="test-target")
    evidence["rust_record"] = {
        "resolution": result.resolutions[-1],
        "nrmse": 1.0,
        "residual_inf": 2.0,
        "boundary_preserved": True,
        "meets_analytic_tolerance": False,
    }
    report = tmp_path / "gs_rust_miss.json"
    mod._write_report(evidence, report)
    assert "does not meet the analytic tolerance" in report.with_suffix(".md").read_text()


def test_main_returns_one_on_failure(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import validation.validate_grad_shafranov_solovev as mod

    real = mod.validate_grad_shafranov
    # A tiny sweep budget leaves the SOR reconstruction unconverged → failure.
    monkeypatch.setattr(
        mod,
        "validate_grad_shafranov",
        lambda **_: real(resolutions=(33, 49), max_sweeps=40, include_rust=False),
    )
    assert mod.main([]) == 1
    assert "Status: fail" in capsys.readouterr().out
