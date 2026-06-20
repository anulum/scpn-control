# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Realtime Efit
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Real-Time EFIT Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from scpn_control.control.realtime_efit import (
    EFITLiteClaimEvidence,
    MU0,
    MagneticDiagnostics,
    RealtimeEFIT,
    ReconstructionResult,
    ShapeParams,
    _finite_float,
    _relative_array_error,
    _trapezoid_integral,
    assert_efit_lite_facility_claim_admissible,
    efit_lite_claim_evidence,
    save_efit_lite_claim_evidence,
)


def create_mock_diagnostics() -> MagneticDiagnostics:
    flux_loops = [(2.0, 1.0), (3.0, 1.5), (4.0, 1.0)]
    b_probes = [(2.0, 1.0, "R"), (2.0, 1.0, "Z"), (4.0, 1.0, "R")]
    return MagneticDiagnostics(flux_loops, b_probes, rogowski_radius=3.0)


def _solovev_efit_and_result() -> tuple[RealtimeEFIT, ReconstructionResult]:
    diag = create_mock_diagnostics()
    R = np.linspace(4.2, 8.2, 33)
    Z = np.linspace(-3.0, 3.0, 33)
    efit = RealtimeEFIT(diag, R, Z)
    res = efit.reconstruct(
        {"flux_loops": np.zeros(3), "b_probes": np.zeros(3), "Ip": 15.0e6, "coil_currents": np.zeros(5)}
    )
    return efit, res


def test_simulate_measurements():
    diag = create_mock_diagnostics()
    R = np.linspace(2.0, 10.0, 30)
    Z = np.linspace(-6.0, 6.0, 30)

    efit = RealtimeEFIT(diag, R, Z)

    # Generic psi field
    R2, Z2 = np.meshgrid(R, Z, indexing="ij")
    psi = (R2 - 6.0) ** 2 + Z2**2

    coils = np.zeros(5)
    meas = efit.response.simulate_measurements(psi, coils)

    assert len(meas["flux_loops"]) == len(diag.flux_loops)
    assert len(meas["b_probes"]) == len(diag.b_probes)
    assert "Ip" in meas


def test_simulate_measurements_derives_rogowski_current_from_flux_source():
    R = np.linspace(2.0, 4.0, 81)
    Z = np.linspace(-1.0, 1.0, 81)
    diag = MagneticDiagnostics(
        flux_loops=[(2.4, -0.5), (3.0, 0.0), (3.6, 0.5)],
        b_probes=[(2.4, 0.0, "R"), (3.0, 0.0, "Z"), (3.6, 0.0, "Z")],
        rogowski_radius=3.0,
    )
    efit = RealtimeEFIT(diag, R, Z)

    mu0 = 4.0e-7 * np.pi
    current_density = 2.5e6
    rr, _ = np.meshgrid(R, Z, indexing="ij")
    psi = -(mu0 * current_density / 3.0) * rr**3

    meas = efit.response.simulate_measurements(psi, np.zeros(3))

    expected_ip = current_density * (R[-1] - R[0]) * (Z[-1] - Z[0])
    np.testing.assert_allclose(meas["Ip"], expected_ip, rtol=0.02)
    assert not np.isclose(meas["Ip"], 15.0e6)


def test_reconstruction_solovev():
    diag = create_mock_diagnostics()
    R = np.linspace(4.2, 8.2, 33)
    Z = np.linspace(-3.0, 3.0, 33)

    efit = RealtimeEFIT(diag, R, Z)

    meas = {"flux_loops": np.zeros(3), "b_probes": np.zeros(3), "Ip": 15.0e6, "coil_currents": np.zeros(5)}

    res = efit.reconstruct(meas)

    # Check shape params
    assert np.isclose(res.shape.R0, 6.2)
    assert np.isclose(res.shape.a, 2.0)
    assert res.shape.Ip_reconstructed == 15.0e6
    # CI variance across OS/VM classes can exceed 100 ms while preserving the
    # same reconstructed physics result.
    assert res.wall_time_ms < 150.0
    assert res.n_iterations > 0


def test_efit_lite_claim_evidence_records_synthetic_boundary(tmp_path):
    diag = create_mock_diagnostics()
    R = np.linspace(4.2, 8.2, 33)
    Z = np.linspace(-3.0, 3.0, 33)
    efit = RealtimeEFIT(diag, R, Z)
    meas = {"flux_loops": np.zeros(3), "b_probes": np.zeros(3), "Ip": 15.0e6, "coil_currents": np.zeros(5)}
    res = efit.reconstruct(meas)

    evidence = efit_lite_claim_evidence(
        res,
        diag,
        source="synthetic_regression_reference",
        source_id="tests/test_realtime_efit.py::synthetic_solovev",
        diagnostic_source="synthetic diagnostic response",
    )
    out = tmp_path / "efit_claim.json"
    save_efit_lite_claim_evidence(evidence, out)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert isinstance(evidence, EFITLiteClaimEvidence)
    assert evidence.grid_shape == res.psi.shape
    assert evidence.n_flux_loops == len(diag.flux_loops)
    assert evidence.n_b_probes == len(diag.b_probes)
    assert evidence.ip_reconstructed_A == pytest.approx(15.0e6)
    assert evidence.psi_relative_error is None
    assert evidence.facility_claim_allowed is False
    assert evidence.claim_status.startswith("bounded synthetic EFIT-lite regression evidence")
    assert payload["schema_version"] == 1
    assert payload["facility_claim_allowed"] is False


def test_efit_lite_facility_admission_requires_matched_reference():
    diag = create_mock_diagnostics()
    R = np.linspace(4.2, 8.2, 33)
    Z = np.linspace(-3.0, 3.0, 33)
    efit = RealtimeEFIT(diag, R, Z)
    meas = {"flux_loops": np.zeros(3), "b_probes": np.zeros(3), "Ip": 15.0e6, "coil_currents": np.zeros(5)}
    res = efit.reconstruct(meas)
    reference_shape = ShapeParams(
        R0=res.shape.R0,
        a=res.shape.a,
        kappa=res.shape.kappa,
        delta_upper=res.shape.delta_upper,
        delta_lower=res.shape.delta_lower,
        q95=res.shape.q95,
        beta_pol=res.shape.beta_pol,
        li=res.shape.li,
        Ip_reconstructed=res.shape.Ip_reconstructed,
    )
    rejected_shape = ShapeParams(
        R0=res.shape.R0,
        a=res.shape.a,
        kappa=res.shape.kappa,
        delta_upper=res.shape.delta_upper,
        delta_lower=res.shape.delta_lower,
        q95=res.shape.q95 + 0.5,
        beta_pol=res.shape.beta_pol,
        li=res.shape.li,
        Ip_reconstructed=res.shape.Ip_reconstructed * 1.2,
    )

    admitted = efit_lite_claim_evidence(
        res,
        diag,
        source="efit_reference",
        source_id="reference_efit_solovev_case",
        diagnostic_source="reference flux loops, B probes, and Rogowski",
        reference_psi=res.psi.copy(),
        reference_shape=reference_shape,
    )
    rejected = efit_lite_claim_evidence(
        res,
        diag,
        source="efit_reference",
        source_id="reference_efit_mismatch_case",
        diagnostic_source="reference flux loops, B probes, and Rogowski",
        reference_psi=res.psi * 1.2,
        reference_shape=rejected_shape,
        psi_relative_tolerance=0.05,
        ip_relative_tolerance=0.02,
        q95_abs_tolerance=0.1,
    )

    assert assert_efit_lite_facility_claim_admissible(admitted) == admitted
    assert admitted.facility_claim_allowed is True
    assert admitted.psi_relative_error == pytest.approx(0.0)
    assert admitted.ip_relative_error == pytest.approx(0.0)
    assert rejected.facility_claim_allowed is False
    with pytest.raises(ValueError, match="not admissible"):
        assert_efit_lite_facility_claim_admissible(rejected)


def test_efit_lite_claim_evidence_rejects_invalid_reference_inputs():
    diag = create_mock_diagnostics()
    R = np.linspace(4.2, 8.2, 17)
    Z = np.linspace(-3.0, 3.0, 17)
    efit = RealtimeEFIT(diag, R, Z)
    res = efit.reconstruct(
        {"flux_loops": np.zeros(3), "b_probes": np.zeros(3), "Ip": 15.0e6, "coil_currents": np.zeros(5)}
    )

    with pytest.raises(ValueError, match="source"):
        efit_lite_claim_evidence(res, diag, source="mock", source_id="case", diagnostic_source="synthetic")
    with pytest.raises(ValueError, match="source_id"):
        efit_lite_claim_evidence(
            res, diag, source="synthetic_regression_reference", source_id="", diagnostic_source="synthetic"
        )
    with pytest.raises(ValueError, match="diagnostic_source"):
        efit_lite_claim_evidence(
            res, diag, source="synthetic_regression_reference", source_id="case", diagnostic_source=""
        )
    with pytest.raises(ValueError, match="psi reference"):
        efit_lite_claim_evidence(
            res,
            diag,
            source="efit_reference",
            source_id="case",
            diagnostic_source="reference",
            reference_psi=np.zeros((3, 3)),
        )


def test_gs_solver_satisfies_constant_source_residual():
    diag = create_mock_diagnostics()
    R = np.linspace(4.2, 8.2, 41)
    Z = np.linspace(-3.0, 3.0, 41)
    efit = RealtimeEFIT(diag, R, Z)

    p_prime = 2.0e5
    ff_prime = 0.4
    psi = efit._solve_gs_with_sources(np.array([p_prime]), np.array([ff_prime]))

    dpsi_dR = np.gradient(psi, R, axis=0, edge_order=2)
    d2psi_dR2 = np.gradient(dpsi_dR, R, axis=0, edge_order=2)
    dpsi_dZ = np.gradient(psi, Z, axis=1, edge_order=2)
    d2psi_dZ2 = np.gradient(dpsi_dZ, Z, axis=1, edge_order=2)
    delta_star = d2psi_dR2 - dpsi_dR / R[:, np.newaxis] + d2psi_dZ2
    source = np.broadcast_to(
        -(MU0 * R[:, np.newaxis] ** 2 * p_prime + ff_prime),
        psi.shape,
    )

    interior = np.s_[2:-2, 2:-2]
    residual = delta_star[interior] - source[interior]
    assert np.linalg.norm(residual) / np.linalg.norm(source[interior]) < 0.08
    assert np.allclose(psi[[0, -1], :], 0.0)
    assert np.allclose(psi[:, [0, -1]], 0.0)


def test_xpoint_detection():
    diag = create_mock_diagnostics()
    R = np.linspace(4.2, 8.2, 33)
    Z = np.linspace(-3.0, 3.0, 33)

    efit = RealtimeEFIT(diag, R, Z)

    # Just need an arbitrary psi
    psi = np.zeros((33, 33))
    xp = efit.find_xpoint(psi)

    assert xp is not None
    assert xp[0] > 0.0


def test_find_lcfs_extracts_elliptical_boundary():
    diag = create_mock_diagnostics()
    R = np.linspace(4.2, 8.2, 81)
    Z = np.linspace(-3.0, 3.0, 81)
    efit = RealtimeEFIT(diag, R, Z)

    rr, zz = np.meshgrid(R, Z, indexing="ij")
    R0 = 6.2
    a = 1.3
    kappa = 1.6
    psi = 1.0 - ((rr - R0) / a) ** 2 - (zz / (kappa * a)) ** 2
    psi = np.maximum(psi, 0.0)

    lcfs = efit.find_lcfs(psi)

    assert lcfs.shape[1] == 2
    assert lcfs.shape[0] > 20
    np.testing.assert_allclose(np.ptp(lcfs[:, 0]), 2.0 * a, rtol=0.2, atol=0.2)
    np.testing.assert_allclose(np.ptp(lcfs[:, 1]), 2.0 * kappa * a, rtol=0.2, atol=0.2)
    assert np.all(np.isfinite(lcfs))


# ── Trapezoidal integration helper ───────────────────────────────────


def test_trapezoid_integral_rejects_non_one_dimensional_grid():
    with pytest.raises(ValueError, match="grid must be one-dimensional"):
        _trapezoid_integral(np.ones(3), np.ones((2, 2)))


def test_trapezoid_integral_rejects_length_mismatch():
    with pytest.raises(ValueError, match="values and grid lengths must match"):
        _trapezoid_integral(np.ones(3), np.array([0.0, 1.0]))


def test_trapezoid_integral_returns_zero_for_degenerate_grid():
    result = _trapezoid_integral(np.ones(1), np.array([0.0]))
    assert result.shape == ()
    assert float(result) == 0.0


# ── Numeric / array validation helpers ───────────────────────────────


def test_finite_float_rejects_non_finite_and_sign_violations():
    with pytest.raises(ValueError, match="must be finite"):
        _finite_float("x", float("inf"))
    with pytest.raises(ValueError, match="must be positive"):
        _finite_float("x", 0.0, positive=True)
    with pytest.raises(ValueError, match="must be non-negative"):
        _finite_float("x", -1.0, nonnegative=True)


def test_relative_array_error_rejects_non_finite_reference():
    with pytest.raises(ValueError, match="reference must be finite"):
        _relative_array_error("psi", np.ones((2, 2)), np.array([[1.0, np.inf], [1.0, 1.0]]))


# ── Claim-evidence builder guards ────────────────────────────────────


def test_claim_evidence_rejects_blank_model_id():
    efit, res = _solovev_efit_and_result()
    with pytest.raises(ValueError, match="model_id must be a non-empty string"):
        efit_lite_claim_evidence(
            res,
            efit.diagnostics,
            source="synthetic_regression_reference",
            source_id="case",
            diagnostic_source="synthetic",
            model_id="   ",
        )


def test_claim_evidence_rejects_non_two_dimensional_psi():
    efit, res = _solovev_efit_and_result()
    tampered = replace(res, psi=np.ones((2, 2)))
    with pytest.raises(ValueError, match="two-dimensional grid with both dimensions"):
        efit_lite_claim_evidence(
            tampered,
            efit.diagnostics,
            source="synthetic_regression_reference",
            source_id="case",
            diagnostic_source="synthetic",
        )


def test_claim_evidence_rejects_non_finite_psi():
    efit, res = _solovev_efit_and_result()
    psi = res.psi.copy()
    psi[0, 0] = np.nan
    tampered = replace(res, psi=psi)
    with pytest.raises(ValueError, match="result.psi must be finite"):
        efit_lite_claim_evidence(
            tampered,
            efit.diagnostics,
            source="synthetic_regression_reference",
            source_id="case",
            diagnostic_source="synthetic",
        )


def test_claim_evidence_rejects_non_positive_iteration_count():
    efit, res = _solovev_efit_and_result()
    tampered = replace(res, n_iterations=0)
    with pytest.raises(ValueError, match="n_iterations must be positive"):
        efit_lite_claim_evidence(
            tampered,
            efit.diagnostics,
            source="synthetic_regression_reference",
            source_id="case",
            diagnostic_source="synthetic",
        )


def test_claim_evidence_flags_external_source_without_complete_comparison():
    efit, res = _solovev_efit_and_result()
    evidence = efit_lite_claim_evidence(
        res,
        efit.diagnostics,
        source="efit_reference",
        source_id="reference_without_comparison",
        diagnostic_source="reference diagnostics",
    )
    assert evidence.facility_claim_allowed is False
    assert "comparison is missing" in evidence.claim_status


def test_facility_admission_rejects_non_evidence_object():
    with pytest.raises(ValueError, match="must be EFITLiteClaimEvidence"):
        assert_efit_lite_facility_claim_admissible({"not": "evidence"})


def test_facility_admission_rejects_unsupported_schema_version():
    efit, res = _solovev_efit_and_result()
    evidence = efit_lite_claim_evidence(
        res,
        efit.diagnostics,
        source="synthetic_regression_reference",
        source_id="case",
        diagnostic_source="synthetic",
    )
    tampered = replace(evidence, schema_version=99)
    with pytest.raises(ValueError, match="schema_version is unsupported"):
        assert_efit_lite_facility_claim_admissible(tampered)


# ── DiagnosticResponse guards ────────────────────────────────────────


def test_simulate_measurements_rejects_psi_shape_mismatch():
    efit, _res = _solovev_efit_and_result()
    with pytest.raises(ValueError, match="psi shape must match the diagnostic R/Z grid"):
        efit.response.simulate_measurements(np.ones((3, 3)), np.zeros(5))


def test_simulate_measurements_rejects_non_finite_psi():
    efit, _res = _solovev_efit_and_result()
    psi = np.zeros((efit.nR, efit.nZ))
    psi[0, 0] = np.inf
    with pytest.raises(ValueError, match="psi must be finite"):
        efit.response.simulate_measurements(psi, np.zeros(5))


# ── Grad-Shafranov solver guards ─────────────────────────────────────


def test_gs_solver_rejects_non_one_dimensional_coefficients():
    efit, _res = _solovev_efit_and_result()
    with pytest.raises(ValueError, match="source coefficients must be one-dimensional"):
        efit._solve_gs_with_sources(np.ones((2, 2)), np.ones(1))


def test_gs_solver_rejects_empty_coefficients():
    efit, _res = _solovev_efit_and_result()
    with pytest.raises(ValueError, match="source coefficient arrays must be non-empty"):
        efit._solve_gs_with_sources(np.array([]), np.array([1.0]))


def test_gs_solver_rejects_non_finite_coefficients():
    efit, _res = _solovev_efit_and_result()
    with pytest.raises(ValueError, match="source coefficients must be finite"):
        efit._solve_gs_with_sources(np.array([np.nan]), np.array([1.0]))


def test_gs_solver_rejects_too_few_grid_points():
    diag = create_mock_diagnostics()
    efit = RealtimeEFIT(diag, np.linspace(2.0, 4.0, 2), np.linspace(-1.0, 1.0, 5))
    with pytest.raises(ValueError, match="at least three R and Z points"):
        efit._solve_gs_with_sources(np.array([1.0]), np.array([1.0]))


def test_gs_solver_rejects_non_uniform_spacing():
    diag = create_mock_diagnostics()
    efit = RealtimeEFIT(diag, np.array([2.0, 2.5, 5.0]), np.linspace(-1.0, 1.0, 3))
    with pytest.raises(ValueError, match="uniform R/Z spacing"):
        efit._solve_gs_with_sources(np.array([1.0]), np.array([1.0]))


def test_gs_solver_raises_when_solve_produces_non_finite_flux(monkeypatch):
    import scipy.sparse.linalg

    diag = create_mock_diagnostics()
    efit = RealtimeEFIT(diag, np.linspace(4.2, 8.2, 33), np.linspace(-3.0, 3.0, 33))
    monkeypatch.setattr(scipy.sparse.linalg, "spsolve", lambda *_a, **_k: np.array([np.nan, np.nan]))
    with pytest.raises(RuntimeError, match="non-finite flux"):
        efit._solve_gs_with_sources(np.array([1.0]), np.array([1.0]))


# ── LCFS tracing guards ──────────────────────────────────────────────


def test_find_lcfs_rejects_psi_shape_mismatch():
    efit, _res = _solovev_efit_and_result()
    with pytest.raises(ValueError, match="psi shape must match the EFIT R/Z grid"):
        efit.find_lcfs(np.ones((3, 3)))


def test_find_lcfs_rejects_non_finite_psi():
    efit, _res = _solovev_efit_and_result()
    psi = np.zeros((efit.nR, efit.nZ))
    psi[0, 0] = np.nan
    with pytest.raises(ValueError, match="psi must be finite"):
        efit.find_lcfs(psi)


def test_find_lcfs_returns_empty_for_non_positive_flux():
    efit, _res = _solovev_efit_and_result()
    lcfs = efit.find_lcfs(np.zeros((efit.nR, efit.nZ)))
    assert lcfs.shape == (0, 2)
