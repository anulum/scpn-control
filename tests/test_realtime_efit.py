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

import numpy as np

from scpn_control.control.realtime_efit import (
    MU0,
    MagneticDiagnostics,
    RealtimeEFIT,
)


def create_mock_diagnostics() -> MagneticDiagnostics:
    flux_loops = [(2.0, 1.0), (3.0, 1.5), (4.0, 1.0)]
    b_probes = [(2.0, 1.0, "R"), (2.0, 1.0, "Z"), (4.0, 1.0, "R")]
    return MagneticDiagnostics(flux_loops, b_probes, rogowski_radius=3.0)


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
