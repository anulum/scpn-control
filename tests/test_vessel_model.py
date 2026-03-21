# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851  Contact: protoscience@anulum.li
"""
Tests for the vacuum vessel eddy current model.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.vessel_model import (
    TPF,
    VesselElement,
    VesselModel,
    halo_current,
    halo_em_force,
    vessel_time_constant,
)


def test_vessel_current_decay():
    """Verify that currents decay exponentially with the L/R time constant."""
    # Single element for simplicity
    L = 1e-6
    R_res = 1e-3
    tau = L / R_res  # 1ms

    el = VesselElement(R=1.0, Z=0.0, resistance=R_res, cross_section=0.01, inductance=L)
    model = VesselModel([el])

    # Initial current
    model.I = np.array([1000.0])

    # Use small dt to minimize Euler error
    dt = 1e-6  # 0.01ms
    # Evolve 500 steps (0.5ms)
    for _ in range(500):
        model.step(dt, np.array([0.0]))

    # Expected: I = I0 * exp(-t/tau) = 1000 * exp(-0.5) approx 606.5
    expected = 1000.0 * np.exp(-0.5e-3 / tau)
    # With dt=1e-6, Euler error should be small enough for rel=0.01
    assert model.I[0] == pytest.approx(expected, rel=0.01)


def test_vessel_symmetry():
    """Verify that a symmetric vessel responds symmetrically to symmetric drive."""
    # Two symmetric elements at +/- Z
    el1 = VesselElement(R=1.5, Z=0.5, resistance=1e-3, cross_section=0.01, inductance=1e-6)
    el2 = VesselElement(R=1.5, Z=-0.5, resistance=1e-3, cross_section=0.01, inductance=1e-6)

    model = VesselModel([el1, el2])

    # Symmetric drive (e.g. from plasma current change)
    drive = np.array([10.0, 10.0])
    model.step(0.01, drive)

    assert model.I[0] == pytest.approx(model.I[1], rel=1e-10)


def test_psi_vessel_axisymmetry():
    """Verify that the flux contribution is axisymmetric (constant in theta).

    Actually, our model is already 2D (R, Z), so we check that psi(R, Z)
    is calculated correctly at multiple points.
    """
    el = VesselElement(R=1.0, Z=0.0, resistance=1e-3, cross_section=0.01, inductance=1e-6)
    model = VesselModel([el])
    model.I = np.array([100.0])

    # Check at two points with same R but different Z (should differ)
    # and check at two points with same Z but different R (should differ)
    # This just ensures the Green's function is working.
    R_obs = np.array([1.5, 1.5, 2.0])
    Z_obs = np.array([0.0, 0.5, 0.0])

    psi = model.psi_vessel(R_obs, Z_obs)

    assert psi[0] != psi[1]
    assert psi[0] != psi[2]
    assert np.all(psi > 0)  # Current is positive, R is positive -> psi should be positive


def test_halo_current_bounded():
    """I_halo = f_halo × TPF × I_p (ITER Physics Basis 1999, §3.8.3).

    f_halo ∈ [0.1, 0.5] is the halo current fraction; TPF ≈ 2 toroidal peaking.
    Peak halo current must always be < I_p.
    """
    I_p = 15.0e6  # 15 MA — ITER full-current scenario
    f_halo = 0.3  # ITER default
    I_halo = halo_current(I_p, f_halo=f_halo)
    assert I_halo == pytest.approx(f_halo * TPF * I_p)
    assert I_halo < I_p  # halo current must be < plasma current


def test_vessel_time_constant():
    """τ_vessel = μ₀ σ d R must be positive.

    Wesson 2011, "Tokamaks", 4th ed., Eq. 6.6.6.
    Stainless steel 316L: σ ≈ 1.35e6 S/m; d = 0.04 m; R = 6.2 m (ITER).
    """
    tau = vessel_time_constant(conductivity=1.35e6, wall_thickness=0.04, major_radius=6.2)
    assert tau > 0.0
    # Sanity: τ should be in the range [1 ms, 10 s] for metallic tokamak walls.
    assert 1e-3 < tau < 10.0


def test_halo_em_force_positive():
    """F = I_halo × B_pol × L must be positive for positive inputs.

    Noll et al. 1993, Fusion Eng. Des. 22, 315.
    """
    F = halo_em_force(halo_current_a=1e6, b_poloidal=0.5, path_length=2.0)
    assert F > 0.0
    assert F == pytest.approx(1e6 * 0.5 * 2.0)


def test_singular_inductance_matrix(monkeypatch):
    el1 = VesselElement(R=1.0, Z=0.0, resistance=1e-3, cross_section=0.01, inductance=1e-6)
    el2 = VesselElement(R=2.0, Z=0.0, resistance=1e-3, cross_section=0.01, inductance=1e-6)

    def _inv_raise(m):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(np.linalg, "inv", _inv_raise)
    model = VesselModel([el1, el2])
    np.testing.assert_array_equal(model.M_inv, np.zeros((2, 2)))


def test_mutual_inductance_coincident_returns_zero():
    el = VesselElement(R=0.0, Z=0.0, resistance=1e-3, cross_section=0.01, inductance=1e-6)
    model = VesselModel([el])
    assert model._mutual_inductance(0.0, 0.0, 0.0, 0.0) == 0.0


def test_step_nonpositive_dt_returns_unchanged():
    el = VesselElement(R=1.0, Z=0.0, resistance=1e-3, cross_section=0.01, inductance=1e-6)
    model = VesselModel([el])
    model.I = np.array([42.0])
    result = model.step(0.0, np.array([10.0]))
    assert result[0] == pytest.approx(42.0)
    result = model.step(-1.0, np.array([10.0]))
    assert result[0] == pytest.approx(42.0)


def test_psi_vessel_skips_zero_current():
    el1 = VesselElement(R=1.0, Z=0.0, resistance=1e-3, cross_section=0.01, inductance=1e-6)
    el2 = VesselElement(R=2.0, Z=0.0, resistance=1e-3, cross_section=0.01, inductance=1e-6)
    model = VesselModel([el1, el2])
    model.I = np.array([0.0, 100.0])
    R_obs = np.array([1.5])
    Z_obs = np.array([0.0])
    psi = model.psi_vessel(R_obs, Z_obs)
    model2 = VesselModel([el2])
    model2.I = np.array([100.0])
    psi2 = model2.psi_vessel(R_obs, Z_obs)
    assert psi[0] == pytest.approx(psi2[0], rel=1e-6)
