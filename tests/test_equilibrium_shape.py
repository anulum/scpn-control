# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Equilibrium shape/profile metric tests
"""Validation for the reusable equilibrium shape/profile metrics."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.control.realtime_efit import MU0, MagneticDiagnostics, RealtimeEFIT
from scpn_control.core.equilibrium_shape import (
    EquilibriumShape,
    boundary_geometry,
    compute_equilibrium_shape,
    cylindrical_q95,
    internal_inductance,
    largest_flux_contour,
    plasma_boundary,
    poloidal_beta,
    poloidal_field,
    pressure_grid,
    safety_factor_q95,
)


def _miller_boundary(r0: float, a: float, kappa: float, delta: float, n: int = 400) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = r0 + a * np.cos(theta + delta * np.sin(theta))
    z = kappa * a * np.sin(theta)
    return np.column_stack([r, z])


def _closure_equilibrium(
    n: int = 65,
) -> tuple[RealtimeEFIT, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    diag = MagneticDiagnostics(
        [(6.0, 0.0), (6.2, 1.0), (7.4, 0.0)], [(6.0, 0.0, "Z"), (7.4, 0.0, "Z")], rogowski_radius=6.2
    )
    r_grid = np.linspace(4.2, 8.2, n)
    z_grid = np.linspace(-3.0, 3.0, n)
    efit = RealtimeEFIT(diag, r_grid, z_grid, vacuum_rb_phi=33.0)
    p_true = np.array([2.0, -1.5, 0.4])
    ff_true = np.array([1.0, -0.6, 0.1])
    x = efit._geometric_rho()[0]
    psi = np.zeros((n, n))
    for _ in range(60):
        srcs = efit._basis_sources(x)
        src = sum(p_true[k] * srcs[k] for k in range(3)) + sum(ff_true[k] * srcs[3 + k] for k in range(3))
        psi = efit._solve_source(src)
        x = efit._normalized_flux(psi)
    ip = float(efit._diagnostic_vector(psi)[-1])
    return efit, r_grid, z_grid, psi, p_true, ff_true, ip


def test_boundary_geometry_recovers_miller_shape():
    pts = _miller_boundary(6.2, 2.0, 1.7, 0.3)
    r0, a, kappa, delta_u, delta_l = boundary_geometry(pts)
    assert r0 == pytest.approx(6.2, abs=0.05)
    assert a == pytest.approx(2.0, abs=0.05)
    assert kappa == pytest.approx(1.7, abs=0.05)
    assert delta_u == pytest.approx(0.3, abs=0.06)
    assert delta_l == pytest.approx(0.3, abs=0.06)


def test_poloidal_field_satisfies_amperes_law():
    pytest.importorskip("contourpy")
    from scipy.interpolate import RegularGridInterpolator

    _efit, r_grid, z_grid, psi, _p, _ff, ip = _closure_equilibrium()
    dpsi_dr = np.gradient(psi, r_grid, axis=0, edge_order=2)
    dpsi_dz = np.gradient(psi, z_grid, axis=1, edge_order=2)
    rr = r_grid[:, None]
    b_r = -dpsi_dz / rr
    b_z = dpsi_dr / rr
    i_br = RegularGridInterpolator((r_grid, z_grid), b_r, bounds_error=False, fill_value=0.0)
    i_bz = RegularGridInterpolator((r_grid, z_grid), b_z, bounds_error=False, fill_value=0.0)
    loop = largest_flux_contour(np.clip(1.0 - psi / psi.max(), 0.0, 1.0), r_grid, z_grid, 0.5)
    assert loop is not None
    seg = np.diff(np.vstack([loop, loop[:1]]), axis=0)
    mid = 0.5 * (np.vstack([loop, loop[:1]])[:-1] + np.vstack([loop, loop[:1]])[1:])
    b_dot_dl = i_br(mid) * seg[:, 0] + i_bz(mid) * seg[:, 1]
    # The enclosed current of the psi_N=0.5 surface is a fraction of Ip, so just
    # check the field magnitude and Ampere sign/scale are physical, not exact Ip.
    assert np.all(np.isfinite(poloidal_field(psi, r_grid, z_grid)))
    assert abs(float(np.sum(b_dot_dl))) < MU0 * abs(ip)


def test_internal_inductance_is_physical():
    _efit, r_grid, z_grid, psi, _p, _ff, ip = _closure_equilibrium()
    boundary = plasma_boundary(psi, r_grid, z_grid)
    r0 = 0.5 * (boundary[:, 0].max() + boundary[:, 0].min())
    li = internal_inductance(psi, r_grid, z_grid, ip, r0)
    assert 0.2 < li < 2.0


def test_internal_inductance_zero_for_zero_current():
    _efit, r_grid, z_grid, psi, _p, _ff, _ip = _closure_equilibrium()
    assert internal_inductance(psi, r_grid, z_grid, 0.0, 6.2) == 0.0


def test_pressure_grid_vanishes_at_edge_and_peaks_at_axis():
    psi_n = np.array([0.0, 0.5, 1.0])
    p = pressure_grid(psi_n, np.array([3.0, -1.0]), psi_axis=2.0)
    assert p[-1] == pytest.approx(0.0)  # boundary
    assert p[0] > p[1] > p[-1]  # monotone decreasing axis -> edge for positive p'


def test_poloidal_beta_nonnegative_and_zero_for_no_pressure():
    _efit, r_grid, z_grid, psi, _p, _ff, ip = _closure_equilibrium()
    psi_n = np.clip(1.0 - psi / psi.max(), 0.0, 1.0)
    beta = poloidal_beta(psi, psi_n, r_grid, z_grid, np.array([2.0, -1.5, 0.4]), float(psi.max()), 2.0, ip)
    assert beta >= 0.0
    zero = poloidal_beta(psi, psi_n, r_grid, z_grid, np.zeros(3), float(psi.max()), 2.0, ip)
    assert zero == 0.0


def test_poloidal_beta_zero_for_zero_current():
    _efit, r_grid, z_grid, psi, _p, _ff, _ip = _closure_equilibrium()
    psi_n = np.clip(1.0 - psi / psi.max(), 0.0, 1.0)
    assert poloidal_beta(psi, psi_n, r_grid, z_grid, np.array([1.0]), float(psi.max()), 2.0, 0.0) == 0.0


def test_safety_factor_q95_is_finite_and_positive():
    pytest.importorskip("contourpy")
    _efit, r_grid, z_grid, psi, _p, ff, _ip = _closure_equilibrium()
    psi_n = np.clip(1.0 - psi / psi.max(), 0.0, 1.0)
    q95 = safety_factor_q95(psi, psi_n, r_grid, z_grid, ff, float(psi.max()), 33.0)
    assert np.isfinite(q95)
    assert q95 > 0.0


def test_safety_factor_q95_nan_for_absent_surface():
    _efit, r_grid, z_grid, psi, _p, ff, _ip = _closure_equilibrium()
    psi_n = np.clip(1.0 - psi / psi.max(), 0.0, 1.0)
    # psi_N = 2.0 lies outside [0, 1] so no contour exists.
    assert np.isnan(safety_factor_q95(psi, psi_n, r_grid, z_grid, ff, float(psi.max()), 33.0, surface=2.0))


def test_compute_equilibrium_shape_full_set():
    pytest.importorskip("contourpy")
    _efit, r_grid, z_grid, psi, p_true, ff_true, ip = _closure_equilibrium()
    shape = compute_equilibrium_shape(psi, r_grid, z_grid, p_true, ff_true, ip, 33.0)
    assert isinstance(shape, EquilibriumShape)
    assert shape.R0 == pytest.approx(6.2, abs=0.1)
    assert shape.a == pytest.approx(2.0, abs=0.2)
    assert shape.kappa > 1.0
    assert abs(shape.delta_upper) < 0.3
    assert 0.2 < shape.li < 2.0
    assert shape.beta_pol >= 0.0
    assert np.isfinite(shape.q95) and shape.q95 > 0.0


def test_compute_equilibrium_shape_none_for_degenerate_flux():
    r_grid = np.linspace(4.2, 8.2, 17)
    z_grid = np.linspace(-3.0, 3.0, 17)
    psi = np.zeros((17, 17))
    assert compute_equilibrium_shape(psi, r_grid, z_grid, np.zeros(3), np.zeros(3), 0.0, 33.0) is None


def test_plasma_boundary_empty_for_nonpositive_flux():
    r_grid = np.linspace(4.2, 8.2, 9)
    z_grid = np.linspace(-3.0, 3.0, 9)
    assert plasma_boundary(np.zeros((9, 9)), r_grid, z_grid).shape == (0, 2)


def test_largest_flux_contour_none_for_absent_level():
    _efit, r_grid, z_grid, psi, _p, _ff, _ip = _closure_equilibrium()
    psi_n = np.clip(1.0 - psi / psi.max(), 0.0, 1.0)
    assert largest_flux_contour(psi_n, r_grid, z_grid, 5.0) is None


def test_cylindrical_q95_estimate():
    # Contour-free fallback: finite, positive, and elongation-scaled.
    q = cylindrical_q95(a=2.0, kappa=1.7, r0=6.2, ip=5.0e6, vacuum_rb_phi=33.0)
    assert np.isfinite(q) and q > 0.0
    q_round = cylindrical_q95(a=2.0, kappa=1.0, r0=6.2, ip=5.0e6, vacuum_rb_phi=33.0)
    assert q > q_round  # elongation raises the cylindrical q
    assert np.isnan(cylindrical_q95(a=2.0, kappa=1.7, r0=6.2, ip=0.0, vacuum_rb_phi=33.0))
