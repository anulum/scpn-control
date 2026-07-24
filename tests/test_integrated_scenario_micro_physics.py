# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for scenario micro-physics leaf

"""Drive production Spitzer / gyro-Bohm / diffusion helpers on real profiles."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_control.core.integrated_scenario as owner
import scpn_control.core.integrated_scenario_micro_physics as leaf


def test_owner_symbols_bind_to_leaf() -> None:
    """Owner re-exports are the production micro-physics leaf callables."""
    assert owner._spitzer_resistivity is leaf._spitzer_resistivity
    assert owner._gyro_bohm_chi is leaf._gyro_bohm_chi
    assert owner._diffusion_step is leaf._diffusion_step


def test_spitzer_resistivity_positive_and_clamps_low_te() -> None:
    """Spitzer η decreases with Te and clamps near-zero Te."""
    eta = leaf._spitzer_resistivity(np.array([0.001, 1.0, 4.0]), Z_eff=1.5)
    assert eta.shape == (3,)
    assert np.all(np.isfinite(eta))
    assert eta[0] > eta[1] > eta[2]
    with pytest.raises(ValueError, match="Te_keV"):
        leaf._spitzer_resistivity(np.array([1.0, -0.1]), Z_eff=1.5)
    with pytest.raises(ValueError, match="Z_eff"):
        leaf._spitzer_resistivity(np.array([1.0]), Z_eff=0.0)


def test_gyro_bohm_chi_positive_and_rejects_bad_q() -> None:
    """Gyro-Bohm χ is positive with a floor and rejects non-positive q."""
    rho = np.linspace(0.0, 1.0, 20)
    Te = 2.0 * (1.0 - 0.5 * rho**2)
    Ti = Te.copy()
    ne = 5.0 * np.ones_like(rho)
    q = 1.0 + 2.0 * rho**2
    chi = leaf._gyro_bohm_chi(rho, Te, Ti, ne, q, a=0.58, B0=1.0)
    assert chi.shape == rho.shape
    assert np.all(chi >= leaf._CHI_FLOOR)
    with pytest.raises(ValueError, match="q must be positive"):
        leaf._gyro_bohm_chi(rho, Te, Ti, ne, np.zeros_like(q), a=0.58, B0=1.0)


def test_diffusion_step_reduces_peaked_axis_and_fail_closed() -> None:
    """Explicit diffusion cools a peaked profile; invalid grids fail closed."""
    rho = np.linspace(0.0, 1.0, 30)
    T = 5.0 * (1.0 - rho**2) + 0.5
    chi = 0.5 * np.ones_like(rho)
    ne = 5.0 * np.ones_like(rho)
    S = np.zeros_like(rho)
    T_new = leaf._diffusion_step(T, rho, chi, ne, S, dt=1e-4, a=0.58)
    assert T_new.shape == T.shape
    assert float(T_new[0]) <= float(T[0]) + 1e-12
    assert float(T_new[-1]) == pytest.approx(float(T[-1]), abs=1e-12)
    with pytest.raises(ValueError, match="rho"):
        leaf._diffusion_step(T, rho[::-1], chi, ne, S, dt=1e-4, a=0.58)
    with pytest.raises(ValueError, match="dt"):
        leaf._diffusion_step(T, rho, chi, ne, S, dt=-0.01, a=0.58)
