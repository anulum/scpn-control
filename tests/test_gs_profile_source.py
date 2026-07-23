# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for GS profile / plasma source

"""Drive production mTanh profile and plasma-source helpers on real surfaces."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scpn_control.core.gs_profile_source as prof
from scpn_control.core.fusion_kernel import FusionKernel

_PED = {"ped_top": 0.7, "ped_width": 0.08, "ped_height": 2.0, "core_alpha": 0.5}


def _write_config(path: Path, *, mode: str = "l-mode", extra_profiles: dict | None = None) -> Path:
    profiles: dict = {"mode": mode}
    if mode in ("h-mode", "H-mode", "hmode"):
        profiles["p_prime"] = dict(_PED)
        profiles["ff_prime"] = {**_PED, "ped_height": 1.0, "core_alpha": 0.3}
    if extra_profiles:
        profiles.update(extra_profiles)
    raw = {
        "reactor_name": "Profile-Source-Test",
        "dimensions": {"R_min": 1.0, "R_max": 3.0, "Z_min": -1.5, "Z_max": 1.5},
        "grid_resolution": [13, 13],
        "physics": {
            "plasma_current_target": 1.0,
            "vacuum_permeability": 1.0,
            "profiles": profiles,
        },
        "solver": {"boundary_variant": "fixed", "solver_method": "sor", "sor_omega": 1.5},
        "coils": [{"r": 3.2, "z": 0.0, "current": 1.0e5, "turns": 1}],
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_mtanh_profile_zero_outside_plasma_and_finite_inside() -> None:
    """Production mtanh is zero outside the plasma and finite inside."""
    psi = np.array([-0.1, 0.0, 0.3, 0.7, 0.99, 1.0, 1.2])
    out = prof.mtanh_profile(psi, _PED)
    assert out[0] == 0.0
    assert out[5] == 0.0
    assert out[6] == 0.0
    assert np.all(np.isfinite(out[1:5]))
    assert out[2] > 0.0


def test_mtanh_derivative_matches_finite_difference() -> None:
    """Production mtanh slope matches a central finite-difference check."""
    psi = np.linspace(0.05, 0.95, 21)
    analytic = prof.mtanh_profile_derivative(psi, _PED)
    eps = 1.0e-6
    fd = (prof.mtanh_profile(psi + eps, _PED) - prof.mtanh_profile(psi - eps, _PED)) / (2.0 * eps)
    np.testing.assert_allclose(analytic, fd, rtol=2.0e-4, atol=1.0e-6)


def test_normalised_flux_denominator_fail_closed() -> None:
    """Degenerate axis/boundary flux fails closed with ValueError."""
    assert prof.normalised_flux_denominator(0.0, 1.0) == 1.0
    with pytest.raises(ValueError, match="degenerate equilibrium"):
        prof.normalised_flux_denominator(0.5, 0.5)
    with pytest.raises(ValueError, match="degenerate equilibrium"):
        FusionKernel._normalised_flux_denominator(1.0, 1.0)


def test_owner_mtanh_wrappers_match_leaf(tmp_path: Path) -> None:
    """FusionKernel mtanh wrappers are the production leaf functions."""
    kernel = FusionKernel(_write_config(tmp_path / "cfg.json", mode="h-mode"))
    psi = np.linspace(0.0, 0.95, kernel.NZ * kernel.NR).reshape(kernel.NZ, kernel.NR)
    leaf = prof.mtanh_profile(psi, kernel.ped_params_p)
    owner = kernel.mtanh_profile(psi, kernel.ped_params_p)
    np.testing.assert_allclose(owner, leaf, rtol=1.0e-14, atol=1.0e-14)
    leaf_d = prof.mtanh_profile_derivative(psi, kernel.ped_params_p)
    owner_d = kernel.mtanh_profile_derivative(psi, kernel.ped_params_p)
    np.testing.assert_allclose(owner_d, leaf_d, rtol=1.0e-14, atol=1.0e-14)


def test_owner_source_and_jacobian_match_leaf_lmode(tmp_path: Path) -> None:
    """Owner source and L-mode Jacobian match the pure leaf helpers."""
    kernel = FusionKernel(_write_config(tmp_path / "cfg_l.json", mode="l-mode"))
    kernel.Psi = np.linspace(0.0, 1.0, kernel.NZ * kernel.NR).reshape(kernel.NZ, kernel.NR)
    axis, boundary = 0.0, 1.0
    mu0 = 1.0
    i_target = float(kernel.cfg["physics"]["plasma_current_target"])
    leaf_j = prof.update_plasma_source_nonlinear(
        kernel.Psi,
        kernel.RR,
        kernel.dR,
        kernel.dZ,
        axis,
        boundary,
        mu0=mu0,
        I_target=i_target,
        profile_mode=kernel.profile_mode,
        ped_params_p=kernel.ped_params_p,
        ped_params_ff=kernel.ped_params_ff,
    )
    owner_j = kernel.update_plasma_source_nonlinear(axis, boundary)
    np.testing.assert_allclose(owner_j, leaf_j, rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose(kernel.J_phi, owner_j, rtol=1.0e-14, atol=1.0e-14)
    leaf_jac = prof.compute_profile_jacobian(
        kernel.Psi,
        kernel.RR,
        kernel.dR,
        kernel.dZ,
        axis,
        boundary,
        mu0,
        profile_mode=kernel.profile_mode,
        ped_params_p=kernel.ped_params_p,
        ped_params_ff=kernel.ped_params_ff,
        I_target=i_target,
    )
    owner_jac = kernel._compute_profile_jacobian(axis, boundary, mu0)
    np.testing.assert_allclose(owner_jac, leaf_jac, rtol=1.0e-14, atol=1.0e-14)
    assert np.all(np.isfinite(owner_j))
    assert np.all(np.isfinite(owner_jac))


def test_owner_source_and_jacobian_match_leaf_hmode(tmp_path: Path) -> None:
    """H-mode owner source/Jacobian are the production leaf path."""
    kernel = FusionKernel(_write_config(tmp_path / "cfg_h.json", mode="h-mode"))
    kernel.Psi = np.linspace(0.0, 1.0, kernel.NZ * kernel.NR).reshape(kernel.NZ, kernel.NR)
    axis, boundary = 0.0, 1.0
    mu0 = 1.0
    i_target = float(kernel.cfg["physics"]["plasma_current_target"])
    leaf_j = prof.update_plasma_source_nonlinear(
        kernel.Psi,
        kernel.RR,
        kernel.dR,
        kernel.dZ,
        axis,
        boundary,
        mu0=mu0,
        I_target=i_target,
        profile_mode=kernel.profile_mode,
        ped_params_p=kernel.ped_params_p,
        ped_params_ff=kernel.ped_params_ff,
    )
    owner_j = kernel.update_plasma_source_nonlinear(axis, boundary)
    np.testing.assert_allclose(owner_j, leaf_j, rtol=1.0e-14, atol=1.0e-14)
    leaf_jac = prof.compute_profile_jacobian(
        kernel.Psi,
        kernel.RR,
        kernel.dR,
        kernel.dZ,
        axis,
        boundary,
        mu0,
        profile_mode=kernel.profile_mode,
        ped_params_p=kernel.ped_params_p,
        ped_params_ff=kernel.ped_params_ff,
        I_target=i_target,
    )
    owner_jac = kernel._compute_profile_jacobian(axis, boundary, mu0)
    np.testing.assert_allclose(owner_jac, leaf_jac, rtol=1.0e-14, atol=1.0e-14)
    assert float(np.std(owner_jac)) > 1.0e-9


def test_external_profile_mode_requires_tables() -> None:
    """External profile mode fails closed when tables are missing."""
    psi = np.ones((5, 5)) * 0.5
    rr = np.full((5, 5), 2.0)
    with pytest.raises(ValueError, match="external profile mode requires"):
        prof.update_plasma_source_nonlinear(
            psi,
            rr,
            0.1,
            0.1,
            0.0,
            1.0,
            mu0=1.0,
            I_target=1.0,
            profile_mode="external",
            ped_params_p=_PED,
            ped_params_ff=_PED,
            ext_psi_grid=None,
            ext_pprime=None,
            ext_ffprime=None,
        )


def test_external_profile_mode_interpolates(tmp_path: Path) -> None:
    """External tables drive a finite renormalised J_phi on the owner path."""
    psi_grid = np.linspace(0.0, 1.0, 11)
    pprime = 1.0 - psi_grid
    ffprime = 0.5 * (1.0 - psi_grid)
    path = _write_config(
        tmp_path / "cfg_ext.json",
        mode="external",
        extra_profiles={
            "psi_grid": psi_grid.tolist(),
            "pprime_values": pprime.tolist(),
            "ffprime_values": ffprime.tolist(),
        },
    )
    kernel = FusionKernel(path)
    kernel.Psi = np.linspace(0.0, 0.9, kernel.NZ * kernel.NR).reshape(kernel.NZ, kernel.NR)
    j_phi = kernel.update_plasma_source_nonlinear(0.0, 1.0)
    assert np.all(np.isfinite(j_phi))
    assert float(np.sum(np.abs(j_phi))) > 0.0
