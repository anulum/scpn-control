# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GS profile functions and plasma source

"""Pure mTanh pedestal profiles and nonlinear plasma source construction.

This leaf owns modified-tanh profile evaluation, profile derivatives for
Newton linearisation, normalised-flux guards, and J_phi source updates used
by the CONTROL Grad-Shafranov solver. The CONTROL
:class:`~scpn_control.core.fusion_kernel.FusionKernel` product surface remains
first-class under dual-home C and keeps thin wrappers that supply runtime
state (mesh, config, external profiles).
"""

from __future__ import annotations

import numpy as np

from scpn_control._typing import FloatArray


def mtanh_profile(psi_norm: FloatArray, params: dict[str, float]) -> FloatArray:
    """Evaluate a modified-tanh pedestal profile (vectorised).

    Parameters
    ----------
    psi_norm :
        Normalised poloidal flux (0 at axis, 1 at separatrix).
    params :
        Profile shape parameters with keys ``ped_top``, ``ped_width``,
        ``ped_height``, ``core_alpha``.

    Returns
    -------
    FloatArray
        Profile value; zero outside the plasma region.
    """
    result = np.zeros_like(psi_norm)
    mask = (psi_norm >= 0) & (psi_norm < 1.0)
    x = psi_norm[mask]

    y = np.clip((params["ped_top"] - x) / params["ped_width"], -20, 20)
    pedestal = 0.5 * params["ped_height"] * (1.0 + np.tanh(y))

    core = np.where(
        x < params["ped_top"],
        np.maximum(0.0, 1.0 - (x / params["ped_top"]) ** 2),
        0.0,
    )

    result[mask] = pedestal + params["core_alpha"] * core
    return result


def mtanh_profile_derivative(psi_norm: FloatArray, params: dict[str, float]) -> FloatArray:
    """Evaluate ``d(mtanh_profile)/dpsi_norm`` for H-mode Newton linearisation."""
    result = np.zeros_like(psi_norm)
    mask = (psi_norm >= 0) & (psi_norm < 1.0)
    x = psi_norm[mask]
    ped_top = params["ped_top"]
    ped_width = params["ped_width"]
    raw_y = (ped_top - x) / ped_width
    y = np.clip(raw_y, -20, 20)
    active_pedestal = (raw_y >= -20.0) & (raw_y <= 20.0)
    sech2 = 1.0 - np.tanh(y) ** 2
    pedestal_slope = np.where(
        active_pedestal,
        -0.5 * params["ped_height"] * sech2 / ped_width,
        0.0,
    )
    core_slope = np.where(
        x < ped_top,
        -2.0 * params["core_alpha"] * x / (ped_top * ped_top),
        0.0,
    )
    result[mask] = pedestal_slope + core_slope
    return result


def normalised_flux_denominator(Psi_axis: float, Psi_boundary: float) -> float:
    """Return (Psi_boundary - Psi_axis) or fail closed on a degenerate equilibrium."""
    denom = float(Psi_boundary) - float(Psi_axis)
    if not np.isfinite(denom) or abs(denom) < 1e-9:
        raise ValueError("degenerate equilibrium: separatrix flux is indistinguishable from magnetic-axis flux")
    return denom


def update_plasma_source_nonlinear(
    Psi: FloatArray,
    RR: FloatArray,
    dR: float,
    dZ: float,
    Psi_axis: float,
    Psi_boundary: float,
    *,
    mu0: float,
    I_target: float,
    profile_mode: str,
    ped_params_p: dict[str, float],
    ped_params_ff: dict[str, float],
    ext_psi_grid: FloatArray | None = None,
    ext_pprime: FloatArray | None = None,
    ext_ffprime: FloatArray | None = None,
) -> FloatArray:
    """Compute toroidal current density J_phi from the GS source term.

    Uses ``J_phi = R p'(psi) + FF'(psi) / (mu0 R)`` with L-mode (linear),
    H-mode (mtanh), or external profile tables, then renormalises to match
    the target plasma current.
    """
    denom = normalised_flux_denominator(Psi_axis, Psi_boundary)

    Psi_norm = (Psi - Psi_axis) / denom
    mask_plasma = (Psi_norm >= 0) & (Psi_norm < 1.0)

    if profile_mode == "external":
        if ext_psi_grid is None or ext_pprime is None or ext_ffprime is None:
            raise ValueError("external profile mode requires ext_psi_grid, ext_pprime, and ext_ffprime")
        psi_flat = np.clip(Psi_norm.ravel(), 0.0, 1.0)
        p_profile = np.interp(psi_flat, ext_psi_grid, ext_pprime).reshape(Psi_norm.shape)
        ff_profile = np.interp(psi_flat, ext_psi_grid, ext_ffprime).reshape(Psi_norm.shape)
        p_profile[~mask_plasma] = 0.0
        ff_profile[~mask_plasma] = 0.0
    elif profile_mode in ("h-mode", "H-mode", "hmode"):
        p_profile = mtanh_profile(Psi_norm, ped_params_p)
        ff_profile = mtanh_profile(Psi_norm, ped_params_ff)
    else:
        p_profile = np.zeros_like(Psi)
        p_profile[mask_plasma] = 1.0 - Psi_norm[mask_plasma]
        ff_profile = p_profile.copy()

    J_p = RR * p_profile
    J_f = (1.0 / (mu0 * RR)) * ff_profile

    J_raw = J_p + J_f

    I_current = float(np.sum(J_raw)) * dR * dZ

    if abs(I_current) > 1e-9:
        return J_raw * (I_target / I_current)
    return np.zeros_like(Psi)


def compute_profile_jacobian(
    Psi: FloatArray,
    RR: FloatArray,
    dR: float,
    dZ: float,
    Psi_axis: float,
    Psi_boundary: float,
    mu0: float,
    *,
    profile_mode: str,
    ped_params_p: dict[str, float],
    ped_params_ff: dict[str, float],
    I_target: float,
) -> FloatArray:
    """Compute dJ_phi/dpsi as a 2D diagonal scaling field."""
    denom = normalised_flux_denominator(Psi_axis, Psi_boundary)

    Psi_norm = (Psi - Psi_axis) / denom
    mask_plasma = (Psi_norm >= 0) & (Psi_norm < 1.0)

    dJ_dpsi = np.zeros_like(Psi)
    if profile_mode in ("h-mode", "H-mode", "hmode"):
        d_p = mtanh_profile_derivative(Psi_norm, ped_params_p)
        d_ff = mtanh_profile_derivative(Psi_norm, ped_params_ff)
        dJ_dpsi[mask_plasma] = (
            RR[mask_plasma] * d_p[mask_plasma] + d_ff[mask_plasma] / (mu0 * RR[mask_plasma])
        ) / denom
    else:
        # For the linear L-mode profile: Source = -mu0 * R * J_phi
        # J_phi = c * (1 - psi_norm) * R  =>  dJ_phi/dpsi_norm = -c * R
        # dJ_phi/dpsi = dJ_phi/dpsi_norm * dpsi_norm/dpsi = -c * R / denom
        # I = ∫ J_phi dA ≈ c · Σ_plasma((1-ψ_norm)·R)·ΔR·ΔZ  (midpoint quadrature)
        s = float(np.sum(np.where(mask_plasma, (1 - Psi_norm) * RR, 0.0))) * dR * dZ
        c = I_target / max(abs(s), 1e-9)

        dJ_dpsi[mask_plasma] = -c * RR[mask_plasma] / denom

    return dJ_dpsi


# Historical private names for FusionKernel wrappers.
_mtanh_profile = mtanh_profile
_mtanh_profile_derivative = mtanh_profile_derivative
_normalised_flux_denominator = normalised_flux_denominator
_update_plasma_source_nonlinear = update_plasma_source_nonlinear
_compute_profile_jacobian = compute_profile_jacobian
