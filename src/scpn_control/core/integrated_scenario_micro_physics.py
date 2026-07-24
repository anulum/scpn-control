# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Integrated scenario transport micro-physics helpers

"""Spitzer resistivity, gyro-Bohm diffusivity, and thermal diffusion step.

This leaf owns the pure transport micro-physics helpers used by the
integrated scenario Strang operator split (CTL-G07 R5-S3). Coupling audit
lives in :mod:`integrated_scenario_coupling_audit`; the simulator
orchestration remains on :class:`IntegratedScenarioSimulator`.
"""

from __future__ import annotations

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.core.integrated_scenario_presets import _finite_scalar, _profile_array

# Spitzer / gyro-Bohm constants (local to this leaf)
_E_CHARGE: float = 1.602176634e-19  # C, CODATA 2018
_LN_LAMBDA: float = 17.0  # Wesson 2011, Ch. 14
_SPITZER_COEFF: float = 1.65e-9  # Spitzer 1962
_C_GB: float = 0.1  # ITPA Transport DB, Nucl. Fusion 39, 2175 (1999)
_CHI_FLOOR: float = 0.01  # m^2/s diffusivity floor


def _spitzer_resistivity(Te_keV: AnyFloatArray, Z_eff: float) -> FloatArray:
    """Spitzer parallel resistivity profile.

    η = SPITZER_COEFF * Z_eff * ln_Λ / T_e^1.5

    Spitzer 1962, "Physics of Fully Ionized Gases", Interscience, Ch. 5.
    Returns Ω·m; T_e in keV.
    """
    Te_keV = np.asarray(Te_keV, dtype=float)
    if not np.all(np.isfinite(Te_keV)) or np.any(Te_keV < 0.0):
        raise ValueError("Te_keV must contain only finite non-negative values")
    Z_eff = _finite_scalar("Z_eff", Z_eff, positive=True)
    return np.asarray(_SPITZER_COEFF * Z_eff * _LN_LAMBDA / np.maximum(Te_keV, 0.01) ** 1.5)


def _gyro_bohm_chi(
    rho: AnyFloatArray,
    Te: AnyFloatArray,
    Ti: AnyFloatArray,
    ne: AnyFloatArray,
    q: AnyFloatArray,
    a: float,
    B0: float,
) -> FloatArray:
    """Gyro-Bohm anomalous electron/ion thermal diffusivity.

    chi_gB = c_gB * rho_i * v_ti * (rho_i / a)^2 * q^2

    ITPA Transport DB, Nucl. Fusion 39, 2175 (1999), Eq. 1.
    c_gB = 0.1 calibrated to L-mode database.
    """
    rho = _profile_array("rho", rho, np.shape(rho))
    Te = _profile_array("Te", Te, rho.shape, nonnegative=True)
    Ti = _profile_array("Ti", Ti, rho.shape, nonnegative=True)
    ne = _profile_array("ne", ne, rho.shape, nonnegative=True)
    q = _profile_array("q", q, rho.shape)
    if np.any(q <= 0.0):
        raise ValueError("q must be positive everywhere")
    a = _finite_scalar("a", a, positive=True)
    B0 = _finite_scalar("B0", B0, positive=True)
    m_p: float = 1.67262192369e-27  # kg, CODATA 2018
    m_i = 2.0 * m_p  # deuterium
    T_i_J = np.maximum(Ti, 0.01) * 1.602176634e-16  # keV → J
    v_ti = np.sqrt(2.0 * T_i_J / m_i)
    rho_i = m_i * v_ti / (_E_CHARGE * B0)

    # Wesson 2011, Ch. 7 — chi_gB proportional to rho_i * v_ti * (rho_i/a)^2
    chi = _C_GB * rho_i * v_ti * (rho_i / a) ** 2 * np.maximum(q, 0.5) ** 2
    return np.asarray(np.maximum(chi, _CHI_FLOOR))


def _diffusion_step(
    T: AnyFloatArray,
    rho: AnyFloatArray,
    chi: AnyFloatArray,
    ne: AnyFloatArray,
    S: AnyFloatArray,
    dt: float,
    a: float,
) -> FloatArray:
    """Explicit cylindrical thermal diffusion step.

    Solves one half-step of:

        (3/2) n dT/dt = (1/r) d/dr [ r chi n dT/dr ] + S

    where r = rho * a.  Wesson 2011, Ch. 14, Eq. (14.5.1).
    Jardin 2010, "Computational Methods in Plasma Physics", Ch. 7.

    Uses forward Euler for simplicity; dt must satisfy the parabolic CFL:
        dt <= drho^2 / (2 * chi_max)
    The caller (step()) enforces sub-stepping when needed.
    """
    rho = _profile_array("rho", rho, np.shape(rho))
    if rho.ndim != 1 or rho.size < 2 or np.any(np.diff(rho) <= 0.0):
        raise ValueError("rho must be a strictly increasing one-dimensional grid")
    T = _profile_array("T", T, rho.shape, nonnegative=True)
    chi = _profile_array("chi", chi, rho.shape, nonnegative=True)
    ne = _profile_array("ne", ne, rho.shape, nonnegative=True)
    S = _profile_array("S", S, rho.shape)
    dt = _finite_scalar("dt", dt, nonnegative=True)
    a = _finite_scalar("a", a, positive=True)
    nr = len(rho)
    drho = rho[1] - rho[0]
    dr = drho * a

    dT_dt = np.zeros(nr)
    n_si = np.maximum(ne, 0.01) * 1e19  # 10^19 m^-3 → m^-3

    for i in range(1, nr - 1):
        r = rho[i] * a
        # Wesson 2011, Ch. 14 — flux: F = -chi * n * dT/dr
        # (1/r) d/dr(r F) expanded in finite differences:
        r_p = 0.5 * (rho[i] + rho[i + 1]) * a
        r_m = 0.5 * (rho[i] + rho[i - 1]) * a
        chi_p = 0.5 * (chi[i] + chi[i + 1])
        chi_m = 0.5 * (chi[i] + chi[i - 1])
        n_p = 0.5 * (n_si[i] + n_si[i + 1])
        n_m = 0.5 * (n_si[i] + n_si[i - 1])

        flux_p = r_p * chi_p * n_p * (T[i + 1] - T[i]) / dr
        flux_m = r_m * chi_m * n_m * (T[i] - T[i - 1]) / dr

        dT_dt[i] = (flux_p - flux_m) / (r * dr * n_si[i]) + S[i] / n_si[i]

    # Axis: symmetry → dT/dr = 0 at r = 0
    dT_dt[0] = dT_dt[1]
    # Edge: fixed-temperature BC (edge radiates away, T stays low)
    dT_dt[-1] = 0.0

    T_new = T + (2.0 / 3.0) * dt * dT_dt
    # Clamp to physical floor: 0.01 keV
    return np.maximum(T_new, 0.01)
