# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Volt-second bootstrap profile proxy.

"""Bootstrap-current proxy from ordered pressure profiles."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from scpn_control.control.volt_second_core import _finite_profile, _finite_scalar, _strict_rho


class BootstrapCurrentEstimate:
    """Bootstrap-current proxy from pressure-gradient profiles."""

    @staticmethod
    def from_profiles(
        ne: NDArray[np.float64],
        Te: NDArray[np.float64],
        Ti: NDArray[np.float64],
        q: NDArray[np.float64],
        rho: NDArray[np.float64],
        R0: float,
        a: float,
    ) -> float:
        """Simplified bootstrap current proxy: I_bs ~ ε^{1/2} · ∫ dp/dr dr.

        Full neoclassical expression in Wesson 2011, Ch. 4.9.
        ε = a/R₀ is the inverse aspect ratio.
        """
        ne = _finite_profile("ne", ne, positive=True)
        Te = _finite_profile("Te", Te, positive=True)
        Ti = _finite_profile("Ti", Ti, positive=True)
        q = _finite_profile("q", q, positive=True)
        rho = _strict_rho(rho)
        if not (len(ne) == len(Te) == len(Ti) == len(q) == len(rho)):
            raise ValueError("ne, Te, Ti, q, and rho profiles must have the same length")
        R0 = _finite_scalar("R0", R0, positive=True)
        a = _finite_scalar("a", a, positive=True)
        if a >= R0:
            raise ValueError("a must be smaller than R0 for tokamak ordering")
        epsilon = a / R0
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            p = 2.0 * ne * 1e19 * Te * 1e3 * 1.6e-19
        if not np.all(np.isfinite(p)):
            raise ValueError("pressure profile must be finite")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            grad_p = np.gradient(p, rho)
        if not np.all(np.isfinite(grad_p)):
            raise ValueError("pressure gradient must be finite")

        # J_bs ~ −ε^{1/2} / B_pol · dp/dr  (Wesson 2011, Eq. 4.9.4, rough scaling)
        with np.errstate(over="ignore", invalid="ignore"):
            J_bs_integral = float(np.sum(-grad_p * math.sqrt(epsilon)) * 1e-5)
        J_bs_integral = _finite_scalar("bootstrap proxy integral", J_bs_integral)

        I_bs_MA = max(0.0, J_bs_integral * 0.1)
        return _finite_scalar("bootstrap current", I_bs_MA, nonnegative=True)
