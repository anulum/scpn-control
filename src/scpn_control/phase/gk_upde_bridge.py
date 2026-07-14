# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK → UPDE Phase Dynamics Bridge
"""
Bridge between gyrokinetic transport fluxes and the 8-layer UPDE Kuramoto phase dynamics system.

Maps GK-computed growth rates and diffusivities into adaptive K_nm
coupling modulation for layers P0 (microturbulence), P1 (zonal flows),
P4 (transport barrier), and P5 (current profile).

Reference layer mappings:
  P0 ← max(gamma_ITG, gamma_TEM): turbulence drive
  P1 ← chi_e suppression ratio: zonal flow damping of transport
  P4 ← chi_i pedestal / chi_i core: transport barrier strength
  P5 ← bootstrap current contribution (via pressure gradient)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scpn_control.core.gk_interface import GKOutput

logger = logging.getLogger(__name__)

# Minimum layer count for the P0-P5 GK->UPDE mapping. adaptive_knm indexes
# layers up to P4/P5, so a smaller coupling matrix cannot be modulated.
_MIN_MODULATED_LAYERS = 6


@dataclass(frozen=True)
class GKCouplingGains:
    """Calibration gains for GK-driven ``K_nm`` coupling modulation.

    These are dimensionless SCPN phase-coupling calibration gains — multipliers
    on the baseline coupling that set how strongly each gyrokinetic transport
    channel modulates its UPDE layer pair. They are model tuning parameters, not
    values imported from external literature; they are exposed here so callers
    can recalibrate the bridge without editing the kernel.

    Parameters
    ----------
    p0_p1_turbulence : float, default 0.5
        Gain on the P0-P1 (microturbulence to zonal-flow) coupling driven by the
        dominant growth rate through ``tanh(max_gamma / gamma_ref)``.
    p1_p4_transport : float, default 0.3
        Gain on the P1-P4 (zonal-flow to transport-barrier) coupling driven by
        the normalised electron heat diffusivity ``chi_e / chi_ref``.
    p1_p4_chi_clip_max : float, default 2.0
        Upper clip on the normalised ``chi_e / chi_ref`` ratio, bounding the
        P1-P4 modulation; the lower clip is fixed at 0.
    p3_p4_pedestal : float, default 0.4
        Gain on the P3-P4 (sawtooth/ELM to transport-barrier) coupling driven by
        the pedestal-to-core ion-diffusivity ratio.
    diffusivity_floor : float, default 1e-10
        Lower floor [m^2/s] applied to mean diffusivities before ratios are
        formed, preventing division by zero at vanishing transport.
    """

    p0_p1_turbulence: float = 0.5
    p1_p4_transport: float = 0.3
    p1_p4_chi_clip_max: float = 2.0
    p3_p4_pedestal: float = 0.4
    diffusivity_floor: float = 1e-10

    def __post_init__(self) -> None:
        """Validate that all gains are finite and physically admissible."""
        for name in ("p0_p1_turbulence", "p1_p4_transport", "p3_p4_pedestal"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not np.isfinite(self.p1_p4_chi_clip_max) or self.p1_p4_chi_clip_max <= 0.0:
            raise ValueError("p1_p4_chi_clip_max must be finite and positive")
        if not np.isfinite(self.diffusivity_floor) or self.diffusivity_floor <= 0.0:
            raise ValueError("diffusivity_floor must be finite and positive")


def _validate_coupling_matrix(K_base: NDArray[np.float64]) -> None:
    if K_base.ndim != 2 or K_base.shape[0] != K_base.shape[1]:
        raise ValueError("K_base must be a square matrix")
    if not np.all(np.isfinite(K_base)):
        raise ValueError("K_base must contain only finite values")


def _positive_growth_drive(gamma: NDArray[np.float64]) -> float:
    if gamma.size == 0:
        return 0.0
    if not np.all(np.isfinite(gamma)):
        raise ValueError("gamma must contain only finite values")
    return max(float(np.max(gamma)), 0.0)


def _validate_nonnegative_finite(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _validate_positive_finite(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def adaptive_knm(
    K_base: NDArray[np.float64],
    gk_output: GKOutput,
    chi_i_profile: NDArray[np.float64] | None = None,
    gamma_ref: float = 0.2,
    chi_ref: float = 1.0,
    gains: GKCouplingGains | None = None,
) -> NDArray[np.float64]:
    """Modulate K_nm based on GK fluxes.

    Parameters
    ----------
    K_base : array, shape (L, L)
        Baseline coupling matrix from build_knm_plasma().
    gk_output : GKOutput
        GK solver output (growth rates, fluxes).
    chi_i_profile : array or None
        Full chi_i(rho) profile for pedestal ratio calculation.
    gamma_ref : float
        Reference growth rate for tanh scaling [c_s/a].
    chi_ref : float
        Reference chi_e for transport modulation [m^2/s].
    gains : GKCouplingGains or None, optional
        Calibration gains for the coupling modulation. Defaults to
        :class:`GKCouplingGains` with the standard SCPN values.

    Returns
    -------
    numpy.ndarray, shape (L, L)
        The modulated coupling matrix. A copy of ``K_base`` is returned
        unmodulated when ``L < 6``: the P0-P5 layer mapping needs at least six
        layers, and a warning is logged in that case rather than silently
        passing the matrix through.

    Raises
    ------
    ValueError
        If ``K_base`` is not square or not finite, ``chi_e`` is negative or
        non-finite, ``gamma_ref`` or ``chi_ref`` is non-positive, or
        ``chi_i_profile`` contains non-finite or negative values.
    """
    gains = gains if gains is not None else GKCouplingGains()
    _validate_coupling_matrix(K_base)
    _validate_nonnegative_finite(gk_output.chi_e, "chi_e")
    _validate_positive_finite(gamma_ref, "gamma_ref")
    _validate_positive_finite(chi_ref, "chi_ref")
    max_gamma = _positive_growth_drive(gk_output.gamma)
    if chi_i_profile is not None and (not np.all(np.isfinite(chi_i_profile)) or np.any(chi_i_profile < 0.0)):
        raise ValueError("chi_i_profile must contain finite non-negative values")
    K = K_base.copy()
    L = K.shape[0]
    if L < _MIN_MODULATED_LAYERS:
        logger.warning(
            "adaptive_knm received a %d-layer coupling matrix (< %d); the GK->UPDE "
            "P0-P5 layer mapping needs at least %d layers, so K_base is returned unmodulated",
            L,
            _MIN_MODULATED_LAYERS,
            _MIN_MODULATED_LAYERS,
        )
        return K

    # P0<->P1: microturbulence <-> zonal flows
    K[0, 1] = K_base[0, 1] * (1.0 + gains.p0_p1_turbulence * np.tanh(max_gamma / gamma_ref))
    K[1, 0] = K[0, 1]

    # P1<->P4: zonal flow <-> transport barrier
    mean_chi_e = max(gk_output.chi_e, gains.diffusivity_floor)
    K[1, 4] = K_base[1, 4] * (
        1.0 + gains.p1_p4_transport * np.clip(mean_chi_e / chi_ref, 0.0, gains.p1_p4_chi_clip_max)
    )
    K[4, 1] = K[1, 4]

    # P3<->P4: sawtooth/ELM <-> transport barrier (pedestal ratio)
    if chi_i_profile is not None and len(chi_i_profile) > 5:
        chi_core = max(float(np.mean(chi_i_profile[: len(chi_i_profile) // 3])), gains.diffusivity_floor)
        chi_ped = max(float(np.mean(chi_i_profile[-len(chi_i_profile) // 5 :])), gains.diffusivity_floor)
        K[3, 4] = K_base[3, 4] * (1.0 + gains.p3_p4_pedestal * (chi_ped / chi_core - 1.0))
        K[4, 3] = K[3, 4]

    return K


def gk_natural_frequencies(
    omega_base: NDArray[np.float64],
    gk_output: GKOutput,
    gamma_scale: float = 0.1,
) -> NDArray[np.float64]:
    """Adjust layer-0 natural frequency based on GK growth rate.

    The turbulence layer's effective frequency increases with the
    dominant instability growth rate.
    """
    if not np.all(np.isfinite(omega_base)):
        raise ValueError("omega_base must contain only finite values")
    if not np.isfinite(gamma_scale) or gamma_scale < 0.0:
        raise ValueError("gamma_scale must be finite and non-negative")
    omega = omega_base.copy()
    max_gamma = _positive_growth_drive(gk_output.gamma)
    omega[0] += gamma_scale * max_gamma
    return omega
