# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — ELM Pedestal and RMP Physics
"""ELM crash, recovery, suppression, and pedestal-coupling model utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scpn_control._typing import AnyFloatArray

# ELM energy loss fraction bounds: ΔW_ELM / W_ped ≈ 0.04–0.15 for Type I.
# Loarte et al. 2003, PPCF 45, 1549, Fig. 12.
ELM_ENERGY_FRACTION_MIN = 0.04
ELM_ENERGY_FRACTION_MAX = 0.15

# Type III ELM: smaller, higher frequency, triggered at lower pedestal pressure than Type I.
# Zohm 1996, PPCF 38, 105.
ELM_TYPE3_FRACTION = 0.02  # Typical Type III ΔW/W_ped (sub-range of Type I bounds)


class PeelingBallooningBoundary:
    """
    Stability against peeling-ballooning modes.

    ELM onset: α > α_crit (ballooning) AND j_edge > j_peel (peeling).
    Snyder et al. 2002, Phys. Plasmas 9, 2037, Eq. 8.
    """

    def __init__(self, q95: float, kappa: float, delta: float, a: float, R0: float):
        if q95 <= 0.0:
            raise ValueError("q95 must be positive")
        if kappa <= 0.0:
            raise ValueError("kappa must be positive")
        if a <= 0.0 or R0 <= 0.0:
            raise ValueError("a and R0 must be positive")
        self.q95 = q95
        self.kappa = kappa
        self.delta = delta
        self.a = a
        self.R0 = R0

    def peeling_limit(self, j_edge: float, n_mode: int = 10) -> float:
        """
        Critical edge current density j_peel.

        Geometry- and mode-aware peeling threshold proxy:

            j_peel,crit ∝ (R0/a) * F(κ, δ) / (q95 * sqrt(n_mode))

        where F(κ, δ) captures edge-shaping stabilisation and ``n_mode``
        captures reduced stability margin for higher-n peeling harmonics.
        This keeps the Snyder-type coupled PB scaling structure while
        avoiding a single-parameter ``1/q95`` closure.
        """
        if n_mode <= 0:
            raise ValueError("n_mode must be positive")
        q95 = max(self.q95, 1e-3)
        n_eff = max(float(n_mode), 1.0)
        aspect = max(self.R0 / max(self.a, 1e-6), 1.0)
        shape_factor = (1.0 + self.kappa**2) / (2.0 * max(self.kappa, 1e-6))
        triangularity_factor = 1.0 + self.delta**2
        j_ref = 1.0e6
        j_crit = j_ref * aspect * shape_factor * triangularity_factor / (q95 * np.sqrt(n_eff))
        return float(j_crit)

    def ballooning_limit(self, s_edge: float) -> float:
        """
        Critical normalised pressure gradient α_crit.

        Shaping factor (1 + κ²(1 + 2δ²)) from Sauter et al. 1999, Phys. Plasmas 6, 2834.
        """
        if s_edge < 0.0:
            raise ValueError("s_edge must be non-negative")
        alpha_crit = 0.5 * max(s_edge, 0.1) * (1.0 + self.kappa**2 * (1.0 + 2.0 * self.delta**2))
        return float(alpha_crit)

    def is_unstable(self, alpha_edge: float, j_edge: float, s_edge: float) -> bool:
        """
        Evaluate the combined peeling-ballooning criterion (elliptical boundary).

        Snyder et al. 2002, Phys. Plasmas 9, 2037, Eq. 8.
        """
        j_crit = self.peeling_limit(j_edge)
        a_crit = self.ballooning_limit(s_edge)

        j_norm = max(0.0, j_edge / j_crit)
        a_norm = max(0.0, alpha_edge / a_crit)

        return (j_norm**2 + a_norm**2) > 1.0

    def stability_margin(self, alpha_edge: float, j_edge: float, s_edge: float) -> float:
        """Distance to boundary (positive = stable)."""
        j_crit = self.peeling_limit(j_edge)
        a_crit = self.ballooning_limit(s_edge)

        j_norm = max(0.0, j_edge / j_crit)
        a_norm = max(0.0, alpha_edge / a_crit)

        return float(1.0 - np.sqrt(j_norm**2 + a_norm**2))


@dataclass
class ELMCrashResult:
    """Outcome of a single ELM crash applied to the pedestal.

    Attributes
    ----------
    delta_W_MJ
        Energy expelled by the crash in MJ.
    T_ped_post
        Post-crash pedestal temperature (same unit as the input).
    n_ped_post
        Post-crash pedestal density (same unit as the input).
    peak_heat_flux_MW_m2
        Peak divertor heat flux during the crash in MW/m².
    duration_ms
        Crash duration in milliseconds.
    """

    delta_W_MJ: float
    T_ped_post: float
    n_ped_post: float
    peak_heat_flux_MW_m2: float
    duration_ms: float


class ELMCrashModel:
    """
    Applies a Type I ELM crash to pedestal profiles.

    ΔW_ELM / W_ped ≈ 0.04–0.15 (Type I).
    Loarte et al. 2003, PPCF 45, 1549, Fig. 12.
    """

    def __init__(self, f_elm_fraction: float = 0.08):
        if not (ELM_ENERGY_FRACTION_MIN <= f_elm_fraction <= ELM_ENERGY_FRACTION_MAX):
            raise ValueError(
                f"f_elm_fraction={f_elm_fraction} outside Type I range "
                f"[{ELM_ENERGY_FRACTION_MIN}, {ELM_ENERGY_FRACTION_MAX}] "
                "(Loarte et al. 2003, PPCF 45, 1549, Fig. 12)"
            )
        self.f_elm_fraction = f_elm_fraction

    def crash(self, T_ped: float, n_ped: float, W_ped: float, A_wet: float = 1.0) -> ELMCrashResult:
        """
        ΔW_ELM = f × W_ped; T and n drop by √(1 − f) (W ∝ n T).

        Loarte et al. 2003, PPCF 45, 1549, Fig. 12.
        """
        if T_ped <= 0.0:
            raise ValueError("T_ped must be positive")
        if n_ped <= 0.0:
            raise ValueError("n_ped must be positive")
        if W_ped <= 0.0:
            raise ValueError("W_ped must be positive")
        if A_wet <= 0.0:
            raise ValueError("A_wet must be positive")
        delta_W_MJ = self.f_elm_fraction * W_ped

        drop_factor = np.sqrt(1.0 - self.f_elm_fraction)

        T_ped_post = T_ped * drop_factor
        n_ped_post = n_ped * drop_factor

        duration_ms = 0.25

        peak_heat_flux = (delta_W_MJ / A_wet) / (duration_ms * 1e-3)

        return ELMCrashResult(delta_W_MJ, T_ped_post, n_ped_post, peak_heat_flux, duration_ms)

    def apply_to_profiles(
        self, rho: AnyFloatArray, Te: AnyFloatArray, ne: AnyFloatArray, rho_ped: float
    ) -> tuple[AnyFloatArray, AnyFloatArray]:
        """Drop the pedestal region of the Te and ne profiles by the crash factor.

        Outside the pedestal top (``rho >= rho_ped``) both profiles are scaled by
        ``√(1 − f_elm_fraction)``.

        Parameters
        ----------
        rho
            Sorted normalised-radius grid, shape ``(n,)``.
        Te
            Electron-temperature profile on ``rho``; positive.
        ne
            Electron-density profile on ``rho``; positive.
        rho_ped
            Pedestal-top normalised radius; must lie within the grid.

        Returns
        -------
        tuple of AnyFloatArray
            The post-crash ``(Te, ne)`` profiles.
        """
        if rho.ndim != 1 or Te.ndim != 1 or ne.ndim != 1:
            raise ValueError("rho, Te, and ne must be one-dimensional")
        if not (len(rho) == len(Te) == len(ne)):
            raise ValueError("rho, Te, and ne must have equal length")
        if len(rho) == 0:
            raise ValueError("profiles must not be empty")
        if np.any(np.diff(rho) < 0.0):
            raise ValueError("rho must be sorted")
        if np.any(Te <= 0.0) or np.any(ne <= 0.0):
            raise ValueError("Te and ne profiles must be positive")
        if rho_ped < rho[0] or rho_ped > rho[-1]:
            raise ValueError("rho_ped must lie within the rho grid")
        Te_new = Te.copy()
        ne_new = ne.copy()

        idx_ped = np.searchsorted(rho, rho_ped)

        drop_factor = np.sqrt(1.0 - self.f_elm_fraction)

        Te_new[idx_ped:] *= drop_factor
        ne_new[idx_ped:] *= drop_factor

        return Te_new, ne_new


class Type3ELMCrashModel(ELMCrashModel):
    """
    Type III ELM: smaller energy loss, higher frequency, triggered at lower pedestal pressure.

    Zohm 1996, PPCF 38, 105.
    """

    def __init__(self, f_elm_fraction: float = ELM_TYPE3_FRACTION):
        if not (0.0 < f_elm_fraction < ELM_ENERGY_FRACTION_MIN):
            raise ValueError("Type III f_elm_fraction must be positive and below the Type I lower bound")
        # Bypass Type I bound check — Type III fraction is below Type I minimum.
        self.f_elm_fraction = f_elm_fraction


@dataclass
class ELMEvent:
    """A single ELM crash event in a control cycle.

    Attributes
    ----------
    time
        Event time in seconds.
    delta_W_MJ
        Energy expelled by the crash in MJ.
    f_elm_Hz
        Instantaneous ELM frequency in Hz.
    crash_type
        ELM type label (e.g. ``"Type I"``).
    """

    time: float
    delta_W_MJ: float
    f_elm_Hz: float
    crash_type: str


class RMPSuppression:
    """Resonant Magnetic Perturbation effects on ELMs."""

    def __init__(self, n_coils: int = 3, I_rmp_kA: float = 90.0, n_toroidal: int = 3):
        if n_coils <= 0:
            raise ValueError("n_coils must be positive")
        if I_rmp_kA < 0.0:
            raise ValueError("I_rmp_kA must be non-negative")
        if n_toroidal <= 0:
            raise ValueError("n_toroidal must be positive")
        self.n_coils = n_coils
        self.I_rmp_kA = I_rmp_kA
        self.n_toroidal = n_toroidal

    def chirikov_parameter(
        self, q_profile: AnyFloatArray, rho: AnyFloatArray, delta_B_r: float, B0: float, R0: float
    ) -> float:
        """σ_Chir = Σ w_mn / dr_mn; w_mn = 4√(R0 q' δB_r / (n B0))."""
        if q_profile.ndim != 1 or rho.ndim != 1:
            raise ValueError("q_profile and rho must be one-dimensional")
        if len(q_profile) != len(rho) or len(q_profile) == 0:
            raise ValueError("q_profile and rho must have equal non-zero length")
        if np.any(q_profile <= 0.0):
            raise ValueError("q_profile values must be positive")
        if np.any(np.diff(rho) <= 0.0):
            raise ValueError("rho must be strictly increasing")
        if B0 <= 0.0 or R0 <= 0.0:
            raise ValueError("B0 and R0 must be positive")
        q_edge = q_profile[-1]

        if len(rho) > 1:
            dq_drho = (q_profile[-1] - q_profile[-2]) / (rho[-1] - rho[-2])
        else:
            dq_drho = 10.0

        if dq_drho <= 0.0:
            raise ValueError("edge magnetic shear must be positive for the Chirikov overlap model")

        shear = dq_drho * rho[-1] / q_edge

        if delta_B_r <= 0.0 or B0 <= 0.0 or self.n_toroidal == 0:
            return 0.0

        w_mn = 4.0 * np.sqrt(R0 * shear * delta_B_r / (self.n_toroidal * B0))
        dr_mn = 1.0 / (self.n_toroidal * max(dq_drho, 1e-3))

        return float(w_mn / dr_mn)

    def suppressed(self, sigma_chir: float) -> bool:
        """Return whether RMP suppresses ELMs (σ_Chirikov > 1).

        Parameters
        ----------
        sigma_chir
            Chirikov overlap parameter at the pedestal.

        Returns
        -------
        bool
            ``True`` when the islands overlap and ELMs are suppressed.
        """
        return sigma_chir > 1.0

    def pedestal_transport_enhancement(self, sigma_chir: float) -> float:
        """Pedestal transport enhancement factor from RMP stochasticity.

        Parameters
        ----------
        sigma_chir
            Chirikov overlap parameter.

        Returns
        -------
        float
            Multiplicative transport enhancement (1.0 below overlap).
        """
        alpha = 2.0
        return 1.0 + alpha * max(0.0, sigma_chir - 1.0)

    def density_pump_out(self, sigma_chir: float) -> float:
        """Fractional pedestal density pump-out under RMP.

        Parameters
        ----------
        sigma_chir
            Chirikov overlap parameter.

        Returns
        -------
        float
            Fractional density reduction (0.2 above overlap, 0.0 otherwise).
        """
        if sigma_chir > 1.0:
            return 0.2
        return 0.0


def elm_power_balance_frequency(P_SOL_MW: float, W_ped_MJ: float, f_elm_fraction: float) -> float:
    """
    f_ELM ∝ (P_loss − P_LH) / ΔW_ELM ≈ P_SOL / (f × W_ped).

    Leonard et al. 1999, J. Nucl. Mater. 266-269, 109.
    """
    if P_SOL_MW < 0.0:
        raise ValueError("P_SOL_MW must be non-negative")
    if W_ped_MJ <= 0.0 or f_elm_fraction <= 0.0:
        return 0.0
    return P_SOL_MW / (f_elm_fraction * W_ped_MJ)


class ELMCycler:
    """Tracks pedestal state and triggers ELM events."""

    def __init__(self, pb_boundary: PeelingBallooningBoundary, crash_model: ELMCrashModel):
        self.pb = pb_boundary
        self.crash_model = crash_model
        self.time = 0.0
        self.last_crash = 0.0

    def step(
        self, dt: float, alpha_edge: float, j_edge: float, s_edge: float, T_ped: float, n_ped: float, W_ped: float
    ) -> ELMEvent | None:
        """Advance the pedestal clock and trigger an ELM if peeling-ballooning unstable.

        Parameters
        ----------
        dt
            Time step in seconds; must be positive.
        alpha_edge
            Normalised edge pressure gradient; must be non-negative.
        j_edge
            Edge current density; must be non-negative.
        s_edge
            Edge magnetic shear; must be non-negative.
        T_ped
            Pedestal temperature passed to the crash model.
        n_ped
            Pedestal density passed to the crash model.
        W_ped
            Pedestal stored energy in MJ.

        Returns
        -------
        ELMEvent or None
            The crash event when the boundary is unstable, otherwise ``None``.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if alpha_edge < 0.0 or j_edge < 0.0 or s_edge < 0.0:
            raise ValueError("alpha_edge, j_edge, and s_edge must be non-negative")
        self.time += dt

        if self.pb.is_unstable(alpha_edge, j_edge, s_edge):
            res = self.crash_model.crash(T_ped, n_ped, W_ped)

            f_elm = 1.0 / max(self.time - self.last_crash, 1e-6)
            self.last_crash = self.time

            return ELMEvent(time=self.time, delta_W_MJ=res.delta_W_MJ, f_elm_Hz=f_elm, crash_type="Type I")

        return None
