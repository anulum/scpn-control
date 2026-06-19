# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Pellet Injection
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Pellet Injection Physics
# ──────────────────────────────────────────────────────────────────────
"""Pellet ablation, trajectory, deposition, and fuelling-control utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray


def _finite_scalar(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _profile_array(
    name: str, values: AnyFloatArray, shape: tuple[int, ...], *, nonnegative: bool = False
) -> FloatArray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{name} must match rho shape")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return arr


@dataclass
class PelletParams:
    """Cryogenic fuel-pellet launch parameters.

    Attributes
    ----------
    r_p_mm
        Pellet radius in millimetres; must be positive.
    v_p_m_s
        Pellet injection speed in m/s; must be positive.
    M_p
        Atomic mass of the pellet species in amu (2.0 for deuterium).
    injection_side
        Launch side, ``"HFS"`` (high-field) or ``"LFS"`` (low-field).
    injection_angle_deg
        Injection angle relative to the midplane in degrees.
    """

    r_p_mm: float  # [mm]
    v_p_m_s: float  # [m/s]
    M_p: float = 2.0
    injection_side: str = "HFS"
    injection_angle_deg: float = 0.0

    def __post_init__(self) -> None:
        self.r_p_mm = _finite_scalar("r_p_mm", self.r_p_mm, positive=True)
        self.v_p_m_s = _finite_scalar("v_p_m_s", self.v_p_m_s, positive=True)
        self.M_p = _finite_scalar("M_p", self.M_p, positive=True)
        self.injection_angle_deg = _finite_scalar("injection_angle_deg", self.injection_angle_deg)
        if self.injection_side not in {"HFS", "LFS"}:
            raise ValueError("injection_side must be 'HFS' or 'LFS'")


@dataclass
class PelletResult:
    """Outcome of a pellet ablation/deposition simulation.

    Attributes
    ----------
    penetration_depth
        Normalised radius ``rho`` at which the pellet was fully ablated.
    deposition_profile
        Particle deposition density per radial cell in m⁻³, drift-shifted.
    total_particles
        Total particles deposited into the plasma.
    drift_displacement
        ∇B drift displacement of the deposition in normalised radius.
    """

    penetration_depth: float
    deposition_profile: AnyFloatArray
    total_particles: float
    drift_displacement: float


@dataclass
class PelletInjectionCommand:
    """Command to launch a pellet at a scheduled time.

    Attributes
    ----------
    inject_time
        Scheduled injection time in seconds.
    pellet_params
        The pellet launch parameters.
    """

    inject_time: float
    pellet_params: PelletParams


def ngs_ablation_rate(r_p: float, ne: float, Te_eV: float, M_p: float) -> float:
    """
    Parks & Turnbull (1978) NGS ablation rate [atoms/s].
    """
    r_p = _finite_scalar("r_p", r_p, positive=True)
    ne = _finite_scalar("ne", ne, positive=True)
    Te_eV = _finite_scalar("Te_eV", Te_eV, positive=True)
    M_p = _finite_scalar("M_p", M_p, positive=True)

    C_abl = 1.12e6
    rate = C_abl * (ne ** (1.0 / 3.0)) * (Te_eV ** (5.0 / 3.0)) * (r_p ** (4.0 / 3.0)) * (M_p ** (-1.0 / 3.0))
    return float(max(0.0, rate))


class PelletTrajectory:
    """Radial pellet trajectory with neutral-gas-shielding ablation and ∇B drift.

    Integrates the Parks-Turnbull NGS ablation rate along an inward radial path,
    depositing the ablated particles and applying a ∇B drift shift of the
    deposition profile (Pegourie 2005, Plasma Phys. Control. Fusion 47, 17).

    Parameters
    ----------
    params
        The pellet launch parameters.
    R0
        Major radius in metres; must be positive.
    a
        Minor radius in metres; must be positive and below ``R0``.
    B0
        Toroidal field on axis in tesla; must be positive.
    """

    def __init__(self, params: PelletParams, R0: float, a: float, B0: float):
        self.params = params
        self.R0 = _finite_scalar("R0", R0, positive=True)
        self.a = _finite_scalar("a", a, positive=True)
        if self.a >= self.R0:
            raise ValueError("a must be smaller than R0 for tokamak ordering")
        self.B0 = _finite_scalar("B0", B0, positive=True)

        # Solid density of deuterium ~ 5e28 atoms/m^3
        self.n_solid = 5.0e28
        volume_m3 = (4.0 / 3.0) * np.pi * (params.r_p_mm * 1e-3) ** 3
        self.N_initial = volume_m3 * self.n_solid

    def simulate(self, rho: AnyFloatArray, ne: AnyFloatArray, Te_eV: AnyFloatArray) -> PelletResult:
        """Integrate ablation and deposition along the inward radial path.

        Parameters
        ----------
        rho
            Strictly increasing normalised-radius grid in [0, 1], shape
            ``(n_rho,)`` with at least two points.
        ne
            Electron-density profile in 10¹⁹ m⁻³ on ``rho``; non-negative.
        Te_eV
            Electron-temperature profile in eV on ``rho``; non-negative.

        Returns
        -------
        PelletResult
            The penetration depth, drift-shifted deposition profile, deposited
            particle count, and drift displacement.

        Raises
        ------
        ValueError
            If ``rho`` is not a strictly increasing grid in [0, 1] or the
            profiles have the wrong shape or are negative.
        """
        rho = np.asarray(rho, dtype=float)
        if rho.ndim != 1 or rho.size < 2:
            raise ValueError("rho must be a one-dimensional grid with at least two points")
        if not np.all(np.isfinite(rho)):
            raise ValueError("rho must contain only finite values")
        if rho[0] < 0.0 or rho[-1] > 1.0 or np.any(np.diff(rho) <= 0.0):
            raise ValueError("rho must be strictly increasing within [0, 1]")
        ne = _profile_array("ne", ne, rho.shape, nonnegative=True)
        Te_eV = _profile_array("Te_eV", Te_eV, rho.shape, nonnegative=True)
        N_p = self.N_initial
        r_p_m = self.params.r_p_mm * 1e-3

        # Path simulation
        # Assume it travels strictly radially for simplicity: s = a * (1 - rho)
        # So v_p = ds/dt = -a * drho/dt

        # We integrate over time steps
        dt = 1e-5

        deposition = np.zeros_like(rho)

        current_rho = 1.0  # start at edge
        dr_dt = -self.params.v_p_m_s / self.a

        while N_p > 0 and current_rho > 0.0:
            # Interpolate plasma parameters
            idx = int(np.searchsorted(rho, current_rho))
            if idx == 0:
                n_local = ne[0]
                T_local = Te_eV[0]
            elif idx >= len(rho):
                n_local = ne[-1]
                T_local = Te_eV[-1]
            else:
                frac = (current_rho - rho[idx - 1]) / (rho[idx] - rho[idx - 1])
                n_local = ne[idx - 1] + frac * (ne[idx] - ne[idx - 1])
                T_local = Te_eV[idx - 1] + frac * (Te_eV[idx] - Te_eV[idx - 1])

            rate = ngs_ablation_rate(r_p_m, n_local * 1e19, T_local, self.params.M_p)

            dN = rate * dt
            if dN > N_p:
                dN = N_p

            N_p -= dN

            # Distribute dN to profile
            if idx < len(rho):
                dep_idx = max(int(idx) - 1, 0)
                r1 = rho[dep_idx] if dep_idx > 0 else 0.0
                r2 = rho[min(dep_idx + 1, len(rho) - 1)]
                vol = 2.0 * np.pi**2 * self.R0 * self.a**2 * (r2**2 - r1**2)
                deposition[dep_idx] += dN / max(vol, 1e-6)

            # Update radius based on lost mass
            # N = 4/3 pi r^3 n_solid => r = (3N / 4 pi n_solid)^(1/3)
            if N_p > 0:
                r_p_m = (0.75 * N_p / (np.pi * self.n_solid)) ** (1.0 / 3.0)
            else:
                r_p_m = 0.0

            current_rho += dr_dt * dt

        # ∇B drift displacement — Pegourie, Plasma Phys. Control. Fusion 47, 17 (2005)
        T_avg = np.mean(Te_eV)
        n_dep = np.mean(deposition[deposition > 0]) if np.any(deposition > 0) else 1.0
        n_e_avg = np.mean(ne) * 1e19
        # Parametric fit: displacement ~ 0.1-0.2 a for ITER
        drift_m = 0.1 * self.a * np.sqrt(n_dep / max(n_e_avg, 1e-3)) * (T_avg / max(self.B0**2, 1e-3)) / 1000.0

        if self.params.injection_side == "HFS":
            drift_rho = -drift_m / self.a
        else:
            drift_rho = drift_m / self.a

        # Shift profile
        shifted_dep = np.zeros_like(deposition)
        shift_idx = int(round(drift_rho / (rho[1] - rho[0])))

        if shift_idx > 0:
            shifted_dep[shift_idx:] = deposition[:-shift_idx]
        elif shift_idx < 0:
            shifted_dep[:shift_idx] = deposition[-shift_idx:]
        else:
            shifted_dep = deposition.copy()

        return PelletResult(
            penetration_depth=float(max(0.0, current_rho)),
            deposition_profile=shifted_dep,
            total_particles=float(self.N_initial - N_p),
            drift_displacement=float(drift_rho),
        )


class PelletFuelingController:
    """Pellet-pacing controller maintaining a volume-averaged density target.

    Parameters
    ----------
    target_density
        Target volume-averaged electron density in 10¹⁹ m⁻³; must be positive.
    pellet_params
        The pellet launch parameters used for each injection.
    """

    def __init__(self, target_density: float, pellet_params: PelletParams):
        self.target_density = _finite_scalar("target_density", target_density, positive=True)
        self.pellet_params = pellet_params

        volume_m3 = (4.0 / 3.0) * np.pi * (pellet_params.r_p_mm * 1e-3) ** 3
        self.N_pellet = volume_m3 * 5.0e28
        self.time = 0.0
        self.last_injection = -100.0

    def required_frequency(self, ne_current: float, tau_p: float, V_plasma: float) -> float:
        """Pellet frequency that sustains the target density inventory.

        From the particle balance ``dN/dt = -N/τ_p + f N_pellet`` at the target,
        ``f = N_target / (τ_p N_pellet)``.

        Parameters
        ----------
        ne_current
            Current volume-averaged density in 10¹⁹ m⁻³; must be non-negative.
        tau_p
            Particle confinement time in seconds; must be positive.
        V_plasma
            Plasma volume in m³; must be positive.

        Returns
        -------
        float
            The required injection frequency in Hz.
        """
        _finite_scalar("ne_current", ne_current, nonnegative=True)
        tau_p = _finite_scalar("tau_p", tau_p, positive=True)
        V_plasma = _finite_scalar("V_plasma", V_plasma, positive=True)
        # N_target = target * V, N_current = current * V
        # dN/dt = -N/tau_p + S_pellet
        # S_pellet = f * N_pellet
        # To maintain target: f = N_target / (tau_p * N_pellet)
        N_targ = self.target_density * 1e19 * V_plasma
        f = N_targ / (tau_p * self.N_pellet)
        return float(f)

    def step(
        self, ne_profile: AnyFloatArray, Te_profile: AnyFloatArray, dt: float, V_plasma: float
    ) -> PelletInjectionCommand | None:
        """Advance the controller clock and decide whether to inject.

        A pellet is scheduled when the mean density falls below 95% of target and
        the time since the last injection exceeds the required pacing period.

        Parameters
        ----------
        ne_profile
            Electron-density profile in 10¹⁹ m⁻³; non-empty and non-negative.
        Te_profile
            Electron-temperature profile in eV, same shape as ``ne_profile``.
        dt
            Time step in seconds; must be positive.
        V_plasma
            Plasma volume in m³; must be positive.

        Returns
        -------
        PelletInjectionCommand or None
            A launch command when an injection is due, otherwise ``None``.
        """
        ne_profile = np.asarray(ne_profile, dtype=float)
        if ne_profile.ndim != 1 or ne_profile.size == 0:
            raise ValueError("ne_profile must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(ne_profile)) or np.any(ne_profile < 0.0):
            raise ValueError("ne_profile must contain only finite non-negative values")
        _profile_array("Te_profile", Te_profile, ne_profile.shape, nonnegative=True)
        dt = _finite_scalar("dt", dt, positive=True)
        V_plasma = _finite_scalar("V_plasma", V_plasma, positive=True)
        self.time += dt

        current_density = np.mean(ne_profile)

        if current_density < self.target_density * 0.95:
            # Need fueling
            freq = self.required_frequency(float(current_density), 5.0, V_plasma)
            period = 1.0 / max(freq, 0.1)

            if self.time - self.last_injection > period:
                self.last_injection = self.time
                return PelletInjectionCommand(self.time, self.pellet_params)

        return None


def pellet_pacing_elm_control(
    f_pellet_Hz: float, f_elm_natural_Hz: float, w_elm_natural_MJ: float
) -> tuple[float, float]:
    """
    Returns (f_ELM, delta_W_ELM).
    """
    f_pellet_Hz = _finite_scalar("f_pellet_Hz", f_pellet_Hz, nonnegative=True)
    f_elm_natural_Hz = _finite_scalar("f_elm_natural_Hz", f_elm_natural_Hz, positive=True)
    w_elm_natural_MJ = _finite_scalar("w_elm_natural_MJ", w_elm_natural_MJ, nonnegative=True)
    if f_pellet_Hz > 1.5 * f_elm_natural_Hz:
        f_elm = f_pellet_Hz
        w_elm = w_elm_natural_MJ * (f_elm_natural_Hz / f_pellet_Hz)
        return float(f_elm), float(w_elm)
    return float(f_elm_natural_Hz), float(w_elm_natural_MJ)
