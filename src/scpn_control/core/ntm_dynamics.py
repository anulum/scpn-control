# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — NTM island dynamics
"""
NTM island width dynamics via the Modified Rutherford Equation (MRE).

Implements the full MRE including classical tearing stability, bootstrap
current drive, polarization stabilization, diamagnetic shear stabilization,
ECCD suppression, and the Glasser-Greene-Johnson resistive-interchange index.

Key references
--------------
Rutherford 1973     : P.H. Rutherford, Phys. Fluids 16, 1903 (1973).
Glasser et al. 1975 : A.H. Glasser, J.M. Greene, J.L. Johnson, Phys. Fluids
                      18, 875 (1975). [GGJ]
Sauter et al. 1997  : O. Sauter, R.J. La Haye, Z. Chang et al., Phys. Plasmas
                      4, 1654 (1997). [diamagnetic + polarization]
Sauter et al. 1999  : O. Sauter et al., Phys. Plasmas 6, 2834 (1999).
                      [L31 bootstrap coefficient]
Carrera et al. 1986 : R. Carrera, R.D. Hazeltine, M. Kotschenreuther, Phys.
                      Fluids 29, 899 (1986). [ECCD efficiency model]
La Haye 2006        : R.J. La Haye, Phys. Plasmas 13, 055501 (2006).
                      [MRE coefficient fits a1–a3]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

# Permeability of free space (CODATA 2018)
MU_0: float = 4.0 * np.pi * 1e-7  # H/m

# ─── MRE fit coefficients ────────────────────────────────────────────────────
# La Haye 2006, Phys. Plasmas 13, 055501, Table I
_A1: float = 6.35  # bootstrap drive coefficient
_A2: float = 1.2  # polarization stabilization coefficient
_A3: float = 9.36  # ECCD stabilization coefficient

# Diamagnetic shear stabilization (Sauter et al. 1997, Eq. 20)
_A4: float = 0.86  # Sauter 1997, Eq. 20 fit coefficient
_S_HAT_REF: float = 1.0  # reference shear, dimensionless


def _finite_scalar(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


@dataclass
class RationalSurface:
    """q = m/n rational surface."""

    rho: float
    r_s: float
    m: int
    n: int
    q: float
    shear: float


def eccd_stabilization_factor(d_cd: float, w: float) -> float:
    """ECCD deposition efficiency factor f_ECCD(w, d_cd).

    f_ECCD = (w / d_cd) * exp(-w² / (4 d_cd²))

    Carrera et al. 1986, Phys. Fluids 29, 899, Eq. 16.
    """
    d_cd = _finite_scalar("d_cd", d_cd)
    w = _finite_scalar("w", w)
    if w <= 0.0 or d_cd <= 0.0:
        return 0.0
    return float((w / d_cd) * np.exp(-(w**2) / (4.0 * d_cd**2)))


def find_rational_surfaces(
    q: AnyFloatArray, rho: AnyFloatArray, a: float, m_max: int = 5, n_max: int = 3
) -> list[RationalSurface]:
    """Locate radii where q(rho) = m/n by zero-crossing interpolation."""
    if q.ndim != 1 or rho.ndim != 1:
        raise ValueError("q and rho must be one-dimensional")
    if len(q) != len(rho) or len(q) < 2:
        raise ValueError("q and rho must have equal length with at least two samples")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(rho)):
        raise ValueError("q and rho values must be finite")
    if np.any(q <= 0.0):
        raise ValueError("q values must be positive")
    if np.any(np.diff(rho) <= 0.0):
        raise ValueError("rho must be strictly increasing")
    a = _finite_scalar("a", a, positive=True)
    if m_max <= 0 or n_max <= 0:
        raise ValueError("m_max and n_max must be positive")
    surfaces: list[RationalSurface] = []
    dq_drho = np.gradient(q, rho)

    for m in range(1, m_max + 1):
        for n in range(1, n_max + 1):
            q_target = m / n
            q_diff = q - q_target
            crossings = np.where(np.diff(np.sign(q_diff)))[0]

            for idx in crossings:
                r1, r2 = rho[idx], rho[idx + 1]
                q1, q2 = q[idx], q[idx + 1]
                # A sign change in (q - q_target) between idx and idx+1 implies
                # q1 != q2; equal endpoints share a sign and produce no crossing,
                # so this divide-by-zero guard is unreachable for detected crossings.
                if q1 == q2:  # pragma: no cover
                    continue

                frac = (q_target - q1) / (q2 - q1)
                rho_s = r1 + frac * (r2 - r1)

                dq_s = dq_drho[idx] + frac * (dq_drho[idx + 1] - dq_drho[idx])
                # magnetic shear: s = (ρ/q) * dq/dρ
                shear_s = (rho_s / q_target) * dq_s

                surfaces.append(
                    RationalSurface(
                        rho=float(rho_s),
                        r_s=float(rho_s * a),
                        m=m,
                        n=n,
                        q=float(q_target),
                        shear=float(shear_s),
                    )
                )

    surfaces.sort(key=lambda x: x.rho)
    return surfaces


def _ggj_delta_prime(
    m: int,
    r_s: float,
    s_hat: float,
    pressure_gradient: float,
    B_pol: float,
    q: float,
    R0: float,
) -> float:
    """GGJ resistive-interchange contribution to Δ'.

    Δ'_GGJ = -m * D_R / (r_s * s_hat)

    where the resistive interchange index is

        D_R = -2 μ_0 q² R0² p' / (s_hat² B_pol²)

    Glasser, Greene & Johnson 1975, Phys. Fluids 18, 875, Eq. 42.

    Parameters
    ----------
    m : int
        Poloidal mode number.
    r_s : float
        Minor radius of rational surface [m].
    s_hat : float
        Magnetic shear at r_s (dimensionless).
    pressure_gradient : float
        dp/dr at the rational surface [Pa/m] (negative for peaked profiles).
    B_pol : float
        Poloidal field at r_s [T].
    q : float
        Safety factor at r_s.
    R0 : float
        Major radius [m].

    Returns
    -------
    float
        Δ'_GGJ [m^-1].
    """
    if m <= 0:
        raise ValueError("m must be positive")
    r_s = _finite_scalar("r_s", r_s, positive=True)
    s_hat = _finite_scalar("s_hat", s_hat)
    pressure_gradient = _finite_scalar("pressure_gradient", pressure_gradient)
    B_pol = _finite_scalar("B_pol", B_pol)
    q = _finite_scalar("q", q, positive=True)
    R0 = _finite_scalar("R0", R0, positive=True)
    if abs(s_hat) < 1e-6 or abs(B_pol) < 1e-10:
        return 0.0
    D_R = -2.0 * MU_0 * q**2 * R0**2 * pressure_gradient / (s_hat**2 * B_pol**2)
    return float(-m * D_R / (r_s * s_hat))


def bootstrap_from_local(
    ne_19: float,
    Te_keV: float,
    Ti_keV: float,
    q: float,
    rho: float,
    R0: float,
    a: float,
    B0: float,
    z_eff: float,
    dne_dr: float,
    dTe_dr: float,
    dTi_dr: float,
) -> float:
    """Local Sauter bootstrap current density from full L31+L32+L34 closure.

    Sauter et al. 1999, Phys. Plasmas 6, 2834, Eqs. 14-16:

        j_bs = -(p_e/B_pol) * [L31*d(ln p_e)/dr + L32*d(ln T_e)/dr
                               + L34*(T_i/T_e)*d(ln T_i)/dr]

    The local collisionality and trapped-particle fractions are evaluated from
    the same fits used in :mod:`scpn_control.core.neoclassical`.
    """
    ne_19 = _finite_scalar("ne_19", ne_19, positive=True)
    Te_keV = _finite_scalar("Te_keV", Te_keV, positive=True)
    Ti_keV = _finite_scalar("Ti_keV", Ti_keV, positive=True)
    q = _finite_scalar("q", q, positive=True)
    rho = _finite_scalar("rho", rho)
    R0 = _finite_scalar("R0", R0, positive=True)
    a = _finite_scalar("a", a, positive=True)
    B0 = _finite_scalar("B0", B0)
    z_eff = _finite_scalar("z_eff", z_eff, positive=True)
    dne_dr = _finite_scalar("dne_dr", dne_dr)
    dTe_dr = _finite_scalar("dTe_dr", dTe_dr)
    dTi_dr = _finite_scalar("dTi_dr", dTi_dr)
    if B0 == 0.0:
        raise ValueError("B0 must be non-zero")

    epsilon = rho * a / R0
    if epsilon < 1e-6:
        return 0.0

    # Keep local imports to avoid module-level dependency cycle risk.
    from scpn_control.core.neoclassical import _sauter_L31, _sauter_L32, _sauter_L34

    e_charge = 1.602176634e-19
    eps0 = 8.8541878128e-12
    m_e = 9.1093837015e-31
    ln_lambda = 17.0

    # Sauter 1999 trapped-fraction fit (same expression as neoclassical.py)
    f_t = 1.0 - (1.0 - epsilon) ** 2 / (np.sqrt(1.0 - epsilon**2) * (1.0 + 1.46 * np.sqrt(epsilon)))

    n_e_si = ne_19 * 1e19
    T_e_J = Te_keV * 1.602176634e-16
    T_i_J = Ti_keV * 1.602176634e-16
    p_e = n_e_si * T_e_J
    # ne_19 and Te_keV are validated strictly positive above, so the electron
    # pressure is strictly positive and this degeneracy guard never fires.
    if p_e <= 0.0:  # pragma: no cover
        return 0.0

    # Wesson 2011, Eq. 14.2.3 (same as neoclassical.py)
    v_the = np.sqrt(2.0 * T_e_J / m_e)
    nu_ee = (n_e_si * e_charge**4 * ln_lambda) / (12.0 * np.pi**1.5 * eps0**2 * m_e**0.5 * T_e_J**1.5)
    nu_star_e = nu_ee * q * R0 / (epsilon**1.5 * v_the)

    L31 = _sauter_L31(f_t, nu_star_e, z_eff)
    L32 = _sauter_L32(f_t, nu_star_e, z_eff)
    L34 = _sauter_L34(f_t, nu_star_e, z_eff)

    dln_pe_dr = (T_e_J * (dne_dr * 1e19) + n_e_si * 1.602176634e-16 * dTe_dr) / p_e
    dln_Te_dr = dTe_dr / max(Te_keV, 1e-12)
    dln_Ti_dr = dTi_dr / max(Ti_keV, 1e-12)

    B_pol = B0 * epsilon / max(q, 1e-3)
    if abs(B_pol) < 1e-10:
        return 0.0

    return float(-(p_e / B_pol) * (L31 * dln_pe_dr + L32 * dln_Te_dr + L34 * (T_i_J / T_e_J) * dln_Ti_dr))


class NTMIslandDynamics:
    """Modified Rutherford Equation solver for a single NTM island.

    Rutherford 1973, Phys. Fluids 16, 1903 — original equation.
    La Haye 2006, Phys. Plasmas 13, 055501 — full MRE with coefficients.
    """

    def __init__(
        self,
        r_s: float,
        m: int,
        n: int,
        a: float,
        R0: float,
        B0: float,
        Delta_prime_0: float | None = None,
        s_hat: float = 1.0,
        q_s: float = 2.0,
    ):
        r_s = _finite_scalar("r_s", r_s, positive=True)
        a = _finite_scalar("a", a, positive=True)
        R0 = _finite_scalar("R0", R0, positive=True)
        B0 = _finite_scalar("B0", B0, positive=True)
        if r_s > a:
            raise ValueError("r_s must not exceed the plasma minor radius a")
        if a >= R0:
            raise ValueError("a must be smaller than R0 for tokamak ordering")
        if m <= 0 or n <= 0:
            raise ValueError("m and n must be positive")
        q_s = _finite_scalar("q_s", q_s, positive=True)
        s_hat = _finite_scalar("s_hat", s_hat)
        self.r_s = r_s
        self.m = m
        self.n = n
        self.a = a
        self.R0 = R0
        self.B0 = B0
        # Sauter et al. 1997, Eq. 20: s_hat enters diamagnetic stabilization
        self.s_hat = s_hat
        self.q_s = q_s

        # Classical tearing index — Rutherford 1973, Phys. Fluids 16, 1903
        self.Delta_prime_0 = (
            _finite_scalar("Delta_prime_0", Delta_prime_0) if Delta_prime_0 is not None else -2.0 * m / max(r_s, 1e-3)
        )

        # La Haye 2006, Phys. Plasmas 13, 055501, Table I
        self.a1 = _A1
        self.a2 = _A2
        self.a3 = _A3

    def delta_prime_model(self, w: float) -> float:
        """Classical Δ' with finite-island-width saturation.

        Rutherford 1973, Phys. Fluids 16, 1903; La Haye 2006, Eq. 3.
        """
        w = _finite_scalar("w", w, positive=True)
        c = 0.5  # saturation width coefficient, La Haye 2006, Eq. 3
        return self.Delta_prime_0 * self.r_s / (self.r_s + c * w)

    def dw_dt(
        self,
        w: float,
        j_bs: float,
        j_phi: float,
        j_cd: float,
        eta: float,
        w_d: float = 1e-3,
        w_pol: float = 5e-4,
        d_cd: float = 0.05,
        # GGJ arguments (optional)
        pressure_gradient: float = 0.0,
        B_pol: float | None = None,
        # Diamagnetic shear stabilization (Sauter 1997)
        rho_theta_i: float | None = None,
        beta_pol: float | None = None,
    ) -> float:
        """RHS of the Modified Rutherford Equation.

        τ_R * (dw/dt) / r_s = Δ' r_s          (Rutherford 1973)
                               + a1*(j_bs/j_φ)*(r_s/w)/(1+(w_d/w)²)
                                               (La Haye 2006, Eq. 4)
                               - a2*(w_pol²/w³) (Sauter 1997, Eq. 19)
                               - a4*(w_d²/w²)*(s/s_ref)
                                               (Sauter 1997, Eq. 20)
                               - a3*(j_cd/j_φ)*(r_s/w)*f_ECCD
                                               (La Haye 2006, Eq. 5;
                                                Carrera 1986, Eq. 16)

        Parameters
        ----------
        w : float
            Island half-width [m].
        j_bs : float
            Bootstrap current density [A/m^2].
        j_phi : float
            Total toroidal current density [A/m^2].
        j_cd : float
            ECCD current density [A/m^2].
        eta : float
            Resistivity [Ω·m].
        w_d : float
            Ion banana width ρ_θi √(2 β_pol) [m]; overridden if rho_theta_i
            and beta_pol are given. Sauter 1997, Eq. 21.
        w_pol : float
            Polarization width threshold [m]. Sauter 1997, Eq. 19.
        d_cd : float
            ECCD deposition half-width [m]. Carrera 1986, Eq. 16.
        pressure_gradient : float
            dp/dr [Pa/m]; used for GGJ Δ' correction when non-zero.
        B_pol : float | None
            Poloidal field [T]; required for GGJ when pressure_gradient ≠ 0.
        rho_theta_i : float | None
            Poloidal ion Larmor radius [m]; if given with beta_pol, overrides w_d.
        beta_pol : float | None
            Poloidal beta; if given with rho_theta_i, overrides w_d.

        Returns
        -------
        float
            dw/dt [m/s].
        """
        w = _finite_scalar("w", w, nonnegative=True)
        j_bs = _finite_scalar("j_bs", j_bs)
        j_phi = _finite_scalar("j_phi", j_phi, positive=True)
        j_cd = _finite_scalar("j_cd", j_cd)
        eta = _finite_scalar("eta", eta, positive=True)
        w_d = _finite_scalar("w_d", w_d, nonnegative=True)
        w_pol = _finite_scalar("w_pol", w_pol, nonnegative=True)
        d_cd = _finite_scalar("d_cd", d_cd, nonnegative=True)
        pressure_gradient = _finite_scalar("pressure_gradient", pressure_gradient)
        if w <= 1e-6:
            return 0.0
        if (rho_theta_i is None) != (beta_pol is None):
            raise ValueError("rho_theta_i and beta_pol must be supplied together")
        if B_pol is not None:
            B_pol = _finite_scalar("B_pol", B_pol)
        if rho_theta_i is not None and beta_pol is not None:
            rho_theta_i = _finite_scalar("rho_theta_i", rho_theta_i, nonnegative=True)
            beta_pol = _finite_scalar("beta_pol", beta_pol, nonnegative=True)

        # Resistive diffusion time — Rutherford 1973, Phys. Fluids 16, 1903, Eq. 2
        tau_R = MU_0 * self.r_s**2 / max(eta, 1e-12)

        # ── Ion banana width override ─────────────────────────────────────────
        # Sauter et al. 1997, Phys. Plasmas 4, 1654, Eq. 21:
        #   w_d = ρ_θi √(2 β_pol)
        if rho_theta_i is not None and beta_pol is not None:
            w_d = rho_theta_i * np.sqrt(2.0 * beta_pol)

        # ── Classical tearing index (with optional GGJ correction) ────────────
        delta_prime_rs = self.delta_prime_model(w)
        if pressure_gradient != 0.0 and B_pol is not None:
            delta_prime_rs += _ggj_delta_prime(
                self.m, self.r_s, self.s_hat, pressure_gradient, B_pol, self.q_s, self.R0
            )
        term_classical = self.r_s * delta_prime_rs

        # ── Bootstrap drive ───────────────────────────────────────────────────
        # La Haye 2006, Phys. Plasmas 13, 055501, Eq. 4
        j_ratio = j_bs / max(j_phi, 1e-6)
        term_bs = self.a1 * j_ratio * (self.r_s / w) * (1.0 / (1.0 + (w_d / w) ** 2))

        # ── Polarization stabilization ────────────────────────────────────────
        # Sauter et al. 1997, Phys. Plasmas 4, 1654, Eq. 19
        term_pol = -self.a2 * (w_pol / w) ** 3

        # ── Diamagnetic shear stabilization ───────────────────────────────────
        # Sauter et al. 1997, Phys. Plasmas 4, 1654, Eq. 20
        #   term_dia = -a4 * (w_d² / w²) * (s_hat / s_hat_ref)
        term_dia = -_A4 * (w_d**2 / w**2) * (self.s_hat / _S_HAT_REF)

        # ── ECCD stabilization ────────────────────────────────────────────────
        # La Haye 2006, Phys. Plasmas 13, 055501, Eq. 5;
        # Carrera et al. 1986, Phys. Fluids 29, 899, Eq. 16
        j_cd_ratio = j_cd / max(j_phi, 1e-6)
        f_eccd = eccd_stabilization_factor(d_cd, w)
        term_cd = -self.a3 * j_cd_ratio * (self.r_s / w) * f_eccd

        dw_dt_val = (1.0 / tau_R) * (term_classical + term_bs + term_pol + term_dia + term_cd)
        return float(dw_dt_val)

    def evolve(
        self,
        w0: float,
        t_span: tuple[float, float],
        dt: float,
        j_bs: float,
        j_phi: float,
        j_cd: float,
        eta: float,
        w_d: float = 1e-3,
        w_pol: float = 5e-4,
        d_cd: float = 0.05,
        pressure_gradient: float = 0.0,
        B_pol: float | None = None,
        rho_theta_i: float | None = None,
        beta_pol: float | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Integrate w(t) via RK4.

        All physics keyword arguments are forwarded to dw_dt unchanged.
        """
        t_start = _finite_scalar("t_start", t_span[0])
        t_end = _finite_scalar("t_end", t_span[1])
        if t_end <= t_start:
            raise ValueError("t_span end must be greater than start")
        dt = _finite_scalar("dt", dt, positive=True)
        w0 = _finite_scalar("w0", w0, positive=True)
        n_steps = int(np.ceil((t_end - t_start) / dt))
        t_arr = np.linspace(t_start, t_end, n_steps + 1, dtype=np.float64)
        w_arr = np.zeros(n_steps + 1)
        w_arr[0] = max(w0, 1e-6)

        def _rhs(w_val: float) -> float:
            return self.dw_dt(
                w_val,
                j_bs=j_bs,
                j_phi=j_phi,
                j_cd=j_cd,
                eta=eta,
                w_d=w_d,
                w_pol=w_pol,
                d_cd=d_cd,
                pressure_gradient=pressure_gradient,
                B_pol=B_pol,
                rho_theta_i=rho_theta_i,
                beta_pol=beta_pol,
            )

        for i in range(n_steps):
            w_curr = w_arr[i]

            k1 = _rhs(w_curr)
            k2 = _rhs(max(w_curr + 0.5 * dt * k1, 1e-6))
            k3 = _rhs(max(w_curr + 0.5 * dt * k2, 1e-6))
            k4 = _rhs(max(w_curr + dt * k3, 1e-6))

            w_next = w_curr + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            w_arr[i + 1] = max(w_next, 1e-6)

        return t_arr, w_arr


class NTMController:
    """ECCD-based NTM controller: triggers at onset, deactivates at target."""

    def __init__(self, w_onset: float = 0.02, w_target: float = 0.005):
        w_onset = _finite_scalar("w_onset", w_onset, positive=True)
        w_target = _finite_scalar("w_target", w_target, positive=True)
        if w_target >= w_onset:
            raise ValueError("w_target must be smaller than w_onset")
        self.w_onset = w_onset
        self.w_target = w_target
        self.active = False
        self.target_rho = 0.0
        self.eccd_power_request = 0.0

    def step(self, w: float, rho_rs: float, max_power: float = 20.0) -> float:
        """Return requested ECCD power [MW]; update controller state."""
        w = _finite_scalar("w", w, nonnegative=True)
        rho_rs = _finite_scalar("rho_rs", rho_rs)
        if not (0.0 <= rho_rs <= 1.0):
            raise ValueError("rho_rs must lie in [0, 1]")
        max_power = _finite_scalar("max_power", max_power, nonnegative=True)
        if not self.active and w > self.w_onset:
            self.active = True
            self.target_rho = rho_rs
            self.eccd_power_request = max_power
        elif self.active:
            self.target_rho = rho_rs
            if w < self.w_target:
                self.active = False
                self.eccd_power_request = 0.0
            else:
                self.eccd_power_request = max_power

        return self.eccd_power_request
