# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851  Contact: protoscience@anulum.li
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

# Permeability of free space (CODATA 2018)
MU_0: float = 4.0 * np.pi * 1e-7  # H/m

# ─── MRE fit coefficients ────────────────────────────────────────────────────
# La Haye 2006, Phys. Plasmas 13, 055501, Table I
_A1: float = 6.35   # bootstrap drive coefficient
_A2: float = 1.2    # polarization stabilization coefficient
_A3: float = 9.36   # ECCD stabilization coefficient

# Diamagnetic shear stabilization (Sauter et al. 1997, Eq. 20)
_A4: float = 0.86         # Sauter 1997, Eq. 20 fit coefficient
_S_HAT_REF: float = 1.0   # reference shear, dimensionless


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
    if w <= 0.0 or d_cd <= 0.0:
        return 0.0
    return float((w / d_cd) * np.exp(-(w**2) / (4.0 * d_cd**2)))


def find_rational_surfaces(
    q: np.ndarray, rho: np.ndarray, a: float, m_max: int = 5, n_max: int = 3
) -> list[RationalSurface]:
    """Locate radii where q(rho) = m/n by zero-crossing interpolation."""
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
                if q1 == q2:
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
    if abs(s_hat) < 1e-6 or abs(B_pol) < 1e-10:
        return 0.0
    D_R = -2.0 * MU_0 * q**2 * R0**2 * pressure_gradient / (s_hat**2 * B_pol**2)
    return float(-m * D_R / (r_s * s_hat))


def bootstrap_from_local(
    pressure_gradient: float,
    epsilon: float,
    B_pol: float,
    L31: float,
) -> float:
    """Simplified local bootstrap current density.

    j_bs = -ε^0.5 * p' * L31 / B_pol

    Sauter et al. 1999, Phys. Plasmas 6, 2834, Eq. 14 (leading-order form).

    Parameters
    ----------
    pressure_gradient : float
        dp/dr [Pa/m] (negative for peaked profile → positive j_bs).
    epsilon : float
        Inverse aspect ratio r/R.
    B_pol : float
        Poloidal field [T].
    L31 : float
        Sauter L31 coefficient (see neoclassical._sauter_L31).

    Returns
    -------
    float
        Bootstrap current density [A/m^2].
    """
    if abs(B_pol) < 1e-10 or epsilon < 0.0:
        return 0.0
    return float(-np.sqrt(epsilon) * pressure_gradient * L31 / B_pol)


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
        self.Delta_prime_0 = Delta_prime_0 if Delta_prime_0 is not None else -2.0 * m / max(r_s, 1e-3)

        # La Haye 2006, Phys. Plasmas 13, 055501, Table I
        self.a1 = _A1
        self.a2 = _A2
        self.a3 = _A3

    def delta_prime_model(self, w: float) -> float:
        """Classical Δ' with finite-island-width saturation.

        Rutherford 1973, Phys. Fluids 16, 1903; La Haye 2006, Eq. 3.
        """
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
        if w <= 1e-6:
            return 0.0

        # Resistive diffusion time — Rutherford 1973, Phys. Fluids 16, 1903, Eq. 2
        tau_R = MU_0 * self.r_s**2 / max(eta, 1e-12)

        # ── Ion banana width override ─────────────────────────────────────────
        # Sauter et al. 1997, Phys. Plasmas 4, 1654, Eq. 21:
        #   w_d = ρ_θi √(2 β_pol)
        if rho_theta_i is not None and beta_pol is not None:
            w_d = rho_theta_i * np.sqrt(2.0 * max(beta_pol, 0.0))

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
        term_pol = -self.a2 * (w_pol**2 / w**3)

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

        dw_dt_val = (self.r_s / tau_R) * (
            term_classical + term_bs + term_pol + term_dia + term_cd
        )
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
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate w(t) via RK4.

        All physics keyword arguments are forwarded to dw_dt unchanged.
        """
        t_start, t_end = t_span
        n_steps = int(np.ceil((t_end - t_start) / dt))
        t_arr = np.linspace(t_start, t_end, n_steps + 1)
        w_arr = np.zeros(n_steps + 1)
        w_arr[0] = max(w0, 1e-6)

        kw = dict(
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

            k1 = self.dw_dt(w_curr, **kw)
            k2 = self.dw_dt(max(w_curr + 0.5 * dt * k1, 1e-6), **kw)
            k3 = self.dw_dt(max(w_curr + 0.5 * dt * k2, 1e-6), **kw)
            k4 = self.dw_dt(max(w_curr + dt * k3, 1e-6), **kw)

            w_next = w_curr + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            w_arr[i + 1] = max(w_next, 1e-6)

        return t_arr, w_arr


class NTMController:
    """ECCD-based NTM controller: triggers at onset, deactivates at target."""

    def __init__(self, w_onset: float = 0.02, w_target: float = 0.005):
        self.w_onset = w_onset
        self.w_target = w_target
        self.active = False
        self.target_rho = 0.0
        self.eccd_power_request = 0.0

    def step(self, w: float, rho_rs: float, max_power: float = 20.0) -> float:
        """Return requested ECCD power [MW]; update controller state."""
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
