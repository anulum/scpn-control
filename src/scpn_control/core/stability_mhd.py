# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MHD Stability: Five-Criterion Suite
"""
MHD stability analysis suite — five criteria:

1. **Mercier** — interchange stability (D_M >= 0)
2. **Ballooning** — first-stability boundary (Connor-Hastie-Taylor)
3. **Kruskal-Shafranov** — external kink (q_edge > 1)
4. **Troyon** — normalised beta limit (beta_N < g)
5. **NTM** — neoclassical tearing mode seeding threshold

References
----------
- Freidberg, *Ideal MHD*, Cambridge (2014), Ch. 12
- Connor, Hastie & Taylor, Phys. Rev. Lett. 40:396 (1978)
- Kruskal & Schwarzschild, Proc. R. Soc. Lond. A 223:348 (1954)
- Troyon et al., Plasma Phys. Control. Fusion 26:209 (1984)
- La Haye, Phys. Plasmas 13:055501 (2006)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# ── Dataclasses ──────────────────────────────────────────────────────


@dataclass
class QProfile:
    """Safety-factor profile and derived quantities."""

    rho: NDArray[np.float64]
    q: NDArray[np.float64]
    shear: NDArray[np.float64]
    alpha_mhd: NDArray[np.float64]
    q_min: float
    q_min_rho: float
    q_edge: float


@dataclass
class MercierResult:
    """Mercier interchange stability result."""

    rho: NDArray[np.float64]
    D_M: NDArray[np.float64]
    stable: NDArray[np.bool_]
    first_unstable_rho: float | None


@dataclass
class BallooningResult:
    """First-stability ballooning boundary result."""

    rho: NDArray[np.float64]
    s: NDArray[np.float64]
    alpha: NDArray[np.float64]
    alpha_crit: NDArray[np.float64]
    stable: NDArray[np.bool_]
    margin: NDArray[np.float64]


@dataclass
class KruskalShafranovResult:
    """External kink stability result (Kruskal-Shafranov criterion)."""

    q_edge: float
    stable: bool  # True if q_edge > 1
    margin: float  # q_edge - 1


@dataclass
class TroyonResult:
    """Troyon normalised-beta-limit result."""

    beta_N: float  # Normalised beta [% m T / MA]
    beta_N_crit_nowall: float  # Critical beta_N without wall (g = 2.8)
    beta_N_crit_wall: float  # Critical beta_N with ideal wall (g = 3.5)
    stable_nowall: bool
    stable_wall: bool
    margin_nowall: float  # beta_N_crit_nowall - beta_N


@dataclass
class NTMResult:
    """Neoclassical tearing mode seeding analysis result."""

    rho: NDArray[np.float64]
    delta_prime: NDArray[np.float64]  # Classical stability index (< 0 = stable)
    j_bs_drive: NDArray[np.float64]  # Bootstrap current fraction drive
    w_marginal: NDArray[np.float64]  # Marginal island width [m]
    ntm_unstable: NDArray[np.bool_]
    most_unstable_rho: float | None


@dataclass
class StabilitySummary:
    """Combined result from all five MHD stability criteria."""

    mercier: MercierResult
    ballooning: BallooningResult
    kruskal_shafranov: KruskalShafranovResult
    troyon: TroyonResult | None  # None if beta_t not provided
    ntm: NTMResult | None  # None if j_bs not provided
    n_criteria_checked: int
    n_criteria_stable: int
    overall_stable: bool


def _require_finite_scalar(
    name: str,
    value: float,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    """Return a finite scalar after enforcing its physical domain."""
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _require_profile_array(
    name: str,
    values: NDArray[np.float64],
    reference_shape: tuple[int, ...] | None = None,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> NDArray[np.float64]:
    """Return a finite one-dimensional profile array in the requested domain."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"{name} must be a one-dimensional profile with at least two points")
    if reference_shape is not None and arr.shape != reference_shape:
        raise ValueError(f"{name} must match the q-profile grid shape")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(arr <= 0.0):
        raise ValueError(f"{name} must be positive everywhere")
    if nonnegative and np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative everywhere")
    return arr


def _require_normalised_radius(rho: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a strictly increasing normalised minor-radius grid."""
    arr = _require_profile_array("rho", rho)
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("rho must stay within the normalised interval [0, 1]")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError("rho must be strictly increasing")
    if not np.isclose(float(arr[0]), 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("rho must start at 0 for axis-to-edge MHD stability profiles")
    if not np.isclose(float(arr[-1]), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("rho must end at 1 for axis-to-edge MHD stability profiles")
    return arr


def _validate_q_profile(qp: QProfile) -> QProfile:
    """Validate safety-factor profile arrays before stability calculations."""
    rho = _require_normalised_radius(qp.rho)
    shape = rho.shape
    q = _require_profile_array("q", qp.q, shape, positive=True)
    _require_profile_array("shear", qp.shear, shape)
    _require_profile_array("alpha_mhd", qp.alpha_mhd, shape, nonnegative=True)
    q_min = _require_finite_scalar("q_min", qp.q_min, positive=True)
    q_min_rho = _require_finite_scalar("q_min_rho", qp.q_min_rho, nonnegative=True)
    q_edge = _require_finite_scalar("q_edge", qp.q_edge, positive=True)
    if q_min_rho > 1.0:
        raise ValueError("q_min_rho must stay within the normalised interval [0, 1]")
    q_min_idx = int(np.argmin(q))
    q_min_rho_expected = float(rho[q_min_idx])
    if not np.isclose(q_min, float(np.min(q)), rtol=1e-7, atol=1e-10):
        raise ValueError("q_min must match the q-profile minimum")
    if not np.isclose(q_min_rho, q_min_rho_expected, rtol=1e-7, atol=1e-10):
        raise ValueError("q_min_rho must match the q-profile minimum radius")
    if not np.isclose(q_edge, float(q[-1]), rtol=1e-7, atol=1e-10):
        raise ValueError("q_edge must match the last q-profile point")
    return qp


# ── Q-profile computation ───────────────────────────────────────────


def compute_q_profile(
    rho: NDArray[np.float64],
    ne: NDArray[np.float64],
    Ti: NDArray[np.float64],
    Te: NDArray[np.float64],
    R0: float,
    a: float,
    B0: float,
    Ip_MA: float,
    kappa: float = 1.0,
    delta: float = 0.0,
) -> QProfile:
    """Compute the safety-factor profile from a shape-aware approximation.

    Uses a parabolic current profile and the Uckan-style geometric correction
    for elongation (kappa) and triangularity (delta).

    Parameters
    ----------
    rho : array — normalised radius [0, 1]
    ne : array — electron density [10^19 m^-3]
    Ti, Te : array — ion/electron temperature [keV]
    R0 : float — major radius [m]
    a : float — minor radius [m]
    B0 : float — toroidal field on axis [T]
    Ip_MA : float — total plasma current [MA]
    kappa : float — elongation
    delta : float — triangularity

    Returns
    -------
    QProfile
    """
    rho = _require_normalised_radius(rho)
    shape = rho.shape
    ne = _require_profile_array("ne", ne, shape, positive=True)
    Ti = _require_profile_array("Ti", Ti, shape, nonnegative=True)
    Te = _require_profile_array("Te", Te, shape, nonnegative=True)
    R0 = _require_finite_scalar("R0", R0, positive=True)
    a = _require_finite_scalar("a", a, positive=True)
    B0 = _require_finite_scalar("B0", B0, positive=True)
    Ip_MA = _require_finite_scalar("Ip_MA", Ip_MA, positive=True)
    kappa = _require_finite_scalar("kappa", kappa, positive=True)
    delta = _require_finite_scalar("delta", delta)
    if abs(delta) >= 1.0:
        raise ValueError("delta must remain inside the physical triangularity interval (-1, 1)")
    if a >= R0:
        raise ValueError("a must be smaller than R0 for tokamak ordering")

    mu0 = 4.0 * np.pi * 1e-7
    Ip = Ip_MA * 1e6  # MA -> A
    epsilon = a / R0

    # Shape correction factor (Uckan formula / ITER-scaling proxy)
    # f_shape accounts for the increased path length in a D-shaped plasma.
    f_shape = (1.0 + kappa**2 * (1.0 + 2.0 * delta**2 - 1.2 * delta**3)) / 2.0
    # Aspect ratio correction
    f_aspect = (1.17 - 0.65 * epsilon) / (1.0 - epsilon**2)
    f_total = f_shape * f_aspect

    # Parabolic current profile: j(rho) ∝ (1 - rho^2)
    rho_safe = np.maximum(rho, 1e-10)
    I_enc = Ip * (2.0 * rho_safe**2 - rho_safe**4)

    # q_cyl(rho) = rho * a * B0 / (R0 * B_theta)
    # B_theta(rho) = mu0 * I_enc / (2 * pi * rho * a)
    B_theta = mu0 * I_enc / (2.0 * np.pi * rho_safe * a)
    B_theta = np.maximum(B_theta, 1e-12)
    q_cyl = rho_safe * a * B0 / (R0 * B_theta)

    # Final shape-aware q-profile
    q = q_cyl * f_total

    # Fix axis: q0 = f_total * (pi * a^2 * B0) / (mu0 * R0 * Ip)
    q0 = f_total * np.pi * a**2 * B0 / (mu0 * R0 * Ip)
    q[0] = q0

    # Magnetic shear: s = (rho/q) * dq/drho
    dq = np.gradient(q, rho_safe)
    shear = (rho_safe / q) * dq
    shear[0] = 0.0  # Zero shear at axis by symmetry

    # Normalised pressure gradient (alpha_MHD)
    # alpha = -2 mu0 R0 q^2 / B0^2 * dp/dr
    # p = n_e * (Ti + Te) in keV * 10^19 m^-3 => convert to Pa
    # 2019 SI exact: 1 eV = 1.602176634e-19 J; 1 keV = 1e3 eV
    e_keV_to_J = 1.602176634e-16
    p_Pa = ne * 1e19 * (Ti + Te) * e_keV_to_J  # pressure in Pa
    dp_drho = np.gradient(p_Pa, rho_safe)  # dp/d(rho)
    dp_dr = dp_drho / a  # dp/dr [Pa/m]

    alpha_mhd = -2.0 * mu0 * R0 * q**2 / (B0**2) * dp_dr
    alpha_mhd = np.maximum(alpha_mhd, 0.0)  # Physical: alpha >= 0

    q_min_idx = int(np.argmin(q))
    q_min = float(q[q_min_idx])
    q_min_rho = float(rho[q_min_idx])
    q_edge_val = float(q[-1])

    return QProfile(
        rho=rho,
        q=q,
        shear=shear,
        alpha_mhd=alpha_mhd,
        q_min=q_min,
        q_min_rho=q_min_rho,
        q_edge=q_edge_val,
    )


# ── Mercier criterion ────────────────────────────────────────────────


def mercier_stability(qp: QProfile) -> MercierResult:
    """Evaluate the Mercier interchange stability criterion.

    The Mercier index is (Freidberg, Ch. 12):
        D_M = s*(s - 1) + alpha*(1 - s/2)

    Stable where D_M >= 0.

    Parameters
    ----------
    qp : QProfile

    Returns
    -------
    MercierResult
    """
    qp = _validate_q_profile(qp)
    s = qp.shear
    alpha = qp.alpha_mhd

    D_M = s * (s - 1.0) + alpha * (1.0 - s / 2.0)

    stable = D_M >= 0.0

    # Find first unstable location (skip axis where s=0 makes D_M=0)
    first_unstable_rho: float | None = None
    for i in range(2, len(qp.rho)):
        if not stable[i]:
            first_unstable_rho = float(qp.rho[i])
            break

    return MercierResult(
        rho=qp.rho,
        D_M=D_M.astype(np.float64),
        stable=stable,
        first_unstable_rho=first_unstable_rho,
    )


# ── Ballooning stability ────────────────────────────────────────────


def ballooning_stability(qp: QProfile) -> BallooningResult:
    """Evaluate the first ballooning stability boundary.

    The critical normalised pressure gradient (Connor-Hastie-Taylor 1978):
        alpha_crit(s) = s*(1 - s/2)   for s < 1
                      = 0.6*s          for s >= 1

    Stable where alpha <= alpha_crit.

    Parameters
    ----------
    qp : QProfile

    Returns
    -------
    BallooningResult
    """
    qp = _validate_q_profile(qp)
    s = qp.shear
    alpha = qp.alpha_mhd

    # Connor, Hastie & Taylor, Phys. Rev. Lett. 40:396 (1978), Eq. 8:
    # low-shear: alpha_crit = s(1 - s/2); high-shear: alpha_crit ≈ 0.6 s
    alpha_crit = np.where(s < 1.0, s * (1.0 - s / 2.0), 0.6 * s)
    alpha_crit = np.maximum(alpha_crit, 0.0)

    stable = alpha <= alpha_crit
    margin = alpha_crit - alpha

    return BallooningResult(
        rho=qp.rho,
        s=s,
        alpha=alpha,
        alpha_crit=alpha_crit,
        stable=stable,
        margin=margin,
    )


# ── Kruskal-Shafranov criterion ────────────────────────────────────


def kruskal_shafranov_stability(qp: QProfile) -> KruskalShafranovResult:
    """Evaluate the Kruskal-Shafranov external kink stability criterion.

    The plasma is stable against the m=1/n=1 external kink mode when the
    edge safety factor satisfies q(edge) > 1.  Physically, when q_edge < 1
    the magnetic field-line pitch allows helical perturbations that are
    not stabilised by line-bending, driving a global kink.

    Parameters
    ----------
    qp : QProfile

    Returns
    -------
    KruskalShafranovResult

    References
    ----------
    Kruskal & Schwarzschild, Proc. R. Soc. Lond. A 223:348 (1954)
    Shafranov, Sov. Phys. Tech. Phys. 15:175 (1970)
    """
    qp = _validate_q_profile(qp)
    stable = qp.q_edge > 1.0
    margin = qp.q_edge - 1.0
    return KruskalShafranovResult(
        q_edge=qp.q_edge,
        stable=stable,
        margin=margin,
    )


# ── Troyon beta limit ──────────────────────────────────────────────


def troyon_beta_limit(
    beta_t: float,
    Ip_MA: float,
    a: float,
    B0: float,
    g_nowall: float = 2.8,
    g_wall: float = 3.5,
) -> TroyonResult:
    r"""Evaluate the Troyon normalised-beta limit.

    The normalised beta is defined as:

    .. math::
        \beta_N = \frac{\beta_t}{I_N}  \quad\text{where}\quad
        I_N = \frac{I_p\,[\text{MA}]}{a\,[\text{m}]\,B_0\,[\text{T}]}

    The factor of 100 converts fractional *beta_t* into the conventional
    percent-based normalisation (units: % m T / MA).

    Stability requires beta_N < g, where g ~ 2.8 (no-wall) or g ~ 3.5
    (ideal-wall).

    Parameters
    ----------
    beta_t : float — total toroidal beta (dimensionless, e.g. 0.025)
    Ip_MA : float — plasma current [MA]
    a : float — minor radius [m]
    B0 : float — toroidal magnetic field on axis [T]
    g_nowall : float — Troyon coefficient without wall (default 2.8)
    g_wall : float — Troyon coefficient with ideal wall (default 3.5)

    Returns
    -------
    TroyonResult

    References
    ----------
    Troyon et al., Plasma Phys. Control. Fusion 26:209 (1984)
    """
    beta_t = _require_finite_scalar("beta_t", beta_t, nonnegative=True)
    Ip_MA = _require_finite_scalar("Ip_MA", Ip_MA, positive=True)
    a = _require_finite_scalar("a", a, positive=True)
    B0 = _require_finite_scalar("B0", B0, positive=True)
    g_nowall = _require_finite_scalar("g_nowall", g_nowall, positive=True)
    g_wall = _require_finite_scalar("g_wall", g_wall, positive=True)
    if g_wall < g_nowall:
        raise ValueError("g_wall must not be below g_nowall")

    I_N = Ip_MA / (a * B0)  # normalised current [MA / (m T)]
    beta_N = 100.0 * beta_t / I_N  # [% m T / MA]

    beta_N_crit_nw = g_nowall
    beta_N_crit_w = g_wall

    stable_nw = beta_N < beta_N_crit_nw
    stable_w = beta_N < beta_N_crit_w
    margin_nw = beta_N_crit_nw - beta_N

    return TroyonResult(
        beta_N=beta_N,
        beta_N_crit_nowall=beta_N_crit_nw,
        beta_N_crit_wall=beta_N_crit_w,
        stable_nowall=stable_nw,
        stable_wall=stable_w,
        margin_nowall=margin_nw,
    )


# ── NTM seeding threshold ──────────────────────────────────────────


def ntm_stability(
    qp: QProfile,
    j_bs: NDArray[np.float64],
    j_total: NDArray[np.float64],
    a: float,
    r_s_delta_prime: float = -2.0,
) -> NTMResult:
    r"""Neoclassical tearing mode (NTM) seeding analysis from profile drives.

    The modified Rutherford equation for island width *w* is:

    .. math::
        \tau_R\,\frac{dw}{dt}
            = r_s\,\Delta'(w)
            + \frac{j_\text{bs}}{j_\phi}\,\frac{a}{w}

    The first term is classically stabilising when :math:`r_s \Delta' < 0`.
    The second term is the bootstrap-current drive that destabilises the
    island once seeded. In this reduced stability scan, the bootstrap drive is
    weighted by local magnetic shear and pressure-gradient proxies from
    ``QProfile``:

    .. math::
        D_\mathrm{eff} = \frac{j_\mathrm{bs}}{j_\phi}
            \left(1 + C_\alpha \frac{\alpha}{1+\alpha}\right)
            \frac{1}{1 + C_s |s|}

    An NTM is potentially unstable at a given radius when the
    bootstrap drive exceeds the classical stabilisation.  The marginal
    island width (below which the island shrinks) is:

    .. math::
        w_\text{marg} = -\frac{j_\text{bs} / j_\phi}{r_s \Delta'}\,a

    NTM instability occurs where :math:`w_\text{marg} > 0` (positive
    bootstrap drive with negative classical :math:`\Delta'`).

    Parameters
    ----------
    qp : QProfile
    j_bs : array — bootstrap current density [A/m^2]
    j_total : array — total current density [A/m^2]
    a : float — minor radius [m]
    r_s_delta_prime : float — classical tearing stability index
        (negative = classically stable, positive = classically unstable,
        zero is a singular marginal boundary and is rejected)

    Returns
    -------
    NTMResult

    References
    ----------
    La Haye, Phys. Plasmas 13:055501 (2006)
    Sauter et al., Phys. Plasmas 4:1654 (1997)
    """
    qp = _validate_q_profile(qp)
    shape = qp.rho.shape
    j_bs = _require_profile_array("j_bs", j_bs, shape, nonnegative=True)
    j_total = _require_profile_array("j_total", j_total, shape, positive=True)
    a = _require_finite_scalar("a", a, positive=True)
    r_s_delta_prime = _require_finite_scalar("r_s_delta_prime", r_s_delta_prime)
    if abs(r_s_delta_prime) <= 1e-10:
        raise ValueError("r_s_delta_prime must be non-zero for marginal island-width evaluation")

    j_bs_frac = j_bs / j_total  # bootstrap fraction

    # Classical Δ' baseline across profile (negative = classically stable).
    delta_prime = np.full_like(qp.rho, r_s_delta_prime)

    # Profile-weighted bootstrap drive:
    # - Lower magnetic shear reduces field-line bending stabilisation.
    # - Higher alpha_MHD increases pressure-driven tearing tendency.
    c_alpha = 0.75
    c_shear = 1.0
    alpha_weight = 1.0 + c_alpha * (qp.alpha_mhd / (1.0 + qp.alpha_mhd))
    shear_weight = 1.0 / (1.0 + c_shear * np.abs(qp.shear))
    j_bs_drive = j_bs_frac * alpha_weight * shear_weight

    # Marginal island width: w_marg = -(j_bs/j_phi) * a / (r_s * Delta')
    # Only meaningful where delta_prime < 0 (classically stable baseline).
    w_marginal = -j_bs_drive * a / delta_prime
    w_marginal = np.maximum(w_marginal, 0.0)  # physical: width >= 0

    # NTM unstable where bootstrap drives a positive marginal width
    ntm_unstable = (w_marginal > 0.0) & (j_bs_drive > 0.0) & (delta_prime < 0.0)

    most_unstable_rho: float | None = None
    if np.any(ntm_unstable):
        # Pick radius with largest marginal island width
        idx = int(np.argmax(np.where(ntm_unstable, w_marginal, 0.0)))
        most_unstable_rho = float(qp.rho[idx])

    return NTMResult(
        rho=qp.rho,
        delta_prime=delta_prime,
        j_bs_drive=j_bs_drive,
        w_marginal=w_marginal,
        ntm_unstable=ntm_unstable,
        most_unstable_rho=most_unstable_rho,
    )


# ── Full stability check (all 5 criteria) ──────────────────────────


def run_full_stability_check(
    qp: QProfile,
    beta_t: float | None = None,
    Ip_MA: float | None = None,
    a: float | None = None,
    B0: float | None = None,
    j_bs: NDArray[np.float64] | None = None,
    j_total: NDArray[np.float64] | None = None,
) -> StabilitySummary:
    """Run all available MHD stability criteria and return a summary.

    Mercier, ballooning, and Kruskal-Shafranov are always evaluated.
    Troyon requires ``beta_t``, ``Ip_MA``, ``a``, and ``B0``.
    NTM requires ``j_bs``, ``j_total``, and ``a``.

    Parameters
    ----------
    qp : QProfile — pre-computed safety-factor profile
    beta_t : float, optional — total toroidal beta (dimensionless)
    Ip_MA : float, optional — plasma current [MA]
    a : float, optional — minor radius [m]
    B0 : float, optional — toroidal field on axis [T]
    j_bs : array, optional — bootstrap current density [A/m^2]
    j_total : array, optional — total current density [A/m^2]

    Returns
    -------
    StabilitySummary
    """
    qp = _validate_q_profile(qp)
    troyon_requested = beta_t is not None or Ip_MA is not None or B0 is not None
    if troyon_requested and not all(arg is not None for arg in (beta_t, Ip_MA, a, B0)):
        raise ValueError("Troyon evaluation requires beta_t, Ip_MA, a, and B0 together")
    ntm_requested = j_bs is not None or j_total is not None
    if ntm_requested and not all(arg is not None for arg in (j_bs, j_total, a)):
        raise ValueError("NTM evaluation requires j_bs, j_total, and a together")

    # --- Always-on criteria ---
    mr = mercier_stability(qp)
    br = ballooning_stability(qp)
    ks = kruskal_shafranov_stability(qp)

    n_checked = 3
    n_stable = 0

    # Mercier: stable if no unstable point found (past the axis)
    mercier_ok = mr.first_unstable_rho is None
    if mercier_ok:
        n_stable += 1

    # Ballooning: stable if all points are stable
    ballooning_ok = bool(np.all(br.stable))
    if ballooning_ok:
        n_stable += 1

    # KS: direct boolean
    if ks.stable:
        n_stable += 1

    # --- Troyon (optional) ---
    troyon_result: TroyonResult | None = None
    if beta_t is not None and Ip_MA is not None and a is not None and B0 is not None:
        troyon_result = troyon_beta_limit(beta_t, Ip_MA, a, B0)
        n_checked += 1
        if troyon_result.stable_nowall:
            n_stable += 1

    # --- NTM (optional) ---
    ntm_result: NTMResult | None = None
    if j_bs is not None and j_total is not None and a is not None:
        ntm_result = ntm_stability(qp, j_bs, j_total, a)
        n_checked += 1
        if not np.any(ntm_result.ntm_unstable):
            n_stable += 1

    overall = n_stable == n_checked

    return StabilitySummary(
        mercier=mr,
        ballooning=br,
        kruskal_shafranov=ks,
        troyon=troyon_result,
        ntm=ntm_result,
        n_criteria_checked=n_checked,
        n_criteria_stable=n_stable,
        overall_stable=overall,
    )
