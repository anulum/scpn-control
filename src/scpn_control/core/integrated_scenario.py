# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851  Contact: protoscience@anulum.li
"""
Top-level time-dependent tokamak shot simulator.

Advances the coupled (psi, Te, Ti, ne) state on a 1D rho grid using
Strang operator splitting (Strang 1968, SIAM J. Numer. Anal. 5, 506):

    a) half-step transport diffusion
    b) full-step sources  (heating, CD, radiation)
    c) full-step MHD events (sawtooth, NTM)
    d) half-step transport diffusion

Transport equation (Wesson 2011, "Tokamaks" 4th ed., Ch. 14):

    (3/2) n d(T)/dt = (1/r) d/dr [ r (chi_neo + chi_anom) n dT/dr ] + S_heat

Current diffusion uses full Sauter bootstrap (Sauter et al. 1999,
Phys. Plasmas 6, 2834).  NTM island width follows the Modified Rutherford
Equation (La Haye 2006, Phys. Plasmas 13, 055501).

Key references
--------------
Strang 1968    : G. Strang, SIAM J. Numer. Anal. 5, 506 (1968).
Spitzer 1962   : L. Spitzer, "Physics of Fully Ionized Gases", Interscience (1962).
Wesson 2011    : J. Wesson, "Tokamaks", 4th ed., Oxford (2011), Ch. 14.
Jardin 2010    : S. Jardin, "Computational Methods in Plasma Physics",
                  CRC Press (2010).
Sauter 1999    : O. Sauter et al., Phys. Plasmas 6, 2834 (1999).
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from scpn_control.core.current_diffusion import (
    CurrentDiffusionSolver,
    psi_from_q,
    q_from_psi,
)
from scpn_control.core.current_drive import CurrentDriveMix, ECCDSource, NBISource
from scpn_control.core.elm_model import ELMCrashModel, ELMCycler, PeelingBallooningBoundary
from scpn_control.core.integrated_transport_solver import (
    TransportSolver,
    chang_hinton_chi_profile,
)
from scpn_control.core.lh_transition import MartinThreshold
from scpn_control.core.neoclassical import sauter_bootstrap
from scpn_control.core.ntm_dynamics import NTMController, NTMIslandDynamics, find_rational_surfaces
from scpn_control.core.sawtooth import SawtoothCycler
from scpn_control.core.sol_model import TwoPointSOL
from scpn_control.core.stability_mhd import (
    QProfile,
    ballooning_stability,
    troyon_beta_limit,
)
from scpn_control.core.tearing_mode_coupling import SawtoothNTMSeeding

# ── Physical constants (CODATA 2018) ─────────────────────────────────────────
_E_CHARGE: float = 1.602176634e-19  # C
_LN_LAMBDA: float = 17.0  # Coulomb logarithm, Wesson 2011, Ch. 14
_MU_0: float = 4.0 * np.pi * 1e-7  # H/m

# Spitzer 1962, "Physics of Fully Ionized Gases" — resistivity coefficient
# η = SPITZER_COEFF * Z_eff * ln_Λ / T_e^1.5   [Ω·m],  T_e in keV
_SPITZER_COEFF: float = 1.65e-9

# Gyro-Bohm anomalous diffusivity coefficient, ITPA Transport DB,
# Nucl. Fusion 39, 2175 (1999), Table II — c_gB = 0.1
_C_GB: float = 0.1

# Minimum diffusivity floor [m^2/s]: prevents singular matrices
_CHI_FLOOR: float = 0.01

# NTM seeding threshold [m]: islands below this do not grow, La Haye 2006, Eq. 6
_W_SEED: float = 5e-3

# Island width above which local T/n flattening is applied [m]
_W_FLAT: float = 1e-2


@dataclass
class ScenarioConfig:
    # Geometry
    R0: float
    a: float
    B0: float
    kappa: float
    delta: float

    # Actuators
    Ip_MA: float
    P_aux_MW: float

    # CD parameters
    P_eccd_MW: float = 0.0
    rho_eccd: float = 0.5
    P_nbi_MW: float = 0.0
    E_nbi_keV: float = 100.0

    # Duration
    t_start: float = 0.0
    t_end: float = 10.0
    dt: float = 0.1

    transport_model: str = "gyro_bohm"

    # Flags
    include_sawteeth: bool = True
    include_ntm: bool = True
    include_sol: bool = True
    include_elm: bool = True
    include_stability: bool = True


@dataclass
class ScenarioState:
    time: float
    rho: np.ndarray
    Te: np.ndarray
    Ti: np.ndarray
    ne: np.ndarray
    q: np.ndarray
    psi: np.ndarray
    j_total: np.ndarray
    j_bs: np.ndarray
    j_cd: np.ndarray

    Ip_MA: float
    beta_N: float
    tau_E: float
    P_loss: float
    W_thermal: float
    li: float

    ballooning_stable: bool
    troyon_stable: bool
    ntm_island_widths: dict[str, float]

    T_target: float
    q_peak: float
    detached: bool

    last_crash_time: float
    n_crashes: int


def iter_baseline_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        R0=6.2,
        a=2.0,
        B0=5.3,
        kappa=1.7,
        delta=0.33,
        Ip_MA=15.0,
        P_aux_MW=50.0,
        P_eccd_MW=17.0,
        rho_eccd=0.5,
        P_nbi_MW=33.0,
        E_nbi_keV=1000.0,
        t_end=100.0,
        dt=1.0,
    )


def iter_hybrid_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        R0=6.2,
        a=2.0,
        B0=5.3,
        kappa=1.7,
        delta=0.33,
        Ip_MA=12.0,
        P_aux_MW=50.0,
        t_end=100.0,
        dt=1.0,
    )


def nstx_u_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        R0=0.93,
        a=0.58,
        B0=1.0,
        kappa=2.0,
        delta=0.4,
        Ip_MA=1.0,
        P_aux_MW=10.0,
        t_end=2.0,
        dt=0.01,
    )


def _spitzer_resistivity(Te_keV: np.ndarray, Z_eff: float) -> np.ndarray:
    """Spitzer parallel resistivity profile.

    η = SPITZER_COEFF * Z_eff * ln_Λ / T_e^1.5

    Spitzer 1962, "Physics of Fully Ionized Gases", Interscience, Ch. 5.
    Returns Ω·m; T_e in keV.
    """
    return np.asarray(_SPITZER_COEFF * Z_eff * _LN_LAMBDA / np.maximum(Te_keV, 0.01) ** 1.5)


def _gyro_bohm_chi(
    rho: np.ndarray,
    Te: np.ndarray,
    Ti: np.ndarray,
    ne: np.ndarray,
    q: np.ndarray,
    a: float,
    B0: float,
) -> np.ndarray:
    """Gyro-Bohm anomalous electron/ion thermal diffusivity.

    chi_gB = c_gB * rho_i * v_ti * (rho_i / a)^2 * q^2

    ITPA Transport DB, Nucl. Fusion 39, 2175 (1999), Eq. 1.
    c_gB = 0.1 calibrated to L-mode database.
    """
    m_p: float = 1.67262192369e-27  # kg, CODATA 2018
    m_i = 2.0 * m_p  # deuterium
    T_i_J = np.maximum(Ti, 0.01) * 1.602176634e-16  # keV → J
    v_ti = np.sqrt(2.0 * T_i_J / m_i)
    rho_i = m_i * v_ti / (_E_CHARGE * B0)

    # Wesson 2011, Ch. 7 — chi_gB proportional to rho_i * v_ti * (rho_i/a)^2
    chi = _C_GB * rho_i * v_ti * (rho_i / a) ** 2 * np.maximum(q, 0.5) ** 2
    return np.asarray(np.maximum(chi, _CHI_FLOOR))


def _diffusion_step(
    T: np.ndarray,
    rho: np.ndarray,
    chi: np.ndarray,
    ne: np.ndarray,
    S: np.ndarray,
    dt: float,
    a: float,
) -> np.ndarray:
    """Explicit cylindrical thermal diffusion step.

    Solves one half-step of:

        (3/2) n dT/dt = (1/r) d/dr [ r chi n dT/dr ] + S

    where r = rho * a.  Wesson 2011, Ch. 14, Eq. (14.5.1).
    Jardin 2010, "Computational Methods in Plasma Physics", Ch. 7.

    Uses forward Euler for simplicity; dt must satisfy the parabolic CFL:
        dt <= drho^2 / (2 * chi_max)
    The caller (step()) enforces sub-stepping when needed.
    """
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


class IntegratedScenarioSimulator:
    """Time-dependent tokamak shot simulator.

    Operator splitting sequence per step() (Strang 1968):
        a) half-step transport diffusion
        b) full-step sources
        c) full-step MHD events
        d) half-step transport diffusion

    State variables carried on 1D rho grid:  Te, Ti, ne, psi (→ q, j).
    """

    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.time = config.t_start

        self.nr = 50
        self.rho = np.linspace(0, 1, self.nr)

        # Current drive sources
        self.cd_mix = CurrentDriveMix(a=self.config.a)
        self.eccd: ECCDSource | None = None
        if self.config.P_eccd_MW > 0:
            self.eccd = ECCDSource(self.config.P_eccd_MW, self.config.rho_eccd, 0.05)
            self.cd_mix.add_source(self.eccd)

        if self.config.P_nbi_MW > 0:
            self.nbi = NBISource(self.config.P_nbi_MW, self.config.E_nbi_keV, 0.2)
            self.cd_mix.add_source(self.nbi)

        self.cd_solver = CurrentDiffusionSolver(self.rho, self.config.R0, self.config.a, self.config.B0)

        self.sawtooth = SawtoothCycler(self.rho, self.config.R0, self.config.a)
        self.n_crashes = 0
        self.last_crash_time = 0.0

        self.sol = TwoPointSOL(
            self.config.R0,
            self.config.a,
            q95=3.0,
            B_pol=0.5,
            kappa=self.config.kappa,
        )

        # NTM state: rational surface key → (NTMIslandDynamics, island_width)
        self.ntm_widths: dict[str, float] = {}
        self._ntm_islands: dict[str, NTMIslandDynamics] = {}
        self.ntm_controller = NTMController()
        self.ntm_seeding = SawtoothNTMSeeding(self.sawtooth)

        # ELM cycler
        pb_boundary = PeelingBallooningBoundary(
            q95=3.0,
            kappa=self.config.kappa,
            delta=self.config.delta,
            a=self.config.a,
            R0=self.config.R0,
        )
        self.elm_cycler = ELMCycler(pb_boundary, ELMCrashModel())

        # L-H transition threshold
        self.lh_threshold = MartinThreshold()

    # ── initialisation ────────────────────────────────────────────────────────

    def _setup_transport_solver(self) -> TransportSolver:
        cfg_dict = {
            "reactor_name": "IntegratedScenario",
            "grid_resolution": [33, 33],
            "tokamak": {
                "R0": self.config.R0,
                "a": self.config.a,
                "B0": self.config.B0,
            },
            "dimensions": {
                "R_min": self.config.R0 - self.config.a - 0.5,
                "R_max": self.config.R0 + self.config.a + 0.5,
                "Z_min": -self.config.a * self.config.kappa - 0.5,
                "Z_max": self.config.a * self.config.kappa + 0.5,
                "nR": 33,
                "nZ": 33,
            },
            "coils": [],
            "grid": {"nr": self.nr},
            "physics": {"transport_model": "gyro_bohm"},
            "control": {"P_aux": self.config.P_aux_MW},
        }
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
            json.dump(cfg_dict, f)
            temp_path = f.name

        solver = TransportSolver(temp_path, nr=self.nr, transport_model=self.config.transport_model)
        solver.Te = np.ones(self.nr) * 1.0  # keV
        solver.Ti = np.ones(self.nr) * 1.0
        solver.ne = np.ones(self.nr) * 5.0  # 10^19 m^-3
        return solver

    def initialize(self, profiles: dict | None = None) -> ScenarioState:
        self.ts_solver = self._setup_transport_solver()
        if profiles:
            if "Te" in profiles:
                self.ts_solver.Te = profiles["Te"]
            if "Ti" in profiles:
                self.ts_solver.Ti = profiles["Ti"]
            if "ne" in profiles:
                self.ts_solver.ne = profiles["ne"]
            if "psi" in profiles:
                self.cd_solver.psi = profiles["psi"]
        return self._build_state()

    # ── internal physics helpers ──────────────────────────────────────────────

    def _chi_total(self, q: np.ndarray) -> np.ndarray:
        """Combined neoclassical + gyro-Bohm anomalous diffusivity.

        Wesson 2011, Ch. 14 — total chi = chi_neo + chi_anom.
        """
        chi_neo = chang_hinton_chi_profile(
            self.rho,
            self.ts_solver.Ti,
            self.ts_solver.ne,
            q,
            self.config.R0,
            self.config.a,
            self.config.B0,
        )
        chi_anom = _gyro_bohm_chi(
            self.rho,
            self.ts_solver.Te,
            self.ts_solver.Ti,
            self.ts_solver.ne,
            q,
            self.config.a,
            self.config.B0,
        )
        return np.asarray(chi_neo + chi_anom)

    def _source_density(self, S_heat_W_m3: np.ndarray) -> np.ndarray:
        """Convert volumetric power density [W/m^3] to keV/s / (10^19 m^-3)."""
        n_si = np.maximum(self.ts_solver.ne, 0.01) * 1e19
        return np.asarray(S_heat_W_m3 / (_E_CHARGE * 1e3 * n_si))

    def _ohmic_power_density(self, q_prof: np.ndarray) -> np.ndarray:
        """Ohmic heating power density.

        P_ohm = η_Spitzer * j_ohm^2  where  j_ohm = E_loop / (η * (1 - f_t))
        and E_loop is set by j_ohm = j_total - j_bs - j_cd.

        For each cell we use η_Spitzer (Spitzer 1962) and reconstruct the
        ohmic contribution from the total current profile.
        Wesson 2011, Ch. 14, Eq. (14.2.6).
        """
        eta = _spitzer_resistivity(self.ts_solver.Te, Z_eff=1.5)
        # Trapped-fraction correction for conductivity: σ_∥ → σ_∥ * (1 - f_t)
        # Sauter, Angioni & Lin-Liu 1999, Phys. Plasmas 6, 2834, Eq. 4.
        f_t = np.zeros(self.nr)
        for i in range(self.nr):
            eps = self.rho[i] * self.config.a / self.config.R0
            if eps > 1e-6:
                f_t[i] = 1.0 - (1.0 - eps) ** 2 / (np.sqrt(1.0 - eps**2) * (1.0 + 1.46 * np.sqrt(eps)))

        # E_loop ≈ μ_0 * R0 * V_loop / (2π), approximate with η·j_total
        # j_total reconstructed from psi gradient (Jardin 2010, Ch. 7)
        j_total = self._j_total_from_psi(q_prof)
        P_ohm = eta * j_total**2 * (1.0 - f_t)
        return np.asarray(np.maximum(P_ohm, 0.0))

    def _j_total_from_psi(self, q_prof: np.ndarray) -> np.ndarray:
        """Approximate toroidal current density from q profile.

        j_tor ≈ B0 * a / (μ_0 * R0) * d/dr(1/q) * (1/r)
        Wesson 2011, Ch. 3, Eq. (3.5.5).
        """
        inv_q = 1.0 / np.maximum(q_prof, 0.5)
        # d(1/q)/dr in physical units: r = rho * a
        dinvq_drho = np.gradient(inv_q, self.rho)
        r = np.maximum(self.rho, 1e-4) * self.config.a
        j_tor = (self.config.B0 * self.config.a / (_MU_0 * self.config.R0)) * dinvq_drho / r
        return np.asarray(j_tor)

    def _aux_source_profile(self, q_prof: np.ndarray) -> np.ndarray:
        """Gaussian auxiliary heating profile [W/m^3].

        Peaked at rho = 0.3 with sigma = 0.15 for NBI-like deposition.
        Jardin 2010, Ch. 9 — auxiliary power modelling.
        """
        if self.config.P_aux_MW <= 0:
            return np.zeros(self.nr)

        # Deposition profile: Gaussian centred at rho=0.3
        rho_aux: float = 0.3
        sigma_aux: float = 0.15
        profile = np.exp(-((self.rho - rho_aux) ** 2) / (2.0 * sigma_aux**2))
        profile /= max(trapezoid(profile, self.rho), 1e-12)

        # Normalise to total power: P = ∫ S dV,  dV ≈ 2π² R0 a² κ * 2ρ dρ
        # We work per unit normalised volume: S_tot [W] = P_aux * 1e6
        vol_norm = 2.0 * np.pi**2 * self.config.R0 * self.config.a**2 * self.config.kappa
        return np.asarray(
            profile * self.config.P_aux_MW * 1e6 / (vol_norm * trapezoid(2.0 * self.rho * profile, self.rho) + 1e-12)
        )

    def _transport_step(self, dt: float, q_prof: np.ndarray) -> None:
        """Explicit cylindrical diffusion for Te and Ti.

        dT/dt = (1/r) d/dr [ r chi n dT/dr ] / n  + S/n
        Wesson 2011, "Tokamaks" 4th ed., Ch. 14, Eq. (14.5.1).

        Sub-steps to satisfy parabolic CFL: dt_sub ≤ drho^2 / (2 chi_max).
        """
        chi = self._chi_total(q_prof)

        # CFL sub-stepping, Jardin 2010, Ch. 7
        drho = self.rho[1] - self.rho[0]
        chi_max = float(np.max(chi))
        dt_cfl = 0.5 * (drho * self.config.a) ** 2 / max(chi_max, 1e-10)
        n_sub = max(1, int(np.ceil(dt / dt_cfl)))
        dt_sub = dt / n_sub

        S_heat = self._aux_source_profile(q_prof)
        S_ohm = self._ohmic_power_density(q_prof)
        S_total = S_heat + S_ohm  # W/m^3

        # Source in keV/s units for the diffusion step
        S_Te = self._source_density(S_total * 0.5)  # split equally e/i
        S_Ti = self._source_density(S_total * 0.5)

        for _ in range(n_sub):
            self.ts_solver.Te = _diffusion_step(
                self.ts_solver.Te,
                self.rho,
                chi,
                self.ts_solver.ne,
                S_Te,
                dt_sub,
                self.config.a,
            )
            self.ts_solver.Ti = _diffusion_step(
                self.ts_solver.Ti,
                self.rho,
                chi,
                self.ts_solver.ne,
                S_Ti,
                dt_sub,
                self.config.a,
            )

    def _ntm_step(self, dt: float, q_prof: np.ndarray) -> None:
        """NTM island dynamics via the Modified Rutherford Equation.

        La Haye 2006, Phys. Plasmas 13, 055501 — full MRE.
        Rutherford 1973, Phys. Fluids 16, 1903 — original equation.

        For each rational surface with q = m/n:
        1. Compute local j_bs from Sauter model.
        2. Evolve island width w via NTMIslandDynamics.evolve().
        3. If w > _W_FLAT, flatten Te/Ti within ±w of the surface.
        """
        surfaces = find_rational_surfaces(q_prof, self.rho, self.config.a)
        j_bs_profile = sauter_bootstrap(
            self.rho,
            self.ts_solver.Te,
            self.ts_solver.Ti,
            self.ts_solver.ne,
            q_prof,
            self.config.R0,
            self.config.a,
            self.config.B0,
        )
        j_cd_profile = self.cd_mix.total_j_cd(self.rho, self.ts_solver.ne, self.ts_solver.Te, self.ts_solver.Ti)

        for surf in surfaces:
            key = f"{surf.m}/{surf.n}"
            w0 = self.ntm_widths.get(key, _W_SEED)

            eta_s = float(
                _spitzer_resistivity(
                    np.array([self.ts_solver.Te[min(int(surf.rho * self.nr), self.nr - 1)]]),
                    Z_eff=1.5,
                )[0]
            )

            # Interpolate j_bs and j_cd to the rational surface
            i_s = int(np.clip(surf.rho * (self.nr - 1), 0, self.nr - 2))
            j_bs_s = float(j_bs_profile[i_s])
            j_cd_s = float(j_cd_profile[i_s])
            j_tor_s = float(abs(self._j_total_from_psi(q_prof)[i_s]))
            j_phi = max(j_tor_s, 1e3)  # A/m^2

            # Pressure gradient at surface: dp/dr ≈ (d/dr)(n_e * (T_e + T_i)) [Pa/m]
            n_si = self.ts_solver.ne * 1e19
            T_total_J = (self.ts_solver.Te + self.ts_solver.Ti) * 1.602176634e-16
            p = n_si * T_total_J
            dp_dr = float(np.gradient(p, self.rho * self.config.a)[i_s])
            B_pol = self.config.B0 * (surf.rho * self.config.a / self.config.R0) / max(q_prof[i_s], 0.1)

            # Bootstrap controller NTM power request
            self.ntm_controller.step(w0, surf.rho)

            if key not in self._ntm_islands:
                self._ntm_islands[key] = NTMIslandDynamics(
                    r_s=surf.r_s,
                    m=surf.m,
                    n=surf.n,
                    a=self.config.a,
                    R0=self.config.R0,
                    B0=self.config.B0,
                    s_hat=surf.shear,
                    q_s=surf.q,
                )

            island = self._ntm_islands[key]
            _, w_arr = island.evolve(
                w0=w0,
                t_span=(0.0, dt),
                dt=dt,
                j_bs=j_bs_s,
                j_phi=j_phi,
                j_cd=j_cd_s,
                eta=eta_s,
                pressure_gradient=dp_dr,
                B_pol=float(B_pol),
            )
            w_new = float(w_arr[-1])
            self.ntm_widths[key] = w_new

            # Flatten Te/Ti in the island if w > threshold
            # Wesson 2011, Ch. 7 — NTM islands suppress temperature gradients
            if w_new > _W_FLAT:
                rho_flat_lo = surf.rho - w_new / self.config.a
                rho_flat_hi = surf.rho + w_new / self.config.a
                mask = (self.rho >= rho_flat_lo) & (self.rho <= rho_flat_hi)
                if mask.sum() > 1:
                    T_mean_e = float(np.mean(self.ts_solver.Te[mask]))
                    T_mean_i = float(np.mean(self.ts_solver.Ti[mask]))
                    self.ts_solver.Te[mask] = T_mean_e
                    self.ts_solver.Ti[mask] = T_mean_i

    def _compute_W_thermal(self) -> float:
        """Total stored thermal energy.

        W = (3/2) ∫ (n_e T_e + n_i T_i) dV
        Wesson 2011, Ch. 1, Eq. (1.1.1).

        Units: n [m^-3], T [J] → W [J].
        """
        vol = 2.0 * np.pi**2 * self.config.R0 * self.config.a**2 * self.config.kappa
        n_si = self.ts_solver.ne * 1e19
        T_e_J = self.ts_solver.Te * 1.602176634e-16
        T_i_J = self.ts_solver.Ti * 1.602176634e-16
        energy_dens = 1.5 * n_si * (T_e_J + T_i_J)
        # Volume-integral over normalised radius: dV/drho = 2π² R0 a² κ * 2ρ
        return float(trapezoid(energy_dens * self.rho, self.rho) * vol * 2.0)

    def _build_state(self) -> ScenarioState:
        q_prof = q_from_psi(self.rho, self.cd_solver.psi, self.config.R0, self.config.a, self.config.B0)

        W_th = self._compute_W_thermal()
        P_aux_W = self.config.P_aux_MW * 1e6

        j_bs = sauter_bootstrap(
            self.rho,
            self.ts_solver.Te,
            self.ts_solver.Ti,
            self.ts_solver.ne,
            q_prof,
            self.config.R0,
            self.config.a,
            self.config.B0,
        )
        j_cd = self.cd_mix.total_j_cd(self.rho, self.ts_solver.ne, self.ts_solver.Te, self.ts_solver.Ti)
        eta_prof = _spitzer_resistivity(self.ts_solver.Te, Z_eff=1.5)

        # Trapped-fraction correction for ohmic current
        f_t = np.zeros(self.nr)
        for i in range(self.nr):
            eps = self.rho[i] * self.config.a / self.config.R0
            if eps > 1e-6:
                f_t[i] = 1.0 - (1.0 - eps) ** 2 / (np.sqrt(1.0 - eps**2) * (1.0 + 1.46 * np.sqrt(eps)))

        # j_ohmic = E_loop / (η * (1 - f_t))
        # Spitzer 1962; Wesson 2011, Ch. 3 — approximate E_loop from η·j_tot
        j_tor = self._j_total_from_psi(q_prof)
        j_ohmic = j_tor - j_bs - j_cd

        # Power balance: dW/dt = P_aux + P_ohm - P_loss
        # P_ohm = ∫ η j_ohm^2 dV
        vol = 2.0 * np.pi**2 * self.config.R0 * self.config.a**2 * self.config.kappa
        P_ohm = float(trapezoid(eta_prof * j_ohmic**2 * self.rho, self.rho) * vol * 2.0)

        P_loss = max(P_aux_W + P_ohm, 1.0)
        tau_E = (W_th / 1e6) / max(P_loss / 1e6, 1.0)

        # INT-7: compute beta_N from profiles — Troyon et al. 1984
        # beta_t = 2 mu_0 <p> / B0^2
        n_si_avg = self.ts_solver.ne * 1e19
        p_avg = n_si_avg * (self.ts_solver.Te + self.ts_solver.Ti) * _E_CHARGE * 1e3
        p_vol = float(trapezoid(p_avg * self.rho, self.rho) * 2.0 / max(trapezoid(self.rho, self.rho), 1e-12))
        beta_t = 2.0 * _MU_0 * p_vol / self.config.B0**2
        I_N = self.config.Ip_MA / max(self.config.a * self.config.B0, 1e-10)
        beta_N = 100.0 * beta_t / max(I_N, 1e-10)

        # INT-7: compute li from current profile — Wesson 2011 Ch. 3
        # li = <B_pol^2> / B_pol(a)^2 ≈ 2 ∫ j^2 rho drho / j_avg^2
        j_tor_sq = j_tor**2
        j2_vol = trapezoid(j_tor_sq * self.rho, self.rho)
        j_vol = trapezoid(np.abs(j_tor) * self.rho, self.rho)
        li = float(j2_vol / max(j_vol**2, 1e-20)) if j_vol > 1e-10 else 1.0

        # INT-5: run stability checks
        ballooning_ok = True
        troyon_ok = True
        if self.config.include_stability:
            shear_prof = np.gradient(q_prof, self.rho) * (self.rho / np.maximum(q_prof, 1e-3))
            # alpha_MHD ≈ -q² R0 dp/dr / (B0²/(2 mu_0))
            p_prof_Pa = n_si_avg * (self.ts_solver.Te + self.ts_solver.Ti) * _E_CHARGE * 1e3
            dp_drho = np.gradient(p_prof_Pa, self.rho)
            alpha_mhd = -(q_prof**2) * self.config.R0 * dp_drho / (self.config.B0**2 / (2.0 * _MU_0))

            qp = QProfile(
                rho=self.rho.astype(np.float64),
                q=q_prof.astype(np.float64),
                shear=shear_prof.astype(np.float64),
                alpha_mhd=alpha_mhd.astype(np.float64),
                q_min=float(np.min(q_prof)),
                q_min_rho=float(self.rho[int(np.argmin(q_prof))]),
                q_edge=float(q_prof[-1]),
            )
            ball_res = ballooning_stability(qp)
            ballooning_ok = bool(np.all(ball_res.stable))

            troyon_res = troyon_beta_limit(beta_t, self.config.Ip_MA, self.config.a, self.config.B0)
            troyon_ok = troyon_res.stable_nowall

        T_t, q_peak, detached = 0.0, 0.0, False
        if self.config.include_sol:
            n_u = self.ts_solver.ne[-1]
            sol_res = self.sol.solve(self.config.P_aux_MW, n_u, f_rad=0.3)
            T_t = sol_res.T_target_eV
            q_peak = sol_res.q_parallel_MW_m2
            detached = T_t < 5.0

        return ScenarioState(
            time=self.time,
            rho=self.rho.copy(),
            Te=self.ts_solver.Te.copy(),
            Ti=self.ts_solver.Ti.copy(),
            ne=self.ts_solver.ne.copy(),
            q=q_prof.copy(),
            psi=self.cd_solver.psi.copy(),
            j_total=j_bs + j_cd + j_ohmic,
            j_bs=j_bs,
            j_cd=j_cd,
            Ip_MA=self.config.Ip_MA,
            beta_N=beta_N,
            tau_E=tau_E,
            P_loss=P_loss,
            W_thermal=W_th,
            li=li,
            ballooning_stable=ballooning_ok,
            troyon_stable=troyon_ok,
            ntm_island_widths=dict(self.ntm_widths),
            T_target=T_t,
            q_peak=q_peak,
            detached=detached,
            last_crash_time=self.last_crash_time,
            n_crashes=self.n_crashes,
        )

    # ── public step interface ─────────────────────────────────────────────────

    def step(self) -> ScenarioState:
        """Advance one time step using Strang operator splitting.

        Strang 1968, SIAM J. Numer. Anal. 5, 506:
            a) half-step transport
            b) full-step sources
            c) full-step MHD events
            d) half-step transport
        """
        dt = self.config.dt
        dt_half = 0.5 * dt

        q_prof = q_from_psi(self.rho, self.cd_solver.psi, self.config.R0, self.config.a, self.config.B0)

        # ── (a) half-step transport ───────────────────────────────────────────
        self._transport_step(dt_half, q_prof)

        # ── (b) full-step current drive + bootstrap + current diffusion ───────
        j_bs = sauter_bootstrap(
            self.rho,
            self.ts_solver.Te,
            self.ts_solver.Ti,
            self.ts_solver.ne,
            q_prof,
            self.config.R0,
            self.config.a,
            self.config.B0,
        )
        j_cd = self.cd_mix.total_j_cd(self.rho, self.ts_solver.ne, self.ts_solver.Te, self.ts_solver.Ti)

        self.cd_solver.step(dt, self.ts_solver.Te, self.ts_solver.ne, 1.5, j_bs, j_cd)
        q_prof = q_from_psi(self.rho, self.cd_solver.psi, self.config.R0, self.config.a, self.config.B0)

        # ── (c) full-step MHD events ──────────────────────────────────────────
        if self.config.include_sawteeth:
            shear = np.gradient(q_prof, self.rho) * (self.rho / np.maximum(q_prof, 1e-3))
            event = self.sawtooth.step(dt, q_prof, shear, self.ts_solver.Te, self.ts_solver.ne)
            if event:
                self.n_crashes += 1
                self.last_crash_time = event.crash_time

                # INT-1: write crashed q back to psi so current diffusion is consistent
                self.cd_solver.psi = psi_from_q(self.rho, q_prof, self.config.R0, self.config.a, self.config.B0)

                # Also flatten Ti inside the mixing radius (sawtooth only crashes Te+ne)
                if event.rho_mix > 0:
                    idx_mix = int(np.searchsorted(self.rho, event.rho_mix))
                    if idx_mix > 1:
                        rho_inner = self.rho[:idx_mix]
                        vol_int = trapezoid(self.ts_solver.Ti[:idx_mix] * rho_inner, rho_inner)
                        vol_tot = trapezoid(rho_inner, rho_inner)
                        Ti_mix = vol_int / vol_tot if vol_tot > 0 else self.ts_solver.Ti[0]
                        self.ts_solver.Ti[:idx_mix] = Ti_mix

                # INT-6: seed NTM from sawtooth crash energy
                for key, island in self._ntm_islands.items():
                    w_seed = self.ntm_seeding.seed_amplitude(event.seed_energy / 1e6, island.r_s)
                    if w_seed > self.ntm_widths.get(key, 0.0):
                        self.ntm_widths[key] = w_seed

                q_prof = q_from_psi(
                    self.rho,
                    self.cd_solver.psi,
                    self.config.R0,
                    self.config.a,
                    self.config.B0,
                )

        if self.config.include_ntm:
            self._ntm_step(dt, q_prof)
            q_prof = q_from_psi(
                self.rho,
                self.cd_solver.psi,
                self.config.R0,
                self.config.a,
                self.config.B0,
            )

        # ── ELM check ──────────────────────────────────────────────────────────
        if self.config.include_elm:
            # Pedestal values at rho ~ 0.95
            i_ped = min(int(0.95 * self.nr), self.nr - 2)
            T_ped = float(self.ts_solver.Te[i_ped])
            n_ped = float(self.ts_solver.ne[i_ped])
            W_ped = self._compute_W_thermal() * 0.3  # pedestal fraction ~30%

            s_edge = float(np.gradient(q_prof, self.rho)[i_ped] * self.rho[i_ped] / max(q_prof[i_ped], 1e-3))
            n_si_edge = n_ped * 1e19
            T_J_edge = T_ped * _E_CHARGE * 1e3
            dp_dr_edge = float(np.gradient(n_si_edge * T_J_edge * np.ones(self.nr), self.rho * self.config.a)[i_ped])
            alpha_edge = float(
                -(q_prof[i_ped] ** 2) * self.config.R0 * dp_dr_edge / (self.config.B0**2 / (2.0 * _MU_0))
            )
            j_edge = float(abs(self._j_total_from_psi(q_prof)[i_ped]))

            elm_event = self.elm_cycler.step(dt, alpha_edge, j_edge, s_edge, T_ped, n_ped, W_ped)
            if elm_event:
                # Apply ELM crash: energy fraction lost from pedestal
                frac_lost = min(elm_event.delta_W_MJ * 1e6 / max(W_ped, 1.0), 0.5)
                drop = np.sqrt(max(1.0 - frac_lost, 0.25))
                ped_mask = self.rho >= 0.9
                self.ts_solver.Te[ped_mask] *= drop
                self.ts_solver.Ti[ped_mask] *= drop
                self.ts_solver.ne[ped_mask] *= drop

        # ── (d) half-step transport ───────────────────────────────────────────
        self._transport_step(dt_half, q_prof)

        self.time += dt
        return self._build_state()

    def run(self) -> list[ScenarioState]:
        if not hasattr(self, "ts_solver"):
            self.initialize()
        n_steps = int((self.config.t_end - self.config.t_start) / self.config.dt)
        states = []
        for _ in range(n_steps):
            states.append(self.step())
        return states

    def to_json(self, path: Path) -> None:
        pass
