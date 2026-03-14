# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Tutorial 08: Gyrokinetic Transport Solver
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Tutorial: native linear gyrokinetic solver + hybrid transport.

Demonstrates:
1. Linear GK eigenvalue solver on the Cyclone Base Case
2. Quasilinear flux computation
3. OOD detection for surrogate validation
4. Hybrid transport with spot-check correction
5. GK → UPDE bridge for phase dynamics
"""

from __future__ import annotations

import numpy as np

# ── 1. Linear GK Spectrum ──────────────────────────────────────────

from scpn_control.core.gk_eigenvalue import solve_linear_gk
from scpn_control.core.gk_species import deuterium_ion, electron

print("=" * 60)
print("1. Cyclone Base Case — Linear GK Spectrum")
print("=" * 60)

species = [
    deuterium_ion(T_keV=2.0, n_19=5.0, R_L_T=6.9, R_L_n=2.2),
    electron(T_keV=2.0, n_19=5.0, R_L_T=6.9, R_L_n=2.2, adiabatic=True),
]

result = solve_linear_gk(
    species_list=species,
    R0=2.78,
    a=1.0,
    B0=2.0,
    q=1.4,
    s_hat=0.78,
    n_ky_ion=8,
    n_ky_etg=0,
    n_theta=32,
    n_period=1,
)

print(f"  k_y range: {result.k_y[0]:.3f} — {result.k_y[-1]:.3f}")
print(f"  gamma_max: {result.gamma_max:.4f} c_s/a")
print(f"  k_y(max):  {result.k_y_max:.3f} rho_s")
print(f"  Modes:     {', '.join(result.mode_type)}")

# ── 2. Quasilinear Fluxes ─────────────────────────────────────────

from scpn_control.core.gk_quasilinear import quasilinear_fluxes_from_spectrum

print("\n" + "=" * 60)
print("2. Quasilinear Transport Fluxes")
print("=" * 60)

fluxes = quasilinear_fluxes_from_spectrum(result, species[0], R0=2.78, a=1.0, B0=2.0)
print(f"  chi_i: {fluxes.chi_i:.4f} m^2/s")
print(f"  chi_e: {fluxes.chi_e:.4f} m^2/s")
print(f"  D_e:   {fluxes.D_e:.4f} m^2/s")
print(f"  Mode:  {fluxes.dominant_mode}")

# ── 3. OOD Detection ──────────────────────────────────────────────

from scpn_control.core.gk_ood_detector import OODDetector, _to_input_vector

print("\n" + "=" * 60)
print("3. Out-of-Distribution Detection")
print("=" * 60)

detector = OODDetector()

# In-distribution (CBC parameters)
x_normal = _to_input_vector(6.9, 6.9, 2.2, 1.4, 0.78, 0.0, 1.0, 1.0, 0.01, 0.0)
result_normal = detector.check(x_normal)
print(f"  CBC params: OOD={result_normal.is_ood}, confidence={result_normal.confidence:.2f}")

# Extreme parameters
x_extreme = _to_input_vector(50.0, 50.0, 20.0, 10.0, 10.0, 5.0, 10.0, 10.0, 100.0, 1.0)
result_extreme = detector.check(x_extreme)
print(f"  Extreme:    OOD={result_extreme.is_ood}, confidence={result_extreme.confidence:.2f}")

# ── 4. Hybrid Correction ──────────────────────────────────────────

from scpn_control.core.gk_corrector import CorrectionRecord, GKCorrector

print("\n" + "=" * 60)
print("4. Hybrid Surrogate + GK Correction")
print("=" * 60)

rho = np.linspace(0, 1, 20)
corrector = GKCorrector(nr=20)
records = [
    CorrectionRecord(
        rho_idx=10,
        rho=0.5,
        chi_i_surrogate=2.0,
        chi_i_gk=3.0,
        chi_e_surrogate=1.0,
        chi_e_gk=1.5,
        D_e_surrogate=0.3,
        D_e_gk=0.4,
    )
]
corrector.update(records, rho)
chi_i_surr = np.ones(20) * 2.0
chi_i_corr, _, _ = corrector.correct(chi_i_surr, np.ones(20), np.ones(20) * 0.3)
print(f"  Surrogate chi_i(0.5): {chi_i_surr[10]:.2f}")
print(f"  Corrected chi_i(0.5): {chi_i_corr[10]:.2f}")
print(f"  Max correction:       {corrector.max_correction_factor:.3f}")

# ── 5. GK → UPDE Bridge ──────────────────────────────────────────

from scpn_control.core.gk_interface import GKOutput
from scpn_control.phase.gk_upde_bridge import adaptive_knm

print("\n" + "=" * 60)
print("5. GK → UPDE Phase Coupling Bridge")
print("=" * 60)

K_base = 0.3 * np.ones((8, 8))
gk_out = GKOutput(
    chi_i=2.5,
    chi_e=1.8,
    D_e=0.4,
    gamma=np.array([0.1, 0.25, 0.15]),
    omega_r=np.array([-0.3, -0.5, 0.2]),
    k_y=np.array([0.1, 0.3, 0.5]),
    dominant_mode="ITG",
)
K_adapted = adaptive_knm(K_base, gk_out)
print(f"  K_base[0,1]:    {K_base[0, 1]:.3f}")
print(f"  K_adapted[0,1]: {K_adapted[0, 1]:.3f}  (P0↔P1: micro↔zonal)")
print(f"  K_adapted[1,4]: {K_adapted[1, 4]:.3f}  (P1↔P4: zonal↔barrier)")

print("\n" + "=" * 60)
print("Tutorial complete.")
print("=" * 60)
