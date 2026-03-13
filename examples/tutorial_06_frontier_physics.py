#!/usr/bin/env python3
"""Tutorial 06: Phase 3 Frontier Physics — Expert Walkthrough.

Demonstrates every Phase 3 physics module end-to-end:
  1. Gyrokinetic transport model (ITG/TEM/ETG spectra, chi profiles)
  2. Ballooning stability (s-alpha diagram, marginal boundary)
  3. Current diffusion (Crank-Nicolson psi evolution, q-profile relaxation)
  4. Current drive sources (ECCD + NBI deposition, total driven current)
  5. NTM island dynamics (Modified Rutherford Equation, ECCD stabilization)
  6. Sawtooth cycler (Kadomtsev crash, trigger detection)
  7. SOL two-point model (upstream density scan, detachment threshold)
  8. Integrated scenario (ITER baseline, coupled time evolution)

Usage:
    pip install scpn-control
    python examples/tutorial_06_frontier_physics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 1: Gyrokinetic Transport Model                          ║
# ╚══════════════════════════════════════════════════════════════════╝

print("=" * 66)
print("SECTION 1: Gyrokinetic Transport Model")
print("=" * 66)

from scpn_control.core.gyrokinetic_transport import (
    GyrokineticsParams,
    GyrokineticTransportModel,
    compute_spectrum,
    quasilinear_fluxes,
)

# ITER mid-radius conditions: R/L_Ti ~ 6, R/L_Te ~ 6, q ~ 1.5, s ~ 1
params_mid = GyrokineticsParams(
    R_L_Ti=6.0,
    R_L_Te=6.0,
    R_L_ne=2.0,
    q=1.5,
    s_hat=1.0,
    alpha_MHD=0.0,
    Te_Ti=1.0,
    Z_eff=1.5,
    nu_star=0.1,
    beta_e=0.01,
)

spec = compute_spectrum(params_mid, n_modes=16, include_etg=True)
fluxes = quasilinear_fluxes(params_mid, spec)

print("  Mid-radius parameters: R/L_Ti=6, R/L_Te=6, q=1.5, s=1.0")
print("  Quasilinear diffusivities (normalized to chi_gB):")
print(f"    chi_i = {fluxes.chi_i:.4f}")
print(f"    chi_e = {fluxes.chi_e:.4f}")
print(f"    D_e   = {fluxes.D_e:.4f}")

print("\n  ITG growth rate spectrum (ion scale, k_theta rho_s):")
print(f"  {'k_y':>6s}  {'gamma':>8s}  {'omega_r':>8s}  {'mode':>5s}")
print(f"  {'-' * 6}  {'-' * 8}  {'-' * 8}  {'-' * 5}")

mode_names = {0: "stab", 1: "ITG", 2: "TEM", 3: "ETG"}
for i in range(len(spec.k_y)):
    if spec.k_y[i] > 2.5:
        break
    print(
        f"  {spec.k_y[i]:6.2f}  {spec.gamma_linear[i]:8.4f}  "
        f"{spec.omega_r[i]:8.4f}  {mode_names[spec.mode_type[i]]:>5s}"
    )

# Critical gradient scan: increase R/L_Ti from 0 to 10
print("\n  Critical gradient scan (R/L_Ti sweep):")
print(f"  {'R/L_Ti':>7s}  {'chi_i':>8s}  {'dominant':>8s}")
print(f"  {'-' * 7}  {'-' * 8}  {'-' * 8}")
for rl in [0.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
    p = GyrokineticsParams(
        R_L_Ti=rl,
        R_L_Te=6.0,
        R_L_ne=2.0,
        q=1.5,
        s_hat=1.0,
        alpha_MHD=0.0,
        Te_Ti=1.0,
        Z_eff=1.5,
        nu_star=0.1,
        beta_e=0.01,
    )
    s = compute_spectrum(p, n_modes=8)
    f = quasilinear_fluxes(p, s)
    dominant = mode_names[int(s.mode_type[s.gamma_linear.argmax()])] if s.gamma_linear.max() > 0 else "stab"
    print(f"  {rl:7.1f}  {f.chi_i:8.4f}  {dominant:>8s}")

# Physical chi_i profile across rho
model = GyrokineticTransportModel(n_modes=16)
rho_grid = np.linspace(0.0, 1.0, 21)
Te_prof = 8.0 * (1.0 - rho_grid**2) + 0.5
Ti_prof = Te_prof * 0.95
ne_prof = 10.0 * (1.0 - 0.5 * rho_grid**2)
q_prof = 1.0 + 2.0 * rho_grid**2
s_hat_prof = 4.0 * rho_grid / np.maximum(q_prof, 0.5)
dTe = np.gradient(Te_prof, rho_grid[1] - rho_grid[0])
dTi = np.gradient(Ti_prof, rho_grid[1] - rho_grid[0])
dne = np.gradient(ne_prof, rho_grid[1] - rho_grid[0])

profiles = {
    "R0": 6.2,
    "a": 2.0,
    "B0": 5.3,
    "Te": Te_prof,
    "Ti": Ti_prof,
    "ne": ne_prof,
    "q": q_prof,
    "s_hat": s_hat_prof,
    "dTe_dr": dTe,
    "dTi_dr": dTi,
    "dne_dr": dne,
}
chi_i, chi_e, D_e = model.evaluate_profile(rho_grid, profiles)

print("\n  Physical chi_i profile [m^2/s] (ITER-like, R0=6.2m, B0=5.3T):")
print(f"  {'rho':>5s}  {'chi_i':>10s}  {'chi_e':>10s}  {'D_e':>10s}")
print(f"  {'-' * 5}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
for i in range(0, len(rho_grid), 4):
    print(f"  {rho_grid[i]:5.2f}  {chi_i[i]:10.4f}  {chi_e[i]:10.4f}  {D_e[i]:10.4f}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 2: Ballooning Stability                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 66)
print("SECTION 2: Ballooning Stability (s-alpha diagram)")
print("=" * 66)

from scpn_control.core.ballooning_solver import (
    BallooningEquation,
    compute_stability_diagram,
)

# Scan the first stability boundary
s_values = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0])
alpha_crit = compute_stability_diagram(s_values)

print("  First stability boundary (ideal MHD ballooning):")
print(f"  {'s':>5s}  {'alpha_crit':>10s}  note")
print(f"  {'-' * 5}  {'-' * 10}  {'-' * 30}")
for s, ac in zip(s_values, alpha_crit):
    note = ""
    if ac < 0.3:
        note = "low pressure limit"
    elif ac > 1.5:
        note = "access to 2nd stability"
    print(f"  {s:5.2f}  {ac:10.3f}  {note}")

# Classify specific (s, alpha) pairs for ITER-like profiles
test_pairs = [
    (0.5, 0.3, "Low beta, moderate shear"),
    (1.0, 0.8, "Typical H-mode pedestal"),
    (1.5, 1.2, "High beta, high shear"),
    (0.8, 0.5, "Core region"),
    (2.0, 0.4, "Edge with strong shear"),
    (0.3, 1.0, "Weak shear, high alpha"),
]

print("\n  Point-by-point stability classification:")
print(f"  {'s':>5s}  {'alpha':>6s}  {'stable':>7s}  description")
print(f"  {'-' * 5}  {'-' * 6}  {'-' * 7}  {'-' * 30}")
for s, alpha, desc in test_pairs:
    eq = BallooningEquation(s, alpha)
    result = eq.solve()
    print(f"  {s:5.2f}  {alpha:6.2f}  {'YES' if result.is_stable else 'NO':>7s}  {desc}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 3: Current Diffusion                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 66)
print("SECTION 3: Current Diffusion (Crank-Nicolson)")
print("=" * 66)

from scpn_control.core.current_diffusion import CurrentDiffusionSolver, q_from_psi

R0, a, B0 = 6.2, 2.0, 5.3
nr = 50
rho = np.linspace(0, 1, nr)

solver = CurrentDiffusionSolver(rho, R0, a, B0)
q_init = q_from_psi(rho, solver.psi, R0, a, B0)

# Plasma profiles for resistivity calculation
Te = 8.0 * (1.0 - rho**2) + 0.3  # keV
ne = 10.0 * (1.0 - 0.5 * rho**2)  # 10^19 m^-3

print(f"  Grid: {nr} points, R0={R0}m, a={a}m, B0={B0}T")
print("  Initial q-profile (parabolic current density):")
print(f"    q(0)   = {q_init[0]:.3f}")
print(f"    q(0.5) = {q_init[nr // 2]:.3f}")
print(f"    q(1.0) = {q_init[-2]:.3f}")

# Evolve for 20 steps at dt = 0.5 s
n_steps = 20
dt = 0.5
j_bs = np.zeros(nr)
j_cd = np.zeros(nr)

print(f"\n  Evolving psi for {n_steps} steps (dt={dt}s, total={n_steps * dt}s):")
print(f"  {'step':>5s}  {'q_axis':>7s}  {'q_mid':>7s}  {'q_edge':>7s}  {'psi_axis':>10s}")
print(f"  {'-' * 5}  {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 10}")

for step in range(n_steps):
    solver.step(dt, Te, ne, Z_eff=1.5, j_bs=j_bs, j_cd=j_cd)
    if step % 5 == 0 or step == n_steps - 1:
        q_now = q_from_psi(rho, solver.psi, R0, a, B0)
        print(f"  {step:5d}  {q_now[0]:7.3f}  {q_now[nr // 2]:7.3f}  {q_now[-2]:7.3f}  {solver.psi[0]:10.6f}")

q_final = q_from_psi(rho, solver.psi, R0, a, B0)
print(f"\n  q(0) drifted {q_init[0]:.3f} -> {q_final[0]:.3f} as current diffused toward steady state")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 4: Current Drive Sources                                ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 66)
print("SECTION 4: Current Drive (ECCD + NBI)")
print("=" * 66)

from scpn_control.core.current_drive import CurrentDriveMix, ECCDSource, NBISource

eccd = ECCDSource(P_ec_MW=17.0, rho_dep=0.45, sigma_rho=0.05, eta_cd=0.03)
nbi = NBISource(P_nbi_MW=33.0, E_beam_keV=1000.0, rho_tangency=0.2, sigma_rho=0.15)

rho_cd = np.linspace(0, 1, 100)
ne_cd = 10.0 * (1.0 - 0.5 * rho_cd**2)
Te_cd = 8.0 * (1.0 - rho_cd**2) + 0.3
Ti_cd = Te_cd * 0.95

j_eccd = eccd.j_cd(rho_cd, ne_cd, Te_cd)
j_nbi = nbi.j_cd(rho_cd, ne_cd, Te_cd, Ti_cd)

mix = CurrentDriveMix(a=2.0)
mix.add_source(eccd)
mix.add_source(nbi)

I_total = mix.total_driven_current(rho_cd, ne_cd, Te_cd, Ti_cd)

print("  ECCD: 17 MW at rho=0.45 (sigma=0.05), eta_cd=0.03")
print("  NBI:  33 MW at rho=0.20 (sigma=0.15), E_beam=1000 keV")
print(f"\n  Total driven current: {I_total / 1e6:.3f} MA")

print("\n  Current density profiles [A/m^2]:")
print(f"  {'rho':>5s}  {'j_ECCD':>12s}  {'j_NBI':>12s}  {'j_total':>12s}")
print(f"  {'-' * 5}  {'-' * 12}  {'-' * 12}  {'-' * 12}")
for i in range(0, 100, 10):
    j_tot = j_eccd[i] + j_nbi[i]
    print(f"  {rho_cd[i]:5.2f}  {j_eccd[i]:12.1f}  {j_nbi[i]:12.1f}  {j_tot:12.1f}")

# Heating power deposition
p_eccd = eccd.P_absorbed(rho_cd)
p_nbi = nbi.P_heating(rho_cd)
idx_ec_peak = np.argmax(p_eccd)
idx_nbi_peak = np.argmax(p_nbi)
print(f"\n  ECCD peak deposition: {p_eccd[idx_ec_peak] / 1e6:.2f} MW/m^3 at rho={rho_cd[idx_ec_peak]:.2f}")
print(f"  NBI  peak deposition: {p_nbi[idx_nbi_peak] / 1e6:.2f} MW/m^3 at rho={rho_cd[idx_nbi_peak]:.2f}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 5: NTM Island Dynamics                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 66)
print("SECTION 5: NTM Dynamics (Modified Rutherford Equation)")
print("=" * 66)

from scpn_control.core.ntm_dynamics import (
    NTMController,
    NTMIslandDynamics,
    find_rational_surfaces,
)

# q=2/1 island on a monotonic q-profile
q_for_ntm = 0.95 + 2.1 * rho**2
surfaces = find_rational_surfaces(q_for_ntm, rho, a=2.0, m_max=3, n_max=2)

print("  Rational surfaces found on q(rho) = 0.95 + 2.1 rho^2:")
print(f"  {'m/n':>5s}  {'rho':>6s}  {'r_s [m]':>8s}  {'q':>5s}  {'shear':>7s}")
print(f"  {'-' * 5}  {'-' * 6}  {'-' * 8}  {'-' * 5}  {'-' * 7}")
for s in surfaces:
    print(f"  {s.m}/{s.n}    {s.rho:6.3f}  {s.r_s:8.3f}  {s.q:5.2f}  {s.shear:7.3f}")

# Evolve a 2/1 island without ECCD (unstable growth)
ntm_21 = NTMIslandDynamics(
    r_s=0.9,
    m=2,
    n=1,
    a=2.0,
    R0=6.2,
    B0=5.3,
)

# Plasma parameters at the rational surface
eta_rs = 1e-7  # Ohm*m (neoclassical)
j_bs_rs = 5e4  # A/m^2 (bootstrap current)
j_phi_rs = 1e6  # A/m^2 (Ohmic)

t_arr, w_no_eccd = ntm_21.evolve(
    w0=0.005,
    t_span=(0, 5.0),
    dt=0.01,
    j_bs=j_bs_rs,
    j_phi=j_phi_rs,
    j_cd=0.0,
    eta=eta_rs,
)

print("\n  2/1 NTM island evolution WITHOUT ECCD:")
print(f"  {'t [s]':>6s}  {'w [m]':>8s}  {'w/a':>6s}")
print(f"  {'-' * 6}  {'-' * 8}  {'-' * 6}")
for i in range(0, len(t_arr), len(t_arr) // 6):
    print(f"  {t_arr[i]:6.2f}  {w_no_eccd[i]:8.5f}  {w_no_eccd[i] / a:6.4f}")
print(f"  {t_arr[-1]:6.2f}  {w_no_eccd[-1]:8.5f}  {w_no_eccd[-1] / a:6.4f}")

# With ECCD stabilization: 10 MW ECCD at q=2 surface
t_arr2, w_eccd = ntm_21.evolve(
    w0=0.005,
    t_span=(0, 5.0),
    dt=0.01,
    j_bs=j_bs_rs,
    j_phi=j_phi_rs,
    j_cd=8e4,
    eta=eta_rs,  # ECCD replaces bootstrap deficit
    d_cd=0.02,
)

print("\n  2/1 NTM island evolution WITH ECCD (j_cd=8e4 A/m^2):")
print(f"  {'t [s]':>6s}  {'w [m]':>8s}  {'w/a':>6s}")
print(f"  {'-' * 6}  {'-' * 8}  {'-' * 6}")
for i in range(0, len(t_arr2), len(t_arr2) // 6):
    print(f"  {t_arr2[i]:6.2f}  {w_eccd[i]:8.5f}  {w_eccd[i] / a:6.4f}")
print(f"  {t_arr2[-1]:6.2f}  {w_eccd[-1]:8.5f}  {w_eccd[-1] / a:6.4f}")

# NTM controller logic
controller = NTMController(w_onset=0.02, w_target=0.005)
print(f"\n  NTM controller (onset={controller.w_onset}m, target={controller.w_target}m):")
test_widths = [0.001, 0.010, 0.020, 0.025, 0.030, 0.015, 0.005, 0.003]
for w in test_widths:
    p = controller.step(w, rho_rs=0.45)
    status = "ACTIVE" if controller.active else "standby"
    print(f"    w={w:.3f}m  -> P_eccd={p:5.1f} MW  [{status}]")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 6: Sawtooth Cycler                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 66)
print("SECTION 6: Sawtooth Cycler (Kadomtsev Reconnection)")
print("=" * 66)

from scpn_control.core.sawtooth import SawtoothCycler

rho_st = np.linspace(0, 1, 80)
R0_st, a_st = 6.2, 2.0

# q-profile with q(0) < 1 to allow sawteeth
q_st = 0.85 + 1.5 * rho_st**2 + 0.65 * rho_st**4
shear_st = np.gradient(q_st, rho_st[1] - rho_st[0]) * rho_st / np.maximum(q_st, 0.01)

# Peaked temperature and density
Te_st = 12.0 * (1.0 - rho_st**2) ** 1.5 + 0.5
ne_st = 10.0 * (1.0 - 0.3 * rho_st**2)

cycler = SawtoothCycler(rho_st, R0_st, a_st, s_crit=0.05)

print(f"  q(0)={q_st[0]:.3f}, q(0.5)={q_st[40]:.3f}, q(1.0)={q_st[-1]:.3f}")
print(f"  Te(0)={Te_st[0]:.1f} keV (peaked), ne(0)={ne_st[0]:.1f} x10^19 m^-3")
print(f"  Sawtooth trigger: shear at q=1 > {cycler.monitor.s_crit}")

n_steps_st = 200
dt_st = 0.05
events = []

print(f"\n  Evolving {n_steps_st} steps (dt={dt_st}s)...")
for step in range(n_steps_st):
    event = cycler.step(dt_st, q_st, shear_st, Te_st, ne_st)
    if event is not None:
        events.append(event)
        # After crash, rebuild shear from modified q
        shear_st = np.gradient(q_st, rho_st[1] - rho_st[0]) * rho_st / np.maximum(q_st, 0.01)
        # Re-peak the temperature gradually (heating between crashes)
    else:
        # Ohmic reheating between crashes: slow peaking
        Te_st[: len(rho_st) // 3] += 0.02 * dt_st

print(f"  Crashes detected: {len(events)}")
if events:
    print(f"\n  {'#':>3s}  {'t_crash':>8s}  {'rho_1':>6s}  {'rho_mix':>8s}  {'dTe_core':>9s}  {'E_seed':>10s}")
    print(f"  {'-' * 3}  {'-' * 8}  {'-' * 6}  {'-' * 8}  {'-' * 9}  {'-' * 10}")
    for i, ev in enumerate(events):
        print(
            f"  {i + 1:3d}  {ev.crash_time:8.2f}  {ev.rho_1:6.3f}  {ev.rho_mix:8.3f}  "
            f"{ev.T_drop:9.2f}  {ev.seed_energy:10.2e}"
        )
    print(f"\n  Post-crash Te(0) = {Te_st[0]:.2f} keV (flattened inside mixing radius)")
else:
    print("  No crashes triggered (shear at q=1 below s_crit)")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 7: SOL Two-Point Model                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 66)
print("SECTION 7: SOL Two-Point Model (Divertor)")
print("=" * 66)

from scpn_control.core.sol_model import TwoPointSOL, eich_heat_flux_width

# ITER-like: R0=6.2m, a=2.0m, q95=3.0, B_pol=0.5T, kappa=1.7
sol = TwoPointSOL(R0=6.2, a=2.0, q95=3.0, B_pol=0.5, kappa=1.7)

print(f"  ITER-like SOL: R0={sol.R0}m, q95={sol.q95}, B_pol={sol.B_pol}T")
print(f"  Connection length L_par = pi * q95 * R0 = {sol.L_par:.1f} m")

lambda_q = eich_heat_flux_width(100.0, 6.2, 0.5, 2.0 / 6.2)
print(f"  Eich lambda_q at P_SOL=100MW: {lambda_q:.2f} mm")

# Upstream density scan: 3 to 12 x10^19 m^-3
P_SOL = 100.0  # MW
n_u_values = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0])

print(f"\n  Density scan at P_SOL={P_SOL} MW (no radiation):")
print(f"  {'n_u':>5s}  {'T_u [eV]':>9s}  {'T_t [eV]':>9s}  {'n_t':>8s}  {'q_par':>8s}  {'status':>10s}")
print(f"  {'-' * 5}  {'-' * 9}  {'-' * 9}  {'-' * 8}  {'-' * 8}  {'-' * 10}")

for n_u in n_u_values:
    res = sol.solve(P_SOL, n_u, f_rad=0.0)
    status = "DETACHED" if res.T_target_eV < 5.0 else "attached"
    print(
        f"  {n_u:5.1f}  {res.T_upstream_eV:9.1f}  {res.T_target_eV:9.1f}  "
        f"{res.n_target_19:8.1f}  {res.q_parallel_MW_m2:8.2f}  {status:>10s}"
    )

# Effect of radiative fraction
print(f"\n  Radiation scan at n_u=8.0, P_SOL={P_SOL} MW:")
print(f"  {'f_rad':>5s}  {'T_target':>9s}  {'detached':>9s}")
print(f"  {'-' * 5}  {'-' * 9}  {'-' * 9}")
for f_rad in [0.0, 0.3, 0.5, 0.7, 0.85, 0.95]:
    res = sol.solve(P_SOL, 8.0, f_rad=f_rad)
    det = "YES" if res.T_target_eV < 5.0 else "no"
    print(f"  {f_rad:5.2f}  {res.T_target_eV:9.2f}  {det:>9s}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 8: Integrated Scenario Simulation                      ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 66)
print("SECTION 8: Integrated Scenario (ITER Baseline)")
print("=" * 66)

from scpn_control.core.integrated_scenario import (
    IntegratedScenarioSimulator,
    iter_baseline_scenario,
)

# Short run: override t_end and dt for a quick demo
cfg = iter_baseline_scenario()
cfg.t_end = 5.0
cfg.dt = 0.5

print(f"  Config: R0={cfg.R0}m, a={cfg.a}m, B0={cfg.B0}T, Ip={cfg.Ip_MA}MA")
print(f"  Heating: P_aux={cfg.P_aux_MW}MW (ECCD={cfg.P_eccd_MW}MW + NBI={cfg.P_nbi_MW}MW)")
print(f"  Duration: {cfg.t_end}s, dt={cfg.dt}s")

sim = IntegratedScenarioSimulator(cfg)
state0 = sim.initialize()

print("\n  Initial state:")
print(f"    Te(0)   = {state0.Te[0]:.2f} keV")
print(f"    ne(0)   = {state0.ne[0]:.2f} x10^19 m^-3")
print(f"    q(0)    = {state0.q[0]:.3f}")
print(f"    q(edge) = {state0.q[-2]:.3f}")
print(f"    W_th    = {state0.W_thermal / 1e6:.3f} MJ")
print(f"    tau_E   = {state0.tau_E:.4f} s")

states = sim.run()

print(f"\n  Time evolution ({len(states)} steps):")
print(f"  {'t':>5s}  {'tau_E':>7s}  {'W_th[MJ]':>9s}  {'q_axis':>7s}  {'q_edge':>7s}  {'crashes':>7s}  {'T_tgt':>7s}")
print(f"  {'-' * 5}  {'-' * 7}  {'-' * 9}  {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 7}")
for st in states:
    print(
        f"  {st.time:5.1f}  {st.tau_E:7.4f}  {st.W_thermal / 1e6:9.3f}  "
        f"{st.q[0]:7.3f}  {st.q[-2]:7.3f}  {st.n_crashes:7d}  {st.T_target:7.1f}"
    )

final = states[-1]
print(f"\n  Final state at t={final.time:.1f}s:")
print(f"    W_thermal    = {final.W_thermal / 1e6:.3f} MJ")
print(f"    tau_E        = {final.tau_E:.4f} s")
print(f"    q_axis       = {final.q[0]:.3f}")
print(f"    Sawteeth     = {final.n_crashes} crashes")
print(f"    Ballooning   = {'stable' if final.ballooning_stable else 'UNSTABLE'}")
print(f"    Troyon       = {'stable' if final.troyon_stable else 'UNSTABLE'}")
print(f"    Divertor T_t = {final.T_target:.1f} eV ({'detached' if final.detached else 'attached'})")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Summary                                                         ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 66)
print("PHASE 3 FRONTIER PHYSICS SUMMARY")
print("=" * 66)
print("  GyrokineticTransportModel   ITG/TEM/ETG spectra -> chi_i, chi_e, D_e")
print("  BallooningEquation          s-alpha ideal MHD stability boundary")
print("  CurrentDiffusionSolver      Crank-Nicolson psi evolution + neoclassical eta")
print("  CurrentDriveMix             ECCD + NBI + LHCD deposition and driven current")
print("  NTMIslandDynamics           Modified Rutherford Equation + ECCD stabilization")
print("  SawtoothCycler              Kadomtsev crash with Porcelli trigger")
print("  TwoPointSOL                 Upstream/target mapping, Eich lambda_q scaling")
print("  IntegratedScenarioSimulator Coupled transport + CD + sawteeth + SOL + NTM")
