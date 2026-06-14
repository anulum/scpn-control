<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Physics Methods Reference

This document catalogs the primary physics models, equations, and scaling laws implemented in `scpn-control`.

## 1. Equilibrium (Grad-Shafranov)

### Grad-Shafranov Equation
$$\Delta^* \psi = R \frac{\partial}{\partial R}\left(\frac{1}{R}\frac{\partial \psi}{\partial R}\right) + \frac{\partial^2 \psi}{\partial Z^2} = -\mu_0 R^2 p'(\psi) - F(\psi) F'(\psi)$$

- **Source**: Grad & Rubin (1958), Shafranov (1966).
- **Implementation**: `src/scpn_control/core/fusion_kernel.py:380` (Picard + SOR).
- **Simplifications**: Axisymmetric assumption. Poloidal current $F(\psi)$ and pressure $p(\psi)$ are specified as polynomial or pedestal functions of $\psi$.
- **Validation**: The production discrete operator
  `FusionKernel._apply_gs_operator` (sharing the `_mg_residual` five-point
  stencil) and the production Red-Black SOR smoother `_sor_step` are checked
  against the exact Solov'ev analytic equilibrium
  $\psi = c_1 R^4/8 + c_2 Z^2$, $\Delta^*\psi = c_1 R^2 + 2 c_2$, in
  `validation/validate_grad_shafranov_solovev.py` with tests in
  `tests/test_grad_shafranov_solovev_validation.py`. Both paths converge at
  second order in the mesh spacing (operator order $\approx 2.00$, SOR
  reconstruction order $\approx 2.02$), recorded as tamper-evident sealed
  evidence in `validation/reports/grad_shafranov_solovev.json`. The Python
  `_multigrid_vcycle` and the Rust `py_multigrid_solve` binding do not reproduce
  the analytic equilibrium on this forcing and are recorded but not admitted.
  This validates the equilibrium **discretisation and SOR solver** against an
  analytic benchmark; it is not a facility-grade EFIT/GEQDSK reconstruction
  claim. References: Solov'ev (1968); Cerfon & Freidberg, *Phys. Plasmas* 17,
  032502 (2010); Jardin, *Computational Methods in Plasma Physics* (2010).

### Evidence context for equilibrium methods

These equations are the implementation basis for bounded controller experiments.
They are not a full replacement for validated external equilibria unless the
relevant reference validator admits external evidence for the specific use case.
The repository therefore treats equation-level implementation and claim-level
admission as separate layers: model code can exist before all claims are fully
validated.

### Green's Function for Coil Flux
$$\psi_{coil}(R,Z) = \frac{\mu_0 I}{\pi k} \sqrt{R R_c} [(1-k^2/2) K(k^2) - E(k^2)]$$
where $k^2 = \frac{4 R R_c}{(R+R_c)^2 + (Z-Z_c)^2}$.

- **Source**: Jackson, *Classical Electrodynamics* (1999) Ch. 5 Eq. 5.37.
- **Implementation**: `src/scpn_control/core/fusion_kernel.py:1250`.
- **Simplifications**: Assumes filamentary circular coils.

---

## 2. Transport (1.5D)

### Cylindrical Heat Diffusion
$$\frac{3}{2} n \frac{\partial T}{\partial t} = \frac{1}{r} \frac{\partial}{\partial r} \left( r n \chi \frac{\partial T}{\partial r} \right) + P_{heat} - P_{rad}$$

- **Source**: Wesson, *Tokamaks* (2011) Ch. 3.
- **Implementation**: `src/scpn_control/core/integrated_transport_solver.py:850`.
- **Simplifications**: 1D radial approximation on flux surfaces.
- **Validation**: The production cylindrical diffusion operator
  `TransportSolver._explicit_diffusion_rhs` is checked against the exact Bessel
  eigenvalue $L[J_0(\lambda\rho)] = -(\chi\lambda^2/a^2)\,J_0(\lambda\rho)$, and
  the Crank-Nicolson tridiagonal solve (`_build_cn_tridiag` + `_thomas_solve`)
  against the manufactured steady state $T^*=1-\rho^3$ with source
  $S=9\chi\rho/a^2$; both converge at second order in $\Delta\rho$ (operator
  order $\approx 2.00$, steady-state order $\approx 1.93$) in
  `validation/validate_transport_diffusion.py`, with tests in
  `tests/test_transport_diffusion_validation.py`. The Python `_thomas_solve` and
  the Rust `scpn_control_rs.py_thomas_solve` (used by the Rust `transport_step`)
  agree to machine precision, validating the polyglot diffusion-solve chain.
  This validates the diffusion discretisation and linear solver against analytic
  references; facility-calibrated integrated-modelling claims still require a
  measured discharge or published benchmark.

### Chang-Hinton Neoclassical Transport
$$\chi_i = 0.66 (1 + 1.54 \epsilon) q^2 \rho_i^2 \nu_{ii} / (\epsilon^{3/2} (1 + 0.74 \nu_*^{2/3}))$$

- **Source**: Chang & Hinton, *Phys. Fluids* 25, 1493 (1982) Eq. 10.
- **Implementation**: `src/scpn_control/core/integrated_transport_solver.py:85`.
- **Simplifications**: Simplified aspect ratio and collisionality dependence.

### Sauter Bootstrap Current
- **Source**: Sauter et al., *Phys. Plasmas* 6, 2834 (1999).
- **Implementation**: `src/scpn_control/core/integrated_transport_solver.py:125`.
- **Simplifications**: Uses analytic fit for trapped particle fraction $f_t$.

### Toroidal Momentum Transport and Rotation
Neutral-beam torque, the neoclassical radial electric field, and the E×B
shearing rate that set the rotation profile and turbulence suppression.

$$E_r = \frac{1}{Z_i e n_i}\frac{dp_i}{dr} + v_\phi B_\theta, \qquad
  \omega_{E\times B} = \left|\frac{R_0 B_\theta}{B}\frac{d\omega_\phi}{dr}\right|,
  \qquad v_{\phi,{\rm intr}} = 3.5\,\frac{W_p}{I_p}$$

- **Source**: Hinton & Hazeltine, *Rev. Mod. Phys.* 48, 239 (1976); Burrell,
  *Phys. Plasmas* 4, 1499 (1997); Rice et al., *Nucl. Fusion* 47, 1618 (2007).
- **Implementation**: `src/scpn_control/core/momentum_transport.py:167`.
- **Validation**: The production `nbi_torque`, `radial_electric_field`,
  `exb_shearing_rate`, `turbulence_suppression_factor`, `rice_intrinsic_velocity`,
  and `RotationDiagnostics.mach_number` are checked against their exact closed
  forms — the NBI torque geometry, the Hinton-Hazeltine force balance (exact for
  constant and linear pressure, where `np.gradient` is exact), the Burrell E×B
  shearing rate (exact for a linear rotation profile), the Biglari-Diamond-Terry
  suppression factor, the Rice $W_p/I_p$ scaling, and the toroidal Mach number —
  all to machine precision, in `validation/validate_momentum_transport.py` with
  tests in `tests/test_momentum_transport_validation.py`. Facility momentum-
  transport claims still require measured NBI rotation cases.

---

## 3. Radiation and Sinks

### Bremsstrahlung Radiation
$$P_{br} = 5.35 \times 10^{-37} n_e^2 Z_{eff} T_e^{1/2} \quad [W/m^3]$$

- **Source**: Wesson, *Tokamaks* (2011) Ch. 14.5.1.
- **Implementation**: `src/scpn_control/core/integrated_transport_solver.py:785`, `src/scpn_control/control/tokamak_digital_twin.py:175`.
- **Simplifications**: Pure Bremsstrahlung; assumes Maxwellian distribution.

### Tungsten Line Radiation
Piecewise power-law fit to ADAS data for tungsten in coronal equilibrium.

- **Source**: Pütterich et al., *Nucl. Fusion* 50, 025012 (2010).
- **Implementation**: `src/scpn_control/core/integrated_transport_solver.py:755`.
- **Simplifications**: Coronal equilibrium (no transport effects on charge states).

### Two-Point Scrape-Off-Layer Model
Upstream and target conditions from the Stangeby two-point model with the
Spitzer-Härm parallel conduction integral, the Eich heat-flux-width regression,
and the sheath-limited target temperature.

$$q_\parallel = \frac{\kappa_0 T_u^{7/2}}{(7/2) L_\parallel}, \qquad
  L_\parallel = \pi q_{95} R_0, \qquad n_u T_u = 2 n_t T_t$$

- **Source**: Stangeby, *The Plasma Boundary of Magnetic Fusion Devices* (2000);
  Eich et al., *Nucl. Fusion* 53, 093031 (2013).
- **Implementation**: `src/scpn_control/core/sol_model.py:111`.
- **Validation**: The production `TwoPointSOL.solve`, `eich_heat_flux_width`,
  `peak_target_heat_flux`, and `detachment_threshold` are checked against their
  exact closed forms — the connection length $L_\parallel = \pi q_{95} R_0$, the
  parallel-flux mapping, the Spitzer-Härm upstream conduction integral, the
  pressure balance $n_u T_u = 2 n_t T_t$, the Eich regression exponents
  ($P^{-0.02}$, $R^{0.04}$, $B_{\rm pol}^{-0.92}$, $\varepsilon^{0.42}$), the
  peak-heat-flux geometry, and the sheath-limited detachment density boundary —
  all to machine precision, in `validation/validate_sol_two_point.py` with tests
  in `tests/test_sol_two_point_validation.py`. Facility-validated edge-transport
  or divertor-heat-load claims still require measured probe-campaign or published
  reference artefacts.

---

## 4. Scaling Laws

### IPB98(y,2) Confinement Scaling
$$\tau_E = 0.0562 \cdot I_p^{0.93} \cdot B_T^{0.15} \cdot \bar{n}_e^{0.41} \cdot M^{0.19} \cdot R^{1.97} \cdot \varepsilon^{0.58} \cdot \kappa^{0.78} \cdot P^{-0.69}$$

- **Source**: ITER Physics Basis, *Nucl. Fusion* 39, 2175 (1999).
- **Implementation**: `src/scpn_control/core/scaling_laws.py:45`.
- **Simplifications**: Global fit; does not capture local profile effects.

### Greenwald Density Limit
$$n_G = \frac{I_p}{\pi a^2} \quad [10^{20} m^{-3}]$$

- **Source**: Greenwald, *Plasma Phys. Control. Fusion* 44, R27 (2002).
- **Implementation**: `src/scpn_control/control/disruption_predictor.py:150`.

---

## 5. Control and Dynamics

### Vertical Stability Growth Rate
Estimated from Naydon instability timescale for elongated plasmas.

- **Source**: Naydon et al., *Phys. Plasmas* (2005).
- **Implementation**: `src/scpn_control/control/h_infinity_controller.py:115`.

### RZIP Rigid Vertical Stability
Linearised rigid-plasma vertical response with the destabilising curvature
spring $K = n\,\mu_0 I_p^2 / (4\pi R_0)$ (field decay index $n$) coupled to the
vessel/coil circuit currents; the growth rate is the largest real eigenvalue of
the state-space matrix for $x = [Z, \dot Z, I_1, \ldots]$.

- **Source**: Lazarus et al., *Nucl. Fusion* 30, 111 (1990); Wesson,
  *Tokamaks* (2011) Ch. 3.10.
- **Implementation**: `src/scpn_control/control/rzip_model.py:251`.
- **Validation**: In the no-wall limit the rigid mode reduces to the exact
  $2\times2$ block with eigenvalues $\pm\sqrt{-K/M_{\mathrm{eff}}}$, so the
  production `vertical_growth_rate` reproduces $\gamma=\sqrt{-n\,\mu_0 I_p^2/
  (4\pi R_0 M_{\mathrm{eff}})}$ for $n<0$, the oscillation frequency
  $\sqrt{K/M_{\mathrm{eff}}}$ for $n>0$, the exact $I_p$, $\sqrt{-n}$, and
  $1/\sqrt{M_{\mathrm{eff}}}$ scaling laws (all to $\sim10^{-16}$ relative), and
  a passive resistive wall reduces the growth rate below the no-wall value, in
  `validation/validate_rzip_vertical_stability.py` with tests in
  `tests/test_rzip_vertical_stability_validation.py`. Facility-validated
  vertical-control claims still require a matched RZIP/CREATE-L/TSC or measured
  vertical-displacement benchmark.

### Ideal-MHD Stability Metrics
Troyon normalised-beta limit, Mercier interchange index, ballooning
first-stability boundary, and the Kruskal-Shafranov external-kink criterion.

$$\beta_N = \frac{100\,\beta_t\, a B_0}{I_p}, \qquad
  D_M = s(s - 1) + \alpha\left(1 - \tfrac{s}{2}\right), \qquad
  \alpha_{\rm crit} = \begin{cases} s(1 - s/2) & s < 1 \\ 0.6\,s & s \ge 1 \end{cases}$$

- **Source**: Troyon et al., *Plasma Phys. Control. Fusion* 26, 209 (1984);
  Freidberg, *Ideal MHD* (2014) Ch. 12; Connor, Hastie & Taylor, *Phys. Rev.
  Lett.* 40, 396 (1978).
- **Implementation**: `src/scpn_control/core/stability_mhd.py:319`.
- **Validation**: The production `troyon_beta_limit`, `mercier_stability`,
  `ballooning_stability`, and `kruskal_shafranov_stability` are checked against
  their exact closed forms — the Troyon $\beta_N$ limit with its $\beta_t$, $a$,
  $B_0$, $1/I_p$ scaling, the Mercier index, the Connor-Hastie-Taylor ballooning
  boundary, and the Kruskal-Shafranov $q_{\rm edge} > 1$ criterion — all to
  machine precision with consistent stability flags, in
  `validation/validate_mhd_stability.py` with tests in
  `tests/test_mhd_stability_validation.py`. Full ideal- or resistive-MHD
  eigenmode claims still require an independent MHD stability code or benchmark
  profiles.

### EPED Pedestal Model
Self-consistent pedestal pressure and width from the peeling-ballooning and
kinetic-ballooning-mode constraints, with a collisionality width-narrowing
correction.

$$\Delta_{\rm KBM} = C_{\rm KBM}\sqrt{\beta_{p,\rm ped}}, \qquad
  p_{\rm ped} = \frac{\alpha_{\rm crit} B_0^2 a \Delta}{2\mu_0 q_{95}^2 R_0},
  \qquad \Delta_{\rm eff} = \frac{\Delta_{\rm KBM}}{1 + 0.4\ln(1 + \nu^*_e)}$$

- **Source**: Snyder et al., *Phys. Plasmas* 16, 056118 (2009); *Nucl. Fusion*
  51, 103016 (2011).
- **Implementation**: `src/scpn_control/core/eped_pedestal.py:252`.
- **Validation**: The production `eped1_predict` and helpers are checked against
  their exact construction relations — the $q_{95}$ formula, the
  $\alpha$-inversion pedestal pressure, the poloidal-beta definition, the
  ideal-gas temperature, the collisionality width narrowing (with the
  $\nu^*=0$ identity), and the shaping-factor reference — all to machine
  precision, plus the KBM width constraint
  $\Delta_{\rm KBM} = C_{\rm KBM}\sqrt{\beta_{p,\rm ped}}$ satisfied at the
  converged collisionless width within the fixed-point iteration tolerance, in
  `validation/validate_eped_pedestal.py` with tests in
  `tests/test_eped_pedestal_validation.py`. Externally validated EPED-database
  claims still require measured pedestal data or published benchmark points.

### ELM Peeling-Ballooning Crash
Edge-localised-mode onset from the coupled peeling-ballooning boundary and a
Type-I crash that sheds a fixed fraction of the pedestal stored energy.

$$\left(\frac{j_{\rm edge}}{j_{\rm crit}}\right)^2 +
  \left(\frac{\alpha_{\rm edge}}{\alpha_{\rm crit}}\right)^2 > 1, \qquad
  \Delta W = f\,W_{\rm ped}, \qquad
  T, n \to T, n\,\sqrt{1 - f}\ \Rightarrow\ W_{\rm post} = (1 - f)\,W_{\rm ped}$$

- **Source**: Snyder et al., *Phys. Plasmas* 9, 2037 (2002); Loarte et al.,
  *Plasma Phys. Control. Fusion* 45, 1549 (2003).
- **Implementation**: `src/scpn_control/core/elm_model.py:26`.
- **Validation**: The production `PeelingBallooningBoundary` and `ELMCrashModel`
  are checked against their exact closed forms — the ballooning $\alpha_{\rm
  crit}$ and peeling $j_{\rm crit}$ limits with their $1/q_{95}$,
  $1/\sqrt{n}$, and $R_0/a$ scalings, the elliptical stability margin
  $1 - \sqrt{(j/j_{\rm crit})^2 + (\alpha/\alpha_{\rm crit})^2}$ (zero on the
  unit ellipse, sign-consistent with `is_unstable`), and the Type-I crash energy
  conservation $W_{\rm post} = (1 - f)\,W_{\rm ped}$ — all to machine precision,
  in `validation/validate_elm_peeling_ballooning.py` with tests in
  `tests/test_elm_peeling_ballooning_validation.py`. Facility ELM/RMP claims
  still require measured H-mode campaign data or published ELM cases.

### Halo-Current L/R Circuit
Post-disruption halo current driven through the wall by the decaying plasma
current during a current quench, modelled as a Fitzpatrick-style L/R circuit
with closed-form circuit parameters and an electromagnetic wall-force estimate.

$$L_h \frac{\mathrm{d} I_h}{\mathrm{d} t} + R_h I_h = M
  \left| \frac{\mathrm{d} I_p}{\mathrm{d} t} \right|, \qquad
  R_h = \frac{\eta\, 2\pi R_0}{d_{\rm wall}\, a\, f_{\rm contact}}, \qquad
  L_h = \mu_0 R_0 \left( \ln \frac{8 R_0}{a} - 1.5 \right)$$

$$M = f_{\rm contact} \sqrt{L_p L_h}, \qquad \tau_h = \frac{L_h}{R_h}, \qquad
  F = \frac{\mu_0 I_{h,\rm peak} I_{p0}}{2\pi a}$$

- **Source**: Fitzpatrick, *Phys. Plasmas* 9, 3459 (2002); Wesson, *Tokamaks*,
  4th ed., Oxford University Press, Ch. 7 (2011).
- **Implementation**: `src/scpn_control/control/halo_re_physics.py:213`.
- **Validation**: The production `HaloCurrentModel` is checked against its exact
  closed forms — the halo resistance $R_h$, the halo inductance $L_h$, the mutual
  inductance $M$, and the time constant $\tau_h = L_h/R_h$, together with the
  $R_h$ scaling laws (linear in $\eta$ and $R_0$, inverse in $f_{\rm contact}$
  and $d_{\rm wall}$), the simulated electromagnetic wall force $F$, and the
  toroidal-peaking product — all to machine precision, plus the fast-circuit
  quasi-static limit in which the halo current tracks $M |\mathrm{d} I_p/\mathrm{d}
  t| / R_h$ with an error that decreases monotonically as $\tau_h/\tau_{cq} \to
  0$, in `validation/validate_halo_current.py` with tests in
  `tests/test_halo_current_validation.py`. Facility mitigation claims still
  require measured disruption-campaign data.

### Runaway-Electron Avalanche
Post-disruption runaway-electron generation from the Connor-Hastie critical and
Dreicer fields and the Rosenbluth-Putvinski avalanche multiplication.

$$E_c = \frac{n_e e^3 \ln\Lambda}{4\pi\varepsilon_0^2 m_e c^2}, \qquad
  E_D = \frac{n_e e^3 \ln\Lambda}{4\pi\varepsilon_0^2 T_e}, \qquad
  \gamma_{\rm av} = \frac{n_{\rm RE}(E/E_c - 1)}{\tau_{\rm av}\ln\Lambda}$$

- **Source**: Connor & Hastie, *Nucl. Fusion* 15, 415 (1975); Rosenbluth &
  Putvinski, *Nucl. Fusion* 37, 1355 (1997).
- **Implementation**: `src/scpn_control/control/halo_re_physics.py:327`.
- **Validation**: The production `RunawayElectronModel` is checked against its
  exact closed forms — the critical field $E_c$ (with total free-plus-bound
  electron density), the Dreicer field $E_D$, the collision time, the avalanche
  time constant $\tau_{\rm av}$ (with $Z_{\rm eff}$ enhancement), and the
  Rosenbluth-Putvinski avalanche rate (zero below $E_c$, linear in $n_{\rm RE}$
  and $(E/E_c-1)$, with the 0.001 RMP deconfinement factor) — all to machine
  precision, in `validation/validate_runaway_electron.py` with tests in
  `tests/test_runaway_electron_validation.py`. Facility mitigation claims still
  require measured disruption-campaign data.

### Resistive-Wall-Mode Feedback
Wall-limited growth rate with rotation stabilisation and active PD feedback for
the $n=1$ resistive wall mode between the no-wall and ideal-wall $\beta$ limits.

$$\gamma_{\mathrm{wall}} = \frac{1}{\tau_{\mathrm{eff}}}
  \frac{\beta_N - \beta_{N,\mathrm{nw}}}{\beta_{N,\mathrm{w}} - \beta_N}, \qquad
  \tau_{\mathrm{eff}} = \tau_{\mathrm{wall}} (b/d)^2$$

- **Source**: Bondeson & Ward, *Phys. Rev. Lett.* 72, 2709 (1994); Fitzpatrick,
  *Phys. Plasmas* 8, 4489 (2001); Garofalo et al., *Phys. Plasmas* 9, 1997
  (2002).
- **Implementation**: `src/scpn_control/control/rwm_feedback.py:295`.
- **Validation**: The production `growth_rate`, `tau_eff`, `critical_rotation`,
  `effective_growth_rate`, and `required_feedback_gain` are checked against their
  exact closed forms — the Bondeson-Ward growth rate, the wall-gap $\tau_{\rm
  eff}$, the Fitzpatrick rotation term, the critical-rotation marginality
  ($\gamma=0$ at $\Omega_{\rm crit}$), and the feedback marginalisation
  ($\gamma_{\rm eff}=0$ at the required gain $(1+\gamma\tau_{\rm ctrl})/M_{\rm
  coil}$) — all to $\sim10^{-16}$ relative, plus the no-wall/ideal-kink window
  boundaries and $1/\tau_{\rm wall}$ scaling, in
  `validation/validate_rwm_feedback.py` with tests in
  `tests/test_rwm_feedback_validation.py`. Facility-validated MHD-stability or
  hardware-control claims still require measured RWM shots or an external MHD
  stability reference.

### Kadomtsev Sawtooth Crash
Full-reconnection sawtooth crash: inside the mixing radius the temperature and
density are replaced by their volume averages and $q$ is reset to one, with the
mixing radius fixed by the helical-flux proxy
$\psi^*(\rho) = \int_0^\rho \rho'(1/q - 1)\,d\rho'$ returning to zero outside the
$q=1$ surface.

- **Source**: Kadomtsev, *Sov. J. Plasma Phys.* 1, 389 (1975); Porcelli et al.,
  *Plasma Phys. Control. Fusion* 38, 2163 (1996).
- **Implementation**: `src/scpn_control/core/sawtooth.py:279`.
- **Validation**: The production `kadomtsev_crash` is checked against exact
  conservation laws — the volume integrals $\int T\rho\,d\rho$ and
  $\int n\rho\,d\rho$ over the mixing region are conserved to machine precision,
  the helical-flux proxy vanishes at the mixing radius
  ($\psi^*(\rho_{\rm mix})=0$), profiles flatten inside and stay invariant
  outside, and the interpolated $q=1$ radius converges at second order to the
  analytic $\rho_1 = \sqrt{(1-q_0)/(q_a-q_0)}$ — in
  `validation/validate_sawtooth_kadomtsev.py` with tests in
  `tests/test_sawtooth_kadomtsev_validation.py`. Full nonlinear MHD sawtooth-crash
  or measured-shot claims still require a measured or published reference.

### Auxiliary Current Drive
Electron-cyclotron, lower-hybrid, and neutral-beam sources with grid-normalised
radial deposition, the Prater ECCD figure of merit, and the Stix beam
slowing-down time and critical energy.

$$\eta_{\rm ECCD} = \eta_0 \frac{T_e}{5 + Z_{\rm eff}}
  \frac{\xi}{1 + \xi^2}, \qquad
  E_{\rm crit} = 14.8\, T_e (A_b/A_i)^{2/3}, \qquad
  \tau_s = \frac{3\sqrt{2\pi}\, m_i T_e^{3/2}}{4\sqrt{m_e}\, n_e e^4 \ln\Lambda\, Z_{\rm eff}}$$

- **Source**: Fisch & Boozer, *Phys. Rev. Lett.* 45, 720 (1980); Prater,
  *Phys. Plasmas* 11, 2349 (2004); Stix, *Plasma Physics* 14, 367 (1972).
- **Implementation**: `src/scpn_control/core/current_drive.py:294`.
- **Validation**: The production `ECCDSource`/`LHCDSource`/`NBISource`,
  `eccd_efficiency`, `nbi_critical_energy`, and `nbi_slowing_down_time` are
  checked against their exact closed forms — grid-normalised deposition power
  conservation ($\int P\,d\rho = P_{\rm source}$), the deposition centroid, the
  Stix critical energy and slowing-down scalings ($T_e^{3/2}$, $1/n_e$,
  $1/Z_{\rm eff}$), the Prater efficiency with the launch-angle factor maximised
  at $N_\parallel = 1$, the driven-current proportionality
  $j_{\rm cd} = \eta_{\rm cd} P_{\rm abs}/(n_e T_e)$, and the neutral-beam
  fast-ion current chain — all to machine precision, in
  `validation/validate_current_drive.py` with tests in
  `tests/test_current_drive_validation.py`. External current-drive claims still
  require ray-tracing, Fokker-Planck, or measured-deposition artefacts.

### Kuramoto-Sakaguchi Phase Dynamics
$$\frac{d\theta_i}{dt} = \omega_i + K R \sin(\psi - \theta_i - \alpha) + \zeta \sin(\Psi - \theta_i)$$

- **Source**: Kuramoto (1975), Sakaguchi & Kuramoto (1986).
- **Implementation**: `src/scpn_control/phase/kuramoto.py:80`.
- **Simplifications**: Mean-field coupling.
- **Validation**: Synchronisation onset and the partially synchronised
  order-parameter branch are checked against the exact mean-field Lorentzian
  results — critical coupling `Kc(α) = 2γ/cos α` and `R∞(K) = sqrt(1 − Kc/K)` —
  in `validation/validate_kuramoto_synchronisation.py`, with tests in
  `tests/test_kuramoto_synchronisation_validation.py`. This validates the
  synchronisation physics only; it is not a validated plasma-phase control law.

### What this section is for

This reference page is used as a physics starting point for implementation and
review, not as a standalone guarantee of facility accuracy. Its practical role is
to align equations, files, and assumptions before benchmark and validation
evidence upgrades are claimed in production-facing documents.

## How to use this methods page in reviews

This page is the assumptions ledger for model code.

Use it when:

- You need a reproducible mapping from equation to file path.
- You need to confirm what is modeled as simplified and what remains approximate.
- You need to decide whether external-code comparisons are already required for your claim.

Physics pages are not admission gates by themselves. Admission is granted only when matching validators admit the corresponding evidence bundle.

## Practical use and scope

Use this file to trace implemented physics equations to their solver locations.

- Read the model entries before changing equilibrium, transport, or profile settings.
- Use this page to confirm which simplifications are active for a given configuration.
- Validate physics claim scope using `docs/validation.md` before changing public-facing statements.
