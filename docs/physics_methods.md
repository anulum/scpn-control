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
