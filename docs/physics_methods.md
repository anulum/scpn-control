# Physics Methods Reference

This document catalogs the primary physics models, equations, and scaling laws implemented in `scpn-control`.

## 1. Equilibrium (Grad-Shafranov)

### Grad-Shafranov Equation
$$\Delta^* \psi = R \frac{\partial}{\partial R}\left(\frac{1}{R}\frac{\partial \psi}{\partial R}\right) + \frac{\partial^2 \psi}{\partial Z^2} = -\mu_0 R^2 p'(\psi) - F(\psi) F'(\psi)$$

- **Source**: Grad & Rubin (1958), Shafranov (1966).
- **Implementation**: `src/scpn_control/core/fusion_kernel.py:380` (Picard + SOR).
- **Simplifications**: Axisymmetric assumption. Poloidal current $F(\psi)$ and pressure $p(\psi)$ are specified as polynomial or pedestal functions of $\psi$.

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

### Kuramoto-Sakaguchi Phase Dynamics
$$\frac{d\theta_i}{dt} = \omega_i + K R \sin(\psi - \theta_i - \alpha) + \zeta \sin(\Psi - \theta_i)$$

- **Source**: Kuramoto (1975), Sakaguchi & Kuramoto (1986).
- **Implementation**: `src/scpn_control/phase/kuramoto.py:80`.
- **Simplifications**: Mean-field coupling.
