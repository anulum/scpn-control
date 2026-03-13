# Glossary

Terms from fusion plasma physics and control theory used in scpn-control.

---

## A

**Alpha heating** — Self-heating of plasma by 3.52 MeV alpha particles produced in D-T fusion reactions. Becomes the dominant heat source when Q > 5. *MW.* `core.integrated_transport_solver`

**Anomalous transport** — Cross-field particle and heat transport driven by turbulent fluctuations, exceeding neoclassical predictions by 1-2 orders of magnitude. Modelled with gyro-Bohm scaling in scpn-control. *m^2/s.* `core.gyrokinetic_transport`

**Anti-windup** — Controller modification preventing integral wind-up when actuator saturation occurs. The super-twisting SMC and PI controllers in scpn-control clamp the integrator state. `control.sliding_mode_vertical`

**Aspect ratio (A)** — R0/a, the ratio of major to minor radius. ITER: 3.1, SPARC: 3.25. *Dimensionless.* `core.tokamak_config`

## B

**Ballooning mode** — Pressure-driven MHD instability localised on the outboard (bad curvature) side of a tokamak. The infinite-n ideal ballooning criterion sets the critical pressure gradient. `core.ballooning_solver`

**Beta (beta_N, beta_p, beta_t)** — Ratio of plasma kinetic pressure to magnetic pressure. beta_t = 2 mu_0 <p> / B0^2 (toroidal), beta_p = 2 mu_0 <p> / B_pol^2 (poloidal), beta_N = beta_t / (Ip/aB0) (normalised, Troyon). *Dimensionless; beta_N in % MA/m/T.* `core.stability_mhd`

**Bootstrap current** — Self-generated toroidal current driven by trapped-particle pressure gradients in the banana regime. Modelled by the Sauter formula (Phys. Plasmas 6, 2834, 1999). *A/m^2.* `core.integrated_transport_solver`

**Bremsstrahlung** — Electromagnetic radiation from electron-ion Coulomb collisions. Scales as n_e^2 Z_eff T_e^{1/2}. A net power loss in the energy balance. *MW/m^3.* `core.integrated_transport_solver`

**Burning plasma** — A plasma where alpha heating exceeds external heating, i.e. Q > 5. The primary target of ITER and SPARC.

## C

**Confinement time (tau_E)** — Energy confinement time: ratio of stored energy to total heating power minus dW/dt. Characterises how well the plasma retains energy. *s.* `core.scaling_laws`

**Controllability** — A state x is controllable if there exists an input sequence driving the system from x to the origin. Checked via the rank of [B, AB, ..., A^{n-1}B]. `control.h_infinity_controller`

**Critical gradient** — The temperature gradient threshold above which turbulent transport sharply increases (stiff transport). The critical-gradient model approximates ITG/TEM-driven chi as max(0, gradient - gradient_crit). `core.neural_transport`

**Current diffusion** — Resistive evolution of the current profile (q profile) on the skin time. Solved by a 1D diffusion equation in the normalised poloidal flux coordinate. `core.current_diffusion`

**Current drive** — Non-inductive methods of sustaining plasma current: NBI, ECCD, LHCD, bootstrap. Required for steady-state tokamak operation. *A.* `core.current_drive`

## D

**DARE (Discrete Algebraic Riccati Equation)** — The matrix equation X = A^T X A - A^T X B (R + B^T X B)^{-1} B^T X A + Q, solved to obtain LQR/H-infinity gains for a sampled-data plant. `control.h_infinity_controller`

**D-K iteration** — Iterative mu-synthesis algorithm alternating between D-scale fitting and H-infinity controller synthesis to achieve robust performance against structured uncertainty. `control.mu_synthesis`

**Detachment** — Divertor operating regime where the plasma cools and recombines before reaching the target plates, reducing peak heat flux. `core.sol_model`

**Disruption** — Abrupt loss of plasma confinement, causing rapid thermal and current quenches. Generates large halo currents and mechanical loads. The primary off-normal event in tokamak operation. `control.disruption_predictor`

**Divertor** — Magnetic geometry that directs scrape-off-layer flux to dedicated target plates for power exhaust and impurity control. `core.sol_model`

## E

**ECCD (Electron Cyclotron Current Drive)** — Current driven by asymmetric electron heating at electron cyclotron resonance. Used for NTM stabilisation by depositing current at the rational surface. *A.* `core.current_drive`

**ELM (Edge Localised Mode)** — Repetitive MHD instability at the H-mode pedestal, expelling 5-15% of stored energy per burst. Type-I ELMs are peeling-ballooning driven. `core.stability_mhd`

**Elongation (kappa)** — Vertical elongation of the plasma cross-section: ratio of vertical to horizontal extent. ITER: 1.7, SPARC: 1.97. Increases beta limits but requires active vertical stabilisation. *Dimensionless.* `core.tokamak_config`

**Energy confinement scaling (IPB98y,2)** — ITER Physics Basis scaling law tau_E = C I_p^{0.93} B^{0.15} n^{0.41} P^{-0.69} ... derived from multi-machine regression. *s.* `core.scaling_laws`

## F

**Fault detection and isolation (FDI)** — Real-time detection of actuator or sensor failures and reconfiguration of the control system to maintain stability. `control.fault_tolerant_control`

**Flux coordinates** — Coordinate system (psi, theta_p, phi) aligned to magnetic flux surfaces. Eliminates the magnetic field's cross-surface component, simplifying transport equations. `core.fusion_kernel`

**Flux surface** — A toroidal surface of constant poloidal flux psi on which magnetic field lines lie. Equilibrium profiles (p, q, J) are flux functions. `core.fusion_kernel`

**Free-boundary equilibrium** — Grad-Shafranov solution where the plasma boundary is self-consistently determined by external coil currents, not imposed. `core.fusion_kernel`

## G

**Gain margin** — The factor by which loop gain can increase before the closed-loop system goes unstable. Measured in dB. H-infinity synthesis guarantees >= 6 dB. `control.h_infinity_controller`

**Grad-Shafranov equation** — The elliptic PDE for axisymmetric toroidal equilibrium: Delta* psi = -mu_0 R^2 p'(psi) - F F'(psi), where Delta* is the toroidal Laplacian operator. Solved by Picard iteration. `core.fusion_kernel`

**Greenwald limit** — Empirical density limit n_G = I_p / (pi a^2) [10^20 m^-3]. Exceeding it triggers disruptions. *10^20 m^-3.* `core.scaling_laws`

**Gyrokinetic transport** — First-principles turbulent transport modelling based on the gyrokinetic Vlasov equation. Approximated in scpn-control via critical-gradient and QLKNN-10D surrogate models. `core.gyrokinetic_transport`

## H

**H-mode (High confinement mode)** — Operating regime with an edge transport barrier (pedestal), achieving ~2x the L-mode confinement time. Accessed by exceeding a power threshold. `core.fusion_kernel`

**H-infinity control** — Robust control synthesis minimising the worst-case disturbance-to-performance transfer function norm. Solved via two continuous Riccati equations, then discretised by ZOH. `control.h_infinity_controller`

**Halo current** — Current flowing through open field lines and the vessel wall during vertical displacement events. Generates large electromagnetic forces. *A.* `control.halo_re_physics`

## I

**ITER** — International Thermonuclear Experimental Reactor. R0=6.2 m, B0=5.3 T, Ip=15 MA, target Q=10. The reference configuration in scpn-control. `core.tokamak_config`

**Island width** — Half-width of a magnetic island at a rational surface, governed by the Modified Rutherford Equation. Controls NTM stability. *m.* `core.ntm_dynamics`

**Isoflux** — Shape control strategy where external coil currents are adjusted to match the poloidal flux at target control points on the desired plasma boundary. `control.tokamak_flight_sim`

## K

**Kink mode** — Current-driven MHD instability. External kinks limit q_edge > 2; internal kinks (m=1, n=1) trigger sawteeth when q_axis < 1. `core.stability_mhd`

**Kuramoto model** — Coupled oscillator model: d theta_i/dt = omega_i + (K/N) sum_j sin(theta_j - theta_i). Exhibits a synchronisation phase transition at critical coupling K_c. `phase.kuramoto`

## L

**L-mode (Low confinement mode)** — Standard tokamak operating regime without an edge transport barrier. Confinement time follows the L-mode scaling. `core.fusion_kernel`

**Lagrangian PPO** — Proximal Policy Optimisation with Lagrangian constraint enforcement for safe reinforcement learning. Maintains a Lagrange multiplier for each safety constraint (e.g. beta_N < Troyon limit). `control.safe_rl_controller`

**Lawson criterion** — Condition for ignition: n tau_E T > ~3 x 10^21 m^-3 s keV for D-T. Defines the minimum triple product for net energy gain.

**LIF neuron (Leaky Integrate-and-Fire)** — Neuron model: dv/dt = -v/tau + I. Fires when v crosses threshold, then resets. Used as the transition-firing element in the SPN-to-SNN compilation. `scpn.compiler`

**Lyapunov stability** — A system is Lyapunov stable if trajectories starting near an equilibrium remain near it. The Lyapunov guard in scpn-control monitors V(theta) = 1 - R(t) and triggers alerts when dV/dt > 0. `phase.lyapunov_guard`

## M

**MHD (Magnetohydrodynamics)** — Fluid description of plasma. Ideal MHD governs kink, ballooning, and peeling modes; resistive MHD governs tearing and NTM. `core.stability_mhd`

**MPC (Model Predictive Control)** — Receding-horizon optimal control: solve a finite-horizon optimisation at each time step using a plant model, apply the first control action, repeat. `control.fusion_sota_mpc`

**Modified Rutherford Equation (MRE)** — ODE governing NTM island width evolution: dw/dt = eta/(mu_0 r_s) [Delta'(w) + Delta_bs(w) + Delta_ECCD(w)]. Predicts when islands grow or shrink. `core.ntm_dynamics`

**Mu-synthesis (mu-synthesis)** — Robust control framework that accounts for structured (parametric) uncertainty. Uses D-K iteration to minimise the structured singular value mu. `control.mu_synthesis`

## N

**NBI (Neutral Beam Injection)** — Heating and current drive method injecting high-energy neutral atoms (50-500 keV) into the plasma. *MW.* `core.current_drive`

**NTM (Neoclassical Tearing Mode)** — Resistive MHD instability at rational surfaces (q = m/n) sustained by a bootstrap current deficit inside the island. `core.ntm_dynamics`

**Neural transport** — Machine-learning surrogate for turbulent transport coefficients. The critical-gradient model and QLKNN-10D input space are implemented. `core.neural_transport`

## O

**Observability** — A state x is observable if it can be determined from output measurements over a finite interval. Dual of controllability. `control.state_estimator`

**Ohmic heating** — Resistive heating from the plasma current: P_ohm = eta J^2. Dominates at low temperature; diminishes as T_e^{3/2} (Spitzer resistivity). *MW/m^3.* `core.integrated_transport_solver`

**Order parameter (R)** — Kuramoto order parameter: R exp(i psi) = (1/N) sum_j exp(i theta_j). R=0 is incoherence, R=1 is full synchrony. *Dimensionless.* `phase.kuramoto`

## P

**PAC (Phase-Amplitude Coupling)** — Cross-frequency coupling where the phase of a low-frequency oscillation modulates the amplitude of a high-frequency one. Used for SNN closed-loop modulation. `control.advanced_soc_fusion_learning`

**Pedestal** — The steep gradient region at the plasma edge in H-mode. Pedestal height and width set the boundary condition for core transport. *keV.* `core.pedestal`

**Picard iteration** — Fixed-point iteration for solving the GS equation: psi^{k+1} = G(psi^k) with under-relaxation. `core.fusion_kernel`

**Plasma current (Ip)** — Total toroidal current in the plasma, providing the poloidal magnetic field. ITER: 15 MA, SPARC: 8.7 MA. *MA.* `core.tokamak_config`

**Poloidal** — The short way around the torus (in the R-Z plane). Poloidal magnetic field B_pol confines plasma vertically. `core.fusion_kernel`

**PPO (Proximal Policy Optimisation)** — Policy gradient RL algorithm with clipped surrogate objective. The default RL agent in `TokamakEnv`. `control.gym_tokamak_env`

## Q

**Q factor** — Fusion gain: Q = P_fusion / P_external. Q=1 is breakeven, Q=10 is the ITER target, Q=infinity is ignition.

**q (safety factor)** — q = d phi / d theta_p, the number of toroidal transits per poloidal transit of a field line. q < 1 at the axis triggers sawteeth; q < 2 at the edge triggers kinks. *Dimensionless.* `core.fusion_kernel`

## R

**Resistive wall mode (RWM)** — External kink stabilised only by eddy currents in the resistive wall. Grows on the wall time constant (~10 ms). Requires active feedback. `control.rwm_feedback`

**Riccati equation** — Matrix equation arising in LQR and H-infinity synthesis. Continuous: A^T X + X A - X B R^{-1} B^T X + Q = 0. Solved by `scipy.linalg.solve_continuous_are`. `control.h_infinity_controller`

## S

**Safety factor (q)** — See **q (safety factor)**.

**Sawtooth** — Periodic relaxation oscillation when q_axis < 1, caused by the internal kink mode. Flattens the core temperature profile. `core.sawtooth`

**Scrape-off layer (SOL)** — Open-field-line region outside the last closed flux surface, connecting to the divertor or limiter. Carries the exhaust power. `core.sol_model`

**SNN (Spiking Neural Network)** — Neural network operating on discrete spike events. The `FusionCompiler` maps SPN transitions to LIF neurons with bitstream-encoded weights. `scpn.compiler`

**SPN (Stochastic Petri Net)** — Bipartite graph (places, transitions) with stochastic firing rules. Marking vector m(t) encodes the system state; transition firing updates m via the incidence matrix. `scpn.structure`

**Sliding-mode control** — Robust nonlinear control driving the state to a sliding surface s=0. The super-twisting algorithm provides chattering-free second-order convergence. `control.sliding_mode_vertical`

**SPARC** — Compact high-field tokamak (R0=1.85 m, B0=12.2 T, Ip=8.7 MA). Target Q > 2 with HTS magnets. Creely et al., J. Plasma Phys. 86 (2020). `core.tokamak_config`

**Super-twisting algorithm** — Second-order sliding mode: u = -alpha |s|^{1/2} sign(s) + v, dv/dt = -beta sign(s). Finite-time convergence, continuous control signal. `control.sliding_mode_vertical`

## T

**Tearing mode** — Resistive MHD instability that reconnects magnetic field lines at rational surfaces, forming magnetic islands. Degrades confinement. `core.ntm_dynamics`

**Tokamak** — Toroidal magnetic confinement device with a strong toroidal field and a plasma current providing the poloidal field. The dominant fusion concept.

**Toroidal** — The long way around the torus (the phi direction). Toroidal magnetic field B_phi is the dominant confining field.

**Transport** — Particle, heat, and momentum fluxes across magnetic surfaces. Neoclassical transport is the irreducible minimum; anomalous (turbulent) transport dominates in practice. *m^2/s.* `core.integrated_transport_solver`

**Triangularity (delta)** — Triangular shaping of the plasma cross-section. Positive delta stabilises ballooning modes. ITER: 0.33, SPARC: 0.54. *Dimensionless.* `core.tokamak_config`

**Troyon limit** — Empirical stability limit: beta_N < g_Troyon, with g_Troyon ~ 2.8 for conventional profiles. Exceeding it triggers MHD instabilities. *% MA/m/T.* `core.stability_mhd`

## U

**UPDE (Unified Phase Dynamics Equation)** — Coupled Kuramoto system across SCPN layers: d theta_n/dt = omega_n + sum_m K_nm sin(theta_m - theta_n) + zeta sin(Psi - theta_n). Governs inter-layer synchronisation. `phase.upde`

## V

**VDE (Vertical Displacement Event)** — Loss of vertical position control, causing the plasma to move vertically until it contacts the wall. Generates halo currents and large forces. `control.sliding_mode_vertical`

**Vertical stability** — Elongated plasmas are vertically unstable with a growth rate gamma proportional to sqrt(kappa - 1). Requires active feedback with sub-millisecond response. `control.h_infinity_controller`

**Vessel model** — Lumped-element model of the vacuum vessel and passive conductors. Provides the eddy current response needed for vertical stability control design. `core.vessel_model`

## Z

**Z_eff (Effective charge)** — Ion-charge-weighted average: Z_eff = sum(n_i Z_i^2) / n_e. Increases Bremsstrahlung and resistivity. Typical range 1.2-3.0. *Dimensionless.* `core.integrated_transport_solver`
