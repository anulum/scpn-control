# Theory

This page summarises the mathematical foundations implemented in `scpn_control.phase`.

## Stochastic Petri Net (SPN)

An SPN is a bipartite directed graph $G = (P, T, W^{-}, W^{+})$ where $P$ is a
set of places, $T$ a set of transitions, and $W^{\pm}$ are the input/output
weight matrices. The marking vector $\mathbf{m}(t) \in \mathbb{R}^{|P|}_{\ge 0}$
evolves via stochastic firing:

$$
\mathbf{m}(t + 1) = \mathbf{m}(t) + (W^{+} - W^{-})^{\top} \mathbf{f}(t)
$$

where $\mathbf{f}(t) \in \{0,1\}^{|T|}$ is the firing vector, sampled from
transition-specific hazard functions.

The `FusionCompiler` maps each place → LIF neuron membrane potential, each
transition → synaptic connection with weight from $W^{\pm}$, producing a
`CompiledNet` for closed-loop control.

## Kuramoto–Sakaguchi Model

The phase of oscillator $n$ in layer $l$ evolves as (Paper 27, Eq. 3):

$$
\dot{\theta}^{(l)}_n = \omega^{(l)}_n
  + \frac{1}{N_l} \sum_{m=1}^{N_l} K_{ll} \sin\!\bigl(\theta^{(l)}_m - \theta^{(l)}_n - \alpha\bigr)
  + \zeta_l \sin\!\bigl(\Psi - \theta^{(l)}_n\bigr)
$$

where $\omega^{(l)}_n$ are natural frequencies, $K_{ll}$ is the intra-layer
coupling, $\alpha$ is the Sakaguchi phase-lag, and $\zeta_l$ is the exogenous
global-field coupling strength.

The order parameter for layer $l$ is:

$$
R_l \, e^{i\Psi_l} = \frac{1}{N_l} \sum_{n=1}^{N_l} e^{i\theta^{(l)}_n}
$$

Implementation: `scpn_control.phase.kuramoto.kuramoto_sakaguchi_step`.

## Unified Phase Dynamics Equation (UPDE)

The UPDE extends the Kuramoto model to $L$ coupled layers with inter-layer
coupling matrix $\mathbf{K} \in \mathbb{R}^{L \times L}_{\ge 0}$:

$$
\dot{\theta}^{(l)}_n = \omega^{(l)}_n
  + \frac{1}{N_l} \sum_m K_{ll} \sin(\theta^{(l)}_m - \theta^{(l)}_n)
  + \sum_{l' \neq l} K_{ll'} R_{l'} \sin(\Psi_{l'} - \theta^{(l)}_n)
  + \zeta_l \sin(\Psi - \theta^{(l)}_n)
$$

The cross-layer term uses the mean-field approximation: layer $l'$ acts on
layer $l$ through its order parameter $R_{l'} e^{i\Psi_{l'}}$.

**PAC gating** (Phase-Amplitude Coupling): when enabled, the cross-layer
coupling $K_{ll'}$ is modulated by $R_{l'}$, suppressing influence from
incoherent layers.

Implementation: `scpn_control.phase.upde.UPDESystem`.

## Paper 27 Coupling Matrix ($K_{nm}$)

The $16 \times 16$ inter-layer coupling matrix follows exponential distance
decay with calibration anchors (Paper 27, Table 2):

$$
K_{nm} = K_{\text{base}} \, e^{-\alpha |n - m|}
$$

with $K_{\text{base}} = 0.45$, $\alpha = 0.3$, and four calibration overrides:

| Pair | $K_{nm}$ | Source |
|------|----------|--------|
| (1,2) | 0.302 | Paper 27 Table 2 |
| (2,3) | 0.201 | Paper 27 Table 2 |
| (3,4) | 0.252 | Paper 27 Table 2 |
| (4,5) | 0.154 | Paper 27 Table 2 |

Implementation: `scpn_control.phase.knm.build_knm_paper27`.

## Lyapunov Stability Monitor

A candidate Lyapunov function for the synchronised state:

$$
V(t) = \frac{1}{N} \sum_{n=1}^{N} \bigl(1 - \cos(\theta_n - \Psi)\bigr)
$$

satisfies $V \ge 0$, $V = 0$ iff $\theta_n = \Psi \;\forall n$. The guard
estimates the Lyapunov exponent $\lambda$ via finite differences over a sliding
window:

$$
\lambda \approx \frac{1}{\Delta t} \ln\!\frac{V(t)}{V(t - \Delta t)}
$$

A session is flagged unstable when $\lambda > 0$ persists for more than
`max_violations` consecutive windows.

Implementation: `scpn_control.phase.lyapunov_guard.LyapunovGuard`.

## Bosch-Hale DT Reactivity

The DT fusion reactivity $\langle \sigma v \rangle$ is estimated using the 
Padé polynomial fit (Bosch & Hale 1992):

$$
\langle \sigma v \rangle = 1.1 \times 10^{-24} \cdot \theta \cdot \sqrt{\frac{\xi}{m_r c^2 T^3}} \exp(-3 \xi)
$$

where $\xi = (\frac{B_G^2}{4\theta})^{1/3}$ and $\theta$ is the modified 
temperature. This provides accurate cross-sections from 0.1 keV to 100 keV.

Ref: Bosch, H.-S. and Hale, G. M. (1992). *Nuclear Fusion*, 32, 611.  
Implementation: `scpn_control.core.uncertainty.bosch_hale_reactivity`.

## Bremsstrahlung Radiation

Losses from electron-ion Bremsstrahlung are calculated per unit volume:

$$
P_{br} = 5.35 \times 10^{-37} n_e^2 Z_{eff} \sqrt{T_e} \quad [W/m^3]
$$

where $n_e$ is in $m^{-3}$ and $T_e$ is in keV.

Ref: Wesson, J. (2011). *Tokamaks*. 4th Edition, Chapter 14.  
Implementation: `scpn_control.control.tokamak_digital_twin.TokamakDigitalTwin`.

## Energy Conservation Diagnostic

The transport solver monitors the global energy balance error $\epsilon_W$:

$$
\epsilon_W = \frac{|(W_{n+1} - W_n) - \Delta t \sum (S_i - L_i) V_i|}{W_n}
$$

where $W = \frac{3}{2} \int (n_e T_e + n_i T_i) dV$ is the stored energy.

Implementation: `scpn_control.core.integrated_transport_solver.TransportSolver.evolve_profiles`.

## Disruption Mechanisms

1.  **NTM (Neoclassical Tearing Mode):** Evolved via the Modified Rutherford Equation 
    for island width $w$:
    $$\frac{\tau_R}{r} \frac{dw}{dt} = r \Delta' + r \beta_p \frac{L_q}{L_p} \frac{w}{w^2 + w_c^2}$$
2.  **Greenwald Limit:** Disruption risk increases as density approaches 
    $n_G = I_p / (\pi a^2)$.
3.  **VDE (Vertical Displacement Event):** Exponential growth of vertical 
    position $z$ driven by plasma elongation $\kappa$.

Implementation: `scpn_control.control.disruption_predictor.simulate_tearing_mode`.

## IPB98(y,2) Scaling Law

Global H-mode thermal energy confinement time $\tau_E$ scaling:

$$
\tau_{98,y2} = 0.0562 \, I_p^{0.93} B_T^{0.15} P^{-0.69} n_e^{0.41} M^{0.19} R^{1.97} \epsilon^{0.58} \kappa_a^{0.78}
$$

The implementation includes a $\pm 2\sigma$ uncertainty band based on the 
ITPA database variance.

Ref: ITER Physics Basis (1999). *Nuclear Fusion*, 39, 2175.  
Implementation: `scpn_control.core.scaling_laws.ipb98y2_with_uncertainty`.

## Free-Boundary Grad-Shafranov

The free-boundary equilibrium is determined by solving $\Delta^* \psi = -\mu_0 R J_\phi$ 
where the boundary value $\psi_{bc}$ is the sum of contributions from all 
external coils:

$$
\psi_{ext}(R, Z) = \sum_{i} I_i \, G(R, Z; R_i, Z_i)
$$

$G$ is the toroidal Green's function involving complete elliptic integrals 
$K(k)$ and $E(k)$.

Ref: Lao, L. L. et al. (1985). *Nuclear Fusion*, 25, 1611.  
Implementation: `scpn_control.core.fusion_kernel.FusionKernel`.
