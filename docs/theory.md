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

## H-infinity Robust Control

The H-infinity controller solves the DARE (Discrete Algebraic Riccati Equation)
for the worst-case disturbance attenuation level $\gamma$:

$$
A^{\top} X A - X - A^{\top} X B (R + B^{\top} X B)^{-1} B^{\top} X A + Q = 0
$$

where $R = \text{diag}(R_u, -\gamma^2 I_w)$ partitions control and disturbance
inputs. The controller gain $F = -(R + B^{\top} X B)^{-1} B^{\top} X A$
guarantees $\|T_{zw}\|_\infty < \gamma$.

Anti-windup: output is clipped to $[-u_{\max}, u_{\max}]$ and the integrator
state is conditionally frozen.

Implementation: `scpn_control.control.h_infinity_controller.HInfinityController`.
