# Control Theory Primer for Fusion Applications

**Prerequisites**: Linear algebra (eigenvalues, SVD, matrix exponential), calculus (ODEs, Laplace transform basics).

**Scope**: From PID to reinforcement learning, as implemented in `scpn-control`.

---

## 1. Why Control Is Hard in Fusion

A tokamak plasma is a magnetically confined fluid at 100+ million kelvin. Controlling it
differs fundamentally from controlling, say, a chemical reactor or an aircraft.

**Multiple coupled instabilities evolve simultaneously.**
Vertical displacement events (VDEs), edge-localised modes (ELMs), neoclassical tearing
modes (NTMs), and resistive wall modes (RWMs) each have their own timescale and spatial
structure, but they interact. An NTM island that grows large enough can trigger a
disruption; a disruption triggers a VDE. The controller must handle all of these with a
shared set of actuators.

**Timescales span four orders of magnitude.**
VDEs grow on the Alfven timescale (~1 ms). Current diffusion operates on the resistive
timescale (~1 s). Energy confinement time sits between (~0.1-1 s). Transport evolves
over seconds. A single controller architecture must close loops at 10 kHz for vertical
stability while simultaneously planning scenario trajectories over minutes.

**Actuators are constrained.**
Poloidal field coil currents are bounded by power supply voltage limits and thermal
ratings. Auxiliary heating (NBI, ECRH, ICRH) has finite power and limited spatial
steering. Gas puff valves have finite flow rates and response delays. The controller
cannot demand arbitrary inputs.

**The plant model is uncertain.**
Turbulent transport depends on local gradients in ways that first-principles models
(gyrokinetic codes) can predict only at enormous computational cost. Impurity profiles
(tungsten, beryllium) are partially observable and change the radiation losses. Pedestal
height in H-mode depends on micro-stability thresholds that vary shot to shot. Any
controller that assumes perfect knowledge of the plant will fail.

**Sensors are imperfect.**
Magnetic diagnostics measure field integrals, not pointwise quantities. Thomson
scattering gives temperature and density profiles but fires discretely (typically
10-100 Hz). Interferometry gives line-integrated density. Bolometry gives line-integrated
radiation. The controller must reconstruct a high-dimensional state from partial,
noisy, delayed measurements.

---

## 2. Classical Control: PID and State-Space

### PID Control

The proportional-integral-derivative controller computes an actuator command from the
tracking error $e(t) = x_{\text{ref}}(t) - x(t)$:

$$
u(t) = K_p \, e(t) + K_i \int_0^t e(\tau) \, d\tau + K_d \, \frac{de}{dt}
$$

$K_p$ drives the error toward zero. $K_i$ eliminates steady-state offset. $K_d$ adds
damping. For a single-input single-output (SISO) linear system, PID is often adequate.

**Why PID alone fails in fusion:**

1. **MIMO coupling.** Plasma position, shape, current, density, and temperature are
   coupled. Changing heating power affects both temperature and current profile. PID on
   each channel independently ignores cross-coupling, leading to instability or sluggish
   response.
2. **Nonlinearity.** The L-H transition is a bifurcation. Confinement jumps
   discontinuously. A controller tuned for L-mode becomes unstable at the transition.
3. **Constraints.** PID has no mechanism to respect actuator limits. Integral windup
   occurs when the actuator saturates: the integral term accumulates error that cannot
   be acted upon, causing large overshoot when the constraint releases.

### State-Space Representation

Modern control works with the state-space form:

$$
\dot{x} = A x + B u, \qquad y = C x
$$

where $x \in \mathbb{R}^n$ is the state vector (plasma current, temperatures, density,
position), $u \in \mathbb{R}^m$ is the control input, $y \in \mathbb{R}^p$ is the
measured output, and $A, B, C$ are matrices derived from linearising the plant dynamics
around an operating point.

**Controllability.** The pair $(A, B)$ is controllable if and only if the controllability
matrix has full rank:

$$
\mathcal{C} = \begin{bmatrix} B & AB & A^2 B & \cdots & A^{n-1} B \end{bmatrix}, \qquad \text{rank}(\mathcal{C}) = n
$$

If the system is controllable, there exists a state feedback $u = -K x$ that places
the closed-loop eigenvalues of $A - BK$ at any desired locations. Unstable eigenvalues
(positive real part) must be moved to the left half-plane.

**Observability.** Dually, $(A, C)$ is observable if:

$$
\mathcal{O} = \begin{bmatrix} C \\ CA \\ \vdots \\ CA^{n-1} \end{bmatrix}, \qquad \text{rank}(\mathcal{O}) = n
$$

An observable system allows full state reconstruction from output measurements via a
Luenberger observer or Kalman filter.

**Eigenvalue analysis.** The open-loop eigenvalues of $A$ determine stability. For
vertical stability, $A$ typically has one unstable eigenvalue with growth rate
$\gamma \sim 10^2 - 10^3 \, \text{s}^{-1}$. The controller must act faster than
$1/\gamma$ to stabilise the mode.

### Gain-Scheduled PID

A tokamak shot passes through distinct operating regimes: ramp-up, L-mode flat-top,
L-H transition, H-mode flat-top, ramp-down, and (if things go wrong) disruption
mitigation. Each regime has different dynamics. Gain scheduling switches PID gains
based on the detected regime.

The `RegimeDetector` classifies the current operating point from $dI_p/dt$, energy
confinement time $\tau_E$, and disruption probability. The `GainScheduledController`
interpolates gains during regime transitions via bumpless transfer:

$$
K(t) = (1 - \alpha) \, K_{\text{old}} + \alpha \, K_{\text{new}}, \qquad \alpha = \frac{t - t_{\text{switch}}}{\tau_{\text{switch}}}
$$

This avoids discontinuous jumps in the control signal at regime boundaries. Integral
state is reset on entry to disruption mitigation.

**scpn-control**: `GainScheduledController`, `RegimeDetector`, `OperatingRegime` in
`scpn_control.control.gain_scheduled_controller`.

---

## 3. Robust Control: H-infinity

### The Problem

The plant model $(A, B, C)$ is never exact. Unmodelled dynamics, parameter variations,
and external disturbances act on the system. Robust control designs a controller that
guarantees stability and performance for all plants within a defined uncertainty set.

### Generalised Plant

The standard H-infinity setup augments the plant with disturbance and performance
channels:

$$
\dot{x} = A x + B_1 w + B_2 u
$$
$$
z = C_1 x + D_{12} u
$$
$$
y = C_2 x + D_{21} w
$$

Here $w$ is the exogenous disturbance (noise, model error, reference), $z$ is the
performance output (tracking error, control effort), $u$ is the control input, and
$y$ is the measured output.

The closed-loop transfer function from $w$ to $z$ is $T_{wz}(s)$. The H-infinity
norm is the worst-case gain:

$$
\|T_{wz}\|_\infty = \sup_\omega \bar{\sigma}\bigl(T_{wz}(j\omega)\bigr)
$$

where $\bar{\sigma}$ is the maximum singular value. The H-infinity controller
minimises this norm: it minimises the worst-case amplification from disturbance to error
across all frequencies.

### Riccati Equations

The Doyle-Glover-Khargonekar synthesis solves two algebraic Riccati equations (AREs).
For the discrete-time implementation used in real controllers, the plant is first
discretised via zero-order hold:

$$
x_{k+1} = A_d x_k + B_d u_k, \qquad A_d = e^{A \Delta t}, \quad B_d = \int_0^{\Delta t} e^{A\tau} B \, d\tau
$$

The discrete algebraic Riccati equation (DARE) for the state-feedback gain:

$$
X = A_d^\top X A_d - A_d^\top X B_d (R + B_d^\top X B_d)^{-1} B_d^\top X A_d + Q
$$

The stabilising solution $X \succeq 0$ yields the optimal gain $K = (R + B_d^\top X B_d)^{-1} B_d^\top X A_d$.

### Stability Margins

Gain margin: the maximum multiplicative factor by which the loop gain can increase
before instability. Phase margin: the maximum phase lag that can be tolerated. For
vertical stability, typical requirements are gain margin $> 6$ dB and phase margin
$> 30°$. The H-infinity controller provides guaranteed margins via the small gain
theorem: if $\|T_{wz}\|_\infty < \gamma^{-1}$, the system remains stable for all
perturbations $\|\Delta\| < \gamma$.

### Anti-Windup

When coil current saturates, the integrator inside the controller continues accumulating
error. Anti-windup clamps the integral state:

$$
\dot{x}_I = \begin{cases}
e(t) & \text{if } |u| < u_{\max} \\
0 & \text{if } |u| \geq u_{\max} \text{ and } \text{sign}(e) = \text{sign}(u)
\end{cases}
$$

This prevents large overshoot when the actuator exits saturation.

**scpn-control**: `HInfinityController` in `scpn_control.control.h_infinity_controller`.
Implements ZOH discretisation, DARE-based gain synthesis, gamma-bisection for feasibility,
and 20% multiplicative uncertainty tolerance.

---

## 4. Model Predictive Control (MPC)

### Receding Horizon

MPC solves a finite-horizon optimisation problem at each timestep and applies only the
first control action. At the next timestep, the horizon shifts forward and the problem
is re-solved with updated state information.

Given current state $x_0$, find the control sequence $\{u_0, u_1, \ldots, u_{H-1}\}$
that minimises:

$$
J = \sum_{k=0}^{H-1} \bigl[ (x_k - x_{\text{ref}})^\top Q (x_k - x_{\text{ref}}) + u_k^\top R \, u_k \bigr] + (x_H - x_{\text{ref}})^\top P (x_H - x_{\text{ref}})
$$

subject to:

$$
x_{k+1} = f(x_k, u_k), \qquad u_{\min} \leq u_k \leq u_{\max}, \qquad x_{\min} \leq x_k \leq x_{\max}
$$

$Q$ penalises state deviation, $R$ penalises control effort, $P$ is the terminal cost
(typically the DARE solution for the linearised system to guarantee stability).

### Constraints

MPC handles constraints natively, unlike PID or LQR. In fusion applications:

- Coil currents: $|I_{\text{coil}}| \leq I_{\max}$ (thermal limits, typically 50 kA)
- Heating power: $0 \leq P_{\text{aux}} \leq P_{\max}$ (73 MW for ITER)
- Slew rates: $|\Delta u_k| \leq \Delta u_{\max}$ (power supply bandwidth)
- Safety limits: $q_{95} \geq 2$ (kink stability), $\beta_N \leq 3.5$ (ideal wall limit)

### Nonlinear MPC

For nonlinear plants, the prediction model $f(x, u)$ is not a matrix multiplication.
The `NonlinearMPC` in scpn-control linearises $f$ at each step via finite differences to
obtain local Jacobians $(A_k, B_k)$, then solves the resulting QP via sequential
quadratic programming (SQP). Up to `max_sqp_iter` iterations refine the trajectory.

When a neural surrogate replaces $f$, the Jacobians come from backpropagation through
the network, enabling gradient-based trajectory optimisation without an explicit physics
model.

**scpn-control**: `NonlinearMPC` in `scpn_control.control.nmpc_controller` (SQP-based,
6 states, 3 inputs, DARE terminal cost). `ModelPredictiveController` in
`scpn_control.control.fusion_sota_mpc` (extended version with neural surrogate support).

---

## 5. Reinforcement Learning for Plasma Control

### Formulation

RL frames control as a Markov decision process (MDP): at each timestep $k$, the agent
observes state $s_k$, takes action $a_k$ according to policy $\pi(a|s)$, receives
reward $r_k$, and transitions to $s_{k+1}$. The objective is to maximise the expected
discounted return:

$$
J(\pi) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k \, r_k \right]
$$

where $\gamma \in (0, 1)$ is the discount factor.

For plasma control, the state is the observation vector
$s = [T_{\text{axis}}, T_{\text{edge}}, \beta_N, l_i, q_{95}, I_p]$,
the action is the actuator change
$a = [\Delta P_{\text{aux}}, \Delta I_{p,\text{ref}}]$,
and the reward penalises tracking error and control effort:

$$
r_k = -\|s_k - s_{\text{ref}}\|_Q^2 - \|a_k\|_R^2
$$

### PPO (Proximal Policy Optimization)

PPO is a policy gradient method that prevents destructively large policy updates. The
clipped surrogate objective:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\bigl( r_t(\theta) \, \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \, \hat{A}_t \bigr) \right]
$$

where $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{\text{old}}}(a_t|s_t)$ is
the probability ratio, $\hat{A}_t$ is the advantage estimate, and $\epsilon \approx 0.2$
is the clipping parameter. If the policy changes too much (ratio far from 1), the
clipping kills the gradient.

### Constrained RL via Lagrangian Relaxation

Fusion control has hard safety constraints (avoid disruptions, stay within actuator
limits). Standard RL ignores constraints. Constrained RL augments the reward:

$$
r_{\text{aug}} = r - \sum_i \lambda_i \, c_i
$$

where $c_i$ is the cost for constraint $i$ and $\lambda_i$ is the Lagrange multiplier.
The multipliers are updated by dual gradient ascent:

$$
\lambda_i \leftarrow \max\bigl(0, \; \lambda_i + \eta \, (C_i - d_i)\bigr)
$$

where $C_i$ is the cumulative constraint cost over an episode and $d_i$ is the
constraint limit. This drives $\lambda_i$ up when constraints are violated, increasing
the penalty, and back toward zero when constraints are satisfied.

### Why RL Can Beat MPC

MPC requires a differentiable model of the plant. If that model is wrong, MPC performs
poorly. RL learns a policy directly from interaction (or simulated interaction),
absorbing nonlinearities and model errors into the policy itself. In scpn-control,
PPO trained for 500K timesteps on `TokamakEnv` outperforms both MPC and PID on
reference tracking with fewer constraint violations.

The tradeoff: RL policies are opaque (no formal stability certificate), and training
requires many episodes. MPC provides constraint satisfaction guarantees. In practice,
RL is used for scenario-level control (seconds timescale), while classical controllers
handle fast loops (milliseconds).

**scpn-control**: `TokamakEnv` (Gymnasium interface, 0D physics) in
`scpn_control.control.gym_tokamak_env`. `LagrangianPPO` and `ConstrainedGymTokamakEnv`
in `scpn_control.control.safe_rl_controller`.

---

## 6. Spiking Neural Network (SNN) Control

### From Biology to Hardware

Biological neurons communicate via discrete spikes, not continuous values. The
Leaky Integrate-and-Fire (LIF) model captures the essential dynamics:

$$
\tau_m \frac{dV}{dt} = -(V - V_{\text{rest}}) + R_m I_{\text{syn}}
$$
$$
\text{if } V \geq V_{\text{thresh}}: \quad \text{emit spike}, \quad V \leftarrow V_{\text{reset}}
$$

$\tau_m$ is the membrane time constant (~20 ms for cortical neurons, tunable in
hardware), $V_{\text{rest}}$ is the resting potential, $R_m$ is the membrane resistance,
and $I_{\text{syn}}$ is the total synaptic input current.

### Petri Net to SNN Compilation

scpn-control compiles a Stochastic Petri Net (SPN) into an SNN:

1. Each SPN **place** becomes a LIF neuron. The marking (token count) maps to membrane
   potential.
2. Each SPN **transition** becomes a set of synaptic connections. The weight matrices
   $W^+$ and $W^-$ define excitatory and inhibitory weights.
3. Transition **firing** corresponds to a neuron reaching threshold and spiking.
4. The SPN's stochastic firing rates become neuron-level noise parameters.

The `FusionCompiler` takes a `StochasticPetriNet` and produces a `CompiledNet` artifact
(`.scpnctl.json`) containing neuron parameters, weight matrices, and metadata. The
`NeuroSymbolicController` loads this artifact and runs it in two modes:

- **Oracle path**: standard float-precision forward pass for validation.
- **Stochastic computing (SC) path**: binary bitstream encoding via `sc-neurocore`.
  Weights are packed as `uint64` bitstreams; the forward pass uses AND + popcount
  operations, enabling FPGA or neuromorphic hardware deployment.

### Temporal Coding Advantage

SNNs encode information in spike timing, not just firing rate. A burst of spikes
arriving within a short window indicates high confidence. Temporal coincidence
detection (multiple input spikes arriving simultaneously) implements logical AND
without multiplication. This gives SNNs a latency advantage for threshold-crossing
detection tasks (disruption warnings, VDE onset) where the decision must happen
within a single membrane time constant (~100 us on dedicated hardware).

### Real-Time Execution

The compiled SNN runs a forward pass per control cycle. On an FPGA, this takes
$< 1 \, \mu\text{s}$ per inference. The `NeuroCyberneticController` wraps a pool
of `SpikingControllerPool` neurons in push-pull pairs (excitatory/inhibitory) for
multi-channel control output with sub-millisecond latency.

**scpn-control**: `FusionCompiler`, `CompiledNet` in `scpn_control.scpn.compiler`.
`NeuroSymbolicController` in `scpn_control.scpn.controller`.
`NeuroCyberneticController`, `SpikingControllerPool` in
`scpn_control.control.neuro_cybernetic_controller`.

---

## 7. Advanced Controllers

### Sliding-Mode Control for Vertical Stability

Vertical displacement of an elongated plasma is an unstable mode: the plasma column
sits at an equilibrium that is a saddle point in the vertical direction. The growth
rate $\gamma$ depends on the vertical stability index $n$, plasma current $I_p$, and
wall eddy current decay time $\tau_w$.

The super-twisting algorithm is a second-order sliding mode controller. Define the
sliding surface:

$$
s = e + c \, \dot{e}, \qquad e = Z_{\text{meas}} - Z_{\text{ref}}
$$

where $c > 0$ sets the convergence rate on the sliding surface. The control law:

$$
u = -\alpha \, |s|^{1/2} \, \text{sgn}(s) + v, \qquad \dot{v} = -\beta \, \text{sgn}(s)
$$

The integral term $v$ provides robustness against matched disturbances. The
$|s|^{1/2}$ term eliminates chattering (the high-frequency oscillation that plagues
classical sliding mode). Convergence to $s = 0$ occurs in finite time.

**Lyapunov certificate.** Define $V = 2\beta|s| + \frac{1}{2}v^2 + \frac{1}{2}(\alpha|s|^{1/2}\text{sgn}(s) - v)^2$.
Then $\dot{V} \leq 0$ for $\alpha, \beta$ satisfying $\alpha > 0$, $\beta > \alpha$.
Finite-time convergence follows from $\dot{V} \leq -\mu V^{1/2}$ for some $\mu > 0$.

**scpn-control**: `SuperTwistingSMC`, `VerticalStabilizer` in
`scpn_control.control.sliding_mode_vertical`.

### Mu-Synthesis (D-K Iteration)

H-infinity treats uncertainty as a single unstructured norm-bounded block.
Mu-synthesis handles **structured** uncertainty: each uncertain parameter (growth rate,
wall time constant, transport coefficient) has its own block with its own bound.

The structured singular value $\mu$ of a matrix $M$ with respect to uncertainty
structure $\boldsymbol{\Delta}$ is:

$$
\mu_{\boldsymbol{\Delta}}(M) = \frac{1}{\min\bigl\{ \bar{\sigma}(\Delta) : \det(I - M \Delta) = 0, \; \Delta \in \boldsymbol{\Delta} \bigr\}}
$$

$\mu$ cannot be computed exactly in general, but upper and lower bounds are available.
The upper bound uses D-scaling:

$$
\mu_{\boldsymbol{\Delta}}(M) \leq \inf_D \bar{\sigma}(D M D^{-1})
$$

where $D$ commutes with the uncertainty structure. The D-K iteration alternates:

1. **K-step**: fix $D$, synthesise H-infinity controller $K$ for the scaled plant $D G D^{-1}$.
2. **D-step**: fix $K$, fit $D(\omega)$ to minimise $\bar{\sigma}(D \, T_{wz} \, D^{-1})$ at each frequency.
3. Repeat until $\mu < 1$ at all frequencies.

Robust stability holds if and only if $\mu(M(j\omega)) < 1$ for all $\omega$.

**scpn-control**: `MuSynthesisController`, `StructuredUncertainty`, `UncertaintyBlock`
in `scpn_control.control.mu_synthesis`. Uses numerical gradient descent on $D$ as a
lightweight alternative to LMI-based solvers.

### Fault-Tolerant Control

Sensors fail. Actuators fail. The controller must detect faults, isolate the failed
component, and reconfigure.

**Fault Detection and Isolation (FDI)** monitors the innovation sequence $\nu_k = y_k - \hat{y}_k$
(measurement minus prediction). Under normal operation, $\nu_k$ is zero-mean with
covariance $S$. A sensor fault causes a persistent bias in $\nu_k$. Detection uses
a windowed chi-squared test: if $\nu_k$ exceeds $3\sigma$ for $n_{\text{alert}}$
consecutive samples, a fault is declared.

Fault types distinguished:

- **Sensor dropout**: $y = 0$ or NaN
- **Sensor drift**: persistent bias
- **Stuck actuator**: command changes but output does not
- **Open circuit**: actuator output collapses to zero

After isolation, the `ReconfigurableController` removes the faulted channel from the
measurement matrix $C$ (or control matrix $B$) and re-solves for the reduced-order
gains. Performance degrades gracefully rather than catastrophically.

**scpn-control**: `FDIMonitor`, `ReconfigurableController` in
`scpn_control.control.fault_tolerant_control`.

### Plasma Shape Control

The plasma boundary (last closed flux surface, LCFS) must match a target shape defined
by isoflux points, gap distances, X-point locations, and strike point positions. The
shape error $e_{\text{shape}}$ is a vector of deviations at control points.

The shape Jacobian $J = \partial e_{\text{shape}} / \partial I_{\text{coil}}$ relates
coil current changes to shape changes. Computing $J$ requires perturbing each coil
and re-solving the Grad-Shafranov equation.

The control law uses Tikhonov-regularised pseudoinverse:

$$
\Delta I = -(J^\top J + \lambda I)^{-1} J^\top e_{\text{shape}}
$$

The regularisation parameter $\lambda$ prevents large current corrections when $J$ is
ill-conditioned (which happens when coils are far from the plasma or nearly collinear
in their effect on the boundary).

Coil currents are clipped to $|I| \leq I_{\max}$ after the solve, and slew rates
$|\Delta I / \Delta t| \leq \dot{I}_{\max}$ are enforced to respect power supply limits.

**scpn-control**: `PlasmaShapeController`, `ShapeJacobian`, `ShapeTarget` in
`scpn_control.control.shape_controller`.

---

## 8. Multi-Layer Control Architecture

### Hierarchy

Tokamak control is organised in nested loops with decreasing bandwidth:

```
┌─────────────────────────────────────────────────────────┐
│  Supervisory Layer  (1 Hz)                              │
│  ScenarioScheduler, DisruptionMitigationController      │
│  Decisions: scenario phase, emergency shutdown           │
├─────────────────────────────────────────────────────────┤
│  Profile & Shape Layer  (100 Hz - 1 kHz)                │
│  PlasmaShapeController, GainScheduledController         │
│  NonlinearMPC, NeuroCyberneticController                │
│  Decisions: heating mix, density, current profile        │
├─────────────────────────────────────────────────────────┤
│  Fast Stability Layer  (1 kHz - 10 kHz)                 │
│  VerticalStabilizer, RWMFeedbackController              │
│  HInfinityController                                    │
│  Decisions: vertical position, RWM suppression           │
├─────────────────────────────────────────────────────────┤
│  Plant  (tokamak + diagnostics + actuators)              │
└─────────────────────────────────────────────────────────┘
```

**Fast stability** runs at 1-10 kHz. It stabilises the vertical position and suppresses
resistive wall modes. These are unstable on millisecond timescales. Controllers here
must be computationally cheap: a matrix-vector multiply (H-infinity gain) or a
sliding-mode switching law.

**Profile and shape** runs at 100 Hz to 1 kHz. It tracks reference profiles for
temperature, density, and current distribution, and maintains the plasma boundary shape.
MPC, gain-scheduled PID, and neuro-cybernetic controllers operate here.

**Supervisory** runs at ~1 Hz. It manages the scenario timeline (ramp-up, flat-top,
ramp-down), triggers disruption mitigation (SPI injection, fast current quench), and
coordinates between subsystems.

### State Estimation

The controller needs the full plasma state, but diagnostics provide only partial
measurements. EFIT (Equilibrium Fitting) reconstructs the 2D flux map $\psi(R, Z)$
from magnetic probe measurements by solving the inverse Grad-Shafranov problem.
From $\psi$, the boundary shape, current profile, and safety factor $q$ are derived.

Real-time EFIT runs between control cycles (~1 ms). The reconstruction is
under-determined (more unknowns than measurements), so regularisation and prior
constraints are essential.

### Disruption Prediction and Avoidance

A disruption is an abrupt loss of confinement that dumps the plasma energy into the
wall. Forces can exceed structural limits. Disruptions must be predicted early enough
to mitigate (inject impurities to radiate the energy uniformly rather than in a
localised hot spot).

The `DisruptionMitigationController` monitors a disruption probability score from
a transformer-based predictor. When $p_{\text{disrupt}} > 0.8$:

1. Switch `RegimeDetector` to `DISRUPTION_MITIGATION`.
2. Trigger SPI (Shattered Pellet Injection) to radiate stored energy.
3. Ramp down current on the fastest safe trajectory.

### Architecture Diagram (scpn-control)

```
                     ┌─────────────────┐
                     │  Scenario Plan   │
                     │  (waveforms,     │
                     │   phase timing)  │
                     └───────┬─────────┘
                             │ x_ref(t)
                             ▼
           ┌─────────────────────────────────────┐
           │     GainScheduledController          │
           │  ┌──────────┐  ┌──────────────────┐ │
           │  │ Regime    │  │ PID + bumpless   │ │
           │  │ Detector  │─▶│ transfer         │ │
           │  └──────────┘  └────────┬─────────┘ │
           └─────────────────────────┼───────────┘
                                     │ u_profile
                     ┌───────────────┼───────────────┐
                     ▼               ▼               ▼
            ┌────────────┐  ┌────────────┐  ┌────────────┐
            │  Vertical  │  │   Shape    │  │  H-inf /   │
            │ Stabilizer │  │ Controller │  │  Mu-synth  │
            │  (SMC)     │  │ (Tikhonov) │  │  (DARE)    │
            └──────┬─────┘  └──────┬─────┘  └──────┬─────┘
                   │               │               │
                   └───────────────┼───────────────┘
                                   │ u_total
                                   ▼
                          ┌────────────────┐
                          │  Actuators     │
                          │  (coils, NBI,  │
                          │   gas, ECRH)   │
                          └────────┬───────┘
                                   │
                                   ▼
                          ┌────────────────┐
                          │  Tokamak       │
                          │  (plasma)      │
                          └────────┬───────┘
                                   │
                                   ▼
                          ┌────────────────┐
                          │  Diagnostics   │
                          │  (magnetics,   │
                          │   Thomson, ECE)│
                          └────────┬───────┘
                                   │ y
                                   ▼
                          ┌────────────────┐
                          │  EFIT / State  │
                          │  Estimation    │
                          └────────┬───────┘
                                   │ x_hat
                                   ▼
                          ┌────────────────┐
                          │  FDI Monitor   │
                          │  (fault detect │
                          │   + isolate)   │
                          └────────┬───────┘
                                   │ x_hat_clean
                                   └──────────▶ back to controllers
```

---

## Further Reading

1. **Skogestad, S. & Postlethwaite, I.** *Multivariable Feedback Control: Analysis and Design*, 2nd ed., Wiley, 2005.
   Chapters 3-4 (MIMO stability), Chapter 8 (H-infinity), Chapter 9 (mu-synthesis).

2. **Khalil, H. K.** *Nonlinear Systems*, 3rd ed., Prentice Hall, 2002.
   Chapter 14 (sliding mode), Chapter 4 (Lyapunov stability).

3. **Sutton, R. S. & Barto, A. G.** *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018.
   Chapter 13 (policy gradient methods). Free at http://incompleteideas.net/book/the-book.html.

4. **Schulman, J. et al.** "Proximal Policy Optimization Algorithms", arXiv:1707.06347, 2017.
   The PPO algorithm used in `LagrangianPPO`.

5. **Stacey, W. M.** *Fusion Plasma Physics*, 2nd ed., Wiley-VCH, 2012.
   Chapter 13 (plasma control), Chapter 11 (MHD stability and vertical stability).

6. **Humphreys, D. A. et al.** "Novel aspects of plasma control in ITER",
   *Physics of Plasmas* 22, 021806, 2015.
   ITER control architecture: the nested-loop hierarchy that scpn-control implements.
