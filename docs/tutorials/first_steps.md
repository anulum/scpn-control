# First Steps

A hands-on introduction to scpn-control. No plasma physics background required.

---

## Prerequisites

- Python 3.10+
- numpy (installed automatically as a dependency)

## Installation

```bash
pip install scpn-control
```

For optional extras:

```bash
pip install scpn-control[jax]     # JAX autodiff solvers
pip install scpn-control[snn]     # SNN engine (pure NumPy)
pip install scpn-control[rl]      # stable-baselines3 PPO agent
```

---

## Your First Tokamak Configuration

scpn-control ships with frozen dataclass presets for ITER, SPARC, DIII-D, and JET.
Each encodes published machine parameters (major radius, field, current, shaping).

```python
from scpn_control.core.tokamak_config import TokamakConfig

iter_cfg = TokamakConfig.iter()
sparc_cfg = TokamakConfig.sparc()

print(f"ITER:  R0={iter_cfg.R0} m, B0={iter_cfg.B0} T, Ip={iter_cfg.Ip} MA")
print(f"SPARC: R0={sparc_cfg.R0} m, B0={sparc_cfg.B0} T, Ip={sparc_cfg.Ip} MA")
print(f"ITER aspect ratio: {iter_cfg.aspect_ratio:.2f}")
```

Output:

```
ITER:  R0=6.2 m, B0=5.3 T, Ip=15.0 MA
SPARC: R0=1.85 m, B0=12.2 T, Ip=8.7 MA
ITER aspect ratio: 3.10
```

---

## Your First Equilibrium

The `FusionKernel` solves the Grad-Shafranov equation on a 2D (R, Z) grid via
Picard iteration. It reads a JSON configuration specifying the reactor geometry,
coil positions, and solver parameters.

```python
from scpn_control.core.fusion_kernel import FusionKernel

kernel = FusionKernel("iter_config.json")
result = kernel.solve_equilibrium()

print(f"Converged: {result['converged']}")
print(f"Iterations: {result['iterations']}")
print(f"Residual: {result['residual']:.2e}")
print(f"Psi grid shape: {result['psi'].shape}")
```

The `iter_config.json` file ships in the repo root. It defines a 129x129 grid
spanning R=[2, 10] m and Z=[-6, 6] m with seven PF/CS coils.

**What the output means:**

- `psi` -- the poloidal flux on the (NZ, NR) grid. Contours of constant psi are
  flux surfaces. The innermost contour encloses the magnetic axis; the outermost
  closed contour is the Last Closed Flux Surface (LCFS).
- `converged` -- whether the Picard residual dropped below the threshold (default 1e-4).
- After solving, `kernel.Psi` holds the flux grid and `kernel.J_phi` holds the
  toroidal current density.

---

## Your First Controller

The H-infinity controller provides guaranteed robust stability for vertical
position control. `get_radial_robust_controller` builds a 2-state vertical
stability plant and synthesises Riccati-based gains.

```python
from scpn_control.control.h_infinity_controller import (
    HInfinityController,
    get_radial_robust_controller,
)

ctrl = get_radial_robust_controller(gamma_growth=100.0, damping=10.0)
print(f"gamma = {ctrl.gamma:.2f}")
print(f"Robust feasible: {ctrl.robust_feasible}")
print(f"State-feedback gain F: {ctrl.F}")

# Step the controller in a loop
dt = 1e-3
for _ in range(500):
    u = ctrl.step(error=0.1, dt=dt)
```

The `gamma_growth` parameter is the vertical instability growth rate in 1/s.
ITER-like values are ~100; SPARC is ~1000. The controller solves two continuous
Algebraic Riccati Equations, then discretises via zero-order hold at the
requested `dt`.

---

## Your First Transport Run

`TransportSolver` extends `FusionKernel` with 1.5D radial transport. It evolves
temperature and density profiles using Crank-Nicolson implicit diffusion, coupled
to the 2D equilibrium.

```python
from scpn_control.core.integrated_transport_solver import TransportSolver

solver = TransportSolver("iter_config.json", nr=50)

# Evolve 10 time steps of 0.1 s each with 50 MW auxiliary heating
for step in range(10):
    dt, P_aux = 0.1, 50.0
    tau_E, Q = solver.evolve_profiles(dt, P_aux)
    if step % 5 == 0:
        print(f"Step {step}: tau_E={tau_E:.3f} s, Q={Q:.2f}")

print(f"Central Ti: {solver.Ti[0]:.2f} keV")
print(f"Central ne: {solver.ne[0]:.2f} x10^19 m^-3")
```

`evolve_profiles` returns (tau_E, Q) -- the energy confinement time and the
fusion gain factor. The solver updates `solver.Te`, `solver.Ti`, and `solver.ne`
in place.

For multi-ion D-T transport with helium ash:

```python
solver = TransportSolver("iter_config.json", nr=50, multi_ion=True)
```

---

## Your First SPN

A Stochastic Petri Net defines the control logic as a bipartite graph of places
(state variables) and transitions (firing rules). The `FusionCompiler` compiles
this graph into a spiking neural network.

```python
from scpn_control import StochasticPetriNet, FusionCompiler

net = StochasticPetriNet()
net.add_place("plasma_state", initial_tokens=1.0)
net.add_place("control_active", initial_tokens=0.0)
net.add_transition("activate", threshold=0.5)

# Arc from place to transition (input) and transition to place (output)
net.add_arc("plasma_state", "activate", weight=0.8)
net.add_arc("activate", "control_active", weight=0.9)

# Compile to SNN
compiler = FusionCompiler()
compiled = compiler.compile(net)

print(f"Places: {net.n_places}")
print(f"Transitions: {net.n_transitions}")
print(f"W_in shape: {compiled.W_in.shape}")
print(f"W_out shape: {compiled.W_out.shape}")
```

Each place maps to a LIF neuron membrane potential. Each transition maps to a
synaptic connection with the arc weight. The `CompiledNet` stores sparse matrices
`W_in` (transitions x places) and `W_out` (places x transitions).

To run closed-loop control with the compiled net, wrap it in a
`NeuroSymbolicController` with observation-to-place mappings and readout
configuration. See `examples/tutorial_01_closed_loop_control.py` for the
full pipeline.

---

## CLI Quick Reference

```bash
# Closed-loop control demo (PID, SNN, or combined)
scpn-control demo --scenario combined --steps 500

# Timing benchmark: PID vs SNN step latency
scpn-control benchmark --n-bench 5000 --json-out

# Validate solver against reference data
scpn-control validate

# WebSocket phase-dynamics server
scpn-control live --port 8765 --zeta 0.5

# Hardware-in-the-loop test against recorded disruption shots
scpn-control hil-test --shots-dir validation/reference_data/diiid/disruption_shots
```

---

## Next Steps

- [tutorials.md](../tutorials.md) -- five self-contained tutorial scripts covering
  the full stack (GS equilibrium, JAX autodiff, PPO RL, neural transport,
  adaptive phase dynamics).
- [notebooks.md](../notebooks.md) -- interactive Jupyter notebooks (Q10 breakeven,
  SNN compiler walkthrough, H-infinity demo, phase dynamics).
- [theory.md](../theory.md) -- mathematical foundations: SPN formalism,
  Kuramoto-Sakaguchi model, UPDE, Lyapunov stability analysis.
- [glossary.md](../learning/glossary.md) -- definitions of all plasma physics and
  control theory terms used in the codebase.
