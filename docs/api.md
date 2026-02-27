# API Reference

## Top-Level Exports

```python
import scpn_control

scpn_control.__version__       # "0.2.0"
scpn_control.FusionKernel      # Grad-Shafranov equilibrium solver
scpn_control.RUST_BACKEND      # True if Rust acceleration available
scpn_control.StochasticPetriNet
scpn_control.FusionCompiler
scpn_control.CompiledNet
scpn_control.NeuroSymbolicController
scpn_control.kuramoto_sakaguchi_step
scpn_control.order_parameter
scpn_control.KnmSpec
scpn_control.build_knm_paper27
scpn_control.UPDESystem
scpn_control.LyapunovGuard
scpn_control.RealtimeMonitor
```

---

## Core — Physics Solvers

### `FusionKernel`

Grad-Shafranov equilibrium solver with Picard iteration.

```python
from scpn_control.core import FusionKernel

kernel = FusionKernel(R0=6.2, a=2.0, B0=5.3, Nr=64, Nz=64)
result = kernel.solve(max_iter=200, tol=1e-6)
# result.psi     — flux surface array (Nr x Nz)
# result.q       — safety factor profile
# result.p       — pressure profile
```

### `TokamakConfig`

Named machine presets for ITER, SPARC, DIII-D, JET.

```python
from scpn_control.core import TokamakConfig

cfg = TokamakConfig.iter()   # R0=6.2m, B0=5.3T, Ip=15MA
cfg = TokamakConfig.sparc()  # R0=1.85m, B0=12.2T, Ip=8.7MA
cfg = TokamakConfig.diiid()  # R0=1.67m, B0=2.1T, Ip=1.5MA
cfg.aspect_ratio              # R0 / a
```

### `IntegratedTransportSolver`

1.5D coupled transport (Chang-Hinton + Sauter bootstrap).

```python
from scpn_control.core.integrated_transport_solver import IntegratedTransportSolver

solver = IntegratedTransportSolver(n_rho=32)
result = solver.solve(T_e_bc=10.0, n_e_bc=1e20, P_aux=20e6, n_steps=100)
```

### `NeuralEquilibrium`

PCA + MLP surrogate for sub-ms equilibrium reconstruction (CPU-only).

```python
from scpn_control.core.neural_equilibrium import NeuralEquilibrium

neq = NeuralEquilibrium.from_pretrained("weights/neural_eq.pt")
psi = neq.predict(Ip=15.0, B0=5.3, kappa=1.7, delta=0.33)
```

---

## Control — Controllers

### `HInfinityController`

DARE-based H-infinity robust controller with anti-windup.

```python
from scpn_control.control.h_infinity_controller import (
    HInfinityController,
    get_radial_robust_controller,
)

ctrl = get_radial_robust_controller(gamma_growth=100.0, damping=10.0)
ctrl.gamma                     # H-inf attenuation level
ctrl.is_stable                 # continuous closed-loop stability
ctrl.gain_margin_db            # gain margin in dB
ctrl.robust_feasibility_margin()  # gamma^2 - rho(XY)
ctrl.riccati_residual_norms()  # (res_X, res_Y) Frobenius norms

# Step the controller
u = ctrl.step(error=0.1, dt=0.001)

# Saturation
ctrl.u_max = 50.0  # clip output to [-50, 50]
ctrl.reset()       # zero observer state
```

### `FusionSoTAMPC`

Gradient-based MPC with surrogate dynamics.

```python
from scpn_control.control.fusion_sota_mpc import FusionSoTAMPC

mpc = FusionSoTAMPC(horizon=10, dt=0.001)
u = mpc.step(state=x, reference=x_ref)
```

### `NengoSNNWrapper`

Spiking neural network controller (Nengo backend).

```python
from scpn_control.control.nengo_snn_wrapper import NengoSNNWrapper

snn = NengoSNNWrapper(n_neurons=100, dt=0.001)
u = snn.step(error=0.1, dt=0.001)
```

### `DisruptionPredictor`

Transformer-based disruption prediction (experimental, requires torch).

```python
from scpn_control.control.disruption_predictor import DisruptionPredictor

pred = DisruptionPredictor(n_features=8, seq_len=64)
risk = pred.predict(time_series)  # -> float in [0, 1]
```

---

## SCPN — Petri Net Compiler

### `StochasticPetriNet`

Define a stochastic Petri net topology.

```python
from scpn_control import StochasticPetriNet

net = StochasticPetriNet()
net.add_place("plasma_state", initial_tokens=1.0)
net.add_place("control_signal", initial_tokens=0.0)
net.add_transition("sense", threshold=0.5)
net.add_arc("plasma_state", "sense", weight=1.0)
net.add_arc("sense", "control_signal", weight=1.0)
```

### `FusionCompiler`

Compile SPN → spiking neural network.

```python
from scpn_control import FusionCompiler

compiler = FusionCompiler()
compiled = compiler.compile(net)
compiled.n_neurons  # number of LIF neurons
compiled.weight_matrix  # synaptic weight matrix
```

### `NeuroSymbolicController`

Execute a compiled net as a controller.

```python
from scpn_control import NeuroSymbolicController

ctrl = NeuroSymbolicController(compiled_net=compiled, dt=0.001)
u = ctrl.step(state_vector)
```

---

## Phase — Paper 27 Dynamics

### `RealtimeMonitor`

Live Kuramoto-Sakaguchi phase monitor with R/V/lambda tracking.

```python
from scpn_control import RealtimeMonitor

mon = RealtimeMonitor.from_paper27(L=16, N_per=50, zeta_uniform=0.5)
snap = mon.tick()
snap["R_global"]        # global order parameter (0-1)
snap["R_layer"]         # per-layer order parameters (list of floats)
snap["V_global"]        # Lyapunov V(t) = (1/N) Σ (1 - cos(θ - Ψ))
snap["V_layer"]         # per-layer V values
snap["lambda_exp"]      # Lyapunov exponent (negative = converging)
snap["Psi_global"]      # global mean field phase
snap["guard_approved"]  # True if LyapunovGuard allows continuation
snap["guard_score"]     # stability score ∈ [0, 1]
snap["latency_us"]      # tick computation time in microseconds
```

### `kuramoto_sakaguchi_step`

Single Kuramoto-Sakaguchi integration step.

```python
from scpn_control import kuramoto_sakaguchi_step
import numpy as np

theta = np.random.uniform(-np.pi, np.pi, 100)
omega = np.random.normal(0, 0.3, 100)
theta_new = kuramoto_sakaguchi_step(
    theta, omega, dt=0.001, K=2.0, zeta=0.5, psi_driver=0.3
)
```

### `UPDESystem`

Unified Phase Dynamics Equation solver (16-layer coupled system).

```python
from scpn_control import UPDESystem, build_knm_paper27
import numpy as np

spec = build_knm_paper27(L=16, zeta_uniform=0.5)
upde = UPDESystem(spec=spec, dt=1e-3, psi_mode="external")

rng = np.random.default_rng(42)
theta = [rng.uniform(-np.pi, np.pi, 50) for _ in range(16)]
omega = [np.full(50, 1.0) for _ in range(16)]

out = upde.step(theta, omega, psi_driver=0.0)
out["R_global"]   # global order parameter
out["theta1"]     # updated phase arrays (list of 16 arrays)
```

### `LyapunovGuard`

Sliding-window stability monitor — flags instability when λ > 0 persists.

```python
from scpn_control import LyapunovGuard
import numpy as np

guard = LyapunovGuard(window=50, dt=1e-3, max_violations=3)
theta = np.random.uniform(-np.pi, np.pi, 100)
verdict = guard.check(theta, psi=0.0)
verdict.approved       # True if λ stayed below threshold
verdict.lambda_exp     # estimated Lyapunov exponent
verdict.score          # stability score ∈ [0, 1]
```

---

## CLI

```bash
scpn-control demo --scenario combined --steps 1000
scpn-control benchmark --n-bench 5000 --json-out
scpn-control validate --json-out
scpn-control live --port 8765 --zeta 0.5 --layers 16
scpn-control hil-test --shots-dir path/to/shots
```

---

## Rust Acceleration

When `scpn-control-rs` is built via maturin, all core solvers use Rust backends automatically:

```python
from scpn_control import RUST_BACKEND
print(RUST_BACKEND)  # True if Rust available

# Transparent acceleration — same Python API, Rust execution
kernel = FusionKernel(R0=6.2, a=2.0, B0=5.3)
# Uses Rust GS solver if available, falls back to NumPy
```

Build Rust bindings:

```bash
cd scpn-control-rs/crates/control-python
maturin develop --release
```
