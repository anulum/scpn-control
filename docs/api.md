# API Reference

## Top-Level Exports

```python
import scpn_control

scpn_control.__version__       # "0.3.0"
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

::: scpn_control.core.fusion_kernel.FusionKernel

::: scpn_control.core.tokamak_config.TokamakConfig

---

## Phase — Paper 27 Dynamics

::: scpn_control.phase.upde.UPDESystem

::: scpn_control.phase.kuramoto.kuramoto_sakaguchi_step

---

## Control — Controllers

::: scpn_control.control.h_infinity_controller.HInfinityController

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
