# Tutorials

## Quick Start

```bash
pip install scpn-control
scpn-control demo --scenario combined --steps 500
scpn-control benchmark --n-bench 5000 --json-out
scpn-control live --port 8765 --zeta 0.5
```

---

## Controller Walkthrough

Four controller tiers in `scpn_control.control`:

### H-infinity (Riccati DARE)

```python
from scpn_control.control.h_infinity_controller import get_radial_robust_controller

ctrl = get_radial_robust_controller(gamma_growth=100.0, damping=10.0)
print(f"gamma={ctrl.gamma:.1f}, GM={ctrl.gain_margin_db:.1f} dB")

for _ in range(500):
    u = ctrl.step(error=0.1, dt=0.001)
```

### MPC (gradient-based)

```python
from scpn_control.control.fusion_sota_mpc import (
    NeuralSurrogate, ModelPredictiveController,
)
import numpy as np

surrogate = NeuralSurrogate(n_coils=3, n_state=4, verbose=False)
target = np.array([1.65, 0.0, 1.4, -1.1])
mpc = ModelPredictiveController(
    surrogate, target, prediction_horizon=10, iterations=20, action_limit=2.0,
)
action = mpc.plan_trajectory(np.array([1.60, 0.05, 1.35, -1.05]))
```

### SNN (Nengo LIF)

Requires `pip install scpn-control[nengo]`.

```python
from scpn_control.control import get_nengo_controller
import numpy as np

NengoSNNController = get_nengo_controller()
ctrl = NengoSNNController()
u = ctrl.step(np.array([0.05, -0.02]))
```

---

## Phase Dynamics (Paper 27)

```python
from scpn_control import RealtimeMonitor

mon = RealtimeMonitor.from_paper27(L=16, N_per=50, zeta_uniform=0.5)

for _ in range(200):
    snap = mon.tick()

print(f"R={snap['R_global']:.4f}")
print(f"lambda={snap['lambda_exp']:.6f}")
print(f"latency={snap['latency_us']:.1f} us")
```

Per-layer coherence:

```python
for i, r in enumerate(snap["R_layer"]):
    print(f"  L{i:02d}: R={r:.3f}")
```

WebSocket stream:

```bash
scpn-control live --port 8765 --layers 16 --n-per 50 --zeta 0.5
```

---

## SCPN Compiler

```python
from scpn_control import StochasticPetriNet, FusionCompiler

net = StochasticPetriNet()
net.add_place("plasma_state", initial_tokens=1.0)
net.add_place("control_signal", initial_tokens=0.0)
net.add_transition("sense", threshold=0.5)
net.add_arc("plasma_state", "sense", weight=1.0)
net.add_arc("sense", "control_signal", weight=1.0)

compiled = FusionCompiler().compile(net)
print(f"Neurons: {compiled.n_neurons}")
```

---

## Notebooks

| Notebook | Description | Extra deps |
|----------|-------------|------------|
| `q10_breakeven_demo.ipynb` | Transport + breakeven analysis | none |
| `snn_compiler_walkthrough.ipynb` | Petri net → SNN compilation | none |
| `h_infinity_controller_demo.ipynb` | DARE H-inf closed-loop control | matplotlib |
| `neuro_symbolic_control_demo.ipynb` | Full closed-loop controller | `sc_neurocore` |

```bash
pip install -e ".[viz]" jupyter nbconvert
jupyter nbconvert --to notebook --execute examples/q10_breakeven_demo.ipynb \
    --output-dir artifacts/notebook-exec
```
