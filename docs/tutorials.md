# Tutorials

## Quick Start

```bash
pip install scpn-control
scpn-control demo --scenario combined --steps 500
scpn-control benchmark --n-bench 5000 --json-out
scpn-control live --port 8765 --zeta 0.5
```

---

## Tutorial Scripts

Seven self-contained scripts in `examples/` covering the full stack.
Each prints structured output and runs without GPU or optional dependencies.

### Core Tutorials (01-05)

| # | Script | Topics | Extra deps |
|---|--------|--------|------------|
| 01 | `tutorial_01_closed_loop_control.py` | Machine configs, GS equilibrium, transport, SPN compiler, H-inf controller, digital twin | none |
| 02 | `tutorial_02_jax_autodiff.py` | Thomas solver, Crank-Nicolson + `jax.grad`, batched `jax.vmap`, JAX GS solver, d(psi)/d(Ip) | `jax` |
| 03 | `tutorial_03_ppo_rl_agent.py` | TokamakEnv, PID baseline, random rollout, PPO training, pre-trained eval, PPO vs PID vs MPC | `stable-baselines3` (optional) |
| 04 | `tutorial_04_neural_transport.py` | Critical-gradient model, QLKNN-10D input space, regime classification, gradient scan, CN coupling | none |
| 05 | `tutorial_05_adaptive_phase_dynamics.py` | Kuramoto sync, 16-layer UPDE, Lyapunov guard, RealtimeMonitor, adaptive Knm, closed-loop | none |

### Elite Tutorials (06-07) — Expert Level

| # | Script | Topics | Extra deps |
|---|--------|--------|------------|
| 06 | `tutorial_06_frontier_physics.py` | Gyrokinetic transport (ITG/TEM/ETG), ballooning stability (s-alpha), current diffusion, current drive (ECCD+NBI), NTM dynamics (Modified Rutherford), sawtooth cycler, SOL two-point model, integrated scenario (ITER baseline) | none |
| 07 | `tutorial_07_advanced_controllers.py` | Sliding-mode vertical (super-twisting SMC), gain-scheduled controller, RWM feedback, mu-synthesis (D-K iteration), fault-tolerant control (FDI), shape controller (isoflux), scenario scheduler, controller comparison (H-inf vs MPC vs PID) | none |

Run any tutorial:

```bash
python examples/tutorial_01_closed_loop_control.py
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

### PPO Reinforcement Learning

```python
from scpn_control.control.gym_tokamak_env import TokamakEnv

env = TokamakEnv(dt=1e-3, max_steps=500, T_target=20.0)
obs, _ = env.reset(seed=42)
for _ in range(500):
    action = np.array([0.5 * (20.0 - obs[0]), 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, _ = env.step(action)
```

### SNN (LIF+NEF)

```python
from scpn_control.control.nengo_snn_wrapper import NengoSNNController
import numpy as np

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
| `snn_compiler_walkthrough.ipynb` | Petri net to SNN compilation | none |
| `h_infinity_controller_demo.ipynb` | DARE H-inf closed-loop control | matplotlib |
| `neuro_symbolic_control_demo.ipynb` | Full closed-loop controller | `sc_neurocore` |

```bash
pip install -e ".[viz]" jupyter nbconvert
jupyter nbconvert --to notebook --execute examples/q10_breakeven_demo.ipynb \
    --output-dir artifacts/notebook-exec
```
