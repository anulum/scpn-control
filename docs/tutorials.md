<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorials

Tutorials are organised by learning goal. Start with Tutorial 01 if you are new
to the package, then choose a control, physics, validation, or deployment path.

## Quick start

```bash
pip install scpn-control
scpn-control demo --scenario combined --steps 500
scpn-control benchmark --n-bench 5000 --json-out
scpn-control live --host 127.0.0.1 --port 8765 --zeta 0.5
```

## Tutorial map

| # | Script | Learning goal | Extra dependencies |
| --- | --- | --- | --- |
| 01 | `examples/tutorial_01_closed_loop_control.py` | Closed-loop control, machine configs, equilibrium, transport, SPN compiler, digital twin | none |
| 02 | `examples/tutorial_02_jax_autodiff.py` | JAX gradients through solver and transport-facing routines | `jax` |
| 03 | `examples/tutorial_03_ppo_rl_agent.py` | Tokamak environment, PID baseline, PPO training and comparison | `stable-baselines3` optional |
| 04 | `examples/tutorial_04_neural_transport.py` | Critical-gradient model, QLKNN-10D input space, regime scan, CN coupling | none |
| 05 | `examples/tutorial_05_adaptive_phase_dynamics.py` | Kuramoto synchronisation, Lyapunov guard, adaptive coupling, closed loop | none |
| 06 | `examples/tutorial_06_frontier_physics.py` | Gyrokinetic, ballooning, current diffusion, NTM, sawtooth, SOL, integrated scenario | none |
| 07 | `examples/tutorial_07_advanced_controllers.py` | Sliding-mode, gain-scheduled, RWM, bounded mu analysis, FDI, isoflux, scheduler | none |

Run any tutorial:

```bash
python examples/tutorial_01_closed_loop_control.py
```

## Recommended paths

| If you want to... | Run |
| --- | --- |
| Understand the package in one pass | Tutorial 01, then [Onboarding](onboarding.md) |
| Work on differentiable physics | Tutorial 02, then [Benchmarks](benchmarks.md) JAX parity sections |
| Work on controller safety | Tutorial 05, then [API Reference](api.md) formal-verification sections |
| Work on facility replay | Tutorial 01, then [Validation and QA](validation.md) |
| Work on notebooks | [Notebook Gallery](notebooks.md) |

## Controller walkthrough

```python
from scpn_control.control.h_infinity_controller import get_radial_robust_controller

ctrl = get_radial_robust_controller(gamma_growth=100.0, damping=10.0)
for _ in range(500):
    u = ctrl.step(error=0.1, dt=0.001)
```

```python
import numpy as np
from scpn_control.control.nmpc_controller import NMPCController, NMPCProblem

problem = NMPCProblem.example()
ctrl = NMPCController(problem)
result = ctrl.solve(np.zeros(problem.nx), np.zeros(problem.nu))
ctrl.close()
```

```python
from scpn_control import FusionCompiler, StochasticPetriNet

net = StochasticPetriNet()
net.add_place("idle", initial_tokens=1.0)
net.add_place("heating", initial_tokens=0.0)
net.add_transition("ignite", threshold=0.5)
net.add_arc("idle", "ignite", weight=1.0)
net.add_arc("ignite", "heating", weight=1.0)
compiled = FusionCompiler().compile(net)
```

## Notebook execution

```bash
pip install -e ".[viz]" jupyter nbconvert
jupyter nbconvert --to notebook --execute examples/q10_breakeven_demo.ipynb     --output-dir artefacts/notebook-exec
```

Notebook outputs are demonstrations. Claim-bearing results must be captured by
the matching validator and admitted under [Validation and QA](validation.md).
