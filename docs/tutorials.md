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

## What each tutorial should teach

A tutorial is successful when you can identify the controller surface, the
physics or replay input, the produced output, and the claim boundary. If a
script produces only a plot or console value, treat it as a learning artefact.
If it produces JSON/Markdown under `validation/reports/`, treat it as evidence
only after the matching validator admits it.

| Tutorial family | Expected takeaway |
| --- | --- |
| Closed loop | How observations become controller actions and replayable summaries |
| JAX autodiff | Where gradients are available for tuning and where fidelity remains bounded |
| PPO/RL | How learning baselines compare to classical controllers without bypassing safety gates |
| Neural transport/equilibrium | How surrogate evidence is separated from full reference admission |
| Phase dynamics | How SCPN phase contracts, Lyapunov guards, and WebSocket runtime boundaries interact |
| Frontier physics | Which physics modules are local bounded models and which require external validation |
| Advanced controllers | How robust/NMPC/mu-synthesis surfaces expose lifecycle and admission constraints |

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

## Tutorial selection model

Choose a tutorial path by review intent, not by perceived complexity.

- Use tutorial 01 for architecture and command-path confidence.
- Use tutorial 02 for differentiable tuning workflows.
- Use tutorial 05 for phase timing and synchronization behavior.
- Use tutorial 06 for frontier physics coupling context.
- Use tutorial 07 for advanced controller breadth and admission constraints.

Each tutorial is a learning path; only validated artifacts are evidence for claims.

## Practical use and scope

Use this page to choose the correct tutorial for your immediate goal.

- Select a path by intent: first-steps onboarding, control workflow, physics setup, validation, or deployment.
- Prefer the control and validation paths before changing runtime parameters in scripts or campaigns.
- Use tutorial outputs as reproducible checkpoints and record results in your session log.
