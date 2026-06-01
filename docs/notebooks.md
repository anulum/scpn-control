<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Notebook Gallery

The notebooks demonstrate workflows visually. They are useful for onboarding,
teaching, and exploratory analysis. They are not claim-bearing evidence unless
their outputs are converted into the matching validation artefact and admitted by
the relevant validator.

## Core notebooks

| Notebook | Purpose | Extra dependencies |
| --- | --- | --- |
| `examples/q10_breakeven_demo.ipynb` | Transport and breakeven demonstration | none |
| `examples/snn_compiler_walkthrough.ipynb` | Stochastic Petri net to SNN compilation | none |
| `examples/h_infinity_controller_demo.ipynb` | DARE-based robust radial controller demo | `matplotlib` |
| `examples/paper27_phase_dynamics_demo.ipynb` | 16-layer Kuramoto-Sakaguchi phase dynamics | `matplotlib` |
| `examples/snn_pac_closed_loop_demo.ipynb` | SNN controller coupled to PAC-gated phase dynamics | `matplotlib` |

## Extended notebooks

| Notebook | Purpose | Extra dependencies |
| --- | --- | --- |
| `examples/neuro_symbolic_control_demo.ipynb` | Full neuro-symbolic stack with optional hardware-simulation backend | `sc-neurocore`, `matplotlib` |
| `examples/scpn_full_stack_demo_2026.ipynb` | End-to-end control-stack demonstration for the current release line | `matplotlib` |
| `examples/frontier_physics_demo.ipynb` | Gyrokinetic, ballooning, NTM, sawtooth, SOL, and scenario physics surfaces | `matplotlib` |
| `examples/advanced_control_demo.ipynb` | Sliding-mode, gain-scheduled, RWM, mu, FDI, shape-control demonstrations | `matplotlib` |

## Execute a notebook

```bash
pip install -e ".[viz]" jupyter nbconvert
jupyter nbconvert --to notebook --execute examples/q10_breakeven_demo.ipynb     --output-dir artefacts/notebook-exec
```

## Render as HTML

```bash
jupyter nbconvert --to html examples/q10_breakeven_demo.ipynb --output-dir docs/_notebooks
jupyter nbconvert --to html examples/snn_compiler_walkthrough.ipynb --output-dir docs/_notebooks
jupyter nbconvert --to html examples/h_infinity_controller_demo.ipynb --output-dir docs/_notebooks
```

## Notebook to evidence workflow

Use notebooks for exploration, teaching, and communication. Use validators for
claims.

1. Run the notebook locally and record the exact environment.
2. Move any claim-bearing computation into a script under `validation/` or a
   module-specific test.
3. Persist JSON and Markdown evidence with schema version, units, source data,
   checksums, tolerances, and claim boundary.
4. Add or update the matching validator so edited artefacts fail closed.
5. Link the admitted report from validation docs, benchmarks, or release notes.

## Interpretation rules

- Notebook plots are explanatory, not facility evidence.
- Timings from notebooks are local observations unless captured by a benchmark
  artefact with host metadata.
- Physics outputs need the corresponding validator before public claims are
  admissible.
- Optional dependencies should be installed explicitly so notebook failures are
  attributable to environment state rather than hidden imports.
