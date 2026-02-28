# Notebook Gallery

Interactive Jupyter notebooks demonstrating scpn-control capabilities.
Each notebook is self-contained and can be executed locally with `jupyter nbconvert`.

---

## Q10 Breakeven Demo

**File:** `examples/q10_breakeven_demo.ipynb`
**Extra deps:** none

Demonstrates the integrated transport solver running a simulated tokamak
discharge toward Q=10 breakeven. Shows temperature/density profile evolution,
fusion power buildup, and energy confinement time convergence.

```bash
jupyter nbconvert --to notebook --execute examples/q10_breakeven_demo.ipynb
```

---

## SNN Compiler Walkthrough

**File:** `examples/snn_compiler_walkthrough.ipynb`
**Extra deps:** none

Step-by-step compilation of a Stochastic Petri Net into a spiking neural
network. Covers `StochasticPetriNet` graph construction, `FusionCompiler`
compilation, `CompiledNet` inspection (neuron count, weight matrix), and
`NeuroSymbolicController` closed-loop execution.

```bash
jupyter nbconvert --to notebook --execute examples/snn_compiler_walkthrough.ipynb
```

---

## H-infinity Controller Demo

**File:** `examples/h_infinity_controller_demo.ipynb`
**Extra deps:** matplotlib

DARE-based H-infinity controller for radial position regulation.
Demonstrates controller synthesis, gain margin computation, step response,
and disturbance rejection with anti-windup.

```bash
pip install "scpn-control[viz]"
jupyter nbconvert --to notebook --execute examples/h_infinity_controller_demo.ipynb
```

---

## Paper 27 Phase Dynamics Demo

**File:** `examples/paper27_phase_dynamics_demo.ipynb`
**Extra deps:** matplotlib

Full 16-layer Kuramoto-Sakaguchi simulation using the Paper 27 Knm coupling
matrix. Shows order parameter R convergence, Lyapunov exponent λ stabilisation,
per-layer coherence, and PAC gating effects.

```bash
pip install "scpn-control[viz]"
jupyter nbconvert --to notebook --execute examples/paper27_phase_dynamics_demo.ipynb
```

---

## SNN-PAC Closed-Loop Demo

**File:** `examples/snn_pac_closed_loop_demo.ipynb`
**Extra deps:** matplotlib

Combines the SNN controller with PAC-gated Kuramoto dynamics in a closed loop.
The SNN controller adjusts coil currents based on phase coherence feedback,
demonstrating the full neuro-symbolic control pipeline.

```bash
pip install "scpn-control[viz]"
jupyter nbconvert --to notebook --execute examples/snn_pac_closed_loop_demo.ipynb
```

---

## Neuro-Symbolic Control Demo

**File:** `examples/neuro_symbolic_control_demo.ipynb`
**Extra deps:** `sc-neurocore`

Full-stack demonstration requiring the `sc-neurocore` hardware simulation
backend. Shows VectorizedSCLayer compilation, bitstream encoding, and
real-time SNN execution at 512× real-time.

```bash
pip install "scpn-control[neuro,viz]"
jupyter nbconvert --to notebook --execute examples/neuro_symbolic_control_demo.ipynb
```

!!! note
    This notebook requires `sc-neurocore >= 3.8.0` which is not available on PyPI.
    Contact [protoscience@anulum.li](mailto:protoscience@anulum.li) for access.

---

## Running All Notebooks

```bash
pip install -e ".[viz]" jupyter nbconvert

# Core notebooks (no extra deps)
jupyter nbconvert --to notebook --execute examples/q10_breakeven_demo.ipynb
jupyter nbconvert --to notebook --execute examples/snn_compiler_walkthrough.ipynb

# Visualization notebooks (matplotlib)
jupyter nbconvert --to notebook --execute examples/h_infinity_controller_demo.ipynb
jupyter nbconvert --to notebook --execute examples/paper27_phase_dynamics_demo.ipynb
jupyter nbconvert --to notebook --execute examples/snn_pac_closed_loop_demo.ipynb
```

## Rendering Notebooks as HTML

```bash
jupyter nbconvert --to html examples/q10_breakeven_demo.ipynb --output-dir docs/_notebooks
jupyter nbconvert --to html examples/snn_compiler_walkthrough.ipynb --output-dir docs/_notebooks
jupyter nbconvert --to html examples/h_infinity_controller_demo.ipynb --output-dir docs/_notebooks
```
