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

## Full Stack Demo (2026)

**File:** `examples/scpn_full_stack_demo_2026.ipynb`
**Extra deps:** matplotlib

Comprehensive end-to-end demonstration of the v0.15.0 control stack.
Includes equilibrium initialization, transport evolution, SNN controller
coupling, and real-time WebSocket telemetry visualization.

```bash
jupyter nbconvert --to notebook --execute examples/scpn_full_stack_demo_2026.ipynb
```

---

## Frontier Physics Demo

**File:** `examples/frontier_physics_demo.ipynb`
**Extra deps:** matplotlib

Demonstrates all Phase 3 frontier physics modules with matplotlib
visualizations: gyrokinetic transport (ITG spectrum + radial chi profile),
ballooning stability (s-alpha diagram), current diffusion + current drive
(ECCD+NBI profiles), NTM dynamics (Modified Rutherford Equation), sawtooth
cycles (Kadomtsev crash), SOL model (two-point), and a coupled ITER 15MA
integrated scenario.

```bash
pip install "scpn-control[viz]"
jupyter nbconvert --to notebook --execute examples/frontier_physics_demo.ipynb
```

---

## Advanced Control Demo

**File:** `examples/advanced_control_demo.ipynb`
**Extra deps:** matplotlib

Demonstrates all Phase 4 advanced control modules: super-twisting
sliding-mode vertical stabilizer, gain-scheduled multi-regime controller,
RWM feedback stabilization (open vs closed-loop growth rates), mu-synthesis
(structured singular value), fault-tolerant control (sensor dropout FDI),
and isoflux shape controller convergence.

```bash
pip install "scpn-control[viz]"
jupyter nbconvert --to notebook --execute examples/advanced_control_demo.ipynb
```

---

## Example Scripts

In addition to Jupyter notebooks, the following Python scripts demonstrate
high-performance and deployment scenarios.

### Digital Twin Performance

**File:** `examples/digital_twin_demo.py`

Runs the real-time digital twin with 10kHz control loop and simulated
diagnostics. Benchmarks the Rust kernel vs Python fallback and reports
P50/P99 latencies.

### Full Pipeline Benchmark

**File:** `examples/full_pipeline_benchmark.py`

Stresses the entire 16-layer stack (including Kuramoto and SPN logic)
under various CPU/GPU dispatch configurations. Generates performance scaling
reports for large-scale ensembles.

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
