# Use Cases

## Fusion Reactor Control

### Tokamak Control Algorithm Prototyping

> **Status:** Research / Alpha. Not a production PCS. Real hardware
> integration requires significant additional work (EPICS/CODAC interface,
> deterministic OS, safety certification).

scpn-control provides a prototyping platform for control algorithm
development. The 11.9 µs kernel step (Criterion-verified) enables:

- **Shape control algorithm R&D** — develop and test at high loop rates
- **Disruption prediction prototyping** — ML inference pipeline (synthetic training data)
- **SPI mitigation modeling** — halo current and runaway electron physics
- **Profile control R&D** — neuro-cybernetic dual R+Z SNN feedback

**Target users:** Control algorithm researchers, fusion startups in early design,
graduate students.

### ITER / DEMO (Future, Speculative)

The contract-based pre/post-condition checking layer provides runtime
assertion checking — not formal theorem-proved verification. The SNN
controller includes a Lyapunov stability guard, but this has not been
validated against real disruption scenarios or certified for nuclear safety.

---

## Digital Twin & Commissioning

### Offline Plant Commissioning

Before first plasma, use the digital twin to:

- Commission control algorithms against synthetic plasma scenarios
- Train operators on disruption response procedures
- Validate sensor/actuator configurations
- Stress-test control logic with adversarial perturbations

The digital twin runs in real-time on a single CPU core, enabling
rapid iteration without dedicated HPC resources.

### Hardware-in-the-Loop Testing

The HIL harness (`scpn-control hil-test`) replays reference shot data
through the full controller pipeline, verifying end-to-end latency
and control fidelity against experimental baselines.

---

## Research & Education

### Neuro-Symbolic Control Research

scpn-control is the only open-source implementation of:

- Stochastic Petri Net to SNN compilation for fusion control
- Kuramoto-Sakaguchi phase dynamics with Lyapunov stability monitoring
- Phase-amplitude coupling (PAC) between oscillator populations

These are active research frontiers. The implementation is tested but
not yet validated in peer-reviewed fusion publications.

### Graduate-Level Course Material

The codebase includes:

- 5 tutorial notebooks (Jupyter) with step-by-step walkthroughs
- A live Streamlit dashboard for interactive exploration
- 2,417 tests (99.99% coverage) demonstrating expected behavior and edge cases
- Competitive analysis against state-of-the-art codes

---

## Edge & Embedded Deployment

### No GPU Required

The neural equilibrium kernel achieves 0.39 ms on CPU only (not
head-to-head validated against P-EFIT on identical equilibria).
This *potentially* enables deployment on:

- ARM-based edge controllers (not tested on ARM)
- Embedded systems (not tested in radiation-hard environments)
- Air-gapped control networks (no cloud dependency)

### Rust Native Backend

The 5-crate Rust workspace provides:

- Zero-copy PyO3 bindings for Python interop
- No runtime garbage collection (deterministic latency)
- Rayon parallelism for multi-core scaling
- Criterion benchmarks for regression testing

---

## Comparison Matrix

| Use Case | scpn-control | TORAX | FUSE | FreeGS |
|----------|-------------|-------|------|--------|
| Real-time kernel | 11.9 µs (bare step) | No | No | No |
| Disruption prediction | Experimental (synthetic) | No | No | No |
| SPI mitigation | Experimental | No | No | No |
| Digital twin | Experimental | No | No | No |
| Neural equilibrium | 0.39 ms (CPU, not cross-validated) | No | No | No |
| SNN controller | Yes (Nengo, mocked CI) | No | No | No |
| Contract checking | Yes (runtime assertions) | No | No | No |
| QLKNN-10D transport | Yes (trained MLP) | Yes (QLKNN10D) | No | No |
| PPO RL agent | Yes (beats MPC + PID) | No | No | No |
| Edge deployment | Possible (Rust, untested on ARM) | No (JAX) | No (Julia) | Partial |
| Autodifferentiation | **Yes (JAX)** | **Yes (JAX)** | **Yes (Julia)** | No |
| GPU transport | **Yes (JAX)** | **Yes** | **Yes** | No |
| Real tokamak data | **No** | **Yes** | **Yes** | **Yes** |
| Peer-reviewed papers | **No** | **Yes** | **Yes** | **Yes** |

---

## Get Started

```bash
pip install scpn-control
scpn-control demo --steps 1000
```

For commercial licensing: [protoscience@anulum.li](mailto:protoscience@anulum.li)
