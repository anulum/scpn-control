# Use Cases

## Fusion Reactor Control

### Tokamak Real-Time Plasma Control

Deploy scpn-control as the real-time controller for experimental or
commercial tokamaks. The 11.9 us P50 control loop enables:

- **Shape control** at 10--30 kHz (vs. 4 kHz on current DIII-D systems)
- **Disruption prediction** with ML inference under 1 ms
- **SPI mitigation** with halo current and runaway electron physics
- **Profile control** via neuro-cybernetic dual R+Z SNN feedback

**Target users:** Fusion startups (Commonwealth Fusion Systems, TAE, Tokamak
Energy, Zap Energy), national labs (DIII-D, JET, KSTAR, EAST).

### ITER / DEMO Integration

The formal verification layer (contract-based pre/post-condition checking)
directly addresses ITER nuclear safety requirements. The SNN controller
provides fail-safe operation: if the Lyapunov guard detects instability,
the system halts to safe state within one control cycle.

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

- Stochastic Petri Net to SNN compilation
- Kuramoto-Sakaguchi phase dynamics with formal Lyapunov stability
- Phase-amplitude coupling (PAC) between oscillator populations

These are active research frontiers in both fusion plasma physics
and computational neuroscience.

### Graduate-Level Course Material

The codebase includes:

- 5 tutorial notebooks (Jupyter) with step-by-step walkthroughs
- A live Streamlit dashboard for interactive exploration
- 1243 tests demonstrating expected behavior and edge cases
- Full competitive analysis against state-of-the-art codes

---

## Edge & Embedded Deployment

### No GPU Required

The neural equilibrium kernel achieves P-EFIT-class speed (0.39 ms)
on CPU only. This enables deployment on:

- ARM-based edge controllers
- Radiation-hardened embedded systems
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
| Real-time control | 11.9 us | No | No | No |
| Disruption prediction | ML-based | No | No | No |
| SPI mitigation | Yes | No | No | No |
| Digital twin | Real-time | No | No | No |
| Neural equilibrium | 0.39 ms (CPU) | No | No | No |
| SNN controller | Yes | No | No | No |
| Formal verification | Contracts | No | No | No |
| Edge deployment | Yes (Rust) | No (JAX) | No (Julia) | Partial |
| Autodifferentiation | No | Yes (JAX) | Yes (Julia) | No |
| GPU transport | No | Yes | Yes | No |

---

## Get Started

```bash
pip install scpn-control
scpn-control demo --steps 1000
```

For commercial licensing: [protoscience@anulum.li](mailto:protoscience@anulum.li)
