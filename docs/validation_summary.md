# Validation Summary — scpn-control

This document provides a high-level summary of the validation evidence supporting the 
physical and architectural integrity of the `scpn-control` framework.

| Claim | Evidence | Script | Result |
|-------|----------|--------|--------|
| GS solver converges | Solov'ev analytic benchmark | `test_p0_regression.py` | NRMSE < 1% |
| GS solver accuracy | Mesh convergence study | `mesh_convergence_study.py` | 2nd Order ($O(h^2)$) |
| Transport matches IPB98(y,2) | Scaling law comparison | `validate_real_shots.py` | 95% within 2σ |
| H-inf outperforms PID | Controller comparison | `controller_comparison.py` | 30% reward improvement |
| PPO beats MPC+PID | 500K step benchmark | `scpn_pid_mpc_benchmark.py` | Higher mean reward |
| Real-time capable | Rust kernel latency | `benchmark_disturbance_rejection.py` | 11.9 µs P50 |
| Disruption prediction | Synthetic ROC analysis | `disruption_roc_analysis.py` | AUC > 0.85 (Base) |
| Physical Consistency | Energy balance diagnostic | `benchmark_transport.py` | Error < 1% (Internal) |

## Key Benchmarks

### 1. Equilibrium Accuracy
The Grad-Shafranov solver was benchmarked against the Solov'ev analytic solution. 
A mesh convergence study confirmed that the 5-point central difference stencil 
achieves the theoretical second-order spatial convergence rate.

### 2. Transport Fidelity
The 1.5D transport solver was validated against the ITPA H-mode confinement 
database. The predicted energy confinement times ($\tau_E$) show excellent 
agreement with the IPB98(y,2) scaling law across a wide range of plasma 
parameters ($I_p$, $B_T$, $P_{loss}$).

### 3. Control Performance
The robustness of the control stack was verified through a 1000-shot stress 
campaign. Advanced controllers ($H_\infty$, MPC, PPO) consistently outperform 
standard PID baselines in tracking error and disruption avoidance, especially 
under high-noise conditions.

### 4. Real-Time Latency
End-to-end loop latency was measured using the Rust backend on a standard 
workstation. The median step latency of 11.9 µs supports control frequencies 
up to 80 kHz, exceeding the requirements for modern tokamak operations.
