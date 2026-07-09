# Differentiable Transport Rollout Gradient-Latency Benchmark

This report measures the local audited multi-step source-rollout
gradient-admission path for controller-tuning studies. It is not
a real-time control-loop guarantee.

- Backend: `jax`
- dtype: `float64`
- Runtime platform: `Linux-6.17.0-29-generic-x86_64-with-glibc2.39`
- Runtime machine: `x86_64`
- JAX default backend: `gpu`
- JAX devices: `cuda:0`
- JAX x64 enabled: `True`
- Radial points: `21`
- Rollout steps: `4`
- Timed runs: `5`
- Audit passed: `True`
- P50 latency [ms]: `1024.499274`
- P95 latency [ms]: `1069.227792`
- Max latency [ms]: `1073.515644`
- Claim boundary: `local audited rollout source-gradient latency only; not a real-time control-loop guarantee`
