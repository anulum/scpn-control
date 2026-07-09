# Differentiable Transport Gradient-Latency Benchmark

This report measures the local audited gradient-admission path for
controller-tuning studies. It is not a real-time control-loop guarantee.

- Backend: `jax`
- dtype: `float64`
- Runtime platform: `Linux-6.17.0-29-generic-x86_64-with-glibc2.39`
- Runtime machine: `x86_64`
- JAX default backend: `gpu`
- JAX devices: `cuda:0`
- JAX x64 enabled: `True`
- Radial points: `21`
- Timed runs: `5`
- Audit passed: `True`
- P50 latency [ms]: `15.527267`
- P95 latency [ms]: `20.741123`
- Max latency [ms]: `21.153153`
- Claim boundary: `local audited gradient-admission latency only; not a real-time control-loop guarantee`
