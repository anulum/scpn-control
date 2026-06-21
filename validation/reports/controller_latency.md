# Control-loop latency (per controller)

Generated: 2026-06-21T21:18:45.911811+00:00
Iterations: 2000 (warm-up 200), platform: Linux-6.17.0-1018-azure-x86_64-with-glibc2.39

| Controller | Backend | P50 (us) | P95 (us) | P99 (us) | Throughput (kHz) | Note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| PID | numpy | 0.4 | 0.5 | 0.5 | 2375.3 | measured |
| PID | rust | n/a | n/a | n/a | n/a | unavailable: PyPIDController not built |
| SNN | numpy | 23.9 | 25.5 | 43.1 | 41.8 | measured |
| SNN | rust | 0.9 | 0.9 | 1.0 | 1135.1 | measured |
| MPC (Np=10) | internal | 18847.4 | 19026.3 | 25171.8 | 0.1 | measured |
| MPC (Np=10) | scipy | 23302.1 | 23452.8 | 24416.4 | 0.0 | measured |
| MPC (Np=10) | osqp | 16845.9 | 16970.8 | 18090.2 | 0.1 | measured |
| MPC (Np=10) | casadi | 78887.2 | 80546.5 | 82805.9 | 0.0 | measured |
| MPC (Np=10) | acados | 10679.2 | 10841.2 | 11393.3 | 0.1 | measured |
| H-infinity | numpy | 29.4 | 33.3 | 53.4 | 34.0 | general state-space |
| H-infinity | rust | 1.4 | 1.4 | 1.5 | 717.9 | 2-state VDE |
