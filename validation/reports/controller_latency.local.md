# Control-loop latency (per controller)

Generated: 2026-06-21T21:11:34.295126+00:00
Iterations: 2000 (warm-up 200), platform: Linux-6.17.0-35-generic-x86_64-with-glibc2.39

| Controller | Backend | P50 (us) | P95 (us) | P99 (us) | Throughput (kHz) | Note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| PID | numpy | 0.2 | 0.4 | 0.6 | 4032.3 | measured |
| PID | rust | n/a | n/a | n/a | n/a | unavailable: PyPIDController not built |
| SNN | numpy | 13.3 | 22.9 | 26.6 | 75.3 | measured |
| SNN | rust | 0.6 | 0.6 | 1.1 | 1798.6 | measured |
| MPC (Np=10) | internal | 10459.7 | 18050.9 | 25058.2 | 0.1 | measured |
| MPC (Np=10) | scipy | 12750.4 | 19548.1 | 21701.3 | 0.1 | measured |
| MPC (Np=10) | osqp | 9202.4 | 13855.1 | 15622.4 | 0.1 | measured |
| MPC (Np=10) | casadi | 54573.4 | 76801.3 | 88107.3 | 0.0 | measured |
| MPC (Np=10) | acados | 6127.8 | 9617.9 | 10417.2 | 0.2 | measured |
| H-infinity | numpy | 13.3 | 24.7 | 29.8 | 75.3 | general state-space |
| H-infinity | rust | 1.1 | 1.3 | 5.3 | 952.4 | 2-state VDE |
