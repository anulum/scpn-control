# Control-loop latency (per controller)

Generated: 2026-06-21T20:12:56.784274+00:00
Iterations: 2000 (warm-up 200), platform: Linux-6.17.0-35-generic-x86_64-with-glibc2.39

| Controller | Backend | P50 (us) | P95 (us) | P99 (us) | Throughput (kHz) | Note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| PID | numpy | 0.3 | 0.3 | 0.3 | 3984.1 | measured |
| PID | rust | n/a | n/a | n/a | n/a | unavailable: PyPIDController not built |
| SNN | numpy | 14.2 | 15.7 | 22.6 | 70.6 | measured |
| SNN | rust | 0.6 | 0.6 | 1.1 | 1686.3 | measured |
| MPC (Np=10) | internal | 12247.7 | 21066.7 | 22638.7 | 0.1 | measured |
| MPC (Np=10) | scipy | 22331.4 | 23707.1 | 25160.9 | 0.0 | measured |
| MPC (Np=10) | osqp | 12957.2 | 14006.7 | 15880.2 | 0.1 | measured |
| MPC (Np=10) | casadi | n/a | n/a | n/a | n/a | unavailable: casadi not installed |
| MPC (Np=10) | acados | n/a | n/a | n/a | n/a | unavailable: acados_template not installed |
| H-infinity | numpy | 22.3 | 23.5 | 26.1 | 44.8 | general state-space |
| H-infinity | rust | 1.2 | 1.2 | 1.3 | 829.2 | 2-state VDE |
