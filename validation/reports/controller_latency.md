# Control-loop latency (per controller)

Generated: 2026-06-21T20:16:08.878575+00:00
Iterations: 2000 (warm-up 200), platform: Linux-6.17.0-1018-azure-x86_64-with-glibc2.39

| Controller | Backend | P50 (us) | P95 (us) | P99 (us) | Throughput (kHz) | Note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| PID | numpy | 0.4 | 0.5 | 0.5 | 2320.2 | measured |
| PID | rust | n/a | n/a | n/a | n/a | unavailable: PyPIDController not built |
| SNN | numpy | 23.2 | 24.3 | 40.5 | 43.2 | measured |
| SNN | rust | 0.9 | 1.0 | 1.1 | 1084.6 | measured |
| MPC (Np=10) | internal | 23741.5 | 24279.3 | 25160.3 | 0.0 | measured |
| MPC (Np=10) | scipy | 26916.9 | 27631.9 | 28245.0 | 0.0 | measured |
| MPC (Np=10) | osqp | 15895.8 | 16135.6 | 16808.9 | 0.1 | measured |
| MPC (Np=10) | casadi | n/a | n/a | n/a | n/a | unavailable: casadi not installed |
| MPC (Np=10) | acados | n/a | n/a | n/a | n/a | unavailable: acados_template not installed |
| H-infinity | numpy | 28.5 | 31.8 | 48.6 | 35.1 | general state-space |
| H-infinity | rust | 1.3 | 1.4 | 1.7 | 755.9 | 2-state VDE |
