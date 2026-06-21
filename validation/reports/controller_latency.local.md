# Control-loop latency (per controller)

Generated: 2026-06-21T20:06:03.020669+00:00
Iterations: 2000 (warm-up 200), platform: Linux-6.17.0-35-generic-x86_64-with-glibc2.39

| Controller | Backend | P50 (us) | P95 (us) | P99 (us) | Throughput (kHz) |
| --- | --- | ---: | ---: | ---: | ---: |
| PID | numpy | 0.3 | 0.3 | 0.4 | 3289.5 |
| SNN (NumPy) | numpy | 16.1 | 29.5 | 36.7 | 62.0 |
| SNN (Rust) | rust | 0.6 | 0.7 | 1.2 | 1562.5 |
| MPC (Np=10) | numpy | 19768.7 | 23907.6 | 35183.1 | 0.1 |
| H-infinity | numpy | 24.1 | 27.3 | 30.4 | 41.5 |
