# Control-loop latency (per controller)

Generated: 2026-06-21T20:05:33.774435+00:00
Iterations: 2000 (warm-up 200), platform: Linux-6.17.0-1018-azure-x86_64-with-glibc2.39

| Controller | Backend | P50 (us) | P95 (us) | P99 (us) | Throughput (kHz) |
| --- | --- | ---: | ---: | ---: | ---: |
| PID | numpy | 0.5 | 0.5 | 0.5 | 2217.3 |
| SNN (NumPy) | numpy | 23.1 | 24.6 | 41.4 | 43.3 |
| SNN (Rust) | rust | 0.9 | 0.9 | 1.0 | 1146.8 |
| MPC (Np=10) | numpy | 23754.6 | 24176.7 | 25543.1 | 0.0 |
| H-infinity | numpy | 28.2 | 31.2 | 50.0 | 35.4 |
