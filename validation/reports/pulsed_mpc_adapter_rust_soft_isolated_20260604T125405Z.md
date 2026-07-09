# Rust Pulsed MPC Adapter Benchmark

- Evidence class: `local_regression`
- Production claim allowed: `false`
- Steps: `2000`
- Warmup: `200`

## Results

| Case | Samples | Mean us | Median us | p95 us | p99 us |
|---|---:|---:|---:|---:|---:|
| `rust_burn_feasible` | 2000 | 39.155620 | 37.292000 | 47.854000 | 68.045000 |
| `rust_burn_infeasible_safe` | 2000 | 37.025599 | 35.926000 | 45.209000 | 49.520000 |
| `rust_non_burn_mask` | 2000 | 37.359761 | 36.123000 | 44.877000 | 51.780000 |

## Claim boundary

This is local regression evidence for the Rust pulsed MPC admission adapter.
It is not target-hardware timing evidence and does not admit facility PCS claims.

Payload SHA-256: `79469bca420a771a00daa013297ea7f5767160a75e4ed70d5e24352e8c16c4f7`
