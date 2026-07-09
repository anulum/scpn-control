# Pulsed MPC Adapter Benchmark

- Generated UTC: `2026-06-04T12:51:45Z`
- Evidence class: `local_regression`
- Production claim allowed: `False`
- Steps: `2000`
- Warmup: `200`
- CPU affinity: `[4, 5]`
- Load average start: `(4.4228515625, 4.37158203125, 4.220703125)`
- Load average end: `(4.46923828125, 4.38232421875, 4.22509765625)`
- PyO3 status: `optional PyO3 extension unavailable: No module named 'scpn_control_rs'`

## Results

| Case | Samples | Mean us | Median us | p95 us | p99 us |
|---|---:|---:|---:|---:|---:|
| `python_non_burn_mask` | 2000 | 954.450787 | 907.479500 | 1227.176000 | 1387.188000 |
| `python_burn_feasible` | 2000 | 1039.941611 | 992.305500 | 1317.322000 | 1838.956000 |
| `python_burn_infeasible_safe` | 2000 | 1046.881359 | 1021.205500 | 1206.399000 | 1354.851000 |

## Claim boundary

This is local regression evidence for the pulsed MPC admission adapter.
It is not target-hardware timing evidence and does not admit facility PCS claims.

Payload SHA-256: `4e8013b256afe9e1917eaa29138417ab9f91f7b578ef26f7c24d4934e35aa7b1`
