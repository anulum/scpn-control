<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Halo-Current L/R Circuit Validation

- Schema: `scpn-control.halo-current-validation.v1`
- Generated (UTC): 2026-06-14T19:10:38Z
- Target: `halo-current-lr-circuit`
- Status: **pass**

## Exact circuit relations (relative error, gate < 1.0e-09)

| relation | value |
| --- | --- |
| halo resistance R_h | 0.000e+00 |
| halo inductance L_h | 0.000e+00 |
| mutual inductance M | 0.000e+00 |
| time constant tau_h = L_h/R_h | 0.000e+00 |
| R_h scaling laws (max) | 0.000e+00 |
| wall force F = mu0 I_h I_p/(2 pi a) | 0.000e+00 |
| TPF product | 0.000e+00 |

## Quasi-static L/R tracking (fast-circuit limit)

- tau_cq values: [0.5, 1.0, 2.0]
- tracking errors: ['3.235e-02', '1.425e-02', '5.421e-03']
- monotonic decrease: True; finest error: 5.421e-03 (gate < 1.0e-02)
