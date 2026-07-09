# Ideal-MHD Stability Metric Validation

- Schema: `scpn-control.mhd-stability-validation.v1`
- Generated (UTC): 2026-06-14T15:21:10Z
- Target: `local-mhd-stability`
- Status: **pass**

## Exact closed-form references (relative error, gate < 1.0e-09)

| reference | value |
| --- | --- |
| Troyon beta_N = 100 beta_t a B0 / Ip | 0.000e+00 |
| Troyon scaling laws (max) | 0.000e+00 |
| Troyon boundary flags consistent | True |
| Mercier D_M = s(s-1) + alpha(1-s/2) | 0.000e+00 |
| Mercier marginal cases consistent | True |
| ballooning alpha_crit (Connor-Hastie-Taylor) | 0.000e+00 |
| ballooning branch closed forms | True |
| Kruskal-Shafranov q_edge > 1 | True |

## Troyon scaling exponents

| law | measured ratio | expected | rel error |
| --- | --- | --- | --- |
| beta_t_linear | 2.000000 | 2.0 | 0.000e+00 |
| minor_radius_linear | 2.000000 | 2.0 | 0.000e+00 |
| field_linear | 2.000000 | 2.0 | 0.000e+00 |
| current_inverse | 0.500000 | 0.5 | 0.000e+00 |
