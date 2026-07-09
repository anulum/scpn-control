# EPED Pedestal Model Validation

- Schema: `scpn-control.eped-pedestal-validation.v1`
- Generated (UTC): 2026-06-14T15:46:09Z
- Target: `local-eped-pedestal`
- Status: **pass**

## Exact construction relations (relative error, gate < 1.0e-09)

| relation | value |
| --- | --- |
| q95 = a B0/(R0 B_pol)(1+kappa^2)/2 | 0.000e+00 |
| alpha-inversion pressure | 0.000e+00 |
| beta_p = 2 mu_0 p/B_pol^2 | 0.000e+00 |
| T_ped = p/(2 n_e e) | 0.000e+00 |
| collisionality narrowing | 0.000e+00 |
| nu*=0 collisionless identity | True |
| shaping factor reference (=1) | 0.000e+00 |
| shaping monotonic in triangularity | True |

## KBM width constraint (fixed-point iteration tolerance)

- KBM residual at collisionless width: 1.263e-02 (gate < 3.0e-02)
