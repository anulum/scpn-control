# RZIP Rigid Vertical Stability Validation

- Schema: `scpn-control.rzip-vertical-stability-validation.v1`
- Generated (UTC): 2026-06-14T14:24:29Z
- Target: `local-rzip-vertical-stability`
- Geometry: R0=1.7 m, a=0.5 m, kappa=1.8, Ip=1.0 MA, M_eff=2.0 kg
- Status: **pass**

## Exact no-wall references (largest eigenvalue of the 2x2 rigid block)

- Max unstable growth-rate rel error (n<0): 2.140e-16
- Max stable oscillation-frequency rel error (n>0): 0.000e+00
- Max growth-time identity rel error: 1.137e-16
- Marginal growth rate at n=0: 0.000e+00 (gate < 1.0e-06)
- Exact-reference tolerance: 1.0e-09

## Exact scaling laws

| law | measured ratio | expected | rel error |
| --- | --- | --- | --- |
| current_linear | 2.000000 | 2.0 | 0.000e+00 |
| index_sqrt | 2.000000 | 2.0 | 0.000e+00 |
| inertia_inverse_sqrt | 0.500000 | 0.5 | 0.000e+00 |

## Resistive-wall stabilisation

- No-wall growth rate: 1.7150e+02 s^-1
- With-wall growth rate: 4.2162e-01 s^-1
- Wall slows growth: True; finite: True
