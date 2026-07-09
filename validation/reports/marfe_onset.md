# MARFE Radiation-Condensation Validation

- Schema: `scpn-control.marfe-onset-validation.v1`
- Generated (UTC): 2026-06-15T10:11:57Z
- Target: `local-marfe-onset`
- Claim status: `bounded_model`
- Public claim allowed: `False`
- Geometry: R0=6.2 m, a=2.0 m, q95=3.0
- Impurity: W at fraction 0.0001
- Status: **pass**

## Exact closed-form references (relative error, gate < 1.0e-09)

| reference | value |
| --- | --- |
| Greenwald limit | 0.000e+00 |
| Greenwald scaling | 0.000e+00 |
| MARFE density-limit scaling | 0.000e+00 |
| MARFE scaling exponents | 0.000e+00 |
| connection length L_parallel = pi q95 R0 | 0.000e+00 |

## Radiation-condensation boundary

- Critical density: 0.127401 x 10^20 m^-3
- Growth below critical: -0.00131304 s^-1
- Growth above critical: 0.00129997 s^-1
- Onset temperature: 5000 eV
- Cooling slope at onset: -1.123857e-35

## Stability diagram and front detector

- Scan critical density: 75.4938 x 10^20 m^-3
- Below/above limit states: 1 / -1
- Front detection passed: `True`

This is bounded local-regression evidence. Measured MARFE campaign or documented public-reference
artefacts remain required for facility density-limit claims.
