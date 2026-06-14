<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Two-Point Scrape-Off-Layer Model Validation

- Schema: `scpn-control.sol-two-point-validation.v1`
- Generated (UTC): 2026-06-14T15:02:31Z
- Target: `local-sol-two-point`
- Geometry: R0=1.7 m, a=0.5 m, q95=3.5, B_pol=0.4 T
- Status: **pass**

## Exact closed-form references (relative error, gate < 1.0e-09)

| reference | value |
| --- | --- |
| connection length L_par = pi q95 R0 | 0.000e+00 |
| parallel-flux mapping | 0.000e+00 |
| Spitzer-Härm upstream conduction integral | 1.112e-15 |
| pressure balance n_u T_u = 2 n_t T_t | 0.000e+00 |
| Eich regression exponents | 2.101e-16 |
| peak target heat flux | 0.000e+00 |

## Eich regression scaling exponents

| exponent | measured ratio | expected | rel error |
| --- | --- | --- | --- |
| power_-0.02 | 0.986233 | 0.986233 | 1.126e-16 |
| major_radius_0.04 | 1.028114 | 1.028114 | 0.000e+00 |
| b_pol_-0.92 | 0.528509 | 0.528509 | 2.101e-16 |
| epsilon_0.42 | 1.337928 | 1.337928 | 0.000e+00 |

## Detachment onset boundary

- Analytic critical density: 15.5572 x 10^19 m^-3
- Attached below critical (detached=False), detached above critical (detached=True)
