# Density-Control and Interferometry Validation

- Schema: `scpn-control.density-control-validation.v2`
- Generated (UTC): 2026-08-30T03:24:54Z
- Target: `density-control-annular-interferometry`
- Status: **pass**

## Exact relations (relative error, gate < 1.0e-09)

| relation | value |
| --- | --- |
| Greenwald limit I_p/(pi a^2) | 0.000e+00 |
| Greenwald fraction <n>/n_GW | 0.000e+00 |
| flux-surface volumes V', V | 0.000e+00 |
| gas-puff source conservation | 2.705e-12 |
| neutral-beam source conservation | 6.253e-13 |
| recycling source conservation | 3.572e-12 |
| cryopump edge sink | 0.000e+00 |
| Greenwald scaling laws (max) | 0.000e+00 |

## Diffusion operator on a uniform profile

- maximum interior relative change: 0.000e+00 (gate < 1.0e-12)

## Circular interferometer projection

- uniform-profile chord-length relative error: 4.134e-17
- signed-impact symmetry relative error: 0.000e+00
- status: **pass**

## Runtime source SHA-256

- `src/scpn_control/control/density_controller.py`: `0b00368ee8aead8f41232f2ca3564df27149ea82cf8c561869d9fed1cc973425`
- `validation/validate_density_control.py`: `4687fff52eeafd468942b1fa618de5f7502b412d1e86ee32b07da684a0ee5b71`
