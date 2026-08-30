# Normalized DGKF H-infinity validation

- Generated: `2026-08-30T05:31:16.638904Z`
- Source commit: `62e4655fd05a8297d8b97377d7dee453bffe1995`
- Schema: `scpn-control.h-infinity-validation.v1`
- Payload seal: `55477a039cb5b516cbc7bcd0ae0d391426d0bc6becbf15f2cd56c2602cded9fa`
- Overall: `PASS`

| Check | Result |
|---|---:|
| Admitted gamma | 51.4592957065 |
| Maximum normalization residual | 0.000e+00 |
| X Riccati relative residual | 9.959e-11 |
| Y Riccati relative residual | 2.665e-15 |
| Central-controller formula relative error | 0.000e+00 |
| Spectral feasibility margin | 31.6633821513 |
| Dominant augmented closed-loop pole real part | -1.41514923841 |
| Frequency-sweep peak | 51.458655504 |
| Frequency-sweep peak / gamma | 0.999987559051 |
| Frequency samples | 20002 |

## Claim boundary

This admits only `normalized continuous-time standard plant; D11=D22=0`. The frequency sweep is
`finite numerical corroboration, not exact norm proof`. Production admission is
`false`. Excluded claims:

- facility or reactor validation
- saturated H-infinity guarantee
- arbitrary sampled-data stability
- structured uncertainty or D-K synthesis
- classical gain margin
