# MAST EFM Neural-Equilibrium Campaign Evidence

Schema: `scpn-control.mast-efm-neural-equilibrium-campaign.v1`
Status: `blocked`
Reference dataset: `mast-efm-30419-30420-30421-30422-30423-30424-2cb0254259360c05`
Shots: 30419, 30420, 30421, 30422, 30423, 30424
Evaluated equilibria: 527
Candidate payload SHA-256: `8d173f423440243c4362256480e7ec40a8ca16244ac862b727428d6f28f747e5`
Campaign payload SHA-256: `5c5c4fcc5153b0c4f1a916fa6a3b503be750f695c5bddc04d94ca799d2c6bb0f`

## Aggregate metrics

- Flux RMSE mean: `1.5551750363663988` Wb/rad
- Flux RMSE max: `1.6436889101867123` Wb/rad
- Magnetic-axis RMSE mean: `0.7715625800838742` m
- LCFS mean-distance mean: `0.5392084105619522` m
- LCFS p95-distance mean: `1.477022081443005` m
- Pressure RMSE: `None`
- q-profile RMSE: `None`

## Shot metrics

| Shot | Slices | Flux RMSE | Axis RMSE | LCFS mean distance | LCFS p95 distance |
| --- | ---: | ---: | ---: | ---: | ---: |
| 30419 | 107 | 1.574623069235294 | 0.8009795241997876 | 0.5942330825260514 | 1.6152649379194224 |
| 30420 | 53 | 1.6436889101867123 | 0.7833022897116073 | 0.4908012378426153 | 1.4648543346750043 |
| 30421 | 108 | 1.5652227141560995 | 0.7975102800213814 | 0.5924676767199194 | 1.6037861409317404 |
| 30422 | 72 | 1.4860590789763775 | 0.7257250416594768 | 0.4797446498438281 | 1.289618186625269 |
| 30423 | 80 | 1.4995240773693 | 0.7248154920139316 | 0.4846184275076656 | 1.2713755794582875 |
| 30424 | 107 | 1.5619323682746085 | 0.797042852897061 | 0.5933853889316338 | 1.617233309048306 |

## Admission boundary

Repository-published campaign evidence covers public MAST EFM flux and derived geometry evaluation only. Predictive EFIT/P-EFIT admission remains blocked until exact pressure, q-profile, LCFS, magnetic-axis, and matched public-reference or P-EFIT artefacts pass declared tolerances.

Fallback features still present: `Bt_T`, `Ip_MA`, `ffprime_scale`

## Next processing steps

- assemble a full-output supervised dataset from converted MAST EFM bundles
- train or fine-tune a model that predicts flux, pressure, q-profile, LCFS, and magnetic-axis outputs
- replace fallback Ip_MA, Bt_T, and ffprime_scale features with acquired or documented public inputs
- emit strict scpn-control.neural-equilibrium-reference.v1 artefacts only after all required outputs pass tolerances
