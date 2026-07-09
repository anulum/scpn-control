# Guiding-Centre Orbit Conservation Validation

- Schema: `scpn-control.guiding-centre-conservation-validation.v1`
- Generated (UTC): 2026-06-14T09:49:47Z
- Target: `local-guiding-centre-conservation`
- Field: R0=1.7 m, a=0.5 m, B0=2.0 T
- Energy tolerance: 1.0e-03; momentum tolerance: 1.0e-03
- Max energy drift: 2.589e-05; max momentum drift: 3.382e-05
- Status: **pass**

| orbit | steps | dE/E | dp_phi/p_phi | max |v_par|/v_tot | trapped |
| --- | --- | --- | --- | --- | --- |
| passing_deuteron | 6000 | 6.403e-06 | 1.564e-05 | 0.877583 | False |
| trapped_deuteron | 6000 | 5.139e-07 | 6.180e-08 | 0.120503 | True |
| passing_alpha | 6000 | 2.589e-05 | 3.244e-05 | 0.825336 | False |
| trapped_alpha | 6000 | 1.703e-05 | 3.382e-05 | 0.636055 | True |
