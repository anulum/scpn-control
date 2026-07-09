# JAX GK Parity Evidence

- Status: `pass`
- Artifacts: `6`
- Max gamma relative error: `1.5386142994101046e-06`
- Max omega absolute error: `2.9658060068937786e-07`
- Entries payload SHA-256: `7c7d3c7eefd5d2577579d1fd89d1fdaa056eebc13aa9d7f06f14cb1e8e755dfb`
- Claim boundary: backend parity only; external GK validation remains required.

| Case | Backend | Dtype | X64 | Native dominant | JAX dominant | Gamma relative error | Omega absolute error |
|---|---|---|---:|---|---|---:|---:|
| cyclone_base_case | cpu | float32 | False | ITG | ITG | 1.25245e-07 | 2.60196e-07 |
| cyclone_base_case | gpu | float32 | False | ITG | ITG | 2.20946e-07 | 2.49803e-07 |
| stable_mode | cpu | float32 | False | ITG | ITG | 8.52525e-07 | 1.27442e-07 |
| stable_mode | gpu | float32 | False | ITG | ITG | 1.53861e-06 | 1.74585e-07 |
| tem_kinetic_electron | cpu | float32 | False | ITG | ITG | 1.05104e-07 | 1.76129e-07 |
| tem_kinetic_electron | gpu | float32 | False | ITG | ITG | 2.13612e-07 | 2.96581e-07 |
