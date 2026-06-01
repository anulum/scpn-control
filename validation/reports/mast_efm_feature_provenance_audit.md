# MAST EFM Feature-Provenance Audit

Schema: `scpn-control.mast-efm-feature-provenance-audit.v1`
Status: `pass`
Reference dataset: `mast-efm-30419-30420-30421-30422-30423-30424-a50277960ea1ada9`
Reference bundles: 6

## Fallback feature status

| Feature | Status | Present keys | Resolution |
|---|---|---|---|
| `Ip_MA` | `resolved` | `Ip_MA` | direct public bundle key available |
| `Bt_T` | `resolved` | `Bt_T` | direct public bundle key available |
| `ffprime_scale` | `resolved` | `ffprime_rms_T_rad` | direct public bundle key available |

## Available reference keys

`Bt_T`, `Ip_MA`, `ffprime_rms_T_rad`, `lcfs_r_m`, `lcfs_valid_mask`, `lcfs_z_m`, `magnetic_axis_r_m`, `magnetic_axis_z_m`, `pprime_Pa_per_Wb_rad`, `pprime_valid_mask`, `psi_axis_Wb_per_rad`, `psi_boundary_Wb_per_rad`, `psirz_Wb_per_rad`, `psirz_valid_mask`, `q_profile`, `q_profile_valid_mask`, `r_grid_m`, `shot_id`, `time_s`, `z_grid_m`

## Next processing steps

- keep the converted public feature-source keys fixed while training and holdout evaluation are performed
- rebuild the supervised dataset whenever converted reference bundles are regenerated
