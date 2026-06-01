# MAST EFM Feature-Provenance Audit

Schema: `scpn-control.mast-efm-feature-provenance-audit.v1`
Status: `blocked`
Reference dataset: `mast-efm-30419-30420-30421-30422-30423-30424-2cb0254259360c05`
Reference bundles: 6

## Fallback feature status

| Feature | Status | Present keys | Resolution |
|---|---|---|---|
| `Ip_MA` | `blocked` | none | not present in converted public EFM bundles |
| `Bt_T` | `blocked` | none | not present in converted public EFM bundles |
| `ffprime_scale` | `blocked` | none | not present in converted public EFM bundles |

## Available reference keys

`lcfs_r_m`, `lcfs_valid_mask`, `lcfs_z_m`, `magnetic_axis_r_m`, `magnetic_axis_z_m`, `pprime_Pa_per_Wb_rad`, `pprime_valid_mask`, `psi_axis_Wb_per_rad`, `psi_boundary_Wb_per_rad`, `psirz_Wb_per_rad`, `psirz_valid_mask`, `q_profile`, `q_profile_valid_mask`, `r_grid_m`, `shot_id`, `time_s`, `z_grid_m`

## Next processing steps

- inspect the original public MAST Level 1 EFM/Zarr metadata for plasma-current and toroidal-field channels
- acquire or document public FF-prime/fpol provenance or keep ffprime_scale blocked
- rebuild the supervised dataset after any non-fallback feature sources are admitted
