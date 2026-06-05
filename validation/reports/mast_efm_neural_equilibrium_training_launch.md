# MAST EFM Neural-Equilibrium Training Launch

Schema: `scpn-control.mast-efm-neural-equilibrium-training.v1`
Status: `prepared`
Execution mode: `dry_run`
Dataset path: `/data/SCPN-CONTROL/processed/neural_equilibrium/mast_efm_supervised_dataset.npz`
Dataset SHA-256: `3206bd530efdd6fc73bae57b2ac18646aff39e130533c7d5167abe1ae7d136f3`
Dataset exists on this host: `False`
Weights path: `artifacts/neural_equilibrium/mast_efm_full_output_baseline_weights.npz`

## Execution host policy

The storage host is storage-only; execute training only on this workstation or external cloud compute with the storage-host dataset mounted read-only or copied to admitted compute storage.

## Claim boundary

This report prepares or executes a deterministic repository baseline. It is not predictive EFIT/P-EFIT admission evidence.

## Pre-run admission

Status: `fail`
Dataset SHA-256 verified: `False`
Source provenance: `pass`
Compute execution: `fail`

## Run command

```bash
python validation/train_mast_efm_neural_equilibrium.py --execute --compute-host-kind workstation --dataset-path /data/SCPN-CONTROL/processed/neural_equilibrium/mast_efm_supervised_dataset.npz --weights-out artifacts/neural_equilibrium/mast_efm_full_output_baseline_weights.npz --feature-provenance-report validation/reports/mast_efm_feature_provenance_audit.json --original-source-report validation/reports/mast_efm_original_feature_source_audit.json
```

## Required targets

- `psirz_Wb_per_rad`
- `psirz_valid_mask`
- `psi_axis_Wb_per_rad`
- `psi_boundary_Wb_per_rad`
- `pprime_Pa_per_Wb_rad`
- `pprime_valid_mask`
- `q_profile`
- `q_profile_valid_mask`
- `lcfs_r_m`
- `lcfs_z_m`
- `lcfs_valid_mask`
- `magnetic_axis_r_m`
- `magnetic_axis_z_m`

## Admission blockers

- run --execute on workstation or external cloud compute and publish holdout metrics for train, validation, and test splits
- validate the exact trained weight checksum through the strict neural-equilibrium reference gate
