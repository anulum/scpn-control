# MAST EFM Neural-Equilibrium Result Templates

Schema: `scpn-control.mast-efm-neural-equilibrium-result-templates.v1`
Expected dataset SHA-256: `3206bd530efdd6fc73bae57b2ac18646aff39e130533c7d5167abe1ae7d136f3`

These are result schemas for a later admitted compute run. They are not executed training evidence.

## Output policy

weights are written to workstation or external cloud compute storage, not storage-host dataset storage

## Template schemas

### `holdout_metrics`

- Schema: `scpn-control.mast-efm-neural-equilibrium-holdout-metrics.v1`
- Acceptance policy: compact train, validation, and test metrics must be emitted before predictive admission is requested

### `latency_metrics`

- Schema: `scpn-control.mast-efm-neural-equilibrium-latency-metrics.v1`
- Acceptance policy: latency is evidence only after hardware, precision, batch size, and sample count are recorded

### `gpu_cost`

- Schema: `scpn-control.mast-efm-neural-equilibrium-gpu-cost.v1`
- Acceptance policy: cost reports must distinguish planning estimates from measured billing evidence

### `admission_certificate`

- Schema: `scpn-control.mast-efm-neural-equilibrium-admission-certificate.v1`
- Acceptance policy: certificate stays blocked until the strict neural-equilibrium reference gate admits the exact weights
