# Structured Singular Value (mu) Closed-Form Validation

- Schema: `scpn-control.mu-structured-singular-value-validation.v1`
- Generated (UTC): 2026-06-14T09:37:23Z
- Target: `local-mu-structured-singular-value`
- Seed: 20260614; sizes: [2, 3, 4, 5]; samples/size: 4
- Status: **pass**

| case | samples | max error | tolerance | passed |
| --- | --- | --- | --- | --- |
| full_block_equals_sigma_max | 16 | 2.665e-15 | 1.0e-09 | True |
| diagonal_equals_max_abs_entry | 16 | 4.441e-16 | 1.0e-09 | True |
| rank_one_equals_sum_abs_products | 16 | 1.616e-04 | 1.0e-03 | True |
| spectral_sandwich_rho_le_mu_le_sigma_max | 16 | 1.776e-15 | 1.0e-06 | True |

## Diagnostics (recorded, not gated)

The D-scaling invariance probe exercises the 50-step finite-difference descent in `compute_mu_upper_bound`. The bound is invariant in exact arithmetic, but the descent reaches slightly different local minima per orientation, so the spread below is reported but does not affect the pass/fail outcome.

| diagnostic | samples | max relative spread | soft tolerance | within |
| --- | --- | --- | --- | --- |
| d_scaling_invariance | 16 | 6.234e-02 | 2.0e-01 | True |
