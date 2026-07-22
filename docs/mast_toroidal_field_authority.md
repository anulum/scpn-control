# MAST toroidal-field authority

`validation/mast_toroidal_field_authority.py` is the L2F-12a admission gate for
the canonical disruption channel `BT_T`. It does not extract, interpolate, or
rename a field. It answers the narrower question: does one verified FAIR-MAST
source object carry enough physical authority to call a value canonical total
toroidal field?

## Quantity boundary

The pinned FAIR-MAST Level-2 mapping exposes separate `bphi_rmag` and
`bvac_rmag` arrays. The direct route therefore requires all three source
members on the same one-dimensional equilibrium timebase:

- `equilibrium.bphi_rmag`, in T, identified as `EFM_BPHI_RMAG`;
- `equilibrium.magnetic_axis_r`, in m, identified as
  `EFM_MAGNETIC_AXIS_R`; and
- `equilibrium.time`, in s.

The output quantity is the signed total toroidal magnetic field at the
time-varying magnetic-axis major radius. Source sign is preserved. Observed
negative or positive samples are evidence about values, not authority for the
physical positive direction. `bvac_rmag` remains comparison evidence only and
cannot be relabelled or absolute-valued into `BT_T`.

The array names and source relationship are pinned to the
[FAIR-MAST ingestion mapping](https://github.com/ukaea/fair-mast-ingestion/blob/862f08d7d91930b988d674e7ec67f3a03aacafac/mappings/level2/mast.yml#L224-L233).
Dataset provenance follows the [FAIR-MAST SoftwareX paper](https://doi.org/10.1016/j.softx.2024.101869).

## Required authority

Array metadata alone is insufficient. Admission additionally requires a
content-digested declaration from an allowlisted UKAEA or FAIR-MAST primary
source for:

1. total-versus-vacuum quantity semantics;
2. the physical positive toroidal-field direction; and
3. one-standard-deviation uncertainty, either as a positive constant in T or
   as a verified, non-negative per-sample array in T.

The alternative TF-current route is also declared, but remains blocked until
the exact signed current signal, timebase, field-radius calibration
`K_TF` in T m A⁻¹, polarity, positive reference radius, all standard
uncertainties, and input covariance policy are source-attested. The gate does
not substitute an ideal-solenoid constant.

## Real two-shot result

The 2026-07-23 bounded run acquired a new SourceObjectManifest-v2 snapshot for
MAST shots 30421 and 30424 into the external Samsung datasets tree. The
snapshot is generation-pinned and contains `bphi_rmag`, `bvac_rmag`,
`magnetic_axis_r`, and their equilibrium timebase without resampling.

Both shots have 107 joint finite field/radius/time samples. The total-field and
vacuum-field candidates are demonstrably different: their maximum absolute
differences are about 0.7774 T and 0.4328 T respectively. This rejects
`bvac_rmag` as an identity substitute on the observed data as well as by
contract.

The current result remains `blocked`: the public array metadata does not supply
an authoritative physical positive direction or one-standard-deviation
uncertainty, and the gate does not invent them. Canonical binding, training,
scientific validation, facility prediction, and control admission remain
false. The internal digest-bound report is
`.coordination/evidence/SCPN-CONTROL/l2f12a_mast_toroidal_field_authority_2026-07-22T224000Z.json`.

## Reproduction

Run the assessment only against a fully verified SourceObjectManifest-v2 root:

```bash
python -m validation.mast_toroidal_field_authority \
  --manifest /external/source-v3/source_object_manifest.json \
  --artifact-root /external/source-v3 \
  --shot-id 30421 --shot-id 30424 \
  --json-out /external/l2f12a-authority.json
```

The output is deterministic and self-digested. A source-member, metadata,
shape, time-order, reference-radius, finite-coverage, sign-authority, or
uncertainty failure always keeps `canonical_bt_binding_admissible` false.
