# MAST normalised-beta authority

`validation/mast_normalised_beta_authority.py` is the L2F-12b admission gate
for the canonical disruption channel `beta_N`. It diagnoses the public
FAIR-MAST `EFM_BETAN` source without guessing a scale, changing its values, or
promoting a metadata correction into physical authority.

## Definition and metadata boundary

The pinned IMAS Data Dictionary defines `beta_tor_norm` as

```text
100 * beta_tor * a[m] * B0[T] / Ip[MA]
```

with unit `1`. The pinned FAIR-MAST ingestion mapping gives the same definition
for `EFM_BETAN`, but targets the older IMAS leaf `beta_tor_normal`. The live
Level-2 array has source unit `T`. These facts identify a metadata conflict;
they do not define a physical T-to-1 conversion.

The only non-destructive candidate is therefore to preserve the source values
and repair metadata after authority is complete. The gate sets
`numeric_transform` to null, forbids range-based scale inference, and never
clamps or discards negative samples.

The source mapping is pinned to the
[FAIR-MAST ingestion repository](https://github.com/ukaea/fair-mast-ingestion/blob/862f08d7d91930b988d674e7ec67f3a03aacafac/mappings/level2/mast.yml#L201-L207).
The canonical leaf, definition, and unit are pinned to the
[IMAS Data Dictionary](https://github.com/iterorganization/IMAS-Data-Dictionary/blob/ba4fa8a57b0c5f35d3225fdeb5e0ea08f1c8d16f/schemas/equilibrium/dd_equilibrium.xsd#L751-L759).

## Required authority

Canonical binding requires either primary-source authority for the direct
array or an independently reproducible formula route. The reproduction route
must bind, on one source-attested alignment:

- `equilibrium.beta_tor`, including whether values are fractional or percent;
- `equilibrium.minor_radius`, in m;
- `equilibrium.bvac_rgeom`, the exact vacuum field used by EFM at the
  geometric axis;
- `equilibrium.plasma_current_x`, including its sign or magnitude convention;
- `equilibrium.time`;
- an equilibrium validity or reconstruction-quality rule; and
- one-standard-deviation uncertainties plus the covariance policy.

Names, approximate magnitudes, positive majorities, or dimensional
cancellation are not substitutes for those declarations.

## Real two-shot result

The 2026-07-23 bounded run acquired a fresh SourceObjectManifest-v2 snapshot
for MAST shots 30421 and 30424 under the external Samsung datasets tree. It
retains `minor_radius` without resampling and is pinned to source generations
`44d2c93f...a32868` and `1da4ee7c...f6b0`. The complete manifest has payload
digest `aba5e048...d196a` and records 14,699,129 derived bytes.

The direct `beta_N` evidence contains 107 finite samples per shot. Shot 30421
has 3 negative and 104 positive finite samples, with an observed range of
-13.4791 to 3.9432. Shot 30424 has 4 negative and 103 positive finite samples,
with an observed range of -5.8219 to 3.8770. These values are observations,
not scale or validity authority.

`beta_tor`, `minor_radius`, and the equilibrium timebase are present. The exact
`bvac_rgeom` and fitted `plasma_current_x` inputs are absent. Both shots remain
`blocked` for that missing formula lineage, the live source-unit and stale-leaf
conflicts, unresolved sign/scale and negative-value validity, absent
reconstruction-quality authority, and absent one-standard-deviation
uncertainty. Canonical binding, training, scientific validation, facility
prediction, and control admission all remain false.

The self-digested report is
`.coordination/evidence/SCPN-CONTROL/l2f12b_mast_normalised_beta_authority_2026-07-22T225703Z.json`.
Its payload digest is `6c8e8b4e...c9706a1`; its file SHA-256 is
`5dc6c759...a425e`.

## Reproduction

Run the gate only against a verified SourceObjectManifest-v2 root:

```bash
python -m validation.mast_normalised_beta_authority \
  --manifest /external/source-v4/source_object_manifest.json \
  --artifact-root /external/source-v4 \
  --shot-id 30421 --shot-id 30424 \
  --json-out /external/l2f12b-authority.json
```

The report is deterministic and self-digested. A source-member, metadata,
shape, time-order, formula-lineage, quality, sign/scale, negative-value, or
uncertainty failure keeps `canonical_beta_n_binding_admissible` false.
