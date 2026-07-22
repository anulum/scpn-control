# MAST saddle-modal authority

`validation/mast_saddle_modal_authority.py` is the L2F-12c admission gate for
the canonical disruption channels `n1_amp` and `n2_amp`. It verifies the
source field, timebase, and toroidal geometry but deliberately does not execute
a Fourier reduction while the row identities and calibration policy are
incomplete.

## Source and transform boundary

The pinned FAIR-MAST ingestion mapping names a twelve-row field source,
`ASM/SAD/OUT`, with ordered source channels `ASM_SAD/M01` through
`ASM_SAD/M12`. It also exposes lower, middle, and upper saddle-coil geometry
from a separate geometry file. Each geometry surface contains twelve toroidal
polygons in degrees.

The candidate transform is fixed as

```text
A_n(t) = (2/N) * abs(sum_k B_k(t) * exp(-i*n*phi_k)))
```

for `n=1` and `n=2`. This formula is only a candidate until the field rows and
geometry rows have an authoritative join. A matching row count or an evenly
spaced geometry is not that join.

The complete upstream mapping is pinned by commit and whole-file SHA-256:

- [FAIR-MAST MAST Level-2 mapping](https://github.com/ukaea/fair-mast-ingestion/blob/862f08d7d91930b988d674e7ec67f3a03aacafac/mappings/level2/mast.yml#L3498-L3685)
- mapping file SHA-256:
  `cb5420f8a9f78bbf417e3d2314e35bf58f402f221a6c9fb0eed5dfbfcefb9b76`
- dataset context: [FAIR-MAST SoftwareX paper](https://doi.org/10.1016/j.softx.2024.101869)

## Required authority

Canonical modal binding requires all of the following to be persisted in the
source lineage:

1. a released geometry revision tied to the exact geometry-value digest;
2. the ordered identity of every field row;
3. the field-to-geometry row join and the selected lower, middle, or upper
   saddle set;
4. field calibration and one-standard-deviation uncertainty; and
5. baseline, saturation, missing-value, and bad-channel policies.

No observed phase, amplitude, or regular spacing may supply these missing
declarations.

## Real two-shot result

The generation-pinned SourceObjectManifest-v2 snapshot for MAST shots 30421
and 30424 contains all three `12 x 28` geometry arrays. Their circular polygon
centres agree at 15°, 45°, through 345°. The field arrays have shapes
`12 x 38261` and `12 x 39821`; their timebases are finite and strictly
increasing.

Only field rows 1 through 8 contain finite values. Rows 9 through 12 contain no
finite sample in either shot. The source metadata also preserves only the
first UDA row name, does not attest which geometry vertical set joins the
field, gives no positive standard uncertainty or baseline/bad-channel policy,
and describes the geometry as development revision 0 with an empty creator
commit. The array-level geometry unit is the compound string
`SI, degrees, m`, while the pinned mapping declares the toroidal component in
degrees.

The result is therefore `blocked`. No modal amplitude is calculated by the
gate, and the signal-binding and replay reports keep both canonical channels
inadmissible. Training, scientific validation, facility prediction, and
control admission remain false. The byte-reproduced internal report is
`.coordination/evidence/SCPN-CONTROL/l2f12c_mast_saddle_modal_authority_2026-07-22T234159Z.json`;
its file SHA-256 is `b321b337...b2919` and its payload digest is
`0785c0e3...88177`.

## Reproduction

Run only against a verified SourceObjectManifest-v2 root:

```bash
python -m validation.mast_saddle_modal_authority \
  --manifest /external/source-v4/source_object_manifest.json \
  --artifact-root /external/source-v4 \
  --shot-id 30421 --shot-id 30424 \
  --json-out /external/l2f12c-authority.json
```

The output is deterministic and self-digested. The gate records shapes, finite
counts, geometry-value identities, and circular centres, but never emits raw
arrays or derives `n1_amp` or `n2_amp` while any authority blocker remains.
