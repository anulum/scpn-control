# MAST dB/dt source authority

`validation/mast_dbdt_authority.py` is the second L2F-12d admission gate. It
protects canonical `dBdt_gauss_per_s` from a source-quantity ambiguity in the
centre-column poloidal Mirnov array. The gate measures metadata, geometry,
finite coverage, and timebase facts, but never differentiates, filters, reduces,
fills, or converts a source trace.

## Why the old loop is unsafe

The historical replay candidate selects the first row, replaces non-finite
values with zero, applies `np.gradient(field, time) * 1e4`, and then takes a
per-bin peak magnitude. This is retained only as a compatibility recipe. It is
not an authoritative physical binding because the source metadata conflicts:

- the pinned FAIR-MAST profile maps five `xmc/CC/MV/*` channels into
  `b_field_pol_probe_cc_field`, declares units `T`, and applies a `2e-6` scale;
- the live array label says `Tesla/sec`, while its top-level units remain `T`;
- the MAST magnetic-diagnostics paper describes the relevant fast centre-column
  Bv-coil output as non-integrated; and
- the acquired artefact does not preserve a declaration resolving whether its
  stored values represent magnetic field or its time derivative.

Treating a derivative source as field would differentiate twice. Treating a
field source as an existing derivative would omit the derivative. Numeric
smoothness, magnitude, a name containing `field`, or a label containing
`Tesla/sec` cannot resolve that choice.

Primary sources:

- [Pinned FAIR-MAST centre-column profile](https://github.com/ukaea/fair-mast-ingestion/blob/862f08d7d91930b988d674e7ec67f3a03aacafac/mappings/level2/mast.yml#L3934-L4033)
- [Pinned FAIR-MAST geometry mapping](https://github.com/ukaea/fair-mast-ingestion/blob/862f08d7d91930b988d674e7ec67f3a03aacafac/mappings/level2/mast.yml#L3105-L3123)
- [MAST magnetic-diagnostics paper](https://doi.org/10.1063/1.1309009)
- [UKAEA-hosted paper](https://scientific-publications.ukaea.uk/wp-content/uploads/Published/RSIVOL72p421.pdf)

## Exclusive transform contract

A future source can clear the gate through exactly one branch:

| Attested source quantity | Required source units | Only admissible transform |
|---|---|---|
| magnetic field | `T` | differentiate once in physical time, then multiply by `1e4 G/T` |
| magnetic-field time derivative | `T/s` | do not differentiate; multiply by `1e4 G/T` |

The contract units must exactly match the observed array metadata. Its own
schema, pinned mapping commit/file digest, and canonical JSON digest are
recomputed, so a post-attestation mutation cannot silently change branch or
mapping lineage. Neither branch is executed by the authority gate.

## Additional admission requirements

Resolving the quantity alone is insufficient. The source must also attest:

1. the ordered five source rows `201, 210, 220, 230, 240`;
2. the complete ordered geometry inventory `201…240`, finite phi/r/z values,
   and exact source-to-geometry indices `0, 9, 19, 29, 39`;
3. the dimensional meaning of the mapping scale and its evidence digest;
4. measured component, probe orientation, and sign convention;
5. probe reduction, missing-data, and bad-channel policies;
6. filter and edge policies with a positive cutoff below source Nyquist;
7. calibration evidence and positive one-standard-deviation uncertainty;
8. released, signed geometry with a creator commit and value digests; and
9. the pinned primary-source citations and a self-consistent authority digest.

The source timebase must be finite, strictly increasing, and uniform. Existing
non-finite samples remain blockers; the gate never changes them to zero.

## Real two-shot result

Generation-pinned shots 30421 and 30424 contain five field rows with 382,601
and 398,201 samples respectively. Both have a measured period of approximately
`2e-6 s`, or 500 kHz, and every field row has one non-finite value at index zero.
The source exposes three 40-element geometry arrays, but their metadata still
uses the combined unit string `SI, degrees, m`, a development revision, and no
creator commit. It preserves neither the mapping-scale declaration nor an
authority contract.

Both shots therefore remain `blocked` on the metadata-unit/label conflict,
missing scale semantics, non-finite first samples, unreleased geometry, absent
geometry creator commit, and absent dB/dt authority contract. Canonical binding,
training, scientific validation, facility prediction, and control admission all
remain false.

The deterministic reports are:

- `.coordination/evidence/SCPN-CONTROL/l2f12d_mast_dbdt_authority_2026-07-23T002907Z.json`
- `.coordination/evidence/SCPN-CONTROL/l2f12d_mast_dbdt_authority_2026-07-23T002907Z_repeat.json`

They are byte-identical with file SHA-256
`8ae8c8e5c43b621995727f562d8097673910f3c9c8cc44bae720b6f790617a84`
and payload digest
`39c4d603e5275d6f1902d26f032dd870e7f319c936b9506b25c0e5e505604736`.

## Reproduction

Run only against a verified SourceObjectManifest-v2 root:

```bash
python -m validation.mast_dbdt_authority \
  --manifest /external/source-v4/source_object_manifest.json \
  --artifact-root /external/source-v4 \
  --shot-id 30421 --shot-id 30424 \
  --json-out /external/l2f12d-dbdt-authority.json
```

The output is exclusive-create, deterministic, and self-digested. It contains
metadata, shapes, finite-sample positions, timebase measurements, blockers, and
digests only; it contains no raw or derived magnetic trace.
