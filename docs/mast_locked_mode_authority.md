# MAST locked-mode authority

`validation/mast_locked_mode_authority.py` is the first L2F-12d admission gate
for canonical `locked_mode_amp`. It distinguishes a source-authoritative MAST
locked-mode observable from the historical SCPN-CONTROL compatibility recipe.
The gate measures source facts and the candidate filter's physical time scale,
but it never filters the saddle array or emits a locked-mode amplitude.

## Primary-source boundary

Kirk et al. describe MAST locked-mode formation as a growing `n=1` radial
magnetic-field perturbation measured by an outer-midplane saddle-coil array.
The modes in the reported experiment form locked without a rotating `m=2,n=1`
signature. The paper does not define SCPN-CONTROL's 201-sample complex boxcar as
the estimator, filter, calibration, or uncertainty contract.

- [Measurement, correction and implications of the intrinsic error fields on MAST](https://doi.org/10.1088/0741-3335/56/10/104003)
- [UKAEA-hosted paper](https://scientific-publications.ukaea.uk/wp-content/uploads/Published/Miss90.pdf)
- [Pinned FAIR-MAST Level-2 mapping](https://github.com/ukaea/fair-mast-ingestion/blob/862f08d7d91930b988d674e7ec67f3a03aacafac/mappings/level2/mast.yml#L3498-L3685)

The gate therefore requires the existing L2F-12c saddle row/geometry authority
and locked-mode-specific declarations for:

1. a radial measured component on the outer mid-plane;
2. a stationary `n=1` estimator in the machine frame;
3. a physical-frequency filter policy, edge handling, and a positive cutoff
   below the source Nyquist frequency;
4. background and poloidal-field pickup correction policies;
5. a vacuum-vessel response policy;
6. positive one-standard-deviation uncertainty;
7. a content digest for the estimator authority; and
8. the exact primary-source citation.

Free-text naming, a smooth-looking trace, or a chosen sample count cannot supply
these declarations.

## Legacy compatibility candidate

The replay producer currently calculates:

```text
abs(boxcar_window(A_1_complex(t), 201 samples))
```

The rationale is that a rotating complex `n=1` phasor may average towards zero
while a stationary component survives. This is a useful synthetic recipe, but
the window is expressed in samples and was not tied to a source-authoritative
physical cutoff, background/pickup subtraction, vessel response, or uncertainty.
The authority gate records this formula as `compatibility_only_not_source_authorised`.

## Real two-shot result

Generation-pinned shots 30421 and 30424 both have a 50 kHz saddle timebase. A
201-sample window spans 4 ms between its first and last samples and has 4.02 ms
of discrete support. Its exact first boxcar null is `50000/201`, approximately
248.756 Hz. These measured numbers are identical for both shots, but they do not
prove that this frequency is a valid locked-mode boundary.

All fourteen locked-mode-specific declarations listed by the gate are absent.
The inherited L2F-12c blockers also remain: rows 9 through 12 contain no finite
samples and the ordered row join, selected vertical set, released geometry,
calibration, baseline, and bad-channel authority are incomplete. The result is
therefore `blocked`; `locked_mode_amp` remains a compatibility candidate and all
training, scientific-validation, facility-prediction, and control-admission
claims remain false.

The byte-reproduced internal report is
`.coordination/evidence/SCPN-CONTROL/l2f12d_mast_locked_mode_authority_2026-07-23T000114Z.json`;
its file SHA-256 is `1ddc292e...9cac2e` and its payload digest is
`54c9fccb...e31f70`.

## Reproduction

Run only against a verified SourceObjectManifest-v2 root:

```bash
python -m validation.mast_locked_mode_authority \
  --manifest /external/source-v4/source_object_manifest.json \
  --artifact-root /external/source-v4 \
  --shot-id 30421 --shot-id 30424 \
  --json-out /external/l2f12d-locked-mode-authority.json
```

The output is deterministic and self-digested. It contains metadata, blockers,
timebase statistics, and digests only; no raw saddle arrays or derived
locked-mode traces are emitted.
