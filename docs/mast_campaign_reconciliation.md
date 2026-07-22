# MAST Campaign Reconciliation

`validation/reconcile_mast_campaign_lineage.py` is the fail-closed admission
gate for the legacy FAIR-MAST disruption campaign. It verifies what is
preserved, reconstructs exact shot-set relationships, and records why the
campaign is not yet admissible for training or scientific claims.

The gate does not copy or mutate source data. It reads a campaign in place and
writes one deterministic JSON report outside the campaign root.

## Evidence chain

The reconciliation spec uses schema
`scpn-control.mast-campaign-reconciliation-spec.v1.0.0`. It binds exactly six
surfaces by relative path and whole-file SHA-256:

1. the legacy material manifest,
2. the replay report,
3. the replay channel archive,
4. the supervised-dataset manifest,
5. the supervised-dataset report, and
6. the historical evaluation report.

Paths must be unique, relative, POSIX-style, and confined beneath the campaign
root. JSON is read as bytes once, duplicate keys are rejected, and the parsed
snapshot must match the spec digest. The material, replay, and dataset reports
must also pass their embedded canonical payload digests. Every material and
dataset NPZ is verified through the existing production manifest readers. The
replay archive is loaded from the already-hashed byte snapshot: its positive,
unique integer shot IDs must equal the replay-derived set, and every shot must
contain exactly the eleven finite, one-dimensional, equal-length floating-point
channel vectors.

This establishes the integrity of the currently preserved byte surfaces. It
does not retroactively prove producer-time lineage that an older producer did
not record.

## Current bounded result

The read-only campaign reconciliation produced this exact inventory:

| Stage | Count | Meaning |
| --- | ---: | --- |
| Requested material shots | 120 | Legacy campaign request set |
| Acquired and checksum-verified | 95 | Preserved material NPZ files |
| Acquisition failures | 25 | 10 missing saddle, 11 missing interferometer, 4 missing equilibrium |
| Replay-derived shots | 93 | Eleven-channel replay output set |
| Replay channel members | 1,023 | 93 shots × 11 exact channel names; 52,141 aligned samples |
| Supervised-dataset artifacts | 93 | 65 positive and 28 negative proxy labels |
| Historical evaluation shots | 93 | Same shot set, retained as historical-only |

Shots `19463` and `21073` are the exact acquired-but-not-derived exclusions.
Both have a reason in the verified replay report: missing
`summary.line_average_n_e`.

Set equality is exact for replay, dataset, and historical evaluation. That
agreement is useful inventory evidence, but it is not a derivation proof. The
dataset artifacts do not carry per-shot source-parent or transform digests.
The historical evaluation report also has an invalid embedded self-digest.

## Licence handling

The authoritative FAIR-MAST source policy is `CC-BY-SA-4.0` with its policy URL
and citations. The legacy material, dataset, and evaluation files declare
`MIT`. Reconciliation replaces the material declaration only in the verified
in-memory v1-to-v2 migration and records the discrepancy. It never rewrites the
legacy files and never treats the persisted dataset licence as corrected.

## Claim boundary

The report schema is
`scpn-control.mast-campaign-lineage-reconciliation.v1.0.0`. Its status remains
`blocked`, and all of these fields are always false:

- `cohort_admission`
- `control_admission`
- `facility_prediction`
- `reuse_admissible`
- `scientific_validation`
- `training_admission`

The current blockers are missing dataset parent/transform digests, missing
native-Zarr/source-generation preservation, the replay archive not being
producer-digest-bound, three persisted legacy licence declarations, and the
invalid historical evaluation self-digest.

## Running the gate

Create a self-digested spec containing the six relative paths and their current
file SHA-256 values, then run:

```bash
PYTHONPATH=src python -m validation.reconcile_mast_campaign_lineage \
  --campaign-root /path/to/read-only/campaign01 \
  --spec /path/to/reconciliation-spec.json \
  --generated-at 2026-07-22T17:45:21Z \
  --json-out /path/to/reconciliation-report.json
```

Use an explicit timestamp when byte-for-byte reproducibility is required. The
report includes the spec file and payload digests, all six input bindings,
recomputed counters and set differences, licence actions, blockers, and its own
canonical payload SHA-256.

Any malformed schema, duplicate identifier, checksum drift, unsafe path,
counter mismatch, dataset identity mismatch, unsupported status, or claim
promotion causes a domain error. Cross-stage set mismatches remain reportable
blockers, while corrupt or ambiguous input structure fails before a report is
written.

## What closes the gate

Reuse requires a newly generated, immutable dataset manifest that binds each
derived shot to its verified source-parent digest, transform-spec digest, and
explicit exclusion ledger. The replay producer must bind its channel archive,
the corrected licence policy must be persisted without mutating historical
evidence, and evaluation must be regenerated from the sealed admitted cohort.
Independent outcome authority is still required; the current Ip-derived labels
remain capacity-planning proxies only.
