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

## Immutable post-hoc dataset lineage (L2F-90b)

`validation/build_mast_dataset_lineage_manifest.py` consumes the sealed
reconciliation report and recomputes it from the pinned campaign before
emitting any lineage record. Its output schema is
`scpn-control.mast-dataset-lineage-manifest.v1.0.0`.

For every included shot, the manifest binds:

- the exact legacy source NPZ SHA-256,
- the selected-array parent digest reconstructed by SourceObjectManifest v2,
- the source acquisition-transform digest,
- the whole replay archive SHA-256 and a canonical digest of that shot's eleven
  replay channel vectors,
- the exact labelled-dataset NPZ SHA-256, and
- the self-digest of the bounded transform specification.

The exclusion ledger binds every acquired-but-not-derived shot to the same
source-parent tuple and to its reason in the verified replay report. Included
and excluded shot sets must be disjoint and must exactly partition the acquired
source shots. Paths remain confined to the campaign root, all JSON readers
reject duplicate keys, and the output path must be outside the campaign.

This is deliberately labelled `post_hoc_reconciliation`. It proves that the
listed bytes, reconstructed parents, transform declaration, and set
relationships agree now. It does **not** prove that the historical producer
consumed those exact parents or that transform at producer time. Consequently
`producer_time_lineage`, every per-shot `producer_time_attested` flag, and all
six claim/admission fields are structurally false.

The bounded transform specification lives at
`validation/mast_dataset_transform_spec_v1.json`. It fixes the legacy replay
and dataset schemas, the four-stage transform description, `ip_proxy` label
authority, and false independent-label/training admission. Changing any of
those statements invalidates its self-digest or fails validation.

## Producer-bound replay archives (L2F-90c(a))

`validation/build_disruption_replay_channels.py` emits new replay runs with
schema `scpn-control.mast-disruption-replay-channels.v2.0.0`. Before publishing
`channels.npz`, the producer reopens the persisted temporary archive with
pickle disabled and verifies all of these invariants:

- `shot_ids` is a sorted, unique, positive one-dimensional integer vector and
  exactly matches the successfully derived producer inventory,
- the archive contains only `shot_ids` plus the eleven declared channel members
  for every shot,
- every channel is a non-empty, finite, one-dimensional floating-point vector,
  and all eleven vectors for one shot have the same sample count,
- every channel receives a canonical value digest; the ordered channel digests
  are folded into a canonical per-shot digest, and
- the report binds the complete archive by byte length and whole-file SHA-256.

Only a validated temporary archive is hard-linked to the final path, so an
existing destination or concurrent winner fails closed without replacement.
The CLI reserves the report path exclusively, requires the archive and report
to remain outside the immutable material directory, and removes newly created
output after a handled failure if either half cannot be completed. It never
deletes a pre-existing archive or report. An uncatchable process or host
interruption can leave an incomplete fresh destination; the next run refuses
to overwrite it so an operator can quarantine and inspect those bytes first.

The post-hoc lineage builder uses this same archive inspector and digest
algorithm, preventing producer and reconciliation semantics from drifting. The
reconciliation gate accepts report v2 only when the report's exact embedded
binding equals the already pinned archive byte snapshot. Any difference in the
whole-file digest, byte length, shot count, member-digest kind, per-shot digest,
path, or binding shape fails before a reconciliation report is produced.
Historical report-v1 and campaign bytes remain immutable: v2 governs newly
generated replay products and does not retroactively convert the preserved
93-shot campaign into producer-time evidence. It therefore closes the replay
producer implementation gap for future regeneration, not the campaign's
training, scientific, facility, reuse, or control-admission blockers.
The preserved-campaign v1 path remains supported and produces the same
deterministic report byte surface as before this extension. Its
`channels_archive_producer_digest_bound` field remains false and its blocker is
retained. An exactly verified v2 report sets that field true and removes only
`replay_archive_not_digest_bound_by_its_producer_report`; every unrelated
admission and claim boundary remains false.

Create a fresh producer-bound replay output with:

```bash
PYTHONPATH=src python -m validation.build_disruption_replay_channels \
  --material-dir /path/to/read-only/material \
  --out-dir /path/to/fresh/replay-v2 \
  --json-out /path/to/fresh/replay-v2/report.json \
  --generated-at 2026-07-22T22:45:00Z
```

Both output paths must be fresh. The source material is read only; this command
must not target the preserved historical campaign output directory.

## Producer-time labelled-dataset lineage (L2F-90c(c))

`validation/build_mast_lineage_bound_dataset.py` is the fresh-regeneration
boundary between replay-v2 evidence and the proxy-labelled dataset. It requires
four already sealed inputs:

1. a source-object manifest v2 whose referenced local artifacts validate,
2. a replay report v2,
3. the exact replay archive bound by that report, and
4. `validation/mast_dataset_producer_transform_spec_v1.json`.

The producer reads the replay archive into one immutable byte snapshot, verifies
the report's complete archive binding, and requires acquired source shots to
equal the exact union of replay-derived and replay-failed shots. It then creates
a previously absent output directory, builds only the derived shots, and seals
`producer-lineage.json`. Every included shot binds the selected-array parent,
source-cache checksum, acquisition transform, optional native source-generation
pin, replay-member digest, bounded transform digest, proxy-label record digest,
and final dataset-artifact checksum. Every replay failure becomes a source-bound
exclusion with no replay member or dataset artifact.

The output manifest uses
`scpn-control.mast-dataset-producer-lineage.v1.0.0` and has its own fail-closed
validator. Output-directory reuse, placement under an immutable input directory,
duplicate JSON keys, input self-digest drift, inventory mismatch, binding drift,
attestation drift, count drift, blocker removal, or claim promotion is rejected.
Handled failures remove only the newly created output directory. Fixed inputs
and timestamps reproduce every generated byte.

Run a fresh build with:

```bash
PYTHONPATH=src python -m validation.build_mast_lineage_bound_dataset \
  --source-manifest /path/to/source-v2/source-manifest.json \
  --replay-report /path/to/replay-v2/replay-report.json \
  --replay-archive /path/to/replay-v2/channels.npz \
  --transform-spec validation/mast_dataset_producer_transform_spec_v1.json \
  --dataset-id mast-disruption-regenerated-v1 \
  --out-dir /path/to/fresh/dataset-v2 \
  --retrieved-at 2026-07-22T20:00:00Z \
  --generated-at 2026-07-22T21:30:00Z
```

`producer_time_lineage:true` means that this producer observed and bound those
exact inputs and outputs in one run. It does not make the `ip_proxy` labels
independent ground truth. The manifest remains `status:blocked`; all training,
scientific, reuse, facility-prediction, cohort, and control claims remain false.
If any source artifact lacks a native-generation pin, that limitation is also an
explicit blocker.

## Fresh regeneration verification (L2F-90d)

`validation/verify_mast_lineage_bound_regeneration.py` turns two fresh dataset
roots into one fail-closed reproducibility proof. It revalidates the exact
SourceObjectManifest-v2 artifacts, replay-v2 report and archive, and both
`producer-lineage.json` manifests with their complete output trees. It samples
each run tree before and after validation, rejects symlinks and non-regular
entries, and requires the source, replay, archive, dataset identity, generation
time, blockers, claims, filenames, byte lengths, and SHA-256 values to agree.

Run the verifier only after creating two previously absent output roots with
identical explicit timestamps:

```bash
PYTHONPATH=src python -m validation.verify_mast_lineage_bound_regeneration \
  --source-manifest /path/to/source-v2/source_object_manifest.json \
  --replay-report /path/to/replay-v2/report.json \
  --replay-archive /path/to/replay-v2/channels.npz \
  --run-a /path/to/fresh/dataset-run-a \
  --run-b /path/to/fresh/dataset-run-b \
  --generated-at 2026-07-22T21:50:00Z \
  --json-out /path/to/fresh/regeneration-verification.json
```

The verifier emits
`scpn-control.mast-lineage-bound-regeneration-verification.v1.0.0` with status
`reproducible_blocked`. It accepts only producer manifests that pin native
source generation for every record. Byte reproducibility and source-generation
identity close the corresponding lineage gaps, but do not convert the
input-derived `ip_proxy` label into independent outcome authority. The sealed
cohort, training, scientific, reuse, facility-prediction, and control gates
therefore remain false.

The bounded real-object L2F-90d run used FAIR-MAST shots 30421 and 30424. The
verified source snapshot contains 14,695,227 bytes and exact Zarr-v3 generation
pins for both shots. Fresh replay-v2 derived both shots, and the two dataset
roots reproduced all five output files byte-for-byte. The verification payload
digest is `76543760016d8622cdb607066258cdb065341cc08b72a07a46e3ec556732985b`;
its complete output-inventory digest is
`d9f56fc1af099be9549374dc4b8f24ffaf6df1a8f88e17ebc4aed408aaa758b8`.
All seven claim fields remain false.

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

After the reconciliation report exists, build the immutable post-hoc lineage
manifest with:

```bash
PYTHONPATH=src python -m validation.build_mast_dataset_lineage_manifest \
  --campaign-root /path/to/read-only/campaign01 \
  --spec /path/to/reconciliation-spec.json \
  --reconciliation-report /path/to/reconciliation-report.json \
  --transform-spec validation/mast_dataset_transform_spec_v1.json \
  --generated-at 2026-07-22T20:15:00Z \
  --json-out /path/outside/campaign/dataset-lineage.json
```

Use the same explicit timestamp to reproduce identical output bytes. The
builder never rewrites the historical material, replay, dataset, manifest, or
evaluation files.

## What closes the gate

L2F-90b supplies an immutable post-hoc manifest with source-parent,
transform-spec, replay-archive, dataset-artifact, and exclusion bindings. Reuse
still requires the dataset to be regenerated by a lineage-aware producer so
those relationships are attested at production time. The replay producer must
bind its channel archive, native source generation must be pinned, and
evaluation must be regenerated from a sealed admitted cohort. Independent
outcome authority is still required; the current Ip-derived labels remain
capacity-planning proxies only.
