<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — ADR 0004 validation data and evidence strategy. -->

# ADR-0004: Validation, Data, and Evidence Strategy

## Status

Accepted.

## Context

Users need to know which outputs they can replay, which claims are bounded, and
which claims require external data, external code, target hardware, or facility
approval. Narrative claims are not enough for that decision.

The repository already emits validation reports, benchmark reports, generated
traceability, Studio manifests, and release evidence. These artefacts need one
consistent strategy.

## Decision

Treat validation artefacts as claim-admission inputs. Public claims must point
to executable validators, persisted JSON or Markdown reports, traceability rows,
or generated manifests that describe the evidence boundary.

The evidence strategy is:

- Record source URI, shot or case identifier, units, retrieval time, checksum,
  backend, version, dtype, tolerance, and host context where those fields affect
  the claim.
- Use immutable manifests for real or public data ingestion.
- Keep synthetic, fixture, local-regression, and non-isolated benchmark evidence
  labelled as such.
- Fail closed on missing provenance, stale generated surfaces, checksum drift,
  malformed reports, missing metrics, and external dependency unavailability.
- Regenerate public generated surfaces when tracked source, test, validation, or
  documentation inventories change.

## Alternatives Considered

- **Put validation claims only in prose.** Rejected because prose cannot enforce
  freshness, checksums, tolerances, or admission status.
- **Treat CI success as claim admission.** Rejected because CI can show that
  gates ran, but each claim still needs the specific evidence that covers it.
- **Hide blocked evidence behind roadmap language.** Rejected because reviewers
  and users need to see the blocked state before using the surface.

## Consequences

- Public documentation must distinguish bounded repository evidence from
  facility, external-code, hardware, and review evidence.
- Validators and reports become part of the architecture, not after-the-fact
  attachments.
- Benchmarks need provenance and isolation labels that match their claim use.
- Studio manifests and feeds render claim boundaries rather than silently
  promoting unsealed or unchecked evidence.
- TODO work that changes a public claim must update validators, reports,
  generated surfaces, and documentation in the same cycle.
