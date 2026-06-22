// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Control Studio UI remote — the studio's domain data and honesty rendering rules

/**
 * The CONTROL studio's domain data for the panel.
 *
 * This mirrors the Python vertical (`scpn_control.studio`): the verb taxonomy with
 * its gating attributes, and the claim-boundary honesty rule (only a
 * reference-validated claim renders as validated; every other boundary is shown
 * verbatim). In production the panel reads this from the studio backend; the sample
 * lets the remote render and be tested standalone.
 */

/** Safety tier the Hub gates actuation against. */
export type SafetyTier = 'research' | 'certified' | 'production';

/** Side-effect class — the per-tenant actuation gate. */
export type SideEffect = 'read-only' | 'simulated' | 'live-hardware';

/** Timing class of a verb. */
export type TimingClass = 'batch' | 'interactive' | 'realtime';

/** The seven-state claim-boundary lattice (mirrors the platform contract). */
export type ClaimStatus =
  | 'reference-validated'
  | 'bounded-model'
  | 'bounded-support'
  | 'validation-gap'
  | 'external-dependency-blocked'
  | 'roadmap'
  | 'toolchain-gated';

/** The orthogonal evidence modality. */
export type EvidenceKind = 'measured' | 'curated' | 'formally-proven';

/** The runtime admission decision for a claim. */
export type AdmissionDecision = 'admitted' | 'rejected';

/** A CONTROL verb with the attribute contract the Hub federates against. */
export interface ControlVerb {
  readonly name: string;
  readonly safetyTier: SafetyTier;
  readonly sideEffect: SideEffect;
  readonly timingClass: TimingClass;
  readonly deadlineUs?: number;
  readonly domainDistinctive: boolean;
}

/** A claim summary the panel renders, with its boundary and modality. */
export interface ClaimSummary {
  readonly schema: string;
  readonly status: ClaimStatus;
  readonly admission: AdmissionDecision;
  readonly kind: EvidenceKind;
}

/** Whether the Hub must hard-gate this verb per tenant before running it. */
export function requiresLiveHardwareGate(verb: ControlVerb): boolean {
  return verb.sideEffect === 'live-hardware';
}

/**
 * Whether a claim may be presented as validated.
 *
 * Mirrors the platform `ClaimBoundary.is_admissible` exactly: the status must be
 * `reference-validated` AND the runtime admission must be `admitted`. Checking the
 * status alone would wrongly render a `reference-validated` but `rejected` claim as
 * validated, so both axes are required — keeping the TS rendering in lock-step with
 * the Python contract.
 */
export function claimRendersAsValidated(
  status: ClaimStatus,
  admission: AdmissionDecision,
): boolean {
  return status === 'reference-validated' && admission === 'admitted';
}

/** Every verb the CONTROL studio advertises, in manifest order. */
export const CONTROL_VERBS: readonly ControlVerb[] = [
  {
    name: 'reconstruct',
    safetyTier: 'research',
    sideEffect: 'read-only',
    timingClass: 'interactive',
    domainDistinctive: false,
  },
  {
    name: 'simulate',
    safetyTier: 'research',
    sideEffect: 'simulated',
    timingClass: 'batch',
    domainDistinctive: false,
  },
  {
    name: 'analyse',
    safetyTier: 'research',
    sideEffect: 'read-only',
    timingClass: 'batch',
    domainDistinctive: false,
  },
  {
    name: 'validate',
    safetyTier: 'research',
    sideEffect: 'read-only',
    timingClass: 'batch',
    domainDistinctive: false,
  },
  {
    name: 'benchmark',
    safetyTier: 'research',
    sideEffect: 'simulated',
    timingClass: 'batch',
    domainDistinctive: false,
  },
  {
    name: 'replay',
    safetyTier: 'research',
    sideEffect: 'read-only',
    timingClass: 'batch',
    domainDistinctive: false,
  },
  {
    name: 'regulate',
    safetyTier: 'certified',
    sideEffect: 'live-hardware',
    timingClass: 'realtime',
    deadlineUs: 5,
    domainDistinctive: true,
  },
  {
    name: 'certify',
    safetyTier: 'certified',
    sideEffect: 'read-only',
    timingClass: 'batch',
    domainDistinctive: true,
  },
  {
    name: 'predict',
    safetyTier: 'research',
    sideEffect: 'read-only',
    timingClass: 'interactive',
    domainDistinctive: true,
  },
  {
    name: 'mitigate',
    safetyTier: 'certified',
    sideEffect: 'live-hardware',
    timingClass: 'interactive',
    domainDistinctive: true,
  },
  {
    name: 'monitor',
    safetyTier: 'research',
    sideEffect: 'read-only',
    timingClass: 'interactive',
    domainDistinctive: true,
  },
];

/**
 * A representative slice of CONTROL's emitted claims, one per honesty axis: a
 * bounded measured reconstruction, a formally-proven safety certificate that holds,
 * and a bounded measured latency benchmark. Mirrors the Python evidence mappers.
 */
export const CONTROL_CLAIMS: readonly ClaimSummary[] = [
  {
    schema: 'studio.efit-reconstruction.v1',
    status: 'bounded-model',
    admission: 'rejected',
    kind: 'measured',
  },
  {
    schema: 'studio.safety-certificate.v1',
    status: 'reference-validated',
    admission: 'admitted',
    kind: 'formally-proven',
  },
  {
    schema: 'studio.controller-latency.v1',
    status: 'bounded-model',
    admission: 'rejected',
    kind: 'measured',
  },
];
