// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Feed.

/**
 * Load the CONTROL studio feed the Python vertical emits. Transport success does
 * not establish live measurements or evidence freshness.
 *
 * The wire feed (`scpn_control.studio.feed`, schema `studio.control-feed.v1`) is
 * snake_case; this module narrows it to the panel's camelCase domain types at the
 * boundary. When the feed is unreachable or malformed the loader falls back to the
 * bundled domain sample so the standalone remote always renders — the fallback is
 * the same honesty-graded data, never a fabricated "all validated" view.
 */

import type {
  AdmissionDecision,
  ClaimStatus,
  ClaimSummary,
  ControlVerb,
  EvidenceKind,
  Freshness,
  SafetyTier,
  SideEffect,
  TimingClass,
} from './domain.js';
import { CONTROL_CLAIMS, CONTROL_VERBS } from './domain.js';

/** A verb as it appears on the wire (snake_case, from the Python feed). */
export interface RawVerb {
  /** Stable manifest verb name. */
  readonly name: string;
  /** Safety tier enforced by the Hub. */
  readonly safety_tier: SafetyTier;
  /** Declared side-effect class. */
  readonly side_effect: SideEffect;
  /** Declared execution timing class. */
  readonly timing_class: TimingClass;
  /** Optional hard execution deadline in microseconds. */
  readonly deadline_us?: number;
  /** Whether the verb is distinctive to the CONTROL domain. */
  readonly domain_distinctive: boolean;
}

/** A claim as it appears on the wire (snake_case, from the Python feed). */
export interface RawClaim {
  /** Evidence schema identifier. */
  readonly schema: string;
  /** Scientific claim-boundary status. */
  readonly status: ClaimStatus;
  /** Runtime admission decision. */
  readonly admission: AdmissionDecision;
  /** Evidence modality. */
  readonly kind: EvidenceKind;
  /** Optional evidence-freshness classification. */
  readonly freshness?: Freshness;
}

/** The studio feed document as it appears on the wire. */
export interface RawFeed {
  /** Version of the wire-feed schema. */
  readonly feed_schema: string;
  /** Stable studio identifier. */
  readonly studio: string;
  /** Version of the producing Studio package. */
  readonly studio_version: string;
  /** Producer manifest digest, not a digest of the feed bytes. */
  readonly content_digest: string;
  /** Advertised CONTROL verbs. */
  readonly verbs: readonly RawVerb[];
  /** Emitted honesty-graded claims. */
  readonly claims: readonly RawClaim[];
}

/** The narrowed feed the panel consumes. */
export interface StudioFeed {
  /** Transport origin only; never proof of live measurements. */
  readonly source: 'sample' | 'provided' | 'fetched';
  /** Local UTC receipt time of the latest successful load, not measurement time. */
  readonly receivedAt?: string;
  /** Version of the producing Studio package. */
  readonly studioVersion: string;
  /** Producer manifest digest, not a digest of the feed bytes. */
  readonly contentDigest: string;
  /** Advertised CONTROL verbs. */
  readonly verbs: readonly ControlVerb[];
  /** Emitted honesty-graded claims. */
  readonly claims: readonly ClaimSummary[];
}

/** The bundled domain sample, used when the requested feed is unavailable or invalid. */
export const FALLBACK_FEED: StudioFeed = {
  source: 'sample',
  studioVersion: 'fallback',
  contentDigest: 'fallback',
  verbs: CONTROL_VERBS,
  claims: CONTROL_CLAIMS,
};

/** Default location the standalone remote fetches the live feed from. */
export const DEFAULT_FEED_URL = './studio-feed.json';

function toVerb(raw: RawVerb): ControlVerb {
  const base = {
    name: raw.name,
    safetyTier: raw.safety_tier,
    sideEffect: raw.side_effect,
    timingClass: raw.timing_class,
    domainDistinctive: raw.domain_distinctive,
  };
  // exactOptionalPropertyTypes: only carry deadlineUs when the verb declares one.
  return raw.deadline_us === undefined ? base : { ...base, deadlineUs: raw.deadline_us };
}

function toClaim(raw: RawClaim): ClaimSummary {
  const base = {
    schema: raw.schema,
    status: raw.status,
    admission: raw.admission,
    kind: raw.kind,
  };
  return raw.freshness === undefined ? base : { ...base, freshness: raw.freshness };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isText(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0;
}

function member(value: unknown, choices: readonly string[]): boolean {
  return typeof value === 'string' && choices.includes(value);
}

function isVerb(value: unknown): boolean {
  if (!isRecord(value)) return false;
  return (
    isText(value.name) &&
    member(value.safety_tier, ['research', 'certified', 'production']) &&
    member(value.side_effect, ['read-only', 'simulated', 'live-hardware']) &&
    member(value.timing_class, ['batch', 'interactive', 'realtime']) &&
    typeof value.domain_distinctive === 'boolean' &&
    (value.timing_class === 'realtime'
      ? typeof value.deadline_us === 'number' &&
        Number.isFinite(value.deadline_us) &&
        value.deadline_us > 0
      : !Object.hasOwn(value, 'deadline_us'))
  );
}

function isClaim(value: unknown): boolean {
  if (!isRecord(value)) return false;
  return (
    isText(value.schema) &&
    member(value.status, [
      'reference-validated',
      'bounded-model',
      'bounded-support',
      'validation-gap',
      'external-dependency-blocked',
      'roadmap',
      'toolchain-gated',
      'refuted',
    ]) &&
    member(value.admission, ['admitted', 'rejected']) &&
    member(value.kind, [
      'measured',
      'curated',
      'formally-proven',
      'falsified',
      'noise-limited',
      'hardware-validated',
      'producer-asserted',
    ]) &&
    (!Object.hasOwn(value, 'freshness') ||
      member(value.freshness, ['verified-at-source', 'traceable-unchecked', 'untraceable'])) &&
    !(
      value.status === 'reference-validated' &&
      value.admission === 'admitted' &&
      !Object.hasOwn(value, 'freshness')
    )
  );
}

/**
 * Validate CONTROL wire identity and every field consumed by the panel.
 *
 * Additional fields are ignored for forward compatibility. The digest is a
 * manifest identifier: its format is checked, not a signature or a hash of
 * this feed. Legacy claims without freshness cannot assert admitted validation.
 */
export function isRawFeed(value: unknown): value is RawFeed {
  return (
    isRecord(value) &&
    value.feed_schema === 'studio.control-feed.v1' &&
    value.studio === 'scpn-control' &&
    isText(value.studio_version) &&
    typeof value.content_digest === 'string' &&
    /^sha256:[0-9a-f]{64}$/.test(value.content_digest) &&
    Array.isArray(value.verbs) &&
    Array.from(value.verbs).every(isVerb) &&
    Array.isArray(value.claims) &&
    Array.from(value.claims).every(isClaim)
  );
}

/**
 * Validate and map a wire feed to the panel's camelCase domain types.
 * @throws TypeError if identity, item structure or rendering fields are invalid.
 */
export function narrowFeed(raw: unknown): StudioFeed {
  if (!isRawFeed(raw)) throw new TypeError('Invalid CONTROL studio feed');
  return {
    source: 'provided',
    studioVersion: raw.studio_version,
    contentDigest: raw.content_digest,
    verbs: raw.verbs.map(toVerb),
    claims: raw.claims.map(toClaim),
  };
}

/**
 * Fetch and narrow the studio feed, falling back to the bundled sample.
 *
 * @param url - where to fetch the feed from (defaults to {@link DEFAULT_FEED_URL}).
 * @param timeoutMs - total fetch/body deadline, 1–60000 ms; defaults to 5000.
 * @throws RangeError for an invalid deadline.
 * @returns the narrowed fetched feed, or {@link FALLBACK_FEED} when it is unreachable
 *   (non-OK response, network error) or malformed.
 */
export async function loadStudioFeed(
  url: string = DEFAULT_FEED_URL,
  timeoutMs = 5000,
): Promise<StudioFeed> {
  if (!Number.isInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 60000) {
    throw new RangeError('Feed timeout must be an integer from 1 to 60000 ms');
  }
  const controller = new AbortController();
  const timer = setTimeout(() => {
    controller.abort();
  }, timeoutMs);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) {
      return FALLBACK_FEED;
    }
    const payload: unknown = await response.json();
    return { ...narrowFeed(payload), source: 'fetched', receivedAt: new Date().toISOString() };
  } catch {
    return FALLBACK_FEED;
  } finally {
    clearTimeout(timer);
  }
}
