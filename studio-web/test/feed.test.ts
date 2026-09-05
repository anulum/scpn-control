// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Feed.test.

import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  DEFAULT_FEED_URL,
  FALLBACK_FEED,
  isRawFeed,
  loadStudioFeed,
  narrowFeed,
} from '../src/feed.js';

const VALID_FEED = {
  feed_schema: 'studio.control-feed.v1',
  studio: 'scpn-control',
  studio_version: '0.21.0',
  content_digest: `sha256:${'a'.repeat(64)}`,
  verbs: [
    {
      name: 'regulate',
      safety_tier: 'certified',
      side_effect: 'live-hardware',
      timing_class: 'realtime',
      deadline_us: 5,
      domain_distinctive: true,
    },
    {
      name: 'reconstruct',
      safety_tier: 'research',
      side_effect: 'read-only',
      timing_class: 'interactive',
      domain_distinctive: false,
    },
  ],
  claims: [
    {
      schema: 'studio.safety-certificate.v1',
      status: 'reference-validated',
      admission: 'admitted',
      kind: 'formally-proven',
      freshness: 'verified-at-source',
    },
  ],
} as const;

function mockFetch(impl: () => Promise<unknown>): void {
  vi.stubGlobal('fetch', vi.fn(impl));
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('narrowFeed', () => {
  it('maps the snake_case wire feed to camelCase domain types', () => {
    const feed = narrowFeed(VALID_FEED);
    expect(feed.studioVersion).toBe('0.21.0');
    expect(feed.contentDigest).toBe(VALID_FEED.content_digest);
    expect(feed.verbs).toHaveLength(2);
    expect(feed.claims).toHaveLength(1);
  });

  it('carries deadlineUs only for a realtime verb', () => {
    const feed = narrowFeed(VALID_FEED);
    const regulate = feed.verbs.find((v) => v.name === 'regulate');
    const reconstruct = feed.verbs.find((v) => v.name === 'reconstruct');
    expect(regulate?.deadlineUs).toBe(5);
    expect(reconstruct?.deadlineUs).toBeUndefined();
    expect(reconstruct).not.toHaveProperty('deadlineUs');
  });

  it('preserves the claim boundary fields verbatim', () => {
    const [claim] = narrowFeed(VALID_FEED).claims;
    expect(claim).toEqual({
      schema: 'studio.safety-certificate.v1',
      status: 'reference-validated',
      admission: 'admitted',
      kind: 'formally-proven',
      freshness: 'verified-at-source',
    });
  });

  it('keeps legacy claims without a freshness field sparse', () => {
    const feed = narrowFeed({
      ...VALID_FEED,
      claims: [
        {
          schema: 'studio.legacy-claim.v1',
          status: 'bounded-model',
          admission: 'rejected',
          kind: 'measured',
        },
      ],
    });
    expect(feed.claims[0]).not.toHaveProperty('freshness');
  });
});

describe('isRawFeed', () => {
  it.each([
    {},
    [],
    { verbs: [], claims: [] },
    { ...VALID_FEED, feed_schema: 'studio.control-feed.v2' },
    { ...VALID_FEED, studio: 'different-studio' },
    ...[null, 2, '', ' '].map((studio_version) => ({ ...VALID_FEED, studio_version })),
    ...[null, 2, '', 'sha256:abc', `sha256:${'A'.repeat(64)}`].map((content_digest) => ({
      ...VALID_FEED,
      content_digest,
    })),
    { ...VALID_FEED, verbs: null },
    { ...VALID_FEED, claims: null },
    { ...VALID_FEED, verbs: new Array<unknown>(1) },
    { ...VALID_FEED, claims: new Array<unknown>(1) },
    ...[
      null,
      [],
      4,
      {},
      { ...VALID_FEED.verbs[0], name: '' },
      { ...VALID_FEED.verbs[0], safety_tier: 'unknown' },
      { ...VALID_FEED.verbs[0], side_effect: 1 },
      { ...VALID_FEED.verbs[0], timing_class: 'unknown' },
      { ...VALID_FEED.verbs[0], domain_distinctive: 'false' },
      ...[undefined, null, '5', 0, -1, Infinity, NaN].map((deadline_us) => ({
        ...VALID_FEED.verbs[0],
        deadline_us,
      })),
      { ...VALID_FEED.verbs[1], deadline_us: 5 },
    ].map((verb) => ({ ...VALID_FEED, verbs: [verb] })),
    ...[
      null,
      [],
      4,
      {},
      { ...VALID_FEED.claims[0], schema: '' },
      { ...VALID_FEED.claims[0], status: 'unknown' },
      { ...VALID_FEED.claims[0], admission: 'unknown' },
      { ...VALID_FEED.claims[0], kind: 'unknown' },
      ...[undefined, null, 'unknown'].map((freshness) => ({ ...VALID_FEED.claims[0], freshness })),
      {
        schema: 'claim.v1',
        status: 'reference-validated',
        admission: 'admitted',
        kind: 'measured',
      },
    ].map((claim) => ({ ...VALID_FEED, claims: [claim] })),
  ])('rejects malformed identity or rendering fields: %j', (payload) => {
    expect(isRawFeed(payload)).toBe(false);
    expect(() => narrowFeed(payload)).toThrow(TypeError);
  });

  it('preserves stale or rejected claims without upgrading their evidence', () => {
    for (const admission of ['admitted', 'rejected'] as const) {
      const raw = {
        ...VALID_FEED,
        claims: [{ ...VALID_FEED.claims[0], admission, freshness: 'untraceable' }],
      };
      expect(isRawFeed(raw)).toBe(true);
      expect(narrowFeed(raw).claims[0]?.freshness).toBe('untraceable');
    }
  });

  it('accepts a well-formed feed', () => {
    expect(isRawFeed(VALID_FEED)).toBe(true);
  });

  it('rejects non-objects, null, and missing collections', () => {
    expect(isRawFeed(42)).toBe(false);
    expect(isRawFeed(null)).toBe(false);
    expect(isRawFeed({ verbs: 'nope', claims: [] })).toBe(false);
    expect(isRawFeed({ verbs: [], claims: 'nope' })).toBe(false);
  });
});

describe('loadStudioFeed', () => {
  it('fetches and narrows the live feed from the default url', async () => {
    mockFetch(() => Promise.resolve({ ok: true, json: () => Promise.resolve(VALID_FEED) }));
    const feed = await loadStudioFeed();
    expect(globalThis.fetch).toHaveBeenCalledWith(DEFAULT_FEED_URL);
    expect(feed.studioVersion).toBe('0.21.0');
    expect(feed.verbs).toHaveLength(2);
  });

  it('falls back to the bundled sample when the response is not OK', async () => {
    mockFetch(() => Promise.resolve({ ok: false, json: () => Promise.resolve(VALID_FEED) }));
    expect(await loadStudioFeed('/missing.json')).toBe(FALLBACK_FEED);
  });

  it('falls back when the payload is malformed', async () => {
    mockFetch(() => Promise.resolve({ ok: true, json: () => Promise.resolve({ bogus: true }) }));
    expect(await loadStudioFeed()).toBe(FALLBACK_FEED);
  });

  it('falls back when the fetch rejects', async () => {
    mockFetch(() => Promise.reject(new Error('offline')));
    expect(await loadStudioFeed()).toBe(FALLBACK_FEED);
  });
});
