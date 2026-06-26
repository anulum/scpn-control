// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Control Studio UI remote — tests for the live studio feed loader

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
  content_digest: 'sha256:abc',
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
    expect(feed.contentDigest).toBe('sha256:abc');
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
});

describe('isRawFeed', () => {
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
