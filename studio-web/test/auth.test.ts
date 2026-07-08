// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Control Studio UI remote — tests for the portal session loader

import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  DEFAULT_AUTH_URL,
  FALLBACK_AUTH,
  isRawPortalAccount,
  loadPortalAuth,
  narrowPortalAccount,
} from '../src/auth.js';

const RAW_ACCOUNT = {
  user_id: 'user-1',
  email: 'operator@example.invalid',
  display_name: 'Operator',
  email_verified: true,
  tier: 'pro',
  mfa_enabled: true,
} as const;

function mockFetch(impl: () => Promise<unknown>): void {
  vi.stubGlobal('fetch', vi.fn(impl));
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('isRawPortalAccount', () => {
  it('accepts the portal account shape returned by /api/v1/auth/me', () => {
    expect(isRawPortalAccount(RAW_ACCOUNT)).toBe(true);
  });

  it('rejects non-objects and malformed account fields', () => {
    expect(isRawPortalAccount(null)).toBe(false);
    expect(isRawPortalAccount(42)).toBe(false);
    expect(isRawPortalAccount({ ...RAW_ACCOUNT, user_id: 123 })).toBe(false);
    expect(isRawPortalAccount({ ...RAW_ACCOUNT, email: 123 })).toBe(false);
    expect(isRawPortalAccount({ ...RAW_ACCOUNT, display_name: 123 })).toBe(false);
    expect(isRawPortalAccount({ ...RAW_ACCOUNT, email_verified: 'yes' })).toBe(false);
    expect(isRawPortalAccount({ ...RAW_ACCOUNT, tier: 123 })).toBe(false);
    expect(isRawPortalAccount({ ...RAW_ACCOUNT, mfa_enabled: 'yes' })).toBe(false);
  });
});

describe('narrowPortalAccount', () => {
  it('maps the snake_case portal account into camelCase panel state', () => {
    expect(narrowPortalAccount(RAW_ACCOUNT)).toEqual({
      userId: 'user-1',
      email: 'operator@example.invalid',
      displayName: 'Operator',
      emailVerified: true,
      tier: 'pro',
      mfaEnabled: true,
    });
  });
});

describe('loadPortalAuth', () => {
  it('fetches the current portal session with same-origin cookies', async () => {
    mockFetch(() =>
      Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(RAW_ACCOUNT) }),
    );

    const auth = await loadPortalAuth();

    expect(globalThis.fetch).toHaveBeenCalledWith(DEFAULT_AUTH_URL, { credentials: 'include' });
    expect(auth).toEqual({
      status: 'authenticated',
      account: narrowPortalAccount(RAW_ACCOUNT),
    });
  });

  it('returns anonymous when the portal rejects the session', async () => {
    mockFetch(() => Promise.resolve({ ok: false, status: 401, json: () => Promise.resolve({}) }));
    expect(await loadPortalAuth()).toEqual({ status: 'anonymous' });
  });

  it('returns anonymous for forbidden portal sessions', async () => {
    mockFetch(() => Promise.resolve({ ok: false, status: 403, json: () => Promise.resolve({}) }));
    expect(await loadPortalAuth()).toEqual({ status: 'anonymous' });
  });

  it('falls back when the endpoint returns another error', async () => {
    mockFetch(() => Promise.resolve({ ok: false, status: 503, json: () => Promise.resolve({}) }));
    expect(await loadPortalAuth()).toBe(FALLBACK_AUTH);
  });

  it('falls back when the account payload is malformed', async () => {
    mockFetch(() => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({}) }));
    expect(await loadPortalAuth()).toBe(FALLBACK_AUTH);
  });

  it('falls back when the fetch rejects', async () => {
    mockFetch(() => Promise.reject(new Error('offline')));
    expect(await loadPortalAuth()).toBe(FALLBACK_AUTH);
  });
});
