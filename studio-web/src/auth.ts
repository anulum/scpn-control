// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Control Studio UI remote — portal session loader

/** Raw account payload returned by SCPN-STUDIO's portal backend. */
export interface RawPortalAccount {
  /** Stable backend user identifier. */
  readonly user_id: string;
  /** Account email address. */
  readonly email: string;
  /** User-facing account name. */
  readonly display_name: string;
  /** Whether the backend has verified the email address. */
  readonly email_verified: boolean;
  /** Portal entitlement tier. */
  readonly tier: string;
  /** Whether multi-factor authentication is enabled. */
  readonly mfa_enabled: boolean;
}

/** The authenticated account narrowed to the panel's camelCase domain. */
export interface PortalAccount {
  /** Stable backend user identifier. */
  readonly userId: string;
  /** Account email address. */
  readonly email: string;
  /** User-facing account name. */
  readonly displayName: string;
  /** Whether the backend has verified the email address. */
  readonly emailVerified: boolean;
  /** Portal entitlement tier. */
  readonly tier: string;
  /** Whether multi-factor authentication is enabled. */
  readonly mfaEnabled: boolean;
}

/** Portal session state consumed by the CONTROL panel. */
export type PortalAuthState =
  | {
      /** Authenticated-session discriminator. */
      readonly status: 'authenticated';
      /** Narrowed authenticated portal account. */
      readonly account: PortalAccount;
    }
  | {
      /** Anonymous-session discriminator. */
      readonly status: 'anonymous';
    }
  | {
      /** Unavailable-backend discriminator. */
      readonly status: 'unavailable';
    };

/** Default same-origin portal endpoint for session rehydration. */
export const DEFAULT_AUTH_URL = '/api/v1/auth/me';

/** Bundled fallback state for standalone renders without the portal backend. */
export const FALLBACK_AUTH: PortalAuthState = { status: 'unavailable' };

function isStringRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

/** Structural type guard for the portal account payload. */
export function isRawPortalAccount(value: unknown): value is RawPortalAccount {
  if (!isStringRecord(value)) {
    return false;
  }
  return (
    typeof value.user_id === 'string' &&
    typeof value.email === 'string' &&
    typeof value.display_name === 'string' &&
    typeof value.email_verified === 'boolean' &&
    typeof value.tier === 'string' &&
    typeof value.mfa_enabled === 'boolean'
  );
}

/** Narrow the portal account payload to the panel's account type. */
export function narrowPortalAccount(raw: RawPortalAccount): PortalAccount {
  return {
    userId: raw.user_id,
    email: raw.email,
    displayName: raw.display_name,
    emailVerified: raw.email_verified,
    tier: raw.tier,
    mfaEnabled: raw.mfa_enabled,
  };
}

/**
 * Fetch the portal session using the shared same-origin httpOnly cookie.
 *
 * @param url - portal account endpoint, defaults to {@link DEFAULT_AUTH_URL}.
 * @returns authenticated account state, anonymous state for 401/403, or
 *   unavailable when the endpoint is missing, unreachable, or malformed.
 */
export async function loadPortalAuth(url: string = DEFAULT_AUTH_URL): Promise<PortalAuthState> {
  try {
    const response = await fetch(url, { credentials: 'include' });
    if (response.status === 401 || response.status === 403) {
      return { status: 'anonymous' };
    }
    if (!response.ok) {
      return FALLBACK_AUTH;
    }
    const payload: unknown = await response.json();
    return isRawPortalAccount(payload)
      ? { status: 'authenticated', account: narrowPortalAccount(payload) }
      : FALLBACK_AUTH;
  } catch {
    return FALLBACK_AUTH;
  }
}
