// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Control Studio UI remote — tests for the domain data and honesty rules

import { describe, expect, it } from 'vitest';

import {
  claimRendersAsValidated,
  CONTROL_CLAIMS,
  CONTROL_VERBS,
  requiresLiveHardwareGate,
} from '../src/domain.js';

describe('requiresLiveHardwareGate', () => {
  it('gates a live-hardware verb and passes a read-only one', () => {
    const regulate = CONTROL_VERBS.find((v) => v.name === 'regulate');
    const monitor = CONTROL_VERBS.find((v) => v.name === 'monitor');
    expect(regulate && requiresLiveHardwareGate(regulate)).toBe(true);
    expect(monitor && requiresLiveHardwareGate(monitor)).toBe(false);
  });
});

describe('claimRendersAsValidated', () => {
  it('admits only reference-validated that is also admitted', () => {
    expect(claimRendersAsValidated('reference-validated', 'admitted')).toBe(true);
  });

  it('rejects reference-validated when admission is rejected (status alone is not enough)', () => {
    // The parity check: a status-only rule would wrongly render this as validated.
    expect(claimRendersAsValidated('reference-validated', 'rejected')).toBe(false);
  });

  it('floors reference-validated claims when freshness is unchecked', () => {
    expect(claimRendersAsValidated('reference-validated', 'admitted', 'traceable-unchecked')).toBe(
      false,
    );
    expect(claimRendersAsValidated('reference-validated', 'admitted', 'verified-at-source')).toBe(
      true,
    );
  });

  it('renders every other boundary verbatim, not validated', () => {
    for (const status of [
      'bounded-model',
      'bounded-support',
      'validation-gap',
      'external-dependency-blocked',
      'roadmap',
      'toolchain-gated',
    ] as const) {
      expect(claimRendersAsValidated(status, 'admitted')).toBe(false);
    }
  });
});

describe('CONTROL_VERBS', () => {
  it('advertises 11 verbs, 6 core and 5 domain-distinctive', () => {
    expect(CONTROL_VERBS).toHaveLength(11);
    expect(CONTROL_VERBS.filter((v) => !v.domainDistinctive)).toHaveLength(6);
    expect(CONTROL_VERBS.filter((v) => v.domainDistinctive)).toHaveLength(5);
  });

  it('declares the realtime verb with a hard deadline', () => {
    const regulate = CONTROL_VERBS.find((v) => v.name === 'regulate');
    expect(regulate?.timingClass).toBe('realtime');
    expect(regulate?.deadlineUs).toBe(5);
  });

  it('leaves non-realtime verbs without a deadline', () => {
    for (const verb of CONTROL_VERBS.filter((v) => v.timingClass !== 'realtime')) {
      expect(verb.deadlineUs).toBeUndefined();
    }
  });
});

describe('CONTROL_CLAIMS', () => {
  it('spans the honesty axes with one admissible and two bounded claims', () => {
    expect(
      CONTROL_CLAIMS.filter((c) => claimRendersAsValidated(c.status, c.admission, c.freshness)),
    ).toHaveLength(1);
    expect(
      CONTROL_CLAIMS.filter((c) => !claimRendersAsValidated(c.status, c.admission, c.freshness)),
    ).toHaveLength(2);
    expect(CONTROL_CLAIMS.some((c) => c.kind === 'formally-proven')).toBe(true);
    expect(CONTROL_CLAIMS.every((c) => c.freshness !== undefined)).toBe(true);
  });
});
