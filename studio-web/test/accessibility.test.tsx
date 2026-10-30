// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Studio panel accessibility (axe-core)

import { render } from '@testing-library/react';
import axe from 'axe-core';
import { describe, expect, it } from 'vitest';

import type { PortalAuthState } from '../src/auth.js';
import ControlStudioPanel from '../src/ControlStudioPanel.js';
import type { ControlStudioPanelProps } from '../src/ControlStudioPanel.js';

const AUTHENTICATED: PortalAuthState = {
  status: 'authenticated',
  account: {
    userId: 'u-1',
    email: 'operator@example.org',
    displayName: 'Operator',
    emailVerified: true,
    tier: 'research',
    mfaEnabled: true,
  },
};

const STATES: readonly (readonly [string, ControlStudioPanelProps])[] = [
  ['the bundled sample', {}],
  [
    'a fetched feed with its receipt time',
    { feedSource: 'fetched', feedReceivedAt: '2026-09-05T09:00:00.000Z' },
  ],
  ['caller-provided data without claims', { feedSource: 'provided', claims: [] }],
  ['an authenticated portal session', { portalAuth: AUTHENTICATED }],
  ['an anonymous portal session', { portalAuth: { status: 'anonymous' } }],
];

describe('ControlStudioPanel accessibility', () => {
  it.each(STATES)('has no axe-core violations for %s', async (_label, props) => {
    const { container } = render(
      <main>
        <ControlStudioPanel {...props} />
      </main>,
    );
    const results = await axe.run(container);
    expect(results.violations.map((violation) => violation.id)).toEqual([]);
    expect(results.passes.length).toBeGreaterThan(0);
  });

  it('reports a real violation, so the check can fail', async () => {
    const { container } = render(<main />);
    const image = document.createElement('img');
    image.src = 'data:,';
    container.querySelector('main')?.append(image);
    const results = await axe.run(container);
    expect(results.violations.map((violation) => violation.id)).toContain('image-alt');
  });
});
