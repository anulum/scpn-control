// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Control Studio UI remote — tests for the ControlStudioPanel

import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import ControlStudioPanel from '../src/ControlStudioPanel.js';
import type { ClaimSummary, ControlVerb } from '../src/domain.js';

describe('ControlStudioPanel', () => {
  it('renders the studio header', () => {
    render(<ControlStudioPanel />);
    expect(
      screen.getByRole('heading', { name: 'SCPN-CONTROL — Control Studio' }),
    ).toBeInTheDocument();
    expect(screen.getByText('Portal session: unavailable')).toHaveAttribute(
      'data-auth-status',
      'unavailable',
    );
  });

  it('lists every verb as a table row', () => {
    render(<ControlStudioPanel />);
    // 11 verbs + 1 header row.
    expect(screen.getAllByRole('row')).toHaveLength(12);
  });

  it('shows the realtime verb deadline and marks it Hub-gated', () => {
    render(<ControlStudioPanel />);
    const regulateRow = screen.getByText('regulate').closest('tr');
    expect(regulateRow).toHaveAttribute('data-distinctive', 'domain');
    expect(regulateRow).toHaveTextContent('realtime (5 µs)');
    expect(regulateRow).toHaveTextContent('live-hardware (per-tenant)');
  });

  it('renders a core read-only verb without a gate or deadline', () => {
    render(<ControlStudioPanel />);
    const reconstructRow = screen.getByText('reconstruct').closest('tr');
    expect(reconstructRow).toHaveAttribute('data-distinctive', 'core');
    expect(reconstructRow).toHaveTextContent('interactive');
    expect(reconstructRow).not.toHaveTextContent('µs');
  });

  it('renders a held formal proof as validated', () => {
    render(<ControlStudioPanel />);
    const cert = screen.getByText(/studio\.safety-certificate\.v1/);
    expect(cert.closest('li')).toHaveAttribute('data-validated', 'yes');
    expect(cert).toHaveTextContent('validated');
  });

  it('renders a bounded reconstruction verbatim, not validated', () => {
    render(<ControlStudioPanel />);
    const efit = screen.getByText(/studio\.efit-reconstruction\.v1/);
    expect(efit.closest('li')).toHaveAttribute('data-validated', 'no');
    expect(efit).toHaveTextContent('bounded-model');
  });

  it('renders the verbs and claims supplied from the live feed', () => {
    const verbs: readonly ControlVerb[] = [
      {
        name: 'mitigate',
        safetyTier: 'certified',
        sideEffect: 'live-hardware',
        timingClass: 'interactive',
        domainDistinctive: true,
      },
    ];
    const claims: readonly ClaimSummary[] = [
      {
        schema: 'studio.disruption-mitigation.v1',
        status: 'bounded-model',
        admission: 'rejected',
        kind: 'measured',
        freshness: 'traceable-unchecked',
      },
    ];
    render(<ControlStudioPanel verbs={verbs} claims={claims} />);
    // Only the feed-supplied verb is rendered (1 verb + 1 header row), not the sample.
    expect(screen.getAllByRole('row')).toHaveLength(2);
    expect(screen.queryByText('reconstruct')).not.toBeInTheDocument();
    const claim = screen.getByText(/studio\.disruption-mitigation\.v1/);
    expect(claim.closest('li')).toHaveAttribute('data-validated', 'no');
  });

  it('renders the portal session supplied by the Hub origin', () => {
    render(
      <ControlStudioPanel
        portalAuth={{
          status: 'authenticated',
          account: {
            userId: 'user-1',
            email: 'operator@example.invalid',
            displayName: 'Operator',
            emailVerified: true,
            tier: 'pro',
            mfaEnabled: true,
          },
        }}
      />,
    );

    const status = screen.getByText('Portal session: Operator · pro');
    expect(status).toHaveAttribute('data-auth-status', 'authenticated');
    expect(
      screen.queryByRole('button', { name: /login|billing|account/i }),
    ).not.toBeInTheDocument();
  });
});
