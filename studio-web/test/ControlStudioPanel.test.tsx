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

describe('ControlStudioPanel', () => {
  it('renders the studio header', () => {
    render(<ControlStudioPanel />);
    expect(
      screen.getByRole('heading', { name: 'SCPN-CONTROL — Control Studio' }),
    ).toBeInTheDocument();
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
});
