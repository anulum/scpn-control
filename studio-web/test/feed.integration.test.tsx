// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Feed HTTP and panel integration.

import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, expect, it } from 'vitest';

import ControlStudioPanel from '../src/ControlStudioPanel.js';
import { FALLBACK_FEED, isRawFeed, loadStudioFeed } from '../src/feed.js';

afterEach(cleanup);

it('loads the emitted Python feed over HTTP and renders its claims and verbs', async () => {
  const emitted = await readFile('public/studio-feed.json', 'utf8');
  const payload: unknown = JSON.parse(emitted);
  expect(isRawFeed(payload)).toBe(true);
  const server = createServer((request, response) => {
    response.setHeader('Content-Type', 'application/json');
    response.end(request.url === '/feed' ? emitted : JSON.stringify({ verbs: [], claims: [] }));
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  try {
    const url = `http://127.0.0.1:${String((server.address() as AddressInfo).port)}`;
    const feed = await loadStudioFeed(`${url}/feed`);
    expect(feed).not.toBe(FALLBACK_FEED);
    render(<ControlStudioPanel verbs={feed.verbs} claims={feed.claims} />);
    expect(screen.getAllByRole('row')).toHaveLength(feed.verbs.length + 1);
    expect(screen.getByText(/studio\.efit-reconstruction\.v1/).closest('li')).toHaveAttribute(
      'data-validated',
      'no',
    );
    expect(screen.getByText(/studio\.safety-certificate\.v1/).closest('li')).toHaveAttribute(
      'data-validated',
      'yes',
    );
    expect(await loadStudioFeed(`${url}/invalid`)).toBe(FALLBACK_FEED);
  } finally {
    server.closeAllConnections();
    await new Promise<void>((resolve, reject) =>
      server.close((error) => {
        if (error) reject(error);
        else resolve();
      }),
    );
  }
});
