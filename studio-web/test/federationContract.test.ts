// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Control Studio UI remote — tests for the federation contract

import { describe, expect, it } from 'vitest';

import viteConfig from '../vite.config.js';
import {
  FEDERATION_BASE_PATH,
  FEDERATION_OPTIONS,
  FEDERATION_PANEL_EXPOSE,
  FEDERATION_PANEL_MODULE,
  FEDERATION_REMOTE_ENTRY,
  FEDERATION_REMOTE_NAME,
  FEDERATION_SHARED_DEPENDENCIES,
  STUDIO_ID,
} from '../src/federationContract.js';

describe('federation contract', () => {
  it('matches the Studio Hub manifest contract', () => {
    expect(STUDIO_ID).toBe('scpn-control');
    expect(FEDERATION_REMOTE_NAME).toBe('scpn_control');
    expect(FEDERATION_REMOTE_ENTRY).toBe('remoteEntry.js');
    expect(FEDERATION_PANEL_EXPOSE).toBe('./Panel');
    expect(FEDERATION_PANEL_MODULE).toBe('./src/ControlStudioPanel.tsx');
  });

  it('shares the host React runtime dependencies', () => {
    expect(FEDERATION_SHARED_DEPENDENCIES).toEqual(['react', 'react-dom']);
    expect(FEDERATION_OPTIONS.shared).toEqual(['react', 'react-dom']);
  });

  it('exposes only the stable panel entry expected by the manifest', () => {
    expect(FEDERATION_OPTIONS.exposes).toEqual({
      './Panel': './src/ControlStudioPanel.tsx',
    });
  });

  it('builds under the deployed studio base path', () => {
    expect(viteConfig.base).toBe(FEDERATION_BASE_PATH);
  });
});
