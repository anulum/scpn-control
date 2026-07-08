// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Control Studio UI remote — federation contract constants

import type { ModuleFederationOptions } from '@module-federation/vite';

export const STUDIO_ID = 'scpn-control';
export const FEDERATION_REMOTE_NAME = 'scpn_control';
export const FEDERATION_BASE_PATH = '/studios/scpn-control/';
export const FEDERATION_REMOTE_ENTRY = 'remoteEntry.js';
export const FEDERATION_PANEL_EXPOSE = './Panel';
export const FEDERATION_PANEL_MODULE = './src/ControlStudioPanel.tsx';
export const FEDERATION_SHARED_DEPENDENCIES = ['react', 'react-dom'] as const;

export const FEDERATION_OPTIONS: ModuleFederationOptions = {
  name: FEDERATION_REMOTE_NAME,
  filename: FEDERATION_REMOTE_ENTRY,
  exposes: {
    [FEDERATION_PANEL_EXPOSE]: FEDERATION_PANEL_MODULE,
  },
  shared: [...FEDERATION_SHARED_DEPENDENCIES],
};
