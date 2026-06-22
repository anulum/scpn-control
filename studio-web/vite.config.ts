// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Control Studio UI remote — Vite + Module Federation 2.x remote build

import { federation } from '@module-federation/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// This studio is a Module Federation REMOTE. Its federation name is the Hub's
// derived remote name for this studio (studioRemoteName('scpn-control') →
// 'scpn_control'), so the Hub's loadStudioPanel registers and loads it without an
// out-of-band name. It exposes ControlStudioPanel and emits a remoteEntry.js the
// Hub loads at runtime. react/react-dom are shared so the panel renders against
// the host's single React instance.
export default defineConfig({
  plugins: [
    react(),
    federation({
      name: 'scpn_control',
      filename: 'remoteEntry.js',
      exposes: {
        './ControlStudioPanel': './src/ControlStudioPanel.tsx',
      },
      shared: ['react', 'react-dom'],
    }),
  ],
  build: {
    target: 'esnext',
  },
});
