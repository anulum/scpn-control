// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Control Studio UI remote — standalone preview entry (the studio runs on its own)

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import ControlStudioPanel from './ControlStudioPanel.js';

const container = document.getElementById('root');
if (container === null) {
  throw new Error('Control Studio: #root container not found');
}
createRoot(container).render(
  <StrictMode>
    <ControlStudioPanel />
  </StrictMode>,
);
