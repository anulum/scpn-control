<!--
SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Control Studio — UI remote

The SCPN-CONTROL Control Studio's **Module Federation 2.x remote**. It builds
under `/studios/scpn-control/`, exposes `./Panel` under the remote name
`scpn_control` (`studioRemoteName('scpn-control')`), and emits a `remoteEntry.js`
that the SCPN Studio Hub loads at runtime (the Hub's `loadStudioPanel` consumes
exactly this). It also runs standalone, matching the federation model where each
studio is independent.

## What the panel shows

- CONTROL's verbs with their gating attributes: safety tier, side-effect class, and
  timing. A realtime verb (`regulate`) shows its hard deadline; a live-hardware verb
  (`regulate`, `mitigate`) is marked as Hub-gated per tenant.
- CONTROL's claims, each rendered by the shared honesty rule: only a
  reference-validated claim is marked validated; every other boundary
  (bounded-model, …) is shown verbatim — the same grading the Python vertical emits.

## Layout

| Path | Responsibility |
| --- | --- |
| `src/federationContract.ts` | The remote name, deployed base path, `remoteEntry.js`, `./Panel` exposure, and shared React dependencies. |
| `src/domain.ts` | The studio's verb/claim data + the honesty rendering rules. |
| `src/ControlStudioPanel.tsx` | The exposed federated panel. |
| `public/manifest.json` | The deployed schema-A capability manifest copied from `docs/_generated/studio_manifest.json`. |
| `public/studio-feed.json` | The standalone panel feed rendered from `scpn_control.studio.feed`. |

## Develop

```bash
pnpm install
pnpm typecheck   # tsc --noEmit, strict
pnpm lint        # eslint, type-checked + react-hooks
pnpm format:check
pnpm test        # vitest, 100% coverage gate
pnpm build       # vite build — emits dist/remoteEntry.js under /studios/scpn-control/
pnpm dev         # standalone preview
```

Refresh the deployed public manifest after changing the Python Studio manifest:

```bash
python tools/emit_studio_manifest.py
python tools/sync_studio_web_manifest.py
python tools/sync_studio_web_manifest.py --check
```

## Next

Serve `remoteEntry.js` from the studio host and register the `./Panel` exposure
with the Hub (`type: 'module'`); wire the panel's data from the live CONTROL
studio backend (`scpn_control.studio`) instead of the representative sample.
