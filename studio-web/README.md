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
| `src/auth.ts` | Same-origin portal session loader for `GET /api/v1/auth/me` with `credentials: 'include'`; no local login or billing surface. |
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

The CI `studio-web` job deploys `dist/` to the provisioned SCPN Studio space on
pushes to `main`. Keep `public/manifest.json` and `public/studio-feed.json`
current before merge so the deployed bundle contains the remote entry, manifest,
and feed the Hub reads.
