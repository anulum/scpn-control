# Control Studio — UI remote

The SCPN-CONTROL Control Studio's **Module Federation 2.x remote**. Its default
sovereign-host build uses `/studios/scpn-control/`; the Pages workflow supplies
`/scpn-control/studios/scpn-control/` explicitly. It exposes `./Panel` under the remote name
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

The feed loader validates `studio.control-feed.v1`, the `scpn-control` identity,
non-empty version, manifest-digest format, and every displayed verb/claim field.
Realtime deadlines must be finite and positive; other timing classes omit them.
Legacy claims may omit freshness only when they do not assert admitted reference
validation. Explicitly stale or rejected claims remain visible without promotion.
Invalid feeds fall back to the bundled sample. `narrowFeed` also validates direct
inputs and throws `TypeError` on failure. Additional fields are ignored.

The panel labels bundled samples, fetched feeds, and caller-provided data
separately. A fetched feed is not evidence of live reactor measurements or
freshness. Its displayed UTC receipt time is local load time, not measurement
time. The standalone application loads once; it does not poll or retain a stale
feed cache. Invalid or unavailable feeds display the explicitly labelled sample.

Feed and portal-auth requests each abort after 5 seconds, including stalled
response bodies. Both loaders accept an optional integer timeout of 1–60000 ms;
invalid timeouts throw `RangeError`. Feed failures return the sample, while auth
failures remain unavailable. Neither fallback grants execution authority.

`content_digest` identifies the producer's manifest, not the feed bytes. Format
validation does not authenticate the producer or independently verify its claims.
Rendering a feed does not authorise hardware execution. The integration test
serves the committed Python-emitted feed over loopback HTTP and renders the panel;
it is not a facility or deployment test.

```bash
pnpm install
pnpm typecheck   # tsc --noEmit, strict
pnpm lint        # eslint, type-checked + react-hooks
pnpm format:check
pnpm test        # vitest, 100% coverage gate
pnpm build       # vite build — emits dist/remoteEntry.js under the selected base
pnpm dev         # standalone preview
```

Refresh the deployed public manifest after changing the Python Studio manifest:

```bash
python tools/emit_studio_manifest.py
python tools/sync_studio_web_manifest.py
python tools/sync_studio_web_manifest.py --check
```

## Next

The Docs Pages workflow always publishes the reviewed bundle at the public
manifest URL. The CI `studio-web` job can additionally deploy `dist/` to the
provisioned sovereign SCPN Studio space when its SSH credentials are present.
Keep `public/manifest.json` and `public/studio-feed.json` current before merge.
