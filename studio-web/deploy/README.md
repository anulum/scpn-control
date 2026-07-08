<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Studio Web deploy key notes. -->

# Studio Web Deploy Key

`scpn-control-studio-ci-deploy.pub` is the public ed25519 key for the
SCPN-CONTROL Studio Web deploy lane. SCPN-STUDIO provisions it with:

```bash
deploy/provision-studio.sh scpn-control scpn-control-studio-ci-deploy.pub
```

The private key is not tracked. CI deploy work must install the private key as a
repository secret and use it only for the jailed rsync deploy user described in
`SCPN-STUDIO/docs/studio-hosting.md`.

The `studio-web` job in `.github/workflows/ci.yml` deploys only on pushes to
`main`. It expects these repository secrets:

- `SCPN_CONTROL_STUDIO_DEPLOY_KEY`: private key matching
  `scpn-control-studio-ci-deploy.pub`.
- `SCPN_CONTROL_STUDIO_KNOWN_HOSTS`: pinned SSH known-hosts entry for
  `www.anulum.org`.

The deploy command checks that `dist/remoteEntry.js`, `dist/manifest.json`, and
`dist/studio-feed.json` exist, then runs:

```bash
rsync -az --delete dist/ deploy@www.anulum.org:
```

The SCPN-STUDIO jailed key pins that destination to
`/srv/scpn-studio/studios/scpn-control`.
