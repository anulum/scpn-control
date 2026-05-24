<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Capability Manifest Adoption -->

# Capability Manifest

The SCPN-CONTROL capability manifest is the repository-local inventory for
public control, physics, phase, SCPN, Rust, validation, documentation, test, and
CI surfaces, including packaged project script entry points.

The source of truth is:

- `tools/capability_manifest.py`
- `tools/capability_manifest.toml`
- `docs/_generated/capability_manifest.json`
- `docs/_generated/capability_snapshot.md`

The generator is customised to this repository. It scans
`src/scpn_control/core`, `src/scpn_control/control`,
`src/scpn_control/phase`, `src/scpn_control/scpn`, `scpn-control-rs/crates`,
`validation`, `tests`, public `docs`, `.github/workflows`, and
`[project.scripts]` from `pyproject.toml`.

Refresh the generated files after capability changes:

```bash
python tools/capability_manifest.py
```

Check that the repository surfaces and README snapshot are current:

```bash
python tools/capability_manifest.py --check
```

The README embeds `docs/_generated/capability_snapshot.md` between
`capability-snapshot` markers. The generated fragment uses HTML comments for
metadata so SPDX and provenance lines do not render as large page titles.
