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
`capability-snapshot` markers. The generated fragment starts with the capability
inventory content; legal metadata belongs in the repository licence file and the
MkDocs footer, not at the top of rendered Markdown.

## Why this manifest exists

This inventory is the machine-readable map for reviewers, contributors, and
release auditors. It answers two concrete questions:

- Which public modules, scripts, validators, and workflows are present now?
- Which parts changed since the last manifest regeneration?

Use it before broad documentation updates, because regenerated capability data
must match the public claim language in README and release notes.

## How to keep it aligned

Keep this manifest aligned whenever public-facing content changes:

- add or remove module entries in `tools/capability_manifest.toml`,
- rerun `python tools/capability_manifest.py`,
- run `python tools/capability_manifest.py --check`,
- then confirm the README snapshot and release notes reference the same scope.

This is a documentation hygiene step that prevents mismatched release claims and
reviewer confusion during audits.

## Practical use and scope

Use this page to verify declared capabilities before publishing release or onboarding material.

- Use it to validate that documented features have executable counterparts in `tools/capability_manifest.toml`.
- Run the manifest checks whenever public claims are widened.
- Keep scope updates synchronized with `docs/production_readiness.md` and release notes.

## SCPN Studio manifest

The Studio federation manifest is a separate generated artifact:

- `tools/emit_studio_manifest.py`
- `docs/_generated/studio_manifest.json`
- `src/scpn_control/studio/manifest.py`

It declares the CONTROL studio id, verbs, evidence schemas, platform SDK range,
content digest, and federated UI module. The UI module is published for the
Studio Hub as:

- remote entry: `https://www.anulum.org/studios/scpn-control/remoteEntry.js`
- exposure: `./Panel`
- runtime: `module-federation-2`

Refresh it after Studio verb, evidence-schema, platform SDK, or UI-module changes:

```bash
python tools/emit_studio_manifest.py
python tools/emit_studio_manifest.py --check
```

The `--check` mode ignores only the environment-specific `studio_version` stamp.

The deployed Studio Web public manifest is a byte-for-byte copy of the generated
schema-A manifest:

- `tools/sync_studio_web_manifest.py`
- `studio-web/public/manifest.json`

Refresh and check it with:

```bash
python tools/sync_studio_web_manifest.py
python tools/sync_studio_web_manifest.py --check
```
All other manifest fields are compared against the producer.
