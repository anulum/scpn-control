# Capability Manifest

The SCPN-CONTROL capability manifest inventories configured source declarations,
file presence and packaged project script entry points. It does not import the
listed implementations, compile Rust, run tests, measure performance or establish
scientific, facility or release readiness. A declared script/export can be
present in this catalog while its implementation is unavailable at runtime.

`tools/capability_manifest.toml`, project metadata and the selected source tree
supply the input. The generator has three owners:

- `tools/capability_manifest_inventory.py` reads metadata and source declarations;
- `tools/capability_manifest_rendering.py` renders, compares and writes snapshots;
- `tools/capability_manifest.py` exposes the same public API and command.

The generated JSON and Markdown are derived snapshots:

- `docs/_generated/capability_manifest.json`
- `docs/_generated/capability_snapshot.md`

The configured source roots include `src/scpn_control/core`,
`src/scpn_control/control`, `src/scpn_control/phase`, `src/scpn_control/scpn`,
`src/scpn_control/reactor_semantic_admission`, `scpn-control-rs/crates`,
`validation`, `tests`, public `docs`, `.github/workflows`, and
`[project.scripts]` from `pyproject.toml`.

## Discovery contract

Python classes are AST declaration names, including nested classes. Names
starting with an underscore are omitted; equal unqualified names across modules
are counted once. The package export list is its first literal `__all__`
at module scope, and project script targets are checked for declared dotted
`module:callable` spelling. These checks do not establish importability. Invalid
Python syntax or an unsupported export carrier refuses inventory generation.

Rust PyO3 names come from wrapper-text patterns, including matching comments;
they do not establish that an extension was compiled or loaded. Missing scan
roots are omitted. Counts describe entries discovered in the configured scope,
not successful implementations. Public Markdown excludes configured whole path
components, including `internal` and `_generated`.

## Generate and check local output

Run from the repository root; the command uses the current working directory
and its configuration. Generate the local snapshots and replace the single
ordered README `capability-snapshot` marker block:

```bash
python tools/capability_manifest.py
```

Generation prepares all output text and validates the marker pair before
writing. Missing, duplicated or reversed markers preserve existing outputs.
Writes are sequential; an operating-system write failure can leave partial
outputs. Existing configured destinations are overwritten. The command performs
no Git commit or external publication.

Compare current snapshots without rewriting them:

```bash
python tools/capability_manifest.py --check
```

The comparison uses normalised UTF-8 text: LF and CRLF are equivalent, but
other whitespace/formatting changes can make a snapshot stale. Exit zero means
current text matches the fresh source catalog or generation succeeded. Exit one
means stale/missing output or a configuration, inspection, read/decode or write
failure; argument errors exit two. A passing check does not admit feature
readiness or public scientific claims.

The generated fragment begins with the inventory table. Legal metadata belongs
in the repository license/footer. Keep snapshots and the README block aligned
with the source scope being reviewed before publishing source-presence counts;
review runtime and scientific claims using their actual implementation/evidence
contracts separately.

## SCPN Studio manifest

The Studio federation manifest is a separate generated artifact:

- `tools/emit_studio_manifest.py`
- `docs/_generated/studio_manifest.json`
- `src/scpn_control/studio/manifest.py`

It declares the CONTROL studio id, verbs, evidence schemas, platform SDK range,
content digest, and federated UI module. The producer declares the following
Studio Hub UI metadata:

- remote entry: `https://anulum.github.io/scpn-control/studios/scpn-control/remoteEntry.js`
- exposure: `./Panel`
- runtime: `module-federation-2`

Refresh it after Studio verb, evidence-schema, platform SDK, or UI-module changes:

```bash
python tools/emit_studio_manifest.py
python tools/emit_studio_manifest.py --check
```

The `--check` mode compares decoded objects and ignores only the
environment-specific `studio_version` stamp. JSON whitespace and key order do
not affect equality. Duplicate keys at any depth, nonstandard `NaN`/`Infinity`
tokens, floating tokens overflowing to infinity, non-object roots, malformed
JSON and invalid UTF-8 are refused before
stamp removal. Other manifest fields must equal the actual producer.

Select a caller-owned local artifact with `--artifact`; relative paths resolve
from the working directory. The default path is anchored to the script's
repository. Writing creates parent directories and directly overwrites that
file as sorted, indented finite Unicode JSON with a trailing newline; it does
not promise an atomic transaction. Checking reads without writing:

```bash
PYTHONPATH=src python tools/emit_studio_manifest.py --artifact ./local-studio-manifest.json
PYTHONPATH=src python tools/emit_studio_manifest.py --artifact ./local-studio-manifest.json --check
```

Success returns status 0. Missing, stale, invalid or unreadable artifacts,
unavailable producer dependencies and write failures return 1. Inspection
failures use a fixed stderr refusal without a traceback; missing/stale/write
messages use stdout. Invalid CLI arguments exit 2. Help needs only the standard
library; producing a manifest needs the CONTROL source and Studio dependencies.
This compares declared metadata. It does not validate SDK compatibility,
execute verbs, admit the studio to a federation or verify remote availability.

The Docs Pages workflow is configured to build the Studio source and include
the remote beside the documentation artifact. It supplies the explicit Vite
base `/scpn-control/studios/scpn-control/`. These configuration declarations do
not establish the current hosted build or deployed URL state.

The Studio Web sync tool copies the generated schema-A manifest byte for byte
to its local public manifest:

- `tools/sync_studio_web_manifest.py`
- `studio-web/public/manifest.json`

Refresh and check it with:

```bash
python tools/sync_studio_web_manifest.py
python tools/sync_studio_web_manifest.py --check
```

The sync command uses only the standard library. It reads strict UTF-8 without
newline translation, refuses duplicate keys, nonfinite/overflowing JSON numbers
and non-object roots, then checks the CONTROL studio id and the three declared
UI fields (`remote_entry`, `exposes`, `federation`). It does not check the complete
SDK schema, compatibility range, digest or runtime admission. The historical
API name `validate_deployed_contract` describes those local metadata checks;
it does not request the declared URL.

Unlike the emitter's semantic check, sync requires **exact bytes**, including
key order, whitespace, LF versus CRLF and `studio_version`. Select local paths
relative to cwd with `--source` and `--destination`; omitted paths retain the
script-root defaults. `--check` validates both files and reads without writing.
Copy mode creates parent directories and directly overwrites the destination
without atomicity/fsync guarantees. A valid source is inspected before any
destination write:

```bash
python tools/sync_studio_web_manifest.py --source ./local-studio-manifest.json --destination ./local-web-manifest.json
python tools/sync_studio_web_manifest.py --source ./local-studio-manifest.json --destination ./local-web-manifest.json --check
```

Equality or a successful copy returns 0. Missing, unreadable, invalid or stale
files and write failures return 1; CLI argument errors exit 2. Source/check
refusals and stale/success messages use stdout, while write failures use a fixed
stderr refusal without a traceback. This copies a local artifact; deployment
remains a separate workflow action.
