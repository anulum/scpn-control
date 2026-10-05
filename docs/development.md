# Development

## Local Setup

### Python

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .\.venv\Scripts\activate      # Windows
pip install -e ".[dev]"
```

### Rust (optional)

Requires Rust 1.70+ and maturin:

```bash
pip install maturin
cd scpn-control-rs
cargo build --release
maturin develop --release -m crates/control-python/Cargo.toml
```

After `maturin develop`, `RUST_BACKEND` becomes `True` automatically.

---

## Architecture

Four Python packages under `src/scpn_control/`:

| Package | Purpose |
|---------|---------|
| `core` | GS equilibrium solver, transport, scaling laws, TokamakConfig presets |
| `control` | normalized DGKF H-infinity, MPC, SNN, flight sim, disruption predictor, digital twin |
| `scpn` | Stochastic Petri Net → SNN compiler with formal contracts |
| `phase` | Paper 27 oscillator-model engine, UPDE, model-local Lyapunov guard, WebSocket stream; no reactor feedback closure |

Five Rust crates under `scpn-control-rs/crates/`:

| Crate | Purpose |
|-------|---------|
| `control-types` | PlasmaState, EquilibriumConfig |
| `control-math` | LIF neurons, Boris pusher, Kuramoto |
| `control-core` | Rust GS solver, transport |
| `control-control` | Rust PID, MPC, H-inf, SNN |
| `control-python` | PyO3 bindings |

---

## Running Tests

```bash
pytest tests/ -v                         # full suite
pytest tests/test_h_infinity_controller.py  # single file
pytest -m "not slow"                     # skip slow markers
pytest --cov=scpn_control --cov-report=term --cov-fail-under=100
```

The project enforces a **100% configured package coverage gate for admitted CI contexts**
(configured in `pyproject.toml`). This is a statement/branch gate for
the dependencies and variants actually admitted to the merged coverage jobs;
it is not a claim that unavailable hardware, private datasets, facility
services, or every optional package executed. Coverage claims should come from
the latest local coverage run or the GitHub coverage lane, not from static
documentation text.

---

## Phase-video test dependencies

The Python test matrix installs the existing hash-pinned `requirements/ci-viz.txt`
profile before its first Test step, so Matplotlib and Pillow are available on
Linux, Windows and macOS. Real phase-video tests also require FFmpeg with H.264
encoding and ffprobe. The workflow provisions them through apt on Linux,
Homebrew on macOS and Chocolatey package `ffmpeg` version 9.0.2 on Windows, then prints both
executable versions. Native package versions can differ by platform; this is not
a bit-reproducible cross-platform media claim. Local checks use already available
encoders and do not install dependencies automatically.

Run the bounded real cohort after arranging those dependencies:

```bash
python -m pytest tests/test_phase_video_model.py tests/test_generate_phase_video.py -q
```

It exercises actual monitor capture, GIF/MP4 decoding, CLI/API refusals and native
encoder errors. Missing tools fail explicitly. See the
[video contract](api.md#phase-model-video) for output custody and model limits.

## RL test dependencies

The `dev` extra includes Gymnasium and Stable-Baselines3 for the actual RL tests;
the `rl` extra installs them for model evaluation and explicit training commands.
Both require Gymnasium >=1.2.3 and Stable-Baselines3 >=2.8.

The Python test matrix installs the hash-pinned `requirements/ci-rl.txt` before
its first Test step. This profile locks Gymnasium 1.2.3, Stable-Baselines3 2.8.0
and PyTorch 2.14.0: Linux and Windows select the `+cpu` wheel from the official
PyTorch CPU index; macOS selects the ARM64 wheel, which requires macOS >=14.
The [runner image inventory](https://github.com/actions/runner-images#available-images)
identifies the architecture behind the matrix's `macos-latest` label.
The profile constrains shared
NumPy, plotting and test dependencies to their existing CI locks.

The workflow exercises the real planning command before the tests:

```bash
python tools/train_rl_tokamak.py --ci --seed 123 --dry-run
```

This validates and prints the CPU constructor plan. It does not construct PPO,
learn or save weights. The [RL API contract](api.md#seeded-ppo-recipe-evaluation-and-candidate-results)
describes evaluation, candidate output checks and explicit learning modes.

## Type Checking

```bash
mypy
python tools/run_mypy_strict.py
```

Scope: all modules under `src/scpn_control/` (`disallow_untyped_defs = true`, `warn_return_any = true`).
PEP 561 marker: `src/scpn_control/py.typed`.

`tools/run_mypy_strict.py` is the local preflight gate for strict-typing debt. It
first runs the configured repository mypy check, then runs a whole-package
`mypy --strict src/scpn_control/` probe and compares the result with
`tools/mypy_strict_debt.json`. The ratchet may stay flat or fall; increases need
an explicit baseline update with the increase flag so added strict debt remains a
reviewed change.

The configured check includes `src/scpn_control/` and `validation/`; strict mode
is already enabled. `--skip-configured-gate` explicitly selects only the advisory
package/debt comparison. Mypy statuses zero and one represent clean/error
results; fatal probe statuses return two. Parsed total and module counts must
agree before any ledger update.

Ledger counts must be nonnegative integers excluding booleans, labels must be
nonempty strings, and the module sum must equal the total. A present
`mypy_version` must be a string; an omitted legacy version remains empty.
Persisted JSON requires an object root and unique keys. Invalid, missing or
unreadable ledgers return two and are preserved, including when both update
flags are given. Ordinary comparisons leave the ledger unchanged; a refused
increase returns three. Direct constructors and writes do not validate values;
writes do not create parents and are not atomic. The caller owns concurrent
file coordination. Actual Mypy fixtures and the public API example exercise
these boundaries; compiler success alone does not certify scientific results.

## Selected Python preflight checks

`python tools/run_python_preflight.py` executes the following checks in order,
using the current Python executable and the script's repository as the child
working directory:

| Check | Child target | Selection flag |
| --- | --- | --- |
| Version metadata | `tests/test_project_metadata.py` | `--skip-version-metadata` |
| Golden notebook quality | `tests/test_neuro_symbolic_control_demo_notebook.py` | `--skip-notebook-quality` |
| Task 5/6 threshold smoke | `test_task5_campaign_passes_thresholds_smoke` and `test_task6_campaign_passes_thresholds_smoke` in their named test modules | `--skip-threshold-smoke` |
| Strict typing | `tools/run_mypy_strict.py` | `--skip-mypy` |
| Docstrings | `tools/run_docstring_gate.py` | Always selected |

The Golden notebook and Task 5/6 test files are absent from the CONTROL
extraction. A default invocation therefore returns a failing pytest status
at the notebook check after running metadata tests; it does not qualify the
complete profile. The two smoke node IDs still name
`tests/test_task5_disruption_mitigation_integration.py` and
`tests/test_task6_heating_neutronics_realism.py`.

For an explicit documentation-only selection:

```bash
python tools/run_python_preflight.py --skip-version-metadata --skip-notebook-quality --skip-threshold-smoke --skip-mypy
```

Every selected child must return zero. The first nonzero child return code is
propagated and later checks are not executed. `All selected checks passed`
describes that selection; omitting checks supplies no default-profile or
scientific admission. Unknown arguments are rejected with status two before
any child starts. Subprocess launch errors propagate. Printed commands use
shell quoting for readability; execution passes argument vectors without a
shell. This wrapper runs its named targets and does not invoke
`tools/preflight.py` or the complete test suite.

## Wiring Checks

Two local guards keep source modules connected to real repository surfaces:

```bash
python tools/check_test_module_linkage.py
python tools/check_runtime_wiring.py
```

`check_test_module_linkage.py` checks bounded static test references or exact
allowlist paths. It follows recognised test functions/methods, called local
helpers and visible import aliases in initialisers and ordinary module files,
crediting the referenced facade and selected export target while conservatively refusing
shadowed names. Missing/non-directory roots fail; empty existing roots are
supported. Named class references follow unambiguous imported/local base identities;
class initialisation and inherited method dispatch are not evaluated. Selected plain
methods follow lexical body references. A single direct constructor assignment can
identify a receiver for subsequent references. Unrelated data-field writes preserve
the binding; selected-member writes/deletions and receiver type/dictionary changes
refuse it. Imported member writes resolve through visible aliases, and changes to
constructor hooks refuse inferred instances. Source receiver writes to a selected
method and custom attribute hooks remain opaque. Parameters, other name writes/imports
and conditional/annotated/chained/tuple assignments refuse receiver binding.
Receiver aliases, inline constructors, factory returns, class namespaces and
implicit receiver helper dispatch are not inferred. Decorated, conflicting or
source-mutated methods do not add body edges. Decorated classes, class keywords,
wildcard imports and rebindings prevent class attribution. Selected unambiguous
top-level functions follow named body calls through lexical imports and module helpers;
unused functions, defaults, annotations and decorated/rebound declarations do not
add body-call edges. This remains static reference evidence. Fixtures, subprocesses,
dynamic imports, dynamic mutator calls, inherited descriptors, assignment exports and
argument/return flow are outside this scan, and syntactic calls in unreachable branches
can count. See the [complete linkage API contract](api.md#static-test-ownership-linkage)
for path/error/allowlist behaviour. This does not prove exercised coverage. `check_runtime_wiring.py` parses static imports
across `src`, `tests`, `benchmarks`, `examples`, `tools`, and `validation`; it
fails if a non-exempt source module has no import reference in those files.

The reference-command integration tests compare each family export with its
registered root command, then exercise both entry points with optional and
required inspections and persisted reports. They also exercise the shared
report-path converter through registered options, including NUL and directory
refusals. These runtime checks accompany the static references; the linkage
guard does not infer Click registration or callback execution from decorators.
Missing-owner strings use forward slashes on every OS, as specified by the API.
Native directory-read errors remain platform specific: Windows refuses the
allowlist directory with `PermissionError`, while Linux uses `IsADirectoryError`.
Package initialisers and the declared root exemptions do not require a reference.
Nested imports count even in branches that never execute, and the importer may
itself be unreferenced. Dynamic imports and external library consumers are not
discovered. A positive report therefore establishes static source references;
it does not establish entrypoint reachability, importability or exercised behaviour.
Review orphan names before deciding whether to wire, test or remove a public API.

Use `--repo PATH` to inspect a selected checkout and `--json` for the unchanged
`total_modules`/`orphans` report fields. Relative roots follow the working
directory; the no-flag command uses the script's own repository. Empty/missing
source scopes or unreadable, invalid UTF-8 or syntactically invalid inspected
files refuse the report with status one. The checker never imports inspected
code or writes files. Both gates run through `tools/preflight.py`.

## Documentation links

Run the deterministic public-document graph audit before changing docs,
manuscripts, generated navigation, or submission metadata:

```bash
python tools/document_link_audit.py
mkdocs build --strict
python tools/document_link_audit.py --site-dir site
```

The local gate checks tracked Markdown, HTML, TeX, BibTeX, MkDocs navigation,
and public metadata. It resolves relative files and Markdown anchors, validates
the rendered site tree when supplied, and rejects credential-shaped public
URLs. It never crawls the network.

External availability runs only in the scheduled `Docs Link Audit` workflow or
through an explicit `--external` invocation. That mode limits retries and URL
count, paces each host, reuses a TTL cache, records tool/policy/source/URL-set
provenance, and separates reachable, restricted, transient, and permanent
results. Restricted or transient responses remain visible without being
misreported as permanent breakage; confirmed permanent failures fail the job.
Authenticated or secret-bearing URLs are outside this public audit by policy.

The local graph uses the selected Git index and current disk bytes. An existing
untracked target still refuses; stage the verified task files as part of the
final commit preparation. Indexed sources missing from disk raise a native read
error. Use a worktree root for complete enumeration. Set hashes in JSON reports
bind sorted filenames and observed URLs, rather than documentation contents.

`--list-external` prints screened URLs without requesting them. Restricted or
transient HTTP observations alone return zero; that exit does not certify every
URL as reachable. The matching tool/policy TTL cache trusts stored rows and
retains their original observation times. DNS addresses are not checked, and
urllib follows redirects before the final URL check. This audit supplies no
network sandbox or validation of cited content.

The site audit checks existing HTML target files rather than HTML fragments.
An absent or empty site directory has no HTML findings, so establish a successful
build separately. Reports overwrite their chosen destinations through separate
same-directory replacements, without signing or source-content custody.

## Coverage Exclusions

Every source `# pragma: no cover` exclusion must explain why the line is not
covered by the local Python coverage job:

```bash
python tools/check_coverage_pragmas.py
```

Use a one-line reason such as an optional dependency path, native backend path,
or defensive invariant branch. The gate runs through `tools/preflight.py` and
fails on bare or separator-only exclusions, including mixed Unicode dashes and
whitespace. It checks for trailing text rather than reviewing its justification.

The CLI also accepts Python files or directories and `--json`. Relative CLI
paths resolve against the script's repository; relative Python API `Path`
inputs use the caller's working directory. Missing requested paths, unsupported
explicit files and invalid UTF-8 raise before any success output. Directories
select their native `Path.rglob('*.py')` file descendants; empty directories
and empty API requests have no findings, and overlapping requests retain
duplicates. Matching is case-sensitive and lexical over entire lines, including
strings and docstrings. A clean result applies to enumerated files only and
does not certify coverage, exception ownership or readiness. See the
[public API contract](api.md#coverage-pragma-reason-guard).

The actual CLI and public API regression cases are in
`tests/test_tools/test_pragma_reason_gate.py`. The default docstring gate also
requires documentation on every definition in the checker and that test owner.

The broader exception ledger covers source pragmas, coverage.py exclusion
patterns, `skipif`, runtime skips, and strict xfails. Every entry records its
owner, reason/condition, external dependency, executing CI lane or explicit
blocker, review date, and removal condition:

```bash
python tools/coverage_exception_ledger.py --check
```

The committed machine-readable output is
`tools/coverage_exception_ledger.json`. Source locations and conditions are
digest-sealed, so a new or moved exception fails CI until its ownership and
variant classification are reviewed. Runtime skips are not accepted as
coverage success merely because the configured context excludes them.

The [ownership API contract](api.md#coverage-exception-ownership-ledger)
describes the exact lexical/AST scope and native failure behaviour. Lane labels
and workflow-substring checks declare evidence requirements; they do not
establish that a variant executed. `--print-summary` is informational and
bypasses count/digest admission, including when combined with `--check`.
The default docstring gate requires every definition in the ledger and its
two test owners. Real policy/parser/output/CLI refusals are exercised by
`tests/test_tools/test_exception_ledger_command.py`.

Native-dependent modules use a two-environment coverage matrix because the
authoritative Python job runs without the optional `scpn_control_rs` extension
while the interop job builds it and executes every test-file owner with a
Rust/PyO3 conditional entry in the exception ledger. CI uploads `coverage-data-python` and
`coverage-data-rust`, then `native-coverage-combine` runs:

```bash
coverage combine --keep artifacts/coverage/python artifacts/coverage/rust
coverage report --fail-under=100
```

`python tools/native_coverage_matrix.py` checks the
`scpn-control.native-coverage-matrix.v1` declaration contract. It reads the
physical reusable workflows and coordinator, binds enabled producers to pinned
artifact actions, and checks ordered downloads, combination, XML output and the
100-percent report command. The selected Ubuntu/Python 3.12 matrix lane must
exist. The TOML report threshold must be numeric 100; comments do not set it.
Workflow, job and step environments and run defaults are included. The same
guard runs through `tools/preflight.py`.

The accepted shell profile uses direct Python/pytest/coverage calls and literal
inline environment assignments. Conditional blocks, failure masking, early
exit, environment mutation, changed working directories and unsupported shells
do not count as coverage commands. Comments, echoed commands and heredoc bodies
do not supply evidence. This validates declarations without executing shell or
CI; it does not authenticate artifact contents or establish measured coverage.

The per-commit CI is a distributed responsibility graph. The small
`.github/workflows/ci.yml` coordinator owns only triggers, concurrency,
explicit reusable calls, and the fail-closed `ci-gate`; executable jobs live in
the cohesive `.github/workflows/ci-*.yml` owners declared by
`tools/ci_workflow_policy.json`. Run
`python tools/check_ci_workflow_modularity.py` after any workflow change. The
guard verifies exclusive job ownership, original dependency and artifact
ordering, names of conditional steps, explicit secrets, immutable action pins,
workflow sizes, and required aggregate-gate fragments. These checks inspect
declarations; workflow execution requires running CI. The guard is also part
of the normal local preflight surface and the hosted `python-lint` job.

Repository audit and generation helpers under `tools/` are source-tree
commands, not installed package entry points. Release artifacts expose only the
shipped `scpn-control` CLI. Build wheel and sdist through
`python tools/build_release_artifacts.py`. The frontend runs the project's
actual build backend from the script's repository directory with the current
Python interpreter. Epoch precedence is `--source-date-epoch`, then
`SOURCE_DATE_EPOCH`, then the source commit timestamp; the value must fit the
unsigned 32-bit gzip header. Existing wheel/sdist outputs refuse before a
build; `--sdist-only` requests one source distribution instead of the default
pair. Build hooks execute normally and may install isolated requirements.

Source distributions receive sorted members, fixed timestamps/ownership and
cleared PAX overrides while retaining payloads and modes. An exclusive sibling
temporary file is closed before replacement, including on Windows; unrelated
fixed-name temporary files are preserved. Archive inspection refuses parent,
absolute, Windows drive/backslash and private/build-only paths; tar links and
special members also refuse. Wheel checks require the literal SPDX expression
and the declared console module's presence. Successful builds print filenames,
entry counts, compressed SHA-256 and the selected epoch.

These checks inspect archive declarations. They do not execute console
callables, authenticate artifacts, verify all wheel RECORD/CRC entries, lock
directories, guarantee arbitrary-backend reproducibility or approve publication.
Failures retain backend outputs. See the [release-artifact API](api.md#release-artifact-builder)
for caller-visible interfaces and failure behaviour.

## GitHub Token Format Guard

The security lane runs `python tools/check_github_token_format_readiness.py` in
CI. The guard scans tracked text files and workflow files for brittle GitHub
installation-token assumptions: exact-width `ghs_` regexes, fixed token-length
checks, undersized storage columns, and installation-token endpoint calls that
omit `X-GitHub-Stateless-S2S-Token`.

Treat installation tokens as opaque strings. Code may check for presence,
prefixes needed for routing, or provider errors, but it must not assume a fixed
length or storage width. Test fixtures live under `tests/` and are intentionally
excluded from the repository scan so negative examples remain possible without
making the gate flag itself.

## Public Surface Hygiene

```bash
python tools/check_public_surface_hygiene.py
```

This guard scans tracked outward-facing text files and fails on bare
self-applied promotion terms. It also blocks path-specific leaks where a public
surface would expose internal implementation names, local-host details, or
operational gateway wording, and it rejects public bank or wallet coordinates
on payment surfaces. It also blocks stale public tutorial paths that point
outside the repository's `artifacts/` directory. Private operational records
are excluded; bounded negative language and candidate labels remain allowed
because they do not assert an achieved public claim. Public Markdown and JSON
are also rejected when they expose task lists, prioritisation, internal task
identifiers, or private paths.

Selection comes from Git's index, using NUL-delimited native path spellings so
quoted names and filenames with newlines remain visible. The guard reads current
worktree text, including unstaged changes; it does not read staged blobs,
untracked files or history. Selected missing/non-file paths and invalid UTF-8
payloads are skipped. A pass therefore covers inspected text only.

The pattern scan does not parse Markdown, JSON or code, render pages, verify
scientific claims or certify publication readiness. Markdown fence-looking lines
toggle a simple state that suppresses planning checks only; identifier and
promotion checks still apply inside fences. Findings retain source lines and
native path spellings for the local operator. There is no redaction or coherent
concurrent-file snapshot guarantee.

`--repo` resolves a caller-relative directory. Git subdirectories inspect their
own index-relative scope. The CLI returns 0 for no inspected-text findings, 1
for findings, or 2 for a root, Git enumeration or selected-file read refusal.
Expected operational failures use fixed authored stdout text. Public API
contracts, examples and error types are in the
[native API reference](api.md#public-surface-hygiene).

## Benchmark Producer Inventory

```bash
python tools/check_benchmark_producers.py
python tools/check_benchmark_producers.py --repo path/to/repository
```

The auditor reads the selected repository's producer registry and discovers
the maintained Python/Rust path patterns. Optional `--registry` selects a
caller-relative TOML file. Missing/file roots, unreadable or malformed source
and registry files, unclassified producers and source/documentation custody
findings return 1; success returns 0. The printed count is lexical inventory.
No producer runs, measurements are qualified or output files are written by
this check. The [API contract](api.md#benchmark-producer-registry-audit) describes
source-marker limitations, category semantics and public-command scanning.

## Native Reference Generation

```bash
python tools/generate_native_api_reference.py
python tools/generate_native_api_reference.py --check
```

The first command refreshes the tracked C ABI and Lean declaration reference;
the second compares it without writing. Both read the maintained normative
header and proof source. Function comments must be adjacent and nonempty;
comments for preceding types cannot enter a function contract. This lexical
documentation check does not compile the solver or execute the Lean proof
checker. Qualify source changes with their owning language toolchains as well.

For a selected local corpus, pass `--header path/to/solver.h`,
`--lean path/to/PulsedFSM.lean` and `--output path/to/reference.md` together.
Version-one ABI and the maintained declaration-count requirements still apply.
The write refuses aliases of either input; `--check` leaves missing or stale
output untouched and returns 1. Success returns 0, operational or source-contract
refusal returns 1, and parser usage returns 2. The [API contract](api.md#public-surface-hygiene)
describes the selected-file behaviour and lexical scope in full.

## Changelog Mirror

```bash
python tools/check_changelog_sync.py
```

`CHANGELOG.md` is the authoritative release history. `docs/changelog.md` is the
rendered MkDocs mirror and must stay byte-identical to the root file. The guard
runs in CI, local preflight, pre-commit, and `make lint`.

## Tracked Source Headers

```bash
python tools/check_source_headers.py
```

The source-header gate enforces the repository's seven-line semantic identity
block across owned source, tests, workflows, build files, and commentable
configuration. `tools/source_header_policy.toml` records the reviewed format
families that cannot carry this header without changing legal text, generated
integrity data, serialisation, manuscript rendering, or binary content. A new
tracked format must be classified explicitly; it cannot silently bypass the
gate. The same command runs in CI, local preflight, pre-commit, and
`make lint`.

Policy collections must be TOML arrays of strings; scalar strings, numbers,
booleans, nested arrays and mixed-type members are rejected. Omitted collections
are empty, suffix matching case-folds, and names/exact paths preserve case.
Exemption paths must be canonical relative POSIX spellings and disjoint from
other declared scope. Dataclass construction does not perform these checks;
use the validated loader.

Lean carries seven semantic fields inside a nine-line block; HTML and other
comment syntaxes carry seven physical lines. One leading shebang is accepted.
The gate reads current worktree content for Git-index names and queries HEAD
separately. It writes no files and does not establish an atomic commit snapshot
or filesystem containment. Exit zero means no header/classification findings,
one means findings, and two means a policy, filesystem, decoding or Git error.
`--json` emits the complete result only for zero/one; errors use stderr.
The [API contract](api.md#tracked-source-header-policy-contract) details the
native functions and includes an executable rendering example.

## Competitive Evidence

```bash
python tools/check_competitive_evidence.py
```

The competitive-evidence gate treats
`docs/_data/competitive_evidence.json` as the dated source registry for
`docs/competitive_analysis.md`. Release-backed entries require exact tags and
commit SHAs; papers require stable DOI sources. The public page must carry every
source, state the empty numeric-comparison set when no matched protocol is
admitted, use `not assessed` instead of inferred absence, and exclude ranking
language and private planning markers. Any quantitative row must declare the
same problem, inputs, precision, tolerances, convergence, warm-up, samples,
hardware/load, isolation, failures, and result artifact.

## Python lint scope

```bash
ruff check src/scpn_control/
ruff check --extend-ignore D tests/ tools/ validation/
ruff format --check src/scpn_control/ tests/ tools/ validation/
python tools/check_docstring_debt.py
python tools/check_python_lint_contract.py
```

The lint-contract checker inspects literal declarations in Make, static-governance
CI, pre-commit and local preflight. It requires the package lint command, the
separate test/tool/validation lint command with D excluded, formatting across
all four scopes, the docstring-debt ratchet and the generated-ledger spellcheck
exclusion. Its own pre-commit hook must use the declared entry and file filter,
including both `ci.yml` and `ci-static-governance.yml`.

This is a case- and whitespace-sensitive UTF-8 text check after universal newline
normalisation. Comments or inactive strings can satisfy a required fragment;
a pass does not prove command execution, tool-version alignment or semantic
YAML/Make/Python validity. The checker reads only the four declared surfaces,
follows symlinks and changes no files. Missing/non-file surfaces and expected
read/decoding failures produce authored errors. `--repo` is caller-relative;
the CLI resolves it and returns 2 if resolution fails, 1 for declaration/read
errors, or 0 when fragments match. See the
[native API contract](api.md#python-lint-contract).

The separate public-API docstring gate below retains zero debt across
`src/scpn_control/`. Its explicit all-definition owner list additionally covers
selected tool, validation and test helpers; historical files outside that list
are not claimed to have full private/test documentation. Native docstrings and
actual lint/rendering tests remain separate from declaration inspection.

Documentation builds intentionally constrain MkDocs to `>=1.6.1,<2` and
Material for MkDocs to `>=9.7.7,<10`. Material 9.x does not support MkDocs 2;
the upper bound prevents an unsupported resolver combination until the theme,
plugins, navigation, search, and rendered output have a supported parity path.

## Public API Docstrings

The permanent [private documentation guard](api.md#private-documentation-declaration-guard)
also runs in static-governance CI, local preflight, the always-running local
pre-commit hook, and before the Pages documentation build. Both GitHub jobs fetch
complete history. The ordinary docstring gate enforces all definitions in this
guard and its public test module:

```bash
python tools/check_docs_internal_private.py
```

Keep the exact root private-tree ignore rule after all general negations, and
put the exact `internal/**` line after every negation in MkDocs `exclude_docs`.
The guard reads the actual top-level YAML string without constructing custom
tags, checks literal index paths, and scans all locally available Git history
references. Its local `--skip-history` option reports that history was not
checked. A declaration PASS does not audit plugin output, built-site bytes,
package archives, alternate configurations, or remote publication.

```bash
python tools/run_docstring_gate.py
```

The docstring gate runs ruff's public API pydocstyle rules for classes,
functions, methods, packages, and nested classes. The recorded debt is zero, so
new public APIs without docstrings fail CI, local preflight, and `make lint`.
Docstrings should name the technical contract, units, failure modes, and claim
boundaries where those details matter.

The ordinary command checks all default AST owners before its script-root Ruff
probe and ledger read. `--all-definitions` adds caller-relative files without
replacing defaults. Missing/syntactically invalid owners, malformed Ruff output
and invalid ledgers refuse with status two and fixed authored stderr.

The native probe uses the public `parse_ruff_diagnostics` JSON decoder: its root
must be an array, each entry must contain a string filename and a selected rule
code, and other fields are retained. These shape checks do not authenticate the
producer or establish file existence. The pure `evaluate_ratchet` API also
supports historical count comparisons; no CLI flag disables the checked-in zero
floor.

Ledger snapshots require unique JSON keys, nonnegative integer counts excluding
booleans, module counts summing to `total`, and current rule order when `rules`
is present. Legacy snapshots without `rules` remain readable. Python constructors
are unchecked values; reconstruction and writes enforce the contract. An update
cannot overwrite a malformed existing ledger, including when
`--allow-baseline-increase` is supplied. That flag never disables the zero floor.
Valid writes directly replace UTF-8 text without atomic publication or alias
protection. Native doc presence does not establish semantic or scientific validity.

## Studio Custody Guards

```bash
python tools/check_studio_deploy_key.py
python tools/check_studio_offline_sealing.py
```

`check_studio_deploy_key.py` validates the tracked Studio deploy public key, the
CI rsync deploy workflow, and private deploy-key exclusion.
`check_studio_offline_sealing.py` keeps Studio publication signing custody
offline: workflows, Studio surfaces, docs, and tools must not reference
Hub/Studio sealing or signing private-key secrets, and tracked policy surfaces
must not contain private-key blocks. The guard deliberately allows deploy-only
SSH credentials because they do not sign evidence; sealed evidence keys stay
with the Studio keeper.

---

## Release Process

1. Bump version in `pyproject.toml`, `CITATION.cff`, and `.zenodo.json`
   and run `python tools/check_version_sync.py` to verify release notes,
   README PyPI/Python-version badges, the Pepy all-time downloads badge, and
   local version metadata.
2. Tag and push:

    ```bash
    git tag vX.Y.Z && git push origin vX.Y.Z
    ```

3. CI publishes to PyPI via `publish-pypi.yml`
4. Verify: `pip install scpn-control==X.Y.Z`

---

## Docs

```bash
pip install -e ".[docs]"
mkdocs serve     # preview at http://127.0.0.1:8000
mkdocs build     # static site in site/
```

CI deploys to GitHub Pages on push to `main` via `.github/workflows/docs-pages.yml`.

## JOSS Submission Review

Run `python tools/check_joss_submission.py` before sending the paper to an
external JOSS workflow. The read-only guard checks the canonical
`papers/submissions/001_neuro_symbolic_tokamak_control_software/manuscript.md`
and its `references.bib` bibliography, plus the `docs/joss_paper.md` pointer.
All three UTF-8 files must exist and contain non-whitespace text. Required
editorial markers are case-sensitive substrings after whitespace normalisation.
The first lexical `title:` line must be inside the initial `---`-delimited
front matter and appear in the documentation pointer. The bibliography must
have at least one recognised `@word{key,` entry, no duplicate keys, and entries
for all bracketed manuscript citation keys.

This is a local consistency check. It does not parse the complete YAML/JOSS or
BibTeX schema, validate Markdown structure, scan narrative citations outside
brackets or citations in the documentation pointer, render the PDF, resolve
links, or authenticate scientific evidence. Comments and code fences can contain
lexically matched markers or keys. Missing/blank inputs and consistency findings
return status one; present-file read and UTF-8 decode errors retain their native
exception behaviour. Success does not mean the paper has been submitted or
accepted. Paths come from the resolved script location, independent of caller
working directory; the standalone script has no option parser.

The guard runs in local preflight and CI lint. Its public API and real fixture
CLI checks are documented in the [API reference](api.md#joss-local-editorial-and-citation-guard).

## Rust toolchain declarations

Run `python tools/check_rust_toolchain_contract.py` to compare the current
`rust-toolchain.toml` and seven fixed workflow files with the local pinned
policy. `--root PATH` selects another physical repository; a relative path uses
caller cwd. The guard reads actual YAML action steps and each action's own
`with` inputs. Named steps and aliases work; text inside `run`, comments,
environment values and subsequent steps do not count as toolchain inputs.
It returns one for drift, missing files or malformed TOML/YAML/structure, while
invalid UTF-8 retains its native exception. Help and option errors follow
argparse's zero/two exits.

The public API, exact pins, counts and parsing boundaries are documented in the
[API reference](api.md#rust-toolchain-declaration-contract). Success verifies
declarations, without installing or running Rust or establishing hosted CI,
parity, performance or scientific acceptance. The static-governance capability
job installs the existing hash-pinned `requirements/ci-lint.txt` before policy
checks; local development already declares PyYAML in the development extra.

## How to use this guide in practice

This page defines the engineering path to stable work, not the path for first contact.
Use it after onboarding is complete when you need to:

- reproduce an existing result,
- add a new module behind an existing interface,
- or prepare a release candidate with validation and documentation updates.

Each section is intentionally scoped: setup to make the stack runnable, test and
type gates to keep it safe, and release steps to keep claim and version metadata
aligned.

The docs site includes:

- Full API reference via mkdocstrings, including a complete module index for
  every tracked Python module under `src/scpn_control/`
- Theory page with rendered MathJax equations
- Architecture diagrams via Mermaid
- Notebook gallery with execution instructions
- Changelog, benchmarks, and validation reports

## Practical use and scope

Use this guide for change workflow in `scpn-control` itself.

- Run setup and local checks here before editing core modules.
- Use the workflow before opening implementation tasks that affect CI or packaging.
- Keep claim-boundary and admission checks in lockstep with this guide when production-relevant files change.


## API declaration inventory

```bash
python tools/check_api_contracts.py
python tools/check_api_contracts.py --repo . --print-inventory
```

Make, static-governance CI and local preflight use the same read-only checker.
It compares selected source names and lexical ownership against the local v1
TOML registry, then checks literal renderer fragments. It does not execute
renderers, verify semantic documentation, resolve runtime reexports or prove
independent review. See the [native contract](api.md#api-declaration-contract)
for AST/regex scopes, name digests, source ordering and filesystem limits.

Status 0 means declarations/fragments match; 1 means policy drift; 2 means an
authored inspection refusal. `--repo` is resolved by the CLI, while `--registry`
is relative to the caller's working directory. `--print-inventory` omits registry
and renderer checks. The canonical baseline is never automatically rewritten;
a mismatch requires coherent source ownership and review. The ordinary docstring
gate includes both owning modules and their dedicated public API/CLI tests.


## Dependency advisory audits

The required CI security category audits every committed dependency lock on each
push and pull request to `main`. The Python job checks its installed environment
and uses `tools/export_dependency_audit_lock.py` to export all upstream package
name/version pairs from `uv.lock` and `requirements/ci-*.txt` for a strict
`pip-audit --locked` scan. Optional dependencies and pins for other platforms or
Python versions remain in that inventory. Different locked versions of the same
package are all audited; the exporter does not resolve or install dependencies.
It excludes only this repository's local editable project. Direct source archives
require verified name/version metadata bound to the exact declared SHA-256 in
`tools/dependency_advisory_sources.json`; unknown sources or changed bindings
fail the export. The SC-NeuroCore commit archive declares version 3.16.0 and is
audited separately from the 3.15.0 registry version in `uv.lock`. Local build
suffixes such as PyTorch `+cpu` use the public upstream version for advisory
lookup; the complete locked version remains in the export's per-package tool
metadata. This checks upstream advisories and does not certify vendor build
changes. The generated advisory inventory is not an installation
lock or an artifact-integrity verification.

`studio-web-audit` runs unfiltered `pnpm audit` against the frontend lock, using
the same pinned Node.js and pnpm as the frontend build. The Rust job runs
`cargo audit` for the workspace and `cargo audit --file fuzz/Cargo.lock` for the
fuzz workspace. Audit findings and registry/database failures fail the required
CI gate. The audit-policy tests pin these commands and reject unowned committed
locks; introducing another lock requires adding its audit ownership.

Install the `mast-acquisition` extra in a separate environment from `all`,
`dev`, `mast-data` and `fusion`. Acquisition uses Zarr 3 and NumPy 2; the
older MAST data profiles use Zarr 2, and FUSION requires NumPy below 2. The
`tool.uv.conflicts` declaration preserves these separate resolutions in
`uv.lock`; selecting incompatible extras together is refused. This does not
change an existing installed environment or admit acquired data for training.
