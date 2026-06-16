<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Control — Onboarding guide
-->

# Onboarding

This guide explains how to approach SCPN Control without needing to read the
entire codebase first.

## Mental model

Think of SCPN Control as three layers:

| Layer | What it answers |
| --- | --- |
| Controller logic | What should the controller do, and what invariants must never be violated? |
| Physics and replay facades | Which equilibrium, transport, disruption, and digital-twin signals feed the controller? |
| Admission evidence | Which artefacts prove that a result is reproducible and allowed to be claimed? |

A new workflow should normally touch all three layers. A controller without an
evidence gate is not ready for promotion. A physics result without a controller
contract is not yet a control feature. A benchmark without hardware and claim
metadata is only a local timing observation.

## Practical path for a first contribution

The fastest path to useful impact is to follow the evidence ladder in order:

1. Run one tutorial so the team can verify core semantics from the same command
   sequence.
2. Confirm reproducibility metadata in the report surface (`json` + checksum +
   manifest entries).
3. Choose one controller and one validator and keep the workflow inside one
   bounded scope first (for example one machine configuration and one validation
   mode).
4. Only after admissible evidence appears in the strict validator, open the next
   layer (additional transport modes, alternate formal mode, or new physics
   surface).

This sequence is intentional. It prevents spending cycles on fast iterations that
cannot yet be claimed, and it ensures each code change lands with a visible
evidence transition.

## Why this onboarding order matters to teams

SCPN Control is built as a claim-sensitive system. Every module can be run and
every report can be generated, but not every output can be used for the same
decision.

Before making a code contribution, teams are expected to keep three facts
explicit:

1. what changed,
2. what evidence is available now,
3. what decision this evidence can support.

That order is why the onboarding path starts with a tutorial and ends with a
validator run. It is also why local benchmark numbers are retained separately from
deployment-admissible metrics.

You can use this checklist before merging:

- **Scope**: controller path, benchmark mode, and formal mode are all set.
- **Evidence**: report has matching metadata (timestamps, manifests, checksums, and
  claim boundary).
- **Next action**: either close with bounded evidence, or keep in local iteration
  mode when external validation remains unresolved.

## Roles and expected next evidence

For a team lead deciding whether to accept a change quickly:

- **Controllers and control engineers** should confirm API behavior, then run the
  native handoff comparison and at least one `validate` pass.
- **Physics contributors** should confirm source data and traceability entries before
  touching solver parameters.
- **Safety reviewers** should check whether the same change altered any
  admissibility condition, and ensure the change is mirrored in the relevant
  validator report.
- **Partners and investors** should require the bounded claim level and the missing
  blockers list before the result is quoted externally.

## Install paths

```bash
pip install scpn-control
pip install "scpn-control[ws,formal,jax]"
```

Use editable install for development:

```bash
git clone https://github.com/anulum/scpn-control.git
cd scpn-control
pip install -e ".[dev,docs,formal,jax,ws]"
```

## First hour checklist

1. Run `scpn-control demo --steps 1000` to confirm the package is importable.
2. Read [Production Readiness](production_readiness.md) before interpreting any
   benchmark or validation result.
3. Run one tutorial from [Tutorials](tutorials.md) and one notebook from
   [Notebook Gallery](notebooks.md).
4. Inspect [API Reference](api.md) for the surface you want to use.
5. If you want to make a claim, find the matching validator in
   [Validation and QA](validation.md) before writing code.

## First day by role

| Role | Practical path | What to avoid |
| --- | --- | --- |
| Control researcher | Run Tutorial 01, inspect `scpn_control.control`, then persist a bounded validation report | Do not cite local plots as facility evidence |
| Physics modeller | Start with traceability and validation pages before solver-facing examples | Do not duplicate solver kernels already owned by SCPN Fusion Core |
| Safety reviewer | Read production readiness, formal verification API, and certificate bundle validation | Do not treat bounded proofs as facility certification |
| Data collaborator | Read public-data acquisition, MAST EFM evidence, and compute financing pages | Do not commit large numeric payloads to git |
| Funder or partner | Read use cases, market value, production readiness, and compute validation financing | Do not interpret blocked gates as failures; they are the work plan |

## Evidence ladder

1. **Demonstration**: tutorial, notebook, or example output.
2. **Bounded repository evidence**: deterministic report with declared assumptions.
3. **Reference admission**: strict validator admits documented public or measured reference artefacts.
4. **External-code or facility validation**: matched cases, units, tolerances, and provenance from external tools or shots.
5. **Deployment qualification**: target hardware, interlocks, safety review, operator procedures, and facility sign-off.

Most new work starts at levels 1 or 2. The documentation should make that clear
before any stronger language is used.

## Choosing a workflow

| Goal | Start here |
| --- | --- |
| Learn the controller API | `examples/tutorial_01_closed_loop_control.py` |
| Explore differentiable physics | `examples/tutorial_02_jax_autodiff.py` |
| Evaluate safety certificates | `scpn_control.scpn.formal_verification` in [API Reference](api.md) |
| Build a replay or validation report | [Validation and QA](validation.md) |
| Understand deployment limits | [Production Readiness](production_readiness.md) |
| Discuss funding or collaboration | [Compute Validation Funding](compute_validation_financing.md) |

## Smallest Evidence Workflows

These are the shortest existing entry points for checking a workflow boundary.
They do not upgrade any facility, hardware, or external-code claim by
themselves; the corresponding strict validator and traceability entry decide the
claim level.

| Workflow | Smallest current entry point | Evidence check | Boundary |
| --- | --- | --- | --- |
| Control replay | `PYTHONPATH=src python -m scpn_control.scpn.geometry_neutral_replay --steps 12 --strict --output-json artifacts/geometry_neutral_replay.json --output-md artifacts/geometry_neutral_replay.md` | `pytest tests/test_geometry_neutral_replay.py tests/test_geometry_neutral_replay_v1_1.py` | API/module entry point for bounded synthetic replay; device claims still need non-synthetic magnetic provenance and external artefact digests. |
| Physics validation artefact | `scpn-control validate-physics-traceability --json-out` | `pytest tests/test_physics_traceability.py tests/test_generate_physics_traceability_report.py` | Confirms registry structure and claim blocking; it does not create the missing external artefacts. |
| Formal-verification report | `python validation/validate_scpn_z3_formal.py` and `scpn-control validate --json-out` | `pytest tests/test_scpn_formal_verification.py tests/test_native_formal_certificate_evidence.py` | Bounded Petri-net and native formal evidence only; facility certification and hardware timing remain separate gates. |
| Benchmark gate | `PYTHONPATH=src python tools/run_benchmark_suite.py --json-out artifacts/benchmarks/report.json` then `python tools/benchmark_regression_gate.py --report artifacts/benchmarks/report.json --baseline benchmarks/baselines/capacitor_bank.json` | `pytest tests/test_benchmark_suite_runner.py tests/test_benchmark_regression_gate.py tests/test_benchmark_regression_gates.py` | Absolute latency comparison is valid only against a baseline from the same CPU class and recorded benchmark context. |
| Real-data manifest | `scpn-control validate-data-manifests --json-out` or `scpn-control validate-manifest <manifest.json> --verify-artifact --json-out` | `pytest tests/test_validate_data_manifests.py tests/test_real_diiid_shots.py` | Repository manifest admission; live MDSplus acquisition needs authorised access, policy metadata, and `--require-real-acquisition` when promoting facility artefacts. |

When a workflow writes under `artifacts/`, treat the output as a local working
artefact until it is moved into an admitted validation report path with stable
checksums and a traceability entry.

## What not to assume

- A fast local benchmark is not a full control-cycle guarantee.
- A bounded proof is not facility certification.
- A public dataset conversion is not predictive EFIT or P-EFIT validation until
  the strict admission gate passes.
- A simulator bridge is not a TRANSP, TSC, CODAC, or EPICS acceptance test.
- A neural surrogate is not a substitute for external-code or measured-shot
  evidence unless the corresponding validator admits that claim.

## Native runtime evidence path

For the current release line, use the native runtime evidence path when you
need to discuss timing or formal-runtime coverage:

1. Run the normal tutorials first so controller semantics are clear.
2. Use `scripts/benchmark_native_handoff.py` to compare Python orchestration
   with fused Rust/PyO3 execution at the same campaign boundary.
3. Use `scripts/benchmark_native_formal_modes.py` to choose between
   `async_drop`, `sync_stride`, and `aot_certificate` formal modes.
4. Treat `async_drop` as proof sampling, `sync_stride` as ground-truth bounded
   proof timing, and `aot_certificate` as the hot-path monitor admitted by its
   certificate digest.
5. Label workstation runs as local-regression evidence unless the benchmark
   artefact records isolated cores, host load, governor/frequency context,
   runtime versions, and concurrent-job status.

## Where this onboarding path ends

This page should end with a concrete workflow, not just broad understanding.

The onboarding sequence is considered complete when all three are true:

- One tutorial run completes without runtime dependency errors.
- One benchmark command executes from a clean environment context.
- One matching admissibility report is saved with clear host metadata and command line trace.

At that point, move to validation pages before adding a new marketing claim or claiming hardware readiness.

## Practical use and scope

Use this guide as the entry lane for new operators and campaign owners.

- Read it first if you are joining the project and need a repeatable setup and first-run path.
- Move to `docs/tutorials.md` after environment checks, then to `docs/physics_traceability.md` for claim boundaries.
- For production discussions, follow this with `docs/production_readiness.md` before running facility-admissibility conversations.
