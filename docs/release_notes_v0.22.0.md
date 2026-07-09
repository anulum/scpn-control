# SCPN Control v0.22.0 Release Notes

v0.22.0 is a control-evidence, robustness, and federation release on top of
v0.21.0. It adds Rust fuzzing for parser, numeric-adapter, and FFI surfaces; a
benchmark regression gate; runtime-bound formal safety certificates; SCPN Studio
evidence feeds; and bounded differentiable-scenario evidence. The release keeps
facility, target-hardware, external-code, and production-timing claims
fail-closed.

## What changed

### Robustness and security evidence

- Added libFuzzer targets for reactor-configuration JSON, VMEC-like text import,
  BOUT++ stability-output parsing, capacitor-bank discharge ledgers, and the
  Kuramoto phase kernel.
- Added a seed corpus with SHA-256 inventory and a nightly fuzzing workflow that
  uploads reproducer artefacts while keeping fuzzing out of per-commit CI.
- Fixed a capacitor-bank denial-of-service path by bounding the Rust
  scaling-and-squaring exponent and failing closed on non-finite matrix norms.

### Benchmark and release evidence

- Added a polyglot benchmark suite runner that records p50/p95/p99 latency,
  throughput, backend, runtime provenance, and digest-bound report metadata.
- Added a benchmark regression gate that compares fresh reports against tracked
  baselines and rejects tampered reports, missing metrics, cross-CPU comparisons,
  and threshold-policy violations.
- Added nightly evidence-only benchmark workflow coverage so regression evidence
  stays current without turning local development into a full-suite run.

### Runtime-bound formal safety certificates

- Added `scpn_control.scpn.runtime_safety_certificate`, which binds a holding
  bounded CTL/LTL certificate to the deployable controller identity:
  configuration, Petri-net topology digest, SNN parameters, solver mode, runtime
  target, and timing envelope.
- Added issue, replay, and admission APIs that fail closed unless topology,
  target runtime, timing envelope, proof replay, and formal digest all match on
  the declared stack.
- Added user documentation and API reference coverage for the runtime-bound
  certificate workflow.

### SCPN Studio federation

- Added the CONTROL studio vertical with platform verbs, evidence bundles,
  adapter mappings, a schema-A capability manifest, and a standalone panel feed.
- Added generated `docs/_generated/studio_manifest.json` and studio-web feed
  artefacts so the UI consumes the same contract the Python package emits.
- Aligned the manifest's advertised platform SDK range with the package extra
  and CI-pinned `scpn-studio-platform` dependency policy.
- Added tests for producer conformance, manifest drift, adapters, feed payloads,
  and the studio vertical.

### Bounded differentiable-scenario evidence

- Added the coupled differentiable scenario facade, validation scripts, benchmark
  reports, and documentation for analytic Solov'ev-form equilibrium parameters
  coupled to the four-channel differentiable transport rollout.
- The persisted report remains bounded local evidence: external equilibrium or
  integrated-modelling evidence is still required before full-fidelity scenario
  claims can be admitted.

## Evidence boundary

This release expands release evidence and package integration surfaces. It does
not certify plant deployment, PREEMPT_RT operation, external-code equivalence,
P-EFIT predictive validity, TORAX parity, or facility PCS readiness. Fuzzing,
local timing, studio feeds, and analytic differentiable-scenario reports are
review artefacts until their strict admission gates accept matching external or
target-hardware evidence.

## Release checklist

Do not treat `v0.22.0` as published until all items are true:

- Version metadata and documentation updates are committed.
- CI, CodeQL, Pre-commit, OpenSSF Scorecard, Docs Pages, release, and publish
  workflows are green for the `v0.22.0` commit/tag.
- Pull requests and security alerts remain clear.
- Failed or cancelled Actions/deployment records are deleted only when safe and
  only after replacement evidence is green.
- The GitHub release is created from the `v0.22.0` tag.
- PyPI shows `scpn-control==0.22.0` with both wheel and source distribution.

## Practical use and scope

Use this note to verify what changed in the 0.22.0 line.

- Use it as a checkpoint before upgrading local scripts, dashboards, or
  benchmark baselines.
- Confirm any referenced claims with current validation and admission surfaces.
- Keep this page as historical context for reproducibility evidence.
