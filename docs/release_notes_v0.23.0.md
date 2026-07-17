# SCPN Control v0.23.0 Release Notes

v0.23.0 is a minor release for the v0.2x control-evidence line. It carries a
large-scale modularisation of several god-modules, an integrity-and-honesty
hardening pass over the shipped control path, and continued security, studio,
and documentation infrastructure work. Because some public import paths moved
and one verification backend label was withdrawn, this release contains
behaviour-affecting breaking changes and is a minor (not patch) bump under 0.x
SemVer.

## What changed

### Breaking changes

- `FormalPetriNetVerifier` and the `scpn.formal_safety_certificate` validators no
  longer accept `backend="z3"`. The explicit-state reachability engine performs
  exact enumeration, not SMT, so it now raises `ValueError` on `backend="z3"` and
  never stamps an unearned `z3` backend label. Use `Z3BoundedModelChecker` in
  `scpn_control.scpn.z3_model_checking` for z3-backed evidence. `auto` and
  `explicit-state` are unchanged.
- Several symbols moved to new single-responsibility modules and are no longer
  re-exported from their former homes:
  - transport-model tuning entry points → `control.nmpc_transport_tuning`
    (from `control.nmpc_controller`);
  - z3 formal-report I/O and `Z3FormalVerificationReport` → `scpn.z3_formal_report`
    (from `scpn.z3_model_checking`);
  - free-boundary claim-evidence surface → `control.free_boundary_tracking_claims`
    (from `control.free_boundary_tracking`);
  - safety-certificate I/O and admission → `scpn.formal_safety_certificate`
    (from `scpn.formal_verification`).
  The controller, verifier, and runtime engines themselves are unchanged; update
  imports to the new module paths.

### Integrity and honesty hardening

- Un-rigged the SCPN/PID/MPC benchmark: removed the MPC's future-disturbance
  foresight and the pass-by-construction gate, so the comparison uses an honest
  offset-free persistence disturbance model. The published RMSE figures now
  reflect the fair model (PID `0.121`, MPC `0.047`, SCPN `0.050`; SCPN/MPC RMSE
  ratio `1.07`).
- Failed closed across the shipped control path where a bare `np.clip`, a
  magnitude floor, or a skip-if-present loop previously admitted non-finite or
  sign-inverting inputs silently: the actuator decoder rejects non-finite inputs
  and negative `abs_max`/slew rates; the feature-error kernel rejects a negative
  axis scale; the controller marking setter and the live passthrough-observation
  injection reject non-finite values; the physics safety-invariant monitor treats
  a missing safety channel as a critical violation; and the G-EQDSK parser rejects
  any non-finite value at parse time.

### Modularisation

- Split the transport-solver, CLI, NMPC, formal-verification, free-boundary, and
  z3-report god-modules into single-responsibility modules. Behaviour is
  byte-identical except for the moved import paths noted above; the published
  physics formulae and control loops are unchanged.

### Security, studio, and infrastructure

- Hardened the phase-stream WebSocket `Origin` allowlist and malformed-frame
  handling, and ensured the broadcast tick loop is always cancelled on shutdown.
- Continued SCPN Studio manifest/federation, deploy-key, and offline-sealing
  guards, docstring and public-surface hygiene gates, and architecture decision
  records.

See `CHANGELOG.md` for the complete, itemised list.

## Evidence boundary

This release does not certify facility deployment, PREEMPT_RT operation,
external-code equivalence, P-EFIT predictive validity, TORAX parity, or plant
PCS readiness. The release evidence remains a bounded software, CI, dependency,
and local validation checkpoint, and the strict admission gates keep facility,
measured-shot, target-hardware, and external-code claims blocked until matching
evidence exists.

## Release checklist

Do not treat `v0.23.0` as published until all items are true:

- Version metadata and documentation updates are committed.
- CI, CodeQL, Pre-commit, OpenSSF Scorecard, benchmark, and release workflows
  are green for the `v0.23.0` commit/tag.
- Pull requests and security alerts remain clear.
- The GitHub release is created from the `v0.23.0` tag.
- PyPI shows `scpn-control==0.23.0` with both wheel and source distribution.

## Practical use and scope

Use this note to verify what changed in the 0.23.0 line.

- Upgrade from `0.22.1` for the modularised import surface and the fail-closed
  control-path hardening; update any imports that referenced the moved symbols
  or passed `backend="z3"`.
- Keep broader facility claims blocked unless their strict admission gates accept
  matching external, measured-shot, target-hardware, and review evidence.
