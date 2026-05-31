# SCPN Z3 Formal Verification Report

- Schema: `scpn-control.z3-formal-report.v1`
- Status: `pass`
- Backend: `z3`
- Solver: `z3-solver 4.16.0`
- Max depth: `2`
- Payload SHA-256: `ca5e6eacc411dc4a8c97fe77511d315f639393ecca0cc5a422cde127ced96826`
- Scope: bounded SMT evidence for compiled Petri-net control logic.
- Claim boundary: not hardware timing evidence, PCS certification, or unbounded liveness proof.

## Safety

- Holds: `True`
- Solver status: `unsat`

## Temporal

- Holds: `True`
- Solver status: `unsat`
- Checked specs: `marking_bounds, move_eventually_fires, move_marks_sink, exclusive_source_sink`
