# SCPN Z3 Formal Verification Report

- Schema: `scpn-control.z3-formal-report.v2`
- Status: `pass`
- Backend: `z3`
- Solver: `z3-solver 4.16.0`
- Max depth: `2`
- Payload SHA-256: `62be9bbeffa46f509e8c6e0ba144eca743aa8e9ef336226ebfae4e8e54c11bff`
- Scope: bounded SMT evidence for compiled Petri-net control logic.
- Claim boundary: not hardware timing evidence, PCS certification, or unbounded liveness proof.

## Safety

- Holds: `True`
- Solver status: `unsat`

## Temporal

- Holds: `True`
- Aggregate solver status: `mixed`
- Checked specs: `move_eventually_fires, move_marks_sink, exclusive_source_sink`
