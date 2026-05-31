# SCPN Z3 Formal Verification Report

- Status: `pass`
- Backend: `z3`
- Max depth: `2`
- Scope: bounded SMT evidence for compiled Petri-net control logic.
- Claim boundary: not hardware timing evidence, PCS certification, or unbounded liveness proof.

## Safety

- Holds: `True`
- Solver status: `unsat`

## Temporal

- Holds: `True`
- Solver status: `unsat`
- Checked specs: `move_eventually_fires, move_marks_sink, exclusive_source_sink`
