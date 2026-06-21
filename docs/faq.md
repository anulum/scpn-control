<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Frequently Asked Questions (FAQ)

## Technical & Physics

### 1. The Grad-Shafranov (GS) solver is not converging. What should I do?
First, check the `relaxation_factor` (omega) in your configuration. Values between 1.0
and 1.95 are typical for SOR; if it diverges, reduce it towards 1.0 (Gauss-Seidel).
Also, ensure your boundary conditions (coil currents) are physically consistent
with the target plasma current. For complex shapes, increase `max_iterations`
to 5000 or more.

### 2. How accurate is the neural transport surrogate?
The surrogate is trained on the QLKNN-10D dataset and achieves millisecond-scale
predictions comparable to the original QuaLiKiz simulation within a 2-sigma
uncertainty band. It is validated for core transport in standard tokamak
configurations but has not been validated for stellarators or unconventional
magnetic geometries.

### 3. Can I use `scpn-control` for ITER scenarios?
Yes, the framework includes an ITER baseline configuration (`iter_config.json`)
and is suitable for core transport and equilibrium control studies. However,
it does not currently model 3D effects, edge/SOL physics, or complex
pedestal-top stability (ELMs) beyond simple heuristic models.

### 4. What is the real-time latency of the control loop?
The native (Rust) integrated control cycle has a median latency of **5.05 µs**
(P50) on the CI runner (AMD EPYC 7763) and **2.85 µs** on the local workstation
(Intel i5-11600K); the Python-orchestrated path is 9.42 µs (CI) / 4.36 µs (local).
The isolated Rust SNN controller steps in 0.92 µs (P50, CI). The JAX-accelerated
solver can take up to 1 ms due to JIT compilation and dispatch overhead. See
[benchmarks](benchmarks.md) for the full per-controller and per-backend tables.

## Architecture & Development

### 5. How do I add a new controller?
Implement the standard controller interface (a `step(error, dt)` or similar method)
and register it in the `CONTROLLERS` registry within `tokamak_flight_sim.py` or
`controller_comparison.py`. See `scpn_control.control.h_infinity_controller`
for a reference implementation.

### 6. Why use Stochastic Petri Nets (SPN) instead of direct neural networks?
SPNs provide a formal, graph-based representation of control logic that
allows for rigorous verification of safety properties (e.g., boundedness,
deadlock detection) via the contract system. The SPN-to-SNN compiler
then enables execution on neuromorphic or real-time hardware while
maintaining these formal guarantees.

### 7. Is GPU acceleration worth it for this framework?
GPU acceleration via JAX is highly effective for large-scale ensemble runs
(`jax.vmap`) or gradient-based optimization (`jax.grad`). However, for
single-shot real-time control, the overhead of CPU-to-GPU memory transfer
and JAX kernel dispatch often makes the Rust CPU backend faster.

### 8. How should I decide which result to quote outside the repository?

Use this rule set:

- Use local benchmark output only for engineering direction and short-term tuning.
- Use validator-admitted reports for release-bound claims.
- Use traceability entries to identify claim level before any external statement.
- Use timing claims only from production-admitted reports with execution context
  preserved.

If a question asks about deployment suitability, cross-check the answer against
`production_readiness.md` and the matching validation manifest before creating
slides, press material, or funding claims.

## How to read FAQ answers safely

The FAQ is a usage-level layer. Each answer is tied to a current implementation
surface and must be interpreted with the same admission context as the project:

- local-repeatability and benchmark reproducibility are distinct,
- raw timing lines depend on host load and scheduler context,
- physics and transport claims require matching validation artifacts for the same
  intended domain.

If a claim changes after a version bump or a formal-mode change, treat this page
as a starting point and verify through the linked validator and report outputs.

## Practical use and scope

Use this page as operational first aid for common implementation and configuration issues.

- Start with the troubleshooting path that matches the failure symptom.
- Use the linked settings in each answer before changing solver internals.
- Escalate unresolved questions to validation workflows and preserve traceability in your session notes.
