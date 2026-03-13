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
The Rust-accelerated kernel achieves a median step latency of **11.9 µs** (P50) 
on modern hardware. The pure-Python fallback takes approximately 100 µs, 
while the JAX-accelerated solver can take up to 1 ms due to JIT compilation 
and dispatch overhead.

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
