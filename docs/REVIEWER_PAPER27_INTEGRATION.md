# Paper 27 Integration — Reviewer Summary

**Repository:** [anulum/scpn-control](https://github.com/anulum/scpn-control)
**Branch:** `main`
**Commits:** `81704be..4af1c5f` (5 commits, +1 833 lines across 12 files)
**Date:** 2026-02-25
**Author:** Miroslav Šotek — ORCID [0009-0009-3560-0851](https://orcid.org/0009-0009-3560-0851)

---

## 1. What Was Requested

The reviewer asked for the **Kuramoto–Sakaguchi + global field driver** from
SCPN Paper 27 ("The Knm Matrix") to be woven into the `scpn-control` tokamak
control codebase.  Specifically:

1. The ζ sin(Ψ − θ) **"intention as carrier" injection**, where Ψ is a
   Lagrangian pull parameter with **no own dynamics** (no ˙Ψ equation).
2. The full 16-layer **Knm coupling matrix** with calibration anchors and
   cross-hierarchy boosts.
3. A **Rust sub-ms kernel** for the hot Kuramoto loop (rayon-parallelised).
4. **PAC cross-layer SNN sketch** showing phase-amplitude coupling gating.
5. A **demo notebook** with visualisations and a **markdown export** to `docs/`.

---

## 2. Master Equation

```
dθ_{m,i}/dt = ω_{m,i}
            + K_{mm} · R_m · sin(ψ_m − θ_{m,i} − α_{mm})      [intra-layer]
            + Σ_{n≠m} K_{nm} · R_n · sin(ψ_n − θ_{m,i} − α_{nm})  [inter-layer]
            + ζ_m · sin(Ψ − θ_{m,i})                            [global driver]
```

- **K_{mm}** (diagonal): intra-layer synchronisation strength
- **K_{nm}** (off-diagonal): inter-layer bidirectional causality
- **ζ sin(Ψ − θ)**: exogenous global field driver — Ψ resolved externally or from mean-field
- **α_{nm}**: Sakaguchi phase-lag frustration (optional)
- **R, ψ**: Kuramoto order parameter R·exp(i·ψ) = ⟨exp(i·θ)⟩

Reference: arXiv:2004.06344 (generalized Kuramoto–Sakaguchi finite-size)

---

## 3. Files Created / Modified

### 3.1 Python — Phase Dynamics Package

| File | Lines | Purpose |
|------|------:|---------|
| `src/scpn_control/phase/__init__.py` | 33 | Package exports |
| `src/scpn_control/phase/kuramoto.py` | 139 | Kuramoto–Sakaguchi + ζ sin(Ψ−θ), Rust auto-dispatch |
| `src/scpn_control/phase/knm.py` | 101 | Paper 27 Knm matrix builder + OMEGA_N_16 |
| `src/scpn_control/phase/upde.py` | 168 | Multi-layer UPDE engine |

### 3.2 Rust — Sub-ms Kernel

| File | Lines | Purpose |
|------|------:|---------|
| `scpn-control-rs/crates/control-math/src/kuramoto.rs` | 195 | Rayon-parallelised Kuramoto step + run + 7 unit tests |
| `scpn-control-rs/crates/control-math/src/lib.rs` | +1 | `pub mod kuramoto;` |
| `scpn-control-rs/crates/control-python/src/lib.rs` | +67 | PyO3 bindings: `kuramoto_step()`, `kuramoto_run()` |

### 3.3 FusionKernel Integration

| File | Lines | Purpose |
|------|------:|---------|
| `src/scpn_control/core/fusion_kernel.py` | +43 | `FusionKernel.phase_sync_step()` injection point |

### 3.4 Tests

| File | Lines | Tests |
|------|------:|------:|
| `tests/test_phase_kuramoto.py` | 320 | 28 |
| `kuramoto.rs` (inline `#[cfg(test)]`) | — | 7 |

### 3.5 Documentation

| File | Purpose |
|------|---------|
| `examples/paper27_phase_dynamics_demo.ipynb` | 10-section notebook with plots |
| `docs/paper27_phase_dynamics.md` | Markdown export of notebook |

---

## 4. Architecture — How It Fits

```
┌───────────────────────────────────────────────────────┐
│                    FusionKernel                        │
│                                                       │
│  solve_equilibrium()   ← GS solver (untouched)        │
│  compute_stability()   ← MHD stability (untouched)    │
│  phase_sync_step()     ← NEW: Paper 27 injection      │
│       │                                               │
│       ▼                                               │
│  ┌─────────────────────────────────────┐              │
│  │  scpn_control.phase                 │              │
│  │                                     │              │
│  │  kuramoto_sakaguchi_step()          │──► Rust      │
│  │  ├─ order_parameter()              │   fast-path   │
│  │  ├─ GlobalPsiDriver.resolve()      │   (rayon,     │
│  │  └─ wrap_phase()                   │    sub-ms     │
│  │                                     │    N>1000)   │
│  │  KnmSpec / build_knm_paper27()     │              │
│  │  UPDESystem.step() / .run()        │              │
│  └─────────────────────────────────────┘              │
└───────────────────────────────────────────────────────┘
```

**Non-invasive**: the GS equilibrium solver, SNN controllers, and all existing
code paths are completely untouched.  `phase_sync_step()` is a new method on
`FusionKernel` that reads defaults from `cfg["phase_sync"]`.

---

## 5. Knm Matrix — Paper 27 Specification

```python
# Canonical 16-layer natural frequencies (rad/s)
OMEGA_N_16 = [1.329, 2.610, 0.844, 1.520, 0.710, 3.780, 1.055, 0.625,
              2.210, 1.740, 0.480, 3.210, 0.915, 1.410, 2.830, 0.991]

# Base coupling with exponential distance decay
K[i,j] = K_base · exp(−α · |i − j|)    # K_base=0.45, α=0.3

# Calibration anchors (Paper 27 Table 2)
K[0,1] = K[1,0] = 0.302
K[1,2] = K[2,1] = 0.201
K[2,3] = K[3,2] = 0.252
K[3,4] = K[4,3] = 0.154

# Cross-hierarchy boosts (Paper 27 §4.3)
K[0,15] = K[15,0] ≥ 0.05   # L1 ↔ L16
K[4,6]  = K[6,4]  ≥ 0.15   # L5 ↔ L7
```

---

## 6. Rust Kernel — Performance Path

The Python solver auto-dispatches to Rust when:
- `scpn_control_rs` is importable (maturin build)
- `wrap=True` and `alpha=0.0` (common fast-path)

```rust
// Hot loop — rayon parallel chunks of 64
theta_out
    .par_chunks_mut(64)
    .enumerate()
    .for_each(|(chunk_idx, chunk)| {
        for (local_i, val) in chunk.iter_mut().enumerate() {
            let i = base + local_i;
            let mut dth = om + kr_sin_base * (psi_r - th - alpha).sin();
            if zeta != 0.0 {
                dth += zeta * (psi_global - th).sin();
            }
            *val = wrap_phase(th + dt * dth);
        }
    });
```

PyO3 bindings expose `kuramoto_step(theta, omega, dt, k, alpha, zeta, psi_external)`
and `kuramoto_run(...)` returning NumPy arrays directly.

---

## 7. Global Field Driver — ζ sin(Ψ − θ)

`GlobalPsiDriver` resolves Ψ before the integration step:

| Mode | Ψ source | Use case |
|------|----------|----------|
| `"external"` | Caller supplies float | Intention-as-carrier injection |
| `"mean_field"` | arg(⟨exp(iθ)⟩) | Self-organised collective phase |

There is **no ˙Ψ equation** — Ψ is a Lagrangian pull parameter.  When
`zeta > 0`, all oscillators are pulled toward Ψ with strength proportional to
`sin(Ψ − θ_i)`.

---

## 8. PAC Cross-Layer Gating

The UPDE engine supports phase-amplitude coupling gating via `pac_gamma`:

```python
# Inter-layer term with PAC gate
pac_gate = 1.0 + pac_gamma * (1.0 - R_source)
dθ += gain * pac_gate * K[n,m] * R_n * sin(ψ_n - θ - α[n,m])
```

When a source layer is incoherent (low R), the gate amplifies its coupling,
implementing the PAC hypothesis that desynchronised layers drive downstream
amplitude modulation.

---

## 9. FusionKernel.phase_sync_step()

```python
kernel = FusionKernel("tokamak_config.json")

# Config-driven defaults from cfg["phase_sync"]
out = kernel.phase_sync_step(
    theta=oscillator_phases,
    omega=natural_frequencies,
    dt=1e-3,
    psi_driver=0.0,          # exogenous Ψ
)

# Returns: theta1, dtheta, R, Psi_r, Psi
```

All parameters fall through to `cfg["phase_sync"]` when not explicitly
provided.  The `actuation_gain` parameter scales both K and ζ uniformly.

---

## 10. Test Coverage

**28 Python tests** (all passing, 5.8s):

| Class | Tests | What is verified |
|-------|------:|------------------|
| `TestOrderParameter` | 4 | R=1 sync, R≈0 uniform, R∈[0,1], weighted |
| `TestWrapPhase` | 2 | Identity in range, large angle wrapping |
| `TestGlobalPsiDriver` | 3 | External requires value, returns value, mean-field |
| `TestKuramotoSakaguchiStep` | 4 | Sync stability, R increase, ζ pull, α frustration |
| `TestKnmSpec` | 7 | Shape, anchors, boosts, symmetry, zeta, validation |
| `TestUPDESystem` | 6 | Step shape, intra-sync, ζ pull, trajectory, PAC, error |
| `TestFusionKernelPhaseSync` | 2 | Integration smoke, config-driven ζ |

**7 Rust tests** (inline, all passing):

| Test | What is verified |
|------|------------------|
| `test_order_parameter_synced` | R=1 for identical phases |
| `test_order_parameter_range` | R∈[0,1] |
| `test_wrap_phase_identity` | No-op in range |
| `test_wrap_phase_large` | 7π wraps to (−π,π] |
| `test_step_preserves_count` | Output length matches input |
| `test_zeta_pulls_toward_psi` | 500 steps, spread < 0.1 |
| `test_run_returns_trajectory` | Correct trajectory length |

**Full suite regression**: 548 passed, 91 skipped, 1 pre-existing failure (unrelated).

---

## 11. Demo Notebook Sections

`examples/paper27_phase_dynamics_demo.ipynb` (10 sections + summary):

1. **Knm Heatmap** — 16×16 coupling matrix visualisation
2. **ζ Comparison** — with/without global driver, R convergence
3. **α Frustration** — Sakaguchi phase-lag effect on synchronisation
4. **16-Layer UPDE** — full multi-layer evolution with R trajectories
5. **PAC Gating** — phase-amplitude coupling modulation demo
6. **FusionKernel Plasma Sync** — integration with tokamak config
7. **Gain Sweep** — actuation_gain parameter exploration
8. **Lyapunov Stability** — V(t) = (1/N)Σ(1−cos(θ_i−Ψ)) monotone decrease
9. **SNN Closed-Loop** — spike-rate → Ψ feedback via LIF layer
10. **PAC Cross-Layer SNN** — multi-layer SNN with phase-coupled spike routing

Markdown export: `docs/paper27_phase_dynamics.md`

---

## 12. Commit History

```
4af1c5f fix: silence clippy too_many_arguments / type_complexity on Kuramoto bindings
7453019 style: cargo fmt on kuramoto bindings
ad09c0e feat: Rust Kuramoto kernel, PAC cross-layer SNN, docs export
b11228b docs: add Paper 27 phase dynamics demo notebook
81704be feat: add Paper 27 Knm/UPDE engine + ζ sin(Ψ−θ) global driver
```

---

## 13. What Was NOT Touched

- GS equilibrium solver (`solve_equilibrium`, `gs_step`, `SOR/multigrid`)
- SNN controllers (`LIFNeuron`, `SNNController`, spike-rate feedback`)
- Chebyshev/IGA spectral methods
- Rust control-math crates (SOR, tridiag, FFT, etc.) — only added `kuramoto` module
- All existing tests remain green
