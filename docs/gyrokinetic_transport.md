# Gyrokinetic Transport — Three-Path Architecture

scpn-control implements all three gyrokinetic transport paths in a single
codebase. No competing code (TORAX, FUSE, DREAM, FreeGS) has this combination.

## Architecture Overview

```
                    ┌─────────────────────────────┐
                    │   integrated_transport_solver │
                    │   transport_model = ?         │
                    └──────────┬──────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
       "gyro_bohm"     "external_gk"    "gyrokinetic"
       (fast fallback)  (Path A)        (analytic QL)
                               │
                    ┌──────────┴──────────┐
                    │  GKSolverBase ABC   │
                    └──────────┬──────────┘
              ┌────────┬───────┼───────┬────────┐
           TGLF     GENE    GS2   CGYRO   QuaLiKiz

  Path B (native):  gk_eigenvalue → gk_quasilinear → GKOutput
  Path C (hybrid):  surrogate → OOD check → GK spot-check → correction
```

## Path A: External GK Code Coupling

Five first-principles gyrokinetic codes interfaced via subprocess:

| Solver | Module | Input Format | Reference |
|--------|--------|-------------|-----------|
| TGLF | `gk_tglf.py` | Fortran namelist | Staebler et al., Phys. Plasmas 14 (2007) 055909 |
| GENE | `gk_gene.py` | `parameters_gene` namelist | Jenko et al., Phys. Plasmas 7 (2000) 1904 |
| GS2 | `gk_gs2.py` | `gs2_input` namelist + NetCDF | Kotschenreuther et al., CPC 88 (1995) 128 |
| CGYRO | `gk_cgyro.py` | `input.cgyro` | Candy & Waltz, J. Comp. Phys. 186 (2003) 545 |
| QuaLiKiz | `gk_qualikiz.py` | Python API | Bourdelle et al., Phys. Plasmas 14 (2007) 112501 |

All solvers implement the `GKSolverBase` abstract interface:

```python
from scpn_control.core.gk_interface import GKSolverBase, GKLocalParams, GKOutput

class TGLFSolver(GKSolverBase):
    def prepare_input(self, params: GKLocalParams) -> Path: ...
    def run(self, input_path: Path, *, timeout_s: float = 30.0) -> GKOutput: ...
    def is_available(self) -> bool: ...
```

### Usage

```python
from scpn_control.core.integrated_transport_solver import TransportSolver

ts = TransportSolver(config_path, transport_model="external_gk")
ts.neoclassical_params = neo_params
ts.evolve_profiles(dt=0.01, P_aux=20.0)
```

Falls back to gyro-Bohm if the solver binary is unavailable or returns
unconverged results.

## Path B: Native Linear GK Eigenvalue Solver

Self-contained flux-tube linear GK solver — no external binaries needed.

### Physics

Solves the linearised electrostatic gyrokinetic equation in ballooning
representation:

```
(ω - ω_*T) g_k = (ω - ω_D) J_0 φ_k + i v_∥ ∇_∥(g_k) + C[g_k]
```

| Module | Physics |
|--------|---------|
| `gk_geometry.py` | Miller flux-tube parameterisation (R, Z, |B|, Jacobian, curvature) |
| `gk_species.py` | Per-species distribution, Gauss-Legendre (E, λ) grid, Sugama collision |
| `gk_eigenvalue.py` | Response-matrix eigenvalue problem → growth rates γ(k_y) |
| `gk_quasilinear.py` | Mixing-length saturation → chi_i, chi_e, D_e [m²/s] |

### Usage

```python
from scpn_control.core.gk_eigenvalue import solve_linear_gk
from scpn_control.core.gk_quasilinear import quasilinear_fluxes_from_spectrum
from scpn_control.core.gk_species import deuterium_ion, electron

species = [deuterium_ion(R_L_T=6.9, R_L_n=2.2), electron(R_L_T=6.9, R_L_n=2.2)]
result = solve_linear_gk(species_list=species, R0=2.78, a=1.0, B0=2.0,
                          q=1.4, s_hat=0.78, n_ky_ion=12, n_theta=32)

fluxes = quasilinear_fluxes_from_spectrum(result, species[0], R0=2.78, a=1.0, B0=2.0)
print(f"chi_i = {fluxes.chi_i:.4f} m²/s, dominant: {fluxes.dominant_mode}")
```

### Known Limitations

- **Electrostatic only** — no A∥ or B∥ (no KBM, no microtearing)
- **Collision operator**: pitch-angle scattering only (no energy diffusion,
  no inter-species drag). Adequate for growth rate ordering, not for
  quantitative collisional damping
- **Adiabatic electrons**: kinetic electron path exists but not fully wired
- **Resolution**: default (n_E=12, n_lambda=16) gives qualitative instability
  detection. Quantitative γ within 20% of GENE/GS2 requires higher resolution
  and is not yet benchmarked

### Validation

Cyclone Base Case (Dimits et al., Phys. Plasmas 7, 969, 2000):
- Circular geometry, R/a=2.78, q=1.4, s_hat=0.78, R/L_Ti=6.9
- Produces positive γ_max with ITG classification
- Reference data in `validation/reference_data/cyclone_base/`
- **Not yet quantitatively benchmarked** against GENE/GS2 to 20% tolerance

## Path C: Hybrid Surrogate + GK Validation

Real-time QLKNN surrogate with automatic GK spot-check verification.

```
QLKNN surrogate (µs)
    │
    ├── OOD detector → Mahalanobis + range + ensemble
    │                   ↓ (if OOD or scheduled)
    ├── GK spot-check → scheduler selects flux surfaces
    │                   ↓
    ├── Correction    → multiplicative/additive/replace + EMA smoothing
    │                   ↓
    └── Online learner → buffer → retrain → validate → accept/rollback
```

| Module | Role |
|--------|------|
| `gk_ood_detector.py` | Mahalanobis distance, input range checks, ensemble disagreement |
| `gk_scheduler.py` | Periodic / adaptive / critical-region spot-check scheduling |
| `gk_corrector.py` | Error quantification + 3 correction modes + temporal EMA |
| `gk_online_learner.py` | Buffer-based retraining with validation holdout + rollback |
| `gk_verification_report.py` | Per-session stats, JSON export |

## SCPN Phase Bridge

GK fluxes modulate the 8-layer UPDE Kuramoto coupling matrix:

| Coupling | GK Input | Physics |
|----------|----------|---------|
| K[0,1] P0↔P1 | max(γ_ITG, γ_TEM) | Microturbulence ↔ zonal flow drive |
| K[1,4] P1↔P4 | mean(chi_e) / chi_ref | Zonal flow ↔ transport barrier suppression |
| K[3,4] P3↔P4 | chi_pedestal / chi_core | Sawtooth/ELM ↔ barrier strength |

```python
from scpn_control.phase.gk_upde_bridge import adaptive_knm

K_adapted = adaptive_knm(K_base, gk_output, chi_i_profile=chi_i)
```

## References

- Dimits et al., Phys. Plasmas 7 (2000) 969 — Cyclone Base Case
- Miller et al., Phys. Plasmas 5 (1998) 973 — Flux-tube geometry
- Sugama & Watanabe, Phys. Plasmas 13 (2006) 012501 — Collision operator
- Staebler et al., Phys. Plasmas 14 (2007) 055909 — TGLF
- Bourdelle et al., Phys. Plasmas 14 (2007) 112501 — QuaLiKiz
- van de Plassche et al., Phys. Plasmas 27 (2020) 022310 — QLKNN
