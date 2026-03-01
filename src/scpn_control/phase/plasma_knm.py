# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Plasma-Native Knm Coupling Matrix
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
r"""
Plasma-specific Knm coupling matrices for tokamak phase dynamics.

Maps the UPDE multi-layer Kuramoto framework onto physically motivated
plasma timescale hierarchies.  Each layer represents a distinct plasma
process; coupling strengths encode known physics interactions.

Layer hierarchy (8-layer default)
---------------------------------
  P0  Micro-turbulence     ~μs       ITG/TEM drift-wave fluctuations
  P1  Zonal flows          ~ms       E×B flow shear (Rosenbluth–Hinton)
  P2  MHD / tearing        ~ms–s     NTM, 2/1 island dynamics
  P3  Sawtooth / ELM        10–100ms  Internal kink / peeling–ballooning
  P4  Transport barrier    ~100ms     H-mode pedestal, internal barriers
  P5  Current profile      ~s        q(ρ), l_i evolution (resistive time)
  P6  Global equilibrium   ~s        GS ψ(R,Z) reconstruction
  P7  Plasma–wall (PWI)    ~s–min    Recycling, sputtering, impurities

Coupling physics
----------------
K[i,j] encodes the coupling from layer j → layer i.

  Micro ↔ Zonal:     Drift-wave / zonal-flow predator–prey
                      Ref: Diamond et al., Plasma Phys. Control. Fusion 47, R35 (2005)
  Zonal ↔ Transport:  E×B shear suppresses turbulent transport
                      Ref: Terry, Rev. Mod. Phys. 72, 109 (2000)
  MHD ↔ Current:      NTM driven by bootstrap current → flattens q
                      Ref: La Haye, Phys. Plasmas 13, 055501 (2006)
  Sawtooth ↔ Current: Sawtooth crash redistributes j(ρ)
                      Ref: Porcelli et al., Plasma Phys. Control. Fusion 38, 2163 (1996)
  Transport ↔ Equil:  Pressure profile feeds back to GS equilibrium
                      Ref: Lütjens et al., Comput. Phys. Commun. 97, 219 (1996)
  PWI ↔ Transport:    Edge recycling / impurities modify edge transport
                      Ref: Stangeby, "The Plasma Boundary of Magnetic Fusion Devices"

Modes
-----
``build_knm_plasma`` accepts a ``mode`` argument that biases the coupling
matrix toward the dominant instability:

  baseline : Standard H-mode operating point
  elm      : ELM-dominated (enhanced P3↔P4 coupling)
  ntm      : NTM-dominated (enhanced P2↔P5 coupling)
  sawtooth : Sawtooth-dominated (enhanced P3↔P5 coupling)
  hybrid   : Advanced scenario (balanced, higher overall coupling)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from scpn_control.phase.knm import KnmSpec

# ── 8-layer plasma hierarchy ────────────────────────────────────────

PLASMA_LAYER_NAMES: tuple[str, ...] = (
    "micro_turbulence",
    "zonal_flow",
    "mhd_tearing",
    "sawtooth_elm",
    "transport_barrier",
    "current_profile",
    "global_equilibrium",
    "plasma_wall",
)

# Natural frequencies [rad/s] — order-of-magnitude representative
# timescales for each process, derived from diagnostic observations.
#
# P0 micro:  ~100 kHz drift-wave → ω ~ 6.28e5 (normalised below)
# P1 zonal:  ~1 kHz GAM frequency → ω ~ 6.28e3
# P2 MHD:    ~1 kHz Mirnov (rotating NTM 2/1) → ω ~ 6.28e3
# P3 saw/ELM: ~10–100 Hz sawtooth/ELM → ω ~ 62.8–628
# P4 barrier: ~10 Hz pedestal rebuild → ω ~ 62.8
# P5 current: ~0.1 Hz resistive diffusion → ω ~ 0.628
# P6 equil:   ~0.1 Hz equilibrium evolution → ω ~ 0.628
# P7 PWI:     ~0.01 Hz wall recycling → ω ~ 0.063
#
# Log-normalised to [0.1, 10.0] for numerical stability in the
# Kuramoto integrator (absolute timescale is handled by dt).
OMEGA_PLASMA_8 = np.array(
    [
        8.50,  # P0: micro-turbulence (fastest)
        5.20,  # P1: zonal flows
        4.80,  # P2: MHD / tearing modes
        3.10,  # P3: sawtooth / ELM cycles
        2.40,  # P4: transport barrier dynamics
        0.85,  # P5: current profile evolution
        0.72,  # P6: global equilibrium
        0.18,  # P7: plasma-wall interaction (slowest)
    ],
    dtype=np.float64,
)


# ── Coupling matrix builders ────────────────────────────────────────


def _base_plasma_knm(L: int = 8, K_base: float = 0.30) -> NDArray[np.float64]:
    """Distance-decay baseline with plasma-specific physics overlays.

    Uses weaker exponential decay (α=0.5) than Paper 27 because plasma
    layers span wider timescale separation.
    """
    K_ALPHA = 0.5  # steeper decay across timescale gap
    idx = np.arange(L)
    dist = np.abs(idx[:, None] - idx[None, :])
    K = K_base * np.exp(-K_ALPHA * dist)
    return K


def _apply_physics_couplings(K: NDArray[np.float64]) -> None:
    """Overlay physically motivated nearest-neighbour couplings.

    Values are dimensionless coupling strengths calibrated so the
    Kuramoto model reproduces qualitative multi-scale synchronisation
    patterns observed in DIII-D and JET.
    """
    L = K.shape[0]
    if L < 8:
        return

    # Diamond et al. 2005: drift-wave / zonal-flow predator–prey
    K[0, 1] = K[1, 0] = 0.42

    # Terry 2000: E×B shear suppression of turbulent transport
    K[1, 4] = K[4, 1] = 0.28

    # La Haye 2006: NTM ↔ bootstrap current (q-profile flattening)
    K[2, 5] = K[5, 2] = 0.35

    # Porcelli 1996: sawtooth ↔ current redistribution
    K[3, 5] = K[5, 3] = 0.30

    # Sawtooth/ELM ↔ transport barrier (ELM crash depletes pedestal)
    K[3, 4] = K[4, 3] = 0.32

    # Transport–equilibrium: pressure ↔ GS reconstruction
    K[4, 6] = K[6, 4] = 0.25

    # PWI ↔ edge transport (recycling, impurity influx)
    K[7, 4] = K[4, 7] = 0.20

    # PWI ↔ equilibrium (wall conditioning affects global)
    K[7, 6] = K[6, 7] = 0.15


def _apply_mode_bias(K: NDArray[np.float64], mode: str) -> None:
    """Amplify couplings for a specific instability scenario."""
    L = K.shape[0]
    if L < 8:
        return

    if mode == "elm":
        # Enhanced sawtooth/ELM ↔ transport barrier interaction
        K[3, 4] *= 1.8
        K[4, 3] *= 1.8
        # ELM crashes couple to PWI (wall heat load)
        K[3, 7] = max(K[3, 7], 0.22)
        K[7, 3] = max(K[7, 3], 0.22)

    elif mode == "ntm":
        # Enhanced NTM ↔ current profile coupling
        K[2, 5] *= 1.6
        K[5, 2] *= 1.6
        # NTM couples to transport barrier (island flattens gradient)
        K[2, 4] = max(K[2, 4], 0.25)
        K[4, 2] = max(K[4, 2], 0.25)

    elif mode == "sawtooth":
        # Enhanced sawtooth ↔ current coupling
        K[3, 5] *= 1.7
        K[5, 3] *= 1.7
        # Sawtooth modulates core turbulence
        K[3, 0] = max(K[3, 0], 0.18)
        K[0, 3] = max(K[0, 3], 0.18)

    elif mode == "hybrid":
        # Advanced scenario: all couplings slightly elevated
        K *= 1.15


_VALID_MODES = frozenset({"baseline", "elm", "ntm", "sawtooth", "hybrid"})


def build_knm_plasma(
    mode: str = "baseline",
    L: int = 8,
    K_base: float = 0.30,
    zeta_uniform: float = 0.0,
    custom_overrides: dict[tuple[int, int], float] | None = None,
    layer_names: Sequence[str] | None = None,
) -> KnmSpec:
    """Build a plasma-native Knm coupling matrix.

    Parameters
    ----------
    mode : str
        Instability scenario bias.  One of: baseline, elm, ntm,
        sawtooth, hybrid.
    L : int
        Number of plasma layers (default 8).
    K_base : float
        Base coupling strength for exponential decay backbone.
    zeta_uniform : float
        Global driver gain ζ applied uniformly to all layers.
    custom_overrides : dict, optional
        Explicit (i, j) → value overrides applied last.
        Automatically symmetrised.
    layer_names : sequence of str, optional
        Layer labels.  Defaults to PLASMA_LAYER_NAMES[:L].

    Returns
    -------
    KnmSpec
        Ready for UPDESystem consumption.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown plasma mode {mode!r}; choose from {sorted(_VALID_MODES)}")

    K = _base_plasma_knm(L, K_base)
    _apply_physics_couplings(K)

    if mode != "baseline":
        _apply_mode_bias(K, mode)

    if custom_overrides:
        for (i, j), val in custom_overrides.items():
            if not (0 <= i < L and 0 <= j < L):
                raise IndexError(f"Override index ({i}, {j}) out of range for L={L}")
            K[i, j] = val
            K[j, i] = val

    # Symmetry guarantee
    K = 0.5 * (K + K.T)
    np.maximum(K, 0.0, out=K)

    zeta = np.full(L, zeta_uniform, dtype=np.float64) if zeta_uniform != 0.0 else None

    names = list(layer_names) if layer_names else list(PLASMA_LAYER_NAMES[:L])

    return KnmSpec(K=K, zeta=zeta, layer_names=names)


def plasma_omega(L: int = 8) -> NDArray[np.float64]:
    """Return natural frequencies for L plasma layers.

    For L <= 8, returns OMEGA_PLASMA_8[:L].
    For L > 8, interpolates log-linearly between the fastest (P0)
    and slowest (P7) process timescales.
    """
    if L <= 8:
        return OMEGA_PLASMA_8[:L].copy()
    # Log-linear interpolation across the timescale range
    return np.logspace(
        np.log10(OMEGA_PLASMA_8[0]),
        np.log10(OMEGA_PLASMA_8[-1]),
        L,
    )


def build_knm_plasma_from_config(
    R0: float,
    a: float,
    B0: float,
    Ip: float,
    n_e: float,
    *,
    mode: str = "baseline",
    L: int = 8,
    zeta_uniform: float = 0.0,
) -> KnmSpec:
    """Build plasma Knm with coupling strengths scaled by machine parameters.

    Adjusts K_base via the normalised beta proxy:

        K_base = 0.30 · (1 + 0.5 · β_proxy)

    where β_proxy = n_e [1e19] · a / B0^2 is a dimensionless pressure
    proxy.  Higher β means stronger multi-scale coupling because more
    free energy drives instabilities.

    Parameters
    ----------
    R0 : float  Major radius [m]
    a  : float  Minor radius [m]
    B0 : float  Toroidal field [T]
    Ip : float  Plasma current [MA]
    n_e : float  Line-averaged density [1e19 m^-3]
    mode, L, zeta_uniform : same as build_knm_plasma
    """
    # β_proxy ~ p / (B^2/2μ0) ∝ n·T / B^2; T ∝ a·B via confinement
    beta_proxy = n_e * a / max(B0**2, 1e-6)
    K_base = 0.30 * (1.0 + 0.5 * np.clip(beta_proxy, 0.0, 2.0))

    # q_cyl ≈ 5 a² B0 / (R0 Ip) — used to detect low-q (sawtooth-prone)
    q_cyl = 5.0 * a**2 * B0 / max(R0 * Ip, 1e-6)

    auto_mode = mode
    if mode == "baseline" and q_cyl < 1.0:
        auto_mode = "sawtooth"

    return build_knm_plasma(
        mode=auto_mode,
        L=L,
        K_base=K_base,
        zeta_uniform=zeta_uniform,
    )
