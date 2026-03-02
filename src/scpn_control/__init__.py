# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Package Root
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""scpn-control: Neuro-symbolic Stochastic Petri Net controller."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("scpn-control")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"

# Core solver
from scpn_control.core import RUST_BACKEND, FusionKernel, TokamakConfig

# Phase dynamics (Paper 27)
from scpn_control.phase import (
    KnmSpec,
    LyapunovGuard,
    RealtimeMonitor,
    UPDESystem,
    build_knm_paper27,
    kuramoto_sakaguchi_step,
    order_parameter,
)

# Petri net compiler
from scpn_control.scpn import (
    CompiledNet,
    FusionCompiler,
    NeuroSymbolicController,
    StochasticPetriNet,
)

__all__ = [
    "__version__",
    "FusionKernel",
    "RUST_BACKEND",
    "TokamakConfig",
    "StochasticPetriNet",
    "FusionCompiler",
    "CompiledNet",
    "NeuroSymbolicController",
    "kuramoto_sakaguchi_step",
    "order_parameter",
    "KnmSpec",
    "build_knm_paper27",
    "UPDESystem",
    "LyapunovGuard",
    "RealtimeMonitor",
]
