"""scpn-control: Neuro-symbolic Stochastic Petri Net controller."""
from __future__ import annotations

__version__ = "0.2.0"

# Core solver
from scpn_control.core import FusionKernel, RUST_BACKEND, TokamakConfig

# Petri net compiler
from scpn_control.scpn import (
    StochasticPetriNet,
    FusionCompiler,
    CompiledNet,
    NeuroSymbolicController,
)

# Phase dynamics (Paper 27)
from scpn_control.phase import (
    kuramoto_sakaguchi_step,
    order_parameter,
    KnmSpec,
    build_knm_paper27,
    UPDESystem,
    LyapunovGuard,
    RealtimeMonitor,
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
