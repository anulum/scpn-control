# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Phase Dynamics Package
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Paper 27 Knm/UPDE engine + Kuramoto-Sakaguchi phase reduction.

Implements the generalized Kuramoto-Sakaguchi mean-field model with an
exogenous global field driver ζ sin(Ψ − θ), per the reviewer request
referencing arXiv:2004.06344 and SCPN Paper 27.
"""

from scpn_control.phase.kuramoto import (
    kuramoto_sakaguchi_step,
    order_parameter,
    wrap_phase,
    lyapunov_v,
    lyapunov_exponent,
    GlobalPsiDriver,
)
from scpn_control.phase.knm import KnmSpec, build_knm_paper27
from scpn_control.phase.upde import UPDESystem
from scpn_control.phase.lyapunov_guard import LyapunovGuard
from scpn_control.phase.realtime_monitor import RealtimeMonitor

__all__ = [
    "kuramoto_sakaguchi_step",
    "order_parameter",
    "wrap_phase",
    "lyapunov_v",
    "lyapunov_exponent",
    "GlobalPsiDriver",
    "KnmSpec",
    "build_knm_paper27",
    "UPDESystem",
    "LyapunovGuard",
    "RealtimeMonitor",
]
