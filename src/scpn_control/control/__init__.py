# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Control Package Init
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Control modules (heavy imports guarded).

Nengo, matplotlib, and torch are optional — use accessor functions
to defer their import until actually needed.
"""


def get_nengo_controller():
    """Lazy import of NengoSNNController (requires ``pip install scpn-control[nengo]``)."""
    from scpn_control.control.nengo_snn_wrapper import NengoSNNController
    return NengoSNNController


__all__ = ["get_nengo_controller"]
