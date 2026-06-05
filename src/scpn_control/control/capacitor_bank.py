# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Capacitor-bank compatibility import surface.
"""Compatibility surface for the CONTROL-owned capacitor-bank model.

SCPN-MIF-CORE originally contracted the module path
``scpn_control.control.capacitor_bank``.  The production implementation lives in
``capacitor_bank_state``; this module keeps the cross-repository import stable
without duplicating the RLC mathematics.
"""

from __future__ import annotations

from scpn_control.control.capacitor_bank_state import (
    ENERGY_BALANCE_REL_TOLERANCE,
    CapacitorBank,
    CapacitorBankSpec,
    CapacitorBankState,
    EnergyReport,
    PulseSpec,
    RLCRegime,
    WaveformName,
    free_response,
)

__all__ = [
    "CapacitorBank",
    "CapacitorBankSpec",
    "CapacitorBankState",
    "ENERGY_BALANCE_REL_TOLERANCE",
    "EnergyReport",
    "PulseSpec",
    "RLCRegime",
    "WaveformName",
    "free_response",
]
