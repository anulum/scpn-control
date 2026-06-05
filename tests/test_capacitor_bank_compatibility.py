# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Capacitor-bank compatibility tests.
"""Compatibility tests for the MIF-facing capacitor-bank import path."""

from __future__ import annotations

from scpn_control.control import capacitor_bank as mif_contract
from scpn_control.control.capacitor_bank_state import (
    CapacitorBank,
    CapacitorBankSpec,
    CapacitorBankState,
    EnergyReport,
    PulseSpec,
    RLCRegime,
    free_response,
)


def test_mif_contract_import_path_reexports_capacitor_bank_state_model() -> None:
    assert mif_contract.CapacitorBank is CapacitorBank
    assert mif_contract.CapacitorBankSpec is CapacitorBankSpec
    assert mif_contract.CapacitorBankState is CapacitorBankState
    assert mif_contract.EnergyReport is EnergyReport
    assert mif_contract.PulseSpec is PulseSpec
    assert mif_contract.RLCRegime is RLCRegime
    assert mif_contract.free_response is free_response
