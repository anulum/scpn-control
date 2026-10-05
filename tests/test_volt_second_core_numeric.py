# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Volt-second core numeric tests.

"""Exercise the public flux budget at finite arithmetic boundaries."""

import numpy as np
import pytest

from scpn_control.control.volt_second_manager import FluxBudget


def test_flux_budget_refuses_unrepresentable_derived_flux() -> None:
    """Finite current and timing values cannot emit infinite volt-seconds."""
    budget = FluxBudget(120.0, 1.2, 0.08)
    with pytest.raises(ValueError, match="finite"):
        budget.inductive_flux(1e308)
    with pytest.raises(ValueError, match="finite"):
        budget.resistive_flux_ramp(np.array([1e308]), 1.0)
    with pytest.raises(ValueError, match="finite"):
        budget.ejima_startup_flux(6.2, 1e308)


def test_flattop_duration_refuses_unrepresentable_result() -> None:
    """A tiny resistance cannot masquerade as an infinite valid duration."""
    budget = FluxBudget(1e308, 1.2, 1e-302)
    with pytest.raises(ValueError, match="finite"):
        budget.max_flattop_duration(15.0, 15.0, 0.0)


def test_flattop_duration_refuses_underflowed_denominator() -> None:
    """A positive stored resistance can still underflow after current scaling."""
    budget = FluxBudget(120.0, 1.2, 1e-312)
    with pytest.raises(ValueError, match="denominator"):
        budget.max_flattop_duration(15.0, 15.0, 0.0)
