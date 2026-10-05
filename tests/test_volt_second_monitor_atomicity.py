# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Volt-second monitor atomicity tests.

"""Exercise public online flux-monitor refusal without state publication."""

import pytest

from scpn_control.control.volt_second_manager import FluxBudget, FluxConsumptionMonitor, ScenarioFluxAnalysis


def test_monitor_rejects_finite_product_overflow_without_consuming_flux() -> None:
    """A finite input pair cannot publish an infinite consumed flux."""
    monitor = FluxConsumptionMonitor(FluxBudget(120.0, 1.2, 0.08))
    with pytest.raises(ValueError, match="finite"):
        monitor.step(15.0, 1e308, 1e308)
    assert monitor.consumed == 0.0


def test_monitor_rejects_unrepresentable_fraction_without_consuming_flux() -> None:
    """An invalid status estimate cannot commit the monitor state."""
    monitor = FluxConsumptionMonitor(FluxBudget(1e-308, 1.2, 0.08))
    with pytest.raises(ValueError, match="finite"):
        monitor.step(15.0, 1e308, 1.0)
    assert monitor.consumed == 0.0


def test_scenario_analysis_refuses_unrepresentable_phase_flux() -> None:
    """Finite scenario inputs must not publish an infinite flux report."""
    budget = FluxBudget(120.0, 1.2, 0.08)
    with pytest.raises(ValueError, match="finite"):
        ScenarioFluxAnalysis(budget).analyze(80.0, 200.0, 60.0, 1e308, 4.0)
