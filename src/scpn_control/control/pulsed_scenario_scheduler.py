# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Pulsed scheduler compatibility import surface.
"""Compatibility surface for the CONTROL-owned pulsed scenario scheduler.

SCPN-MIF-CORE originally contracted the module path
``scpn_control.control.pulsed_scenario_scheduler``.  The production
implementation lives in ``pulsed_scenario_scheduler_v2``; this module keeps the
cross-repository import stable without duplicating scheduler logic.
"""

from __future__ import annotations

from scpn_control.control.pulsed_scenario_scheduler_v2 import (
    CapacitorBankTelemetry,
    PulsedPlasmaTelemetry,
    PulsedScenarioAction,
    PulsedScenarioCommand,
    PulsedScenarioScheduler,
    PulsedScenarioSpec,
    PulsedScenarioState,
    PulsedScenarioTransition,
)

__all__ = [
    "CapacitorBankTelemetry",
    "PulsedPlasmaTelemetry",
    "PulsedScenarioAction",
    "PulsedScenarioCommand",
    "PulsedScenarioScheduler",
    "PulsedScenarioSpec",
    "PulsedScenarioState",
    "PulsedScenarioTransition",
]
