# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Multi machine validation.

"""Preserve the public multi-machine imports and unavailable campaign boundary.

Diagnostic examples, machine input capture, confinement comparison and actual
fixed-boundary execution have separate defining owners. This facade reexports
the established classes/presets; their numerical statements and call contracts
remain unchanged. Defining-module metadata now identifies the relevant owner.
The seven-domain campaign and its exporters still refuse missing integrations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from validation.confinement_reference import ConfinementReference as ConfinementReference
from validation.equilibrium_execution import EquilibriumExecution as EquilibriumExecution
from validation.machine_inputs import MachineConfig as MachineConfig
from validation.machine_inputs import diiid_h_mode as diiid_h_mode
from validation.machine_inputs import iter_15ma as iter_15ma
from validation.machine_inputs import jet_high_performance as jet_high_performance
from validation.machine_inputs import nstx_u_standard as nstx_u_standard
from validation.machine_inputs import sparc_baseline as sparc_baseline
from validation.synthetic_diagnostics import SyntheticDiagnosticSuite as SyntheticDiagnosticSuite


@dataclass
class ValidationResult:
    """Retain the legacy result record shape for compatibility only.

    Fields are caller-supplied and unchecked; passed is not an admission or
    provenance decision. The multi-machine exporters reject these records until
    the genuine seven-domain evidence workflow is implemented. New bounded
    confinement comparisons carry their own explicit JSON contract.
    """

    test_name: str
    machine_name: str
    metric_name: str
    value: float
    target: float
    passed: bool
    evidence: str


class MultiMachineValidator:
    """Refuse machine-validation claims until real model/reference bindings exist.

    The historical seven random metrics did not evaluate the supplied machine.
    They have been removed. Construction retains the caller's machine list for
    the forthcoming evidence-backed workflow but evaluates no physical model.
    No successful validation or export is available through this class yet.
    """

    def __init__(self, machines: list[MachineConfig]) -> None:
        """Copy the requested machine list and initialize an empty collection of validation results."""
        self.machines = list(machines)
        self.results: list[ValidationResult] = []

    def run_all(self, seed: int = 42) -> MultiMachineValidator:
        """Refuse an unsupported campaign without generating metrics or changing RNG state.

        Parameters
        ----------
        seed
            Retained call parameter; no random validation evidence is generated.

        Raises
        ------
        RuntimeError
            Real machine-bound equilibrium, transport, current/energy conservation,
            beta, vertical-stability and reconstruction evidence are not integrated.
        """
        raise RuntimeError(
            "Machine validation is unavailable: real model runs and machine-specific "
            "reference evidence are not integrated; random passing metrics are not validation"
        )

    def save_json(self, path: Path) -> None:
        """Refuse unsupported validation export before opening or changing any destination.

        Raises
        ------
        RuntimeError
            No verified machine-validation result contract is implemented. Populating
            the legacy public results list does not grant export permission.
        """
        raise RuntimeError("Cannot export machine validation without verified model/reference evidence")

    def save_markdown(self, path: Path) -> None:
        """Refuse a PASS/FAIL evidence table while real validation remains unavailable.

        Raises
        ------
        RuntimeError
            No verified machine-validation result contract is implemented. Existing
            files remain unchanged; callers must surface this unavailable state.
        """
        raise RuntimeError("Cannot export machine validation without verified model/reference evidence")
