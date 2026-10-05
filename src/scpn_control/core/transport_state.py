# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Transport snapshots and thermal balance records.

"""Transport snapshots and thermal balance records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from scpn_control.core.integrated_transport_solver import TransportSolver


class PhysicsError(RuntimeError):
    """Raised when a physics constraint is violated."""


@dataclass(frozen=True)
class ThermalEnergyBalance:
    """Snapshot of a thermal assessment using species-count ion heat capacity.

    Energies are joules; dt_s is seconds and auxiliary_power_mw is megawatts.
    source_energy_j includes the solver's integrated net temperature sources,
    not an independently measured plant power balance. helium_pumping_energy_j
    is the signed ash-removal contribution already included in source_energy_j;
    it must not be added a second time. Positive boundary energies
    add heat: diffusive_boundary_energy_j integrates the two faces bounding the
    evolved interior cells; prescribed_edge_energy_j accounts for forcing the
    stored edge node and cancelling its ignored volumetric source. Fluxes average old-density/old-temperature and new-density/solved-temperature
    contributions before exchange, not later pedestal overrides. relative_error
    subtracts all three source/boundary terms from the energy change and uses
    max(abs(initial_energy_j), 1e-10) as denominator. Nonfinite arithmetic is
    retained for diagnosis; no validity or facility approval follows from this
    record. It can describe a step rejected by enforce_conservation.
    """

    dt_s: float
    auxiliary_power_mw: float
    initial_energy_j: float
    final_energy_j: float
    source_energy_j: float
    helium_pumping_energy_j: float
    diffusive_boundary_energy_j: float
    prescribed_edge_energy_j: float
    relative_error: float


def capture_evolution_state_impl(self: TransportSolver) -> dict[str, Any]:
    """Copy every quantity a transport step may mutate.

    Arrays are copied, so the snapshot is independent of later in-place
    writes. Attributes absent on this instance are omitted rather than
    defaulted, and :meth:`restore_evolution_state` then leaves them alone.
    The momentum sub-solver's rotation is captured under
    :data:`MOMENTUM_ROTATION_KEY` when that sub-solver exists.

    Returns
    -------
    dict[str, Any]
        Snapshot consumable only by :meth:`restore_evolution_state`.
    """
    state: dict[str, Any] = {}
    for name in self.EVOLUTION_STATE_FIELDS:
        if not hasattr(self, name):
            continue
        value = getattr(self, name)
        state[name] = value.copy() if isinstance(value, np.ndarray) else value
    if self._momentum_solver is not None:
        state[self.MOMENTUM_ROTATION_KEY] = np.array(self._momentum_solver.omega_phi, copy=True)
    return state


def restore_evolution_state_impl(self: TransportSolver, state: dict[str, Any]) -> None:
    """Restore a snapshot taken by :meth:`capture_evolution_state`.

    Arrays are copied back so the caller may restore the same snapshot more
    than once, which a Richardson trial sequence does.

    Parameters
    ----------
    state:
        Snapshot from :meth:`capture_evolution_state` on this instance.

    Raises
    ------
    KeyError
        If the snapshot carries a name outside
        :data:`EVOLUTION_STATE_FIELDS` and :data:`MOMENTUM_ROTATION_KEY`,
        which would mean it came from a different contract and cannot be
        trusted to be complete.
    RuntimeError
        If the snapshot holds a sub-solver rotation and this instance has
        no momentum sub-solver to give it back to.
    """
    unknown = set(state) - set(self.EVOLUTION_STATE_FIELDS) - {self.MOMENTUM_ROTATION_KEY}
    if unknown:
        raise KeyError(f"snapshot carries unknown evolution fields: {sorted(unknown)}")
    if self.MOMENTUM_ROTATION_KEY in state and self._momentum_solver is None:
        raise RuntimeError("snapshot holds a momentum rotation but no momentum solver is configured")
    for name, value in state.items():
        if name == self.MOMENTUM_ROTATION_KEY:
            continue
        setattr(self, name, value.copy() if isinstance(value, np.ndarray) else value)
    if self.MOMENTUM_ROTATION_KEY in state:
        assert self._momentum_solver is not None
        self._momentum_solver.omega_phi = np.array(state[self.MOMENTUM_ROTATION_KEY], copy=True)
