# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Detachment-control failure atomicity tests.

"""Exercise public seeding-controller refusal and coordinated rollback."""

import pytest

from scpn_control.control.detachment_controller import (
    DetachmentController,
    DetachmentState,
    MultiImpuritySeeding,
)


def _state(controller: DetachmentController) -> tuple[float, float, DetachmentState]:
    return controller.integral_e, controller.last_cmd, controller.state


def test_finite_input_overflow_does_not_advance_detachment_controller() -> None:
    """An unrepresentable PI candidate must not publish command or state."""
    controller = DetachmentController()
    before = _state(controller)
    with pytest.raises(ValueError, match="finite"):
        controller.step(1e308, 5.0, 10.0, 0.1, 1e308)
    assert _state(controller) == before


def test_multi_impurity_failure_restores_earlier_controller() -> None:
    """A failed later species must leave all previously accepted states intact."""
    nitrogen = DetachmentController("N2")
    neon = DetachmentController("Ne")
    neon.Ki = float("inf")
    coordinator = MultiImpuritySeeding(["N2", "Ne"], {"N2": nitrogen, "Ne": neon})
    before = _state(nitrogen), _state(neon)
    frame = {"T_target_eV": 20.0, "n_target_19": 10.0, "P_rad_MW": 10.0, "rho_front": 0.1}
    with pytest.raises(ValueError, match="Ki must be finite"):
        coordinator.step(frame, dt=0.1)
    assert (_state(nitrogen), _state(neon)) == before


def test_multi_impurity_rejects_duplicate_species() -> None:
    """One species must not advance the same controller twice per frame."""
    nitrogen = DetachmentController("N2")
    with pytest.raises(ValueError, match="duplicate"):
        MultiImpuritySeeding(["N2", "N2"], {"N2": nitrogen})


def test_multi_impurity_rejects_public_list_mutation_before_step() -> None:
    """Mutation of the exposed species list cannot double-advance a controller."""
    nitrogen = DetachmentController("N2")
    coordinator = MultiImpuritySeeding(["N2"], {"N2": nitrogen})
    coordinator.impurities.append("N2")
    before = _state(nitrogen)
    with pytest.raises(ValueError, match="duplicate"):
        coordinator.step({"T_target_eV": 20.0}, dt=0.1)
    assert _state(nitrogen) == before


def test_multi_impurity_rejects_shared_controller_state() -> None:
    """Two species must not use the same mutable controller instance."""
    shared = DetachmentController("N2")
    coordinator = MultiImpuritySeeding(["N2", "Ne"], {"N2": shared, "Ne": shared})
    before = _state(shared)
    with pytest.raises(ValueError, match="share mutable state"):
        coordinator.step({"T_target_eV": 20.0}, dt=0.1)
    assert _state(shared) == before
