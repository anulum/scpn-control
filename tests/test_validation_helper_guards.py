# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Numeric input guards of the validation helper modules.
"""Validation helpers refuse integers too large for a float as their documented ValueError.

``float()`` and ``math.isfinite()`` raise OverflowError for such integers.
Each helper here documents a ValueError for a non-finite input, so the
overflow must surface as that ValueError rather than escape to a caller
that only handles ValueError.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from validation import ballooning_independent_reference as ballooning
from validation import burn_model_provider as burn
from validation.machine_inputs import iter_15ma

TOO_LARGE_FOR_FLOAT = 10**400


def test_ballooning_reference_refuses_an_oversized_shear() -> None:
    """The independent ballooning solver refuses an oversized shear before integrating."""
    with pytest.raises(ValueError, match="^s must be finite$"):
        ballooning.ballooning_alpha_crit(TOO_LARGE_FOR_FLOAT)


def test_ballooning_finite_guard_keeps_its_other_refusals() -> None:
    """Non-finite and, where requested, non-positive values are still refused; finite values pass as floats."""
    with pytest.raises(ValueError, match="^x must be finite$"):
        ballooning._finite("x", float("nan"))
    with pytest.raises(ValueError, match="^x must be positive$"):
        ballooning._finite("x", 0.0, positive=True)
    assert ballooning._finite("x", 2) == 2.0


def test_burn_reference_point_refuses_an_oversized_input() -> None:
    """A burn reference whose major radius is a 400-digit integer is refused as a non-finite input."""
    reference: dict[str, object] = {
        "R_m": TOO_LARGE_FOR_FLOAT,
        "a_m": 2.0,
        "B_t_T": 5.3,
        "I_p_MA": 15.0,
        "kappa": 1.7,
        "n_e_1e19": 10.0,
        "P_aux_MW": 50.0,
    }
    with pytest.raises(ValueError, match="^R_m must be a finite positive number$"):
        burn._point(reference)
    point = burn._point({**reference, "R_m": 6.2})
    assert point.r_m == 6.2
    assert point.ne20 == pytest.approx(1.0)


def test_machine_snapshot_refuses_an_oversized_scalar() -> None:
    """A machine whose major radius is a 400-digit integer cannot be snapshotted."""
    machine = replace(iter_15ma(), R0=TOO_LARGE_FOR_FLOAT)
    with pytest.raises(ValueError, match="^R0 must be finite$"):
        machine.snapshot(np.linspace(0.0, 1.0, 5))
