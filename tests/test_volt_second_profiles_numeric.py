# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Volt-second bootstrap profile numeric tests.

"""Exercise the public bootstrap proxy against derived arithmetic overflow."""

import numpy as np
import pytest

from scpn_control.control.volt_second_manager import BootstrapCurrentEstimate


def test_bootstrap_profile_refuses_overflowing_pressure() -> None:
    """Finite but unrepresentable pressure must not become a zero current."""
    values = np.full(3, 1e308)
    rho = np.array([0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="finite"):
        BootstrapCurrentEstimate.from_profiles(values, values, values, np.ones(3), rho, 6.2, 2.0)


def test_bootstrap_profile_refuses_overflowing_gradient() -> None:
    """Finite pressure on a near-collapsed grid cannot publish a proxy current."""
    density = np.array([1.0, 2.0, 3.0])
    rho = np.array([0.0, 1e-308, 1.0])
    with pytest.raises(ValueError, match="gradient must be finite"):
        BootstrapCurrentEstimate.from_profiles(density, np.ones(3), np.ones(3), np.ones(3), rho, 6.2, 2.0)
