# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — H-infinity tuning contract.
"""Exercise the public H-infinity tuning facade against admitted DGKF plants."""

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_control.control.controller_tuning as tuning
from scpn_control.control.h_infinity_controller import HInfinityController


def _normalized_plant(growth: float = 1.0) -> dict[str, NDArray[np.float64]]:
    """Return a normalized unstable standard plant with a variable pole."""
    return {
        "A": np.array([[0.0, 1.0], [growth, -1.0]]),
        "B1": np.array([[0.0, 0.0], [0.5, 0.0]]),
        "B2": np.array([[0.0], [1.0]]),
        "C1": np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        "C2": np.array([[1.0, 0.0]]),
        "D12": np.array([[0.0], [0.0], [1.0]]),
        "D21": np.array([[0.0, 1.0]]),
    }


def _synthesize(plant: dict[str, NDArray[np.float64]]) -> HInfinityController:
    """Run the same public plant contract independently of the tuning facade."""
    return HInfinityController(
        A=plant["A"],
        B1=plant["B1"],
        B2=plant["B2"],
        C1=plant["C1"],
        C2=plant["C2"],
        D12=plant["D12"],
        D21=plant["D21"],
    )


def test_tune_hinf_uses_plant_and_returns_only_feasible_attenuation() -> None:
    """Distinct admitted plants produce their own feasible DGKF attenuation."""
    first = _normalized_plant(1.0)
    second = _normalized_plant(2.0)
    first_result = tuning.tune_hinf(first)
    second_result = tuning.tune_hinf(second)
    assert first_result == {"gamma": pytest.approx(_synthesize(first).gamma)}
    assert second_result == {"gamma": pytest.approx(_synthesize(second).gamma)}
    assert first_result["gamma"] < second_result["gamma"]


def test_tune_hinf_refuses_missing_normalized_plant_data_without_optuna(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent solver data cannot become a fabricated default gain."""
    monkeypatch.setattr(tuning, "HAS_OPTUNA", False)
    with pytest.raises(ValueError, match="plant"):
        tuning.tune_hinf({})
    valid = tuning.tune_hinf(_normalized_plant())
    assert valid["gamma"] > 0.0


def test_tune_hinf_refuses_invented_bandwidth_or_invalid_normalization() -> None:
    """No undeclared objective or non-DGKF plant can produce a tuning result."""
    with pytest.raises(ValueError, match="plant"):
        tuning.tune_hinf({**_normalized_plant(), "bandwidth": 0.5})
    invalid = _normalized_plant()
    invalid["D12"] = np.array([[1.0], [0.0], [0.0]])
    with pytest.raises(ValueError, match=r"D12.T @ C1"):
        tuning.tune_hinf(invalid)
