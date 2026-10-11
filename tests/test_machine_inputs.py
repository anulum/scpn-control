# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Machine inputs contract tests.

"""Exercise actual machine input snapshots, profile custody and malformed input refusals."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from validation.multi_machine_validation import (
    MachineConfig,
    diiid_h_mode,
    iter_15ma,
    jet_high_performance,
    nstx_u_standard,
    sparc_baseline,
)


@pytest.mark.parametrize("factory", [iter_15ma, jet_high_performance, diiid_h_mode, sparc_baseline, nstx_u_standard])
def test_snapshot_binds_actual_machine_values(factory: Callable[[], MachineConfig]) -> None:
    """Capture actual preset profiles and units without silently changing the zero edge."""
    machine = factory()
    rho = np.linspace(0.0, 1.0, 51, dtype=np.float64)
    encoded = machine.snapshot(rho)
    decoded = json.loads(encoded)
    assert decoded["machine"] == machine.name
    assert decoded["scalars"]["Ip_MA"] == machine.Ip_MA
    assert decoded["units"]["ne"] == "1e19 m^-3"
    assert decoded["units"]["Te"] == "keV"
    np.testing.assert_array_equal(decoded["profiles"]["Te"], machine.Te_profile(rho))
    assert decoded["profiles"]["ne"][-1] == decoded["profiles"]["Ti"][-1] == 0.0
    assert machine.snapshot(rho) == encoded
    changed = dataclasses.replace(machine, Ip_MA=machine.Ip_MA * 0.9).snapshot(rho)
    assert hashlib.sha256(changed.encode()).digest() != hashlib.sha256(encoded.encode()).digest()


def test_snapshot_isolates_callback_inputs_and_outputs() -> None:
    """Mutating callbacks cannot rewrite an earlier sample, the caller grid or captured scalars."""
    machine = iter_15ma()
    rho = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    shared = np.zeros(5)

    def density(grid: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Deliberately modify callback-owned input and shared output."""
        grid[:] = 7.0
        shared[:] = 2.0
        machine.Ip_MA = 3.0
        return shared

    def temperature(grid: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Require the original grid despite mutation by the preceding callback."""
        np.testing.assert_array_equal(grid, rho)
        shared[:] = 4.0
        return shared

    machine.ne_profile = density
    machine.Te_profile = temperature
    result = json.loads(machine.snapshot(rho))
    assert result["scalars"]["Ip_MA"] == 15.0
    assert result["profiles"]["ne"] == [2.0] * 5
    assert result["profiles"]["Te"] == [4.0] * 5
    np.testing.assert_array_equal(rho, np.linspace(0.0, 1.0, 5, dtype=np.float64))


@pytest.mark.parametrize(
    "field,value",
    [
        ("B0", -1.0),
        ("Ip_MA", -10.0),
        ("a", 6.2),
        ("P_aux_MW", -1.0),
        ("delta", 1.0),
        ("R0", float("nan")),
        ("kappa", True),
    ],
)
def test_snapshot_rejects_invalid_machine_scalars(field: str, value: float) -> None:
    """Reject malformed machine state before a model can consume it."""
    machine = iter_15ma()
    setattr(machine, field, value)
    with pytest.raises(ValueError):
        machine.snapshot(np.linspace(0.0, 1.0, 5, dtype=np.float64))


@pytest.mark.parametrize("values", [[0.0, 0.5, float("nan")], [0.0, 0.5, 0.5, 1.0], [0.1, 0.5, 1.0], [0.0, 1.0]])
def test_snapshot_rejects_invalid_grid(values: list[float]) -> None:
    """Require explicit axis-to-edge sampling without implicit grid repairs."""
    with pytest.raises(ValueError, match="rho"):
        iter_15ma().snapshot(np.asarray(values, dtype=np.float64))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_snapshot_rejects_invalid_profile_values(bad: float) -> None:
    """Reject original nonfinite/negative-profile reproducer inputs at capture time."""
    machine = iter_15ma()
    machine.ne_profile = lambda grid: np.full_like(grid, bad)
    with pytest.raises(ValueError, match="ne profile"):
        machine.snapshot(np.linspace(0.0, 1.0, 5, dtype=np.float64))


@pytest.mark.parametrize("kind", ["shape", "complex", "bool"])
def test_snapshot_rejects_invalid_profile_representation(kind: str) -> None:
    """Prevent broadcasting and lossy conversion from changing supplied profile values."""
    machine = iter_15ma()
    values = {"shape": np.ones(1), "complex": np.ones(5, dtype=complex), "bool": np.ones(5, dtype=bool)}
    machine.Te_profile = lambda grid: values[kind]
    with pytest.raises(ValueError, match="Te profile"):
        machine.snapshot(np.linspace(0.0, 1.0, 5, dtype=np.float64))


@pytest.mark.parametrize("values", [np.array([False, True, True]), np.array([0.0, 0.5, 1.0], dtype=complex)])
def test_snapshot_rejects_lossy_grid_conversion(values: npt.NDArray[Any]) -> None:
    """Boolean and complex radius samples cannot be silently converted into physical coordinates."""
    with pytest.raises(ValueError, match="real numeric"):
        iter_15ma().snapshot(values)


def test_snapshot_rejects_blank_machine_identity() -> None:
    """Require a declared machine identity before invoking profile callbacks."""
    with pytest.raises(ValueError, match="machine name"):
        dataclasses.replace(iter_15ma(), name=" ").snapshot(np.linspace(0.0, 1.0, 5, dtype=np.float64))
