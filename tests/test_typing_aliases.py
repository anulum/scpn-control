# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Shared typing alias tests

"""Contract tests for the shared numpy typing aliases."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

import scpn_control._typing as _typing


def test_float_array_is_float64_ndarray() -> None:
    """``FloatArray`` is the float64 ndarray specialisation used for outputs."""

    assert _typing.FloatArray == npt.NDArray[np.float64]


def test_any_float_array_is_floating_ndarray() -> None:
    """``AnyFloatArray`` is the any-precision floating ndarray for inputs."""

    assert _typing.AnyFloatArray == npt.NDArray[np.floating[Any]]


def test_float_array_and_any_float_array_differ() -> None:
    """The output and input aliases are distinct types at module boundaries."""

    assert _typing.FloatArray != _typing.AnyFloatArray


def test_all_exports_sorted_and_complete() -> None:
    """``__all__`` lists both aliases in sorted order."""

    assert _typing.__all__ == ["AnyFloatArray", "FloatArray"]
    assert _typing.__all__ == sorted(_typing.__all__)


def test_float64_array_is_a_runtime_instance_of_ndarray() -> None:
    """A constructed float64 array is a concrete ndarray the alias describes."""

    array = np.zeros(3, dtype=np.float64)
    assert isinstance(array, np.ndarray)
    assert array.dtype == np.float64
