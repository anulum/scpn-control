# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable transport import-path tests.
"""Import-path coverage for optional differentiable-transport dependencies."""

from __future__ import annotations

import builtins
import importlib
import sys
from types import ModuleType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import pytest


def _profiles(rho: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a compact four-channel transport profile fixture."""
    te = 8.0 * np.exp(-((rho - 0.35) ** 2) / 0.03) + 0.2
    ti = 6.0 * np.exp(-((rho - 0.42) ** 2) / 0.04) + 0.2
    ne = 4.0 + 0.8 * (1.0 - rho**2)
    nz = 0.03 + 0.02 * np.exp(-((rho - 0.65) ** 2) / 0.02)
    return np.asarray(np.stack([te, ti, ne, nz]), dtype=np.float64)


def test_import_without_jax_keeps_numpy_transport_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Import-time JAX absence leaves the deterministic NumPy transport path usable."""
    module_name = "scpn_control.core.differentiable_transport"
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals_: dict[str, Any] | None = None,
        locals_: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if level == 0 and (name == "jax" or name.startswith("jax.")):
            raise ImportError("blocked JAX import for fallback coverage")
        return original_import(name, globals_, locals_, fromlist, level)

    original_module = sys.modules.pop(module_name, None)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    try:
        module = importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module

    module_any = cast(Any, module)
    rho = np.linspace(0.05, 1.0, 8, dtype=np.float64)
    profiles = _profiles(rho)
    chi = np.full_like(profiles, 0.05)
    sources = np.zeros_like(profiles)
    edge = profiles[:, -1]

    stepped = module_any.differentiable_transport_step(profiles, chi, sources, rho, 1.0e-4, edge, use_jax=False)

    assert isinstance(module, ModuleType)
    assert module_any._HAS_JAX is False
    assert module_any.jax is None
    assert module_any.jnp is None
    assert np.asarray(stepped).shape == profiles.shape
