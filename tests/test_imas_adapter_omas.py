# ──────────────────────────────────────────────────────────────────────
# SCPN Control — IMAS Adapter omas Path Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for to_omas (lines 87-99) via a mocked omas.ODS."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from scpn_control.core.imas_adapter import EquilibriumIDS, to_omas


class _FakeODS(dict):
    """Minimal ODS stub that supports nested __getitem__ auto-vivification."""

    def __missing__(self, key):
        val = _FakeODS()
        self[key] = val
        return val

    def __setitem__(self, key, value):
        super().__setitem__(key, value)


class TestToOmas:
    def test_to_omas_without_omas_returns_none(self):
        """Without omas installed, to_omas returns None."""
        if "omas" in sys.modules:
            pytest.skip("omas is installed")
        ids = EquilibriumIDS(
            r=np.linspace(4, 8, 5),
            z=np.linspace(-4, 4, 5),
            psi=np.zeros((5, 5)),
            j_tor=np.zeros((5, 5)),
            ip=15e6,
            b0=5.3,
            r0=6.2,
        )
        assert to_omas(ids) is None

    def test_to_omas_with_mock_returns_ods(self):
        """With omas mocked, to_omas populates ODS fields (lines 87-99)."""
        ids = EquilibriumIDS(
            r=np.linspace(4, 8, 10),
            z=np.linspace(-4, 4, 12),
            psi=np.random.default_rng(0).standard_normal((12, 10)),
            j_tor=np.random.default_rng(1).standard_normal((12, 10)),
            ip=15e6,
            b0=5.3,
            r0=6.2,
            time=1.5,
        )
        fake_omas = types.ModuleType("omas")
        fake_omas.ODS = _FakeODS  # type: ignore[attr-defined]
        sys.modules["omas"] = fake_omas
        try:
            # Force reimport of the function to pick up mocked omas
            import importlib
            import scpn_control.core.imas_adapter as mod

            importlib.reload(mod)
            result = mod.to_omas(ids)
            assert result is not None
            eq = result["equilibrium"]["time_slice"][0]
            assert eq["time"] == 1.5
            assert eq["global_quantities"]["ip"] == 15e6
            p2d = eq["profiles_2d"][0]
            np.testing.assert_array_equal(p2d["grid"]["dim1"], ids.r)
            np.testing.assert_array_equal(p2d["grid"]["dim2"], ids.z)
            np.testing.assert_array_equal(p2d["psi"], ids.psi)
            np.testing.assert_array_equal(p2d["j_tor"], ids.j_tor)
        finally:
            del sys.modules["omas"]
            import importlib
            import scpn_control.core.imas_adapter as mod2

            importlib.reload(mod2)
