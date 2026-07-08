# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Bio-holonomic controller coverage hardening tests
"""COV-1 regression tests for bio-holonomic optional-adapter imports."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest

import scpn_control.control.bio_holonomic_controller as bio_mod


class _FakeL4CellularAdapter:
    """Minimal L4 adapter used to exercise optional import wiring."""


class _FakeL5OrganismalAdapter:
    """Minimal L5 adapter used to exercise optional import wiring."""


class _FakeL4HolonomicParameters:
    """Minimal L4 parameter type used to satisfy the imported symbol contract."""


class _FakeL5HolonomicParameters:
    """Minimal L5 parameter type used to satisfy the imported symbol contract."""


def _module(name: str, **attrs: object) -> ModuleType:
    """Create a module populated with dynamic test attributes."""
    module = ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    return module


def test_optional_holonomic_adapter_import_success_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reload with fake sc-neurocore holonomic modules and verify adapter wiring."""
    try:
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "sc_neurocore", _module("sc_neurocore"))
            patch.setitem(sys.modules, "sc_neurocore.adapters", _module("sc_neurocore.adapters"))
            patch.setitem(
                sys.modules,
                "sc_neurocore.adapters.holonomic",
                _module("sc_neurocore.adapters.holonomic"),
            )
            patch.setitem(
                sys.modules,
                "sc_neurocore.adapters.holonomic.l4_cell",
                _module(
                    "sc_neurocore.adapters.holonomic.l4_cell",
                    L4_CellularAdapter=_FakeL4CellularAdapter,
                    L4_HolonomicParameters=_FakeL4HolonomicParameters,
                ),
            )
            patch.setitem(
                sys.modules,
                "sc_neurocore.adapters.holonomic.l5_org",
                _module(
                    "sc_neurocore.adapters.holonomic.l5_org",
                    L5_OrganismalAdapter=_FakeL5OrganismalAdapter,
                    L5_HolonomicParameters=_FakeL5HolonomicParameters,
                ),
            )

            importlib.reload(bio_mod)

            assert bio_mod.SC_NEUROCORE_HOLONOMIC_AVAILABLE is True
            assert bio_mod.L4_CellularAdapter is _FakeL4CellularAdapter
            assert bio_mod.L5_OrganismalAdapter is _FakeL5OrganismalAdapter
    finally:
        importlib.reload(bio_mod)
