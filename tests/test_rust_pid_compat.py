# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Optional PID compatibility tests
"""Exercise PID fallback semantics through the public compatibility facade."""

from __future__ import annotations

import math
import sys
from types import ModuleType

import pytest

from scpn_control.control.pid_controller import PIDController
from scpn_control.core._rust_compat import RustPIDController as FacadePID
from scpn_control.core._rust_pid_compat import RustPIDController as LeafPID


def _without_native_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Present an installed extension that has no optional PID symbol."""
    monkeypatch.setitem(sys.modules, "scpn_control_rs", ModuleType("scpn_control_rs"))


def test_facade_preserves_finite_domain_pid_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """The historical import and new owner produce the canonical PID sequence."""
    _without_native_pid(monkeypatch)
    assert FacadePID is LeafPID
    compat = FacadePID(2.0, 0.5, 0.1)
    canonical = PIDController(2.0, 0.5, 0.1)
    assert "mode=fallback" in repr(compat)
    assert (compat.kp, compat.ki, compat.kd) == (2.0, 0.5, 0.1)
    for error in (1.0, -0.5, 3.0, -2.0, 0.25):
        assert compat.step(error) == canonical.step(error)
    compat.reset()
    canonical.reset()
    assert compat.step(0.75) == canonical.step(0.75)


def test_fallback_refuses_overflow_without_state_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A finite input that overflows cannot poison a later public PID step."""
    _without_native_pid(monkeypatch)
    compat = FacadePID(0.0, 0.0, 0.0)
    canonical = PIDController(0.0, 0.0, 0.0)
    assert compat.step(-1e308) == canonical.step(-1e308)
    with pytest.raises(ValueError, match="arithmetic must remain finite"):
        compat.step(1e308)
    assert compat.step(0.5) == canonical.step(0.5)
    with pytest.raises(ValueError, match="error input must be finite"):
        compat.step(math.inf)
    assert compat.step(-0.25) == canonical.step(-0.25)


def test_fallback_presets_and_gain_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Radial/vertical entry points retain gains and reject invalid setups."""
    _without_native_pid(monkeypatch)
    assert (FacadePID.radial().kp, FacadePID.radial().ki, FacadePID.radial().kd) == (2.0, 0.1, 0.5)
    assert (FacadePID.vertical().kp, FacadePID.vertical().ki, FacadePID.vertical().kd) == (5.0, 0.2, 2.0)
    with pytest.raises(ValueError, match="gains must be finite"):
        FacadePID(math.inf, 0.0, 0.0)


def test_optional_native_pid_selection_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selector wiring uses a present native symbol for all constructors."""

    class NativeSelectorProbe(PIDController):
        """Exercise symbol selection only; native arithmetic is tested in Rust."""

        @classmethod
        def radial(cls) -> NativeSelectorProbe:
            """Expose the native radial constructor shape."""
            return cls(2.0, 0.1, 0.5)

        @classmethod
        def vertical(cls) -> NativeSelectorProbe:
            """Expose the native vertical constructor shape."""
            return cls(5.0, 0.2, 2.0)

    module = ModuleType("scpn_control_rs")
    module.__dict__["PyPIDController"] = NativeSelectorProbe
    monkeypatch.setitem(sys.modules, "scpn_control_rs", module)
    pid = FacadePID(2.0, 0.5, 0.1)
    assert "mode=rust" in repr(pid)
    assert pid.step(1.0) == 2.6
    pid.reset()
    assert pid.step(0.0) == 0.0
    assert FacadePID.radial().kp == 2.0
    assert FacadePID.vertical().kd == 2.0
