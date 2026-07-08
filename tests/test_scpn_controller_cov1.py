# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — SCPN controller coverage hardening tests
"""COV-1 regression tests for SCPN controller import-time runtime guards."""

from __future__ import annotations

import importlib
import sys

import pytest

import scpn_control.scpn.controller as controller_mod


def test_absent_rust_runtime_reload_covers_import_failure_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reload without `scpn_control_rs` so the optional Rust-runtime guard is covered."""
    try:
        with monkeypatch.context() as patch:
            patch.delitem(sys.modules, "scpn_control_rs", raising=False)
            importlib.reload(controller_mod)

            assert controller_mod._HAS_RUST_SCPN_RUNTIME is False
            assert controller_mod._rust_dense_activations is None
            assert controller_mod._rust_marking_update is None
            assert controller_mod._rust_sample_firing is None
    finally:
        importlib.reload(controller_mod)
