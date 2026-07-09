# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tokamak digital-twin IMAS import boundary tests
"""Strict regression tests for optional digital-twin IDS import wiring."""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Mapping, Sequence


def _summary_to_ids(summary: Mapping[str, object], **metadata: object) -> dict[str, object]:
    """Return a minimal fake IDS summary payload."""
    return {"kind": "summary", "summary": dict(summary), **metadata}


def _history_to_ids(snapshots: Sequence[Mapping[str, object]], **metadata: object) -> dict[str, object]:
    """Return a minimal fake IDS history payload."""
    return {"kind": "history", "count": len(snapshots), **metadata}


def _pulse_to_ids(snapshots: Sequence[Mapping[str, object]], **metadata: object) -> dict[str, object]:
    """Return a minimal fake IDS pulse payload."""
    return {"kind": "pulse", "count": len(snapshots), **metadata}


def test_digital_twin_import_marks_imas_available_when_connector_imports() -> None:
    """Reloading with an importable connector covers the optional IDS true branch."""
    module_name = "scpn_control.io.imas_connector"
    original_connector = sys.modules.get(module_name)

    fake_connector = types.ModuleType(module_name)
    fake_connector.__dict__["digital_twin_summary_to_ids"] = _summary_to_ids
    fake_connector.__dict__["digital_twin_history_to_ids"] = _history_to_ids
    fake_connector.__dict__["digital_twin_history_to_ids_pulse"] = _pulse_to_ids

    import scpn_control.control.tokamak_digital_twin as digital_twin

    try:
        sys.modules[module_name] = fake_connector
        reloaded = importlib.reload(digital_twin)

        assert reloaded.HAS_IMAS is True
        assert reloaded.digital_twin_summary_to_ids({"steps": 1})["kind"] == "summary"
    finally:
        if original_connector is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original_connector
        importlib.reload(digital_twin)
