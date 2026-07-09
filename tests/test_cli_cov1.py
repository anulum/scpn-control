# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CLI coverage hardening tests
"""COV-1 regression tests for CLI validation edge paths."""

from __future__ import annotations

import json
import sys
from types import ModuleType
from typing import Any

import pytest
from click.testing import CliRunner

from scpn_control.cli import main

_OPTIONAL_IMPORT_GUARDS = ("matplotlib", "torch", "streamlit")


def _clear_optional_import_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove optional-module sentinels that affect CLI import hygiene checks."""
    for module_name in _OPTIONAL_IMPORT_GUARDS:
        monkeypatch.delitem(sys.modules, module_name, raising=False)


def test_validate_reports_contaminated_import_without_external_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The validate command reports prohibited eager imports before evidence gates."""
    _clear_optional_import_guards(monkeypatch)
    monkeypatch.setitem(sys.modules, "streamlit", ModuleType("streamlit"))
    result = CliRunner().invoke(
        main,
        [
            "validate",
            "--json-out",
            "--no-data-manifests",
            "--no-jax-gk-parity",
            "--no-physics-traceability",
            "--no-multi-shot-campaign-evidence",
            "--no-runtime-admission-evidence",
            "--no-native-formal-certificate",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert isinstance(payload, dict)
    data = dict[str, Any](payload)
    assert data["status"] == "fail"
    assert data["import_clean"] is False
    assert data["contaminated_module"] == "streamlit"
