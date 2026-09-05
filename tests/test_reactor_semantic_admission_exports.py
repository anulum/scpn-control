# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Published SPO review to CONTROL decision integration.

"""Public discovery and dependency isolation of admission exports."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


def test_public_admission_exports_resolve_and_are_discoverable() -> None:
    """Discover all public symbols and reject an unknown export."""
    facade = importlib.import_module("scpn_control.reactor_semantic_admission")
    for name in facade.__all__:
        assert name in dir(facade)
        assert getattr(facade, name) is not None
    with pytest.raises(AttributeError, match="has no attribute"):
        facade.unknown_admission


def test_missing_spo_produces_public_sealed_refusal_without_package_replacement() -> None:
    """Import the real facade without site packages and seal its refusal."""
    root = Path(__file__).resolve().parents[1]
    command = """
import sys
sys.path.insert(0, sys.argv[1])
from scpn_control.reactor_semantic_admission import (
    admit_device_diagnostic_plan_review, tokamak_device_diagnostic_review_policy,
    device_diagnostic_review_decision_to_bytes, device_diagnostic_review_decision_from_bytes,
)
decision = admit_device_diagnostic_plan_review(b"{}", policy=tokamak_device_diagnostic_review_policy())
assert decision.refusal_codes == ("spo_contract_unavailable",)
assert decision.accepted_for_review is False
assert decision.actionable is False
assert device_diagnostic_review_decision_from_bytes(device_diagnostic_review_decision_to_bytes(decision)) == decision
assert "scpn_phase_orchestrator" not in sys.modules
"""
    subprocess.run([sys.executable, "-S", "-c", command, str(root / "src")], check=True, timeout=20)
