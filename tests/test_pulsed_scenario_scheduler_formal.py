# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Pulsed scheduler formal artefact tests.
"""Formal artefact checks for the CON-C.1 pulsed scheduler."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEAN_PROOF = PROJECT_ROOT / "lean" / "SCPNControl" / "PulsedFSM.lean"


def test_pulsed_fsm_lean_proof_declares_required_theorems() -> None:
    proof = LEAN_PROOF.read_text(encoding="utf-8")

    assert "theorem pulsed_fsm_eventually_returns_to_idle" in proof
    assert "theorem adjacent_transition_deterministic" in proof
    assert "theorem manual_transition_cannot_skip_burn_from_idle" in proof
    assert "stepN n state = State.idle" in proof


def test_pulsed_fsm_lean_builds_when_enabled() -> None:
    if os.environ.get("MIF_LEAN_CI") != "1":
        pytest.skip("set MIF_LEAN_CI=1 to require lake build for pulsed FSM proof")
    if shutil.which("lake") is None:
        pytest.skip("Lean lake executable is not installed in this environment")

    subprocess.run(["lake", "build"], cwd=PROJECT_ROOT, check=True)
