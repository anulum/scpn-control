# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Sotek. All rights reserved.
# © Code 2020–2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Project: SCPN Control
# Description: Coverage exception ownership-ledger tests.
"""Tests for complete and drift-gated coverage exception ownership."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import coverage_exception_ledger


def test_live_coverage_exception_inventory_is_complete() -> None:
    """All five exception families are present and fully owned."""
    ledger = coverage_exception_ledger.build_ledger()

    assert ledger["counts"]["pragma-no-cover"] == 177
    assert ledger["counts"]["pytest-skipif"] == 128
    assert ledger["counts"]["pytest-runtime-skip"] == 45
    assert ledger["counts"]["pytest-xfail"] == 2
    assert ledger["counts"]["coverage-exclude-pattern"] == 11
    assert all(entry["reason"] and entry["removal_condition"] for entry in ledger["entries"])


def test_committed_coverage_exception_ledger_is_current() -> None:
    """The generated ledger matches source, policy, and workflow evidence."""
    assert coverage_exception_ledger.main(["--check"]) == 0


def test_rust_variant_owner_cannot_disappear_from_native_lane(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A Rust-conditional test file absent from its real lane fails closed."""
    workflow = tmp_path / "ci.yml"
    workflow.write_text(
        coverage_exception_ledger.WORKFLOW_PATH.read_text(encoding="utf-8").replace(
            "tests/test_boris_pyo3_bridge.py",
            "removed.py",
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(coverage_exception_ledger, "WORKFLOW_PATH", workflow)

    with pytest.raises(ValueError, match="omits conditional test owners"):
        coverage_exception_ledger.build_ledger()
