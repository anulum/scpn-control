# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Exception-ledger policy contract tests.

"""Tests for complete and drift-gated coverage exception ownership."""

from __future__ import annotations

import pytest

from tools import coverage_exception_ledger
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_live_coverage_exception_inventory_is_complete() -> None:
    """All five exception families are present and fully owned."""
    ledger = coverage_exception_ledger.build_ledger()

    assert ledger["counts"]["pragma-no-cover"] == 182
    assert ledger["counts"]["pytest-skipif"] == 128
    assert ledger["counts"]["pytest-runtime-skip"] == 44
    assert ledger["counts"]["pytest-xfail"] == 2
    assert ledger["counts"]["coverage-exclude-pattern"] == 11
    assert all(entry["reason"] and entry["removal_condition"] for entry in ledger["entries"])


def test_lif_environment_guards_name_their_separate_process_evidence() -> None:
    """Executable package-layout refusals are not intrinsic unreachable branches."""
    entries = coverage_exception_ledger.build_ledger()["entries"]
    guards = [
        entry
        for entry in entries
        if entry["path"] == "src/scpn_control/scpn/exact_current_lif_runtime.py"
        and entry["reason"] == "environment guard"
    ]

    assert len(guards) == 7
    assert all(entry["classification"] == "package-layout-boundary" for entry in guards)
    assert all(entry["status"] == "separate-process-evidence" for entry in guards)
    assert all("test_exact_current_lif_runtime.py" in entry["execution_lane"] for entry in guards)
    assert all("installed-wheel" in entry["removal_condition"] for entry in guards)


def test_committed_coverage_exception_ledger_is_current() -> None:
    """The generated ledger matches source, policy, and workflow evidence."""
    assert coverage_exception_ledger.main(["--check"]) == 0


def test_rust_variant_owner_cannot_disappear_from_native_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Rust-conditional test file absent from its real lane fails closed."""
    workflow = read_ci_workflow_source().replace(
        "tests/test_boris_pyo3_bridge.py",
        "removed.py",
    )
    monkeypatch.setattr(coverage_exception_ledger, "read_ci_workflow_source", lambda: workflow)

    with pytest.raises(ValueError, match="omits conditional test owners"):
        coverage_exception_ledger.build_ledger()
