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
    """All five exception families are present and fully owned.

    Notes
    -----
    This retained repository contract inspects the actual maintained ledger.
    Its declared lane labels do not establish executed variant coverage.
    """
    ledger = coverage_exception_ledger.build_ledger()

    assert ledger["counts"]["pragma-no-cover"] == 173
    assert ledger["counts"]["pytest-skipif"] == 140
    assert ledger["counts"]["pytest-runtime-skip"] == 55
    assert ledger["counts"]["pytest-xfail"] == 2
    assert ledger["counts"]["coverage-exclude-pattern"] == 11
    assert all(entry["reason"] and entry["removal_condition"] for entry in ledger["entries"])


def test_real_provider_and_wider_float_skips_name_their_environment() -> None:
    """Provider and host-range skips cannot claim intrinsic baseline coverage.

    Notes
    -----
    This retained repository contract inspects the actual maintained ledger.
    Its declared lane labels do not establish executed variant coverage.
    """
    entries = coverage_exception_ledger.build_ledger()["entries"]
    provider = [entry for entry in entries if "SCPN_TGLF_BINARY must name" in entry["reason"]]
    wider_float = [
        entry
        for entry in entries
        if entry["reason"].startswith(("extended precision is needed", "longdouble has no wider"))
    ]

    assert len(provider) == 4
    assert len(wider_float) == 2
    assert all(entry["classification"] == "tglf-external-runtime" for entry in provider)
    assert all(entry["classification"] == "extended-floating-range" for entry in wider_float)
    assert all(entry["status"] == "explicit-environment-blocker" for entry in provider + wider_float)
    assert all(entry["external_dependency"] and entry["removal_condition"] for entry in provider + wider_float)
    assert all(entry["last_review"] == "2026-09-23" for entry in provider + wider_float)


def test_device_review_exceptions_keep_their_evidence_boundaries() -> None:
    """Separate installed-package refusals from the upstream decoder invariant.

    Notes
    -----
    This retained repository contract inspects the actual maintained ledger.
    Its declared lane labels do not establish executed variant coverage.
    """
    entries = [
        entry
        for entry in coverage_exception_ledger.build_ledger()["entries"]
        if entry["path"] == "src/scpn_control/reactor_semantic_admission/device_diagnostic_review_admission.py"
    ]
    installation = [entry for entry in entries if entry["reason"] == "SPO installation guard"]
    invariants = [entry for entry in entries if entry["reason"] == "decoder invariant"]
    assert len(entries) == 10
    assert len(installation) == 9
    assert len(invariants) == 1
    assert all(entry["classification"] == "spo-installation-boundary" for entry in installation)
    assert all(entry["status"] == "separate-process-evidence" for entry in installation)
    assert all("isolated wheel installations" in entry["execution_lane"] for entry in installation)
    assert all("collect and merge owner branch coverage" in entry["removal_condition"] for entry in installation)
    assert invariants[0]["classification"] == "intrinsic-control-flow"
    assert invariants[0]["status"] == "reasoned-control-flow"


def test_lif_environment_guards_name_their_separate_process_evidence() -> None:
    """Executable package-layout refusals are not intrinsic unreachable branches.

    Notes
    -----
    This retained repository contract inspects the actual maintained ledger.
    Its declared lane labels do not establish executed variant coverage.
    """
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
    """The generated ledger matches source, policy, and workflow evidence.

    Notes
    -----
    This retained repository contract inspects the actual maintained ledger.
    Its declared lane labels do not establish executed variant coverage.
    """
    assert coverage_exception_ledger.main(["--check"]) == 0


def test_unreviewed_coverage_exception_inventory_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A new exception cannot pass only because the generated file exists.

    Notes
    -----
    This retained repository contract inspects the actual maintained ledger.
    Its declared lane labels do not establish executed variant coverage.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Retained injected observation seam for the original negative test;
        real process qualification is supplied by the command test owner.
    """
    ledger = coverage_exception_ledger.build_ledger()
    ledger["entry_count"] += 1
    monkeypatch.setattr(coverage_exception_ledger, "build_ledger", lambda: ledger)

    assert coverage_exception_ledger.main(["--check"]) == 1


def test_rust_variant_owner_cannot_disappear_from_native_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Rust-conditional test file absent from its real lane fails closed.

    Notes
    -----
    This retained repository contract inspects the actual maintained ledger.
    Its declared lane labels do not establish executed variant coverage.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Retained injected observation seam for the original negative test;
        real process qualification is supplied by the command test owner.
    """
    workflow = read_ci_workflow_source().replace(
        "tests/test_boris_pyo3_bridge.py",
        "removed.py",
    )
    monkeypatch.setattr(coverage_exception_ledger, "read_ci_workflow_source", lambda: workflow)

    with pytest.raises(ValueError, match="omits conditional test owners"):
        coverage_exception_ledger.build_ledger()
