# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Test quality policy guard tests

from __future__ import annotations

from pathlib import Path

from tools.check_test_quality_policy import collect_violations


def test_test_quality_policy_rejects_bucket_filename(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_cov_final_gaps.py").write_text(
        "def test_placeholder():\n    assert True\n",
        encoding="utf-8",
    )

    violations = collect_violations(tests)

    assert len(violations) == 1
    assert "forbidden generic bucket test filename" in violations[0].reason


def test_test_quality_policy_rejects_slop_intent_markers(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    forbidden_text = '"""coverage ' + 'gaps for uncovered ' + 'lines."""\n'
    (tests / "test_named_module.py").write_text(forbidden_text, encoding="utf-8")

    violations = collect_violations(tests)

    assert len(violations) == 2
    assert all("forbidden coverage-slop wording" in violation.reason for violation in violations)


def test_test_quality_policy_accepts_named_contract_surface(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_named_module.py").write_text(
        '"""Boundary contract tests for named module behaviour."""\n',
        encoding="utf-8",
    )

    assert collect_violations(tests) == []
