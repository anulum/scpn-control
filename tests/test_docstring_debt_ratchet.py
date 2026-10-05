# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — tests for the docstring debt ratchet.
"""Exercise the debt gate through real Ruff and CLI runs on repository owners.

Copied maintained owners retain the actual project configuration. Mutations
remove native docstrings or damage real source; no probe file is written into
canonical source and no mocked subprocess provides acceptance evidence.
"""

from __future__ import annotations

import ast
import doctest
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools import check_docstring_debt as gate


@pytest.fixture
def repository_scope(tmp_path: Path) -> Path:
    """Copy one actual owner per debt directory with the production Ruff configuration."""
    owners = (
        "tests/test_check_test_quality_policy.py",
        "tools/check_test_quality_policy.py",
        "validation/control_benchmark_suite.py",
    )
    for owner in owners:
        target = tmp_path / owner
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(gate.ROOT / owner, target)
    shutil.copy2(gate.ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    counts = gate.measure(tmp_path)
    _write_ledger(tmp_path, sum(counts.values()), counts)
    return tmp_path


def _write_ledger(repo: Path, total: int, counts: dict[str, int] | None = None) -> Path:
    """Write the explicit scope ceiling without altering the canonical ledger."""
    payload: dict[str, object] = {"total": total, "scope": list(gate.SCOPE)}
    if counts is not None:
        payload["by_code"] = counts
    path = repo / "tools" / gate.LEDGER.name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _cli(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Launch the actual script with its selected repository and a bounded runtime."""
    return subprocess.run(
        [sys.executable, str(gate.ROOT / "tools/check_docstring_debt.py"), "--repo", str(repo), *args],
        cwd=gate.ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _remove_guard_docstring(repo: Path) -> Path:
    """Remove the actual public collector contract from the copied production guard."""
    source = repo / "tools/check_test_quality_policy.py"
    tree = ast.parse(source.read_text())
    collector = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "collect_violations"
    )
    assert ast.get_docstring(collector)
    collector.body.pop(0)
    source.write_text(ast.unparse(tree), encoding="utf-8")
    return source


def test_live_repository_holds_its_ceiling() -> None:
    """Measure current real source; a falling count stays valid without rewriting evidence."""
    counts = gate.measure(gate.ROOT)
    assert sum(counts.values()) <= gate.read_ceiling(gate.LEDGER)
    assert all(code.startswith("D") for code in counts)
    assert "D105" not in counts and "D107" not in counts
    assert gate.main([]) == 0


def test_real_cli_reports_documented_owners(repository_scope: Path) -> None:
    """The exact default measurement and native-doc owners pass the subprocess boundary."""
    result = _cli(repository_scope)
    assert result.returncode == 0, result.stderr
    assert "PASS: docstring debt holds" in result.stdout


def test_new_owner_and_mixed_findings_raise_only_documentation_debt(repository_scope: Path) -> None:
    """A copied maintained gate absent from the ledger cannot evade measurement."""
    before = sum(gate.measure(repository_scope).values())
    source = repository_scope / "tools/check_docstring_debt.py"
    shutil.copy2(gate.ROOT / "tools/check_docstring_debt.py", source)
    original = source.read_text()
    tree = ast.parse(original)
    lines = original.splitlines(keepends=True)
    spans = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node):
            contract = node.body[0]
            assert contract.end_lineno is not None
            spans.append((contract.lineno - 1, contract.end_lineno))
    # Preserve the production source's Python 3.11 quote syntax. ast.unparse on
    # newer runtimes can produce PEP 701 f-strings that the configured Ruff
    # target correctly refuses before it can measure documentation debt.
    for start, stop in sorted(spans, reverse=True):
        del lines[start:stop]
    source.write_text("".join(lines) + "\nimport os\n", encoding="utf-8")
    raw = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--output-format", "json", "--no-cache", str(source)],
        capture_output=True,
        text=True,
        cwd=repository_scope,
        check=False,
        timeout=30,
    )
    codes = [finding["code"] for finding in json.loads(raw.stdout)]
    assert "F401" in codes and "D103" in codes
    assert sum(gate.measure(repository_scope).values()) - before == sum(code.startswith("D") for code in codes)
    assert _cli(repository_scope).returncode == 1


@pytest.mark.parametrize("update", [False, True])
def test_breach_refuses_without_rewriting(repository_scope: Path, update: bool) -> None:
    """Removing a real contract cannot pass or raise the recorded ceiling."""
    ledger = repository_scope / "tools" / gate.LEDGER.name
    before = ledger.read_bytes()
    _remove_guard_docstring(repository_scope)
    result = _cli(repository_scope, *(["--update"] if update else []))
    assert result.returncode == 1
    assert ledger.read_bytes() == before
    assert "refusing to raise" in result.stdout if update else "debt rose" in result.stdout


@pytest.mark.parametrize("update", [False, True])
def test_falling_debt_only_rewrites_on_request(repository_scope: Path, update: bool) -> None:
    """A documented owner reduces the count, with update retaining the exact breakdown."""
    counts = gate.measure(repository_scope)
    ledger = _write_ledger(repository_scope, sum(counts.values()) + 1)
    before = ledger.read_bytes()
    result = _cli(repository_scope, *(["--update"] if update else []))
    assert result.returncode == 0, result.stderr
    if update:
        payload = json.loads(ledger.read_text())
        assert payload["total"] == sum(counts.values()) and payload["by_code"] == counts
        assert gate.read_ceiling(ledger) == sum(counts.values())
    else:
        assert ledger.read_bytes() == before
        assert "run --update" in result.stdout


@pytest.mark.parametrize("total", [True, False, 2.9, "100", -1, None])
def test_ledger_refuses_coerced_or_negative_totals(repository_scope: Path, total: object) -> None:
    """Numeric coercion cannot silently change the accepted documentation threshold."""
    ledger = repository_scope / "tools" / gate.LEDGER.name
    ledger.write_text(json.dumps({"total": total, "scope": list(gate.SCOPE)}))
    before = ledger.read_bytes()
    with pytest.raises(RuntimeError, match="nonnegative integer"):
        gate.read_ceiling(ledger)
    result = _cli(repository_scope, "--update")
    assert result.returncode == 2 and "nonnegative integer" in result.stderr
    assert ledger.read_bytes() == before


@pytest.mark.parametrize(
    "payload",
    [
        "{}",
        "not json",
        "[]",
        '{"total": 0, "total": 100, "scope": ["tests", "tools", "validation"]}',
        '{"total": 0, "scope": ["tools", "tests", "validation"]}',
        '{"total": 0, "scope": ["tests", "tools", "validation"], "by_code": []}',
        '{"total": 0, "scope": ["tests", "tools", "validation"], "by_code": {"F401": 0}}',
        '{"total": 0, "scope": ["tests", "tools", "validation"], "by_code": {"D103": true}}',
        '{"total": 0, "scope": ["tests", "tools", "validation"], "by_code": {"D103": -1}}',
        '{"total": 0, "scope": ["tests", "tools", "validation"], "by_code": {"D103": 1}}',
    ],
)
def test_cli_refuses_malformed_ledger(repository_scope: Path, payload: str) -> None:
    """JSON shape, uniqueness, scope and count consistency are admission requirements."""
    ledger = repository_scope / "tools" / gate.LEDGER.name
    ledger.write_text(payload)
    result = _cli(repository_scope)
    assert result.returncode == 2 and "malformed docstring debt ledger" in result.stderr


@pytest.mark.parametrize("defect", ["missing", "utf8", "directory"])
def test_cli_refuses_unreadable_ledger(repository_scope: Path, defect: str) -> None:
    """Missing, undecodable or non-file ledgers fail at the real filesystem boundary."""
    ledger = repository_scope / "tools" / gate.LEDGER.name
    if defect == "utf8":
        ledger.write_bytes(b"\xff")
    else:
        ledger.rename(ledger.with_suffix(".saved"))
        if defect == "directory":
            ledger.mkdir()
    result = _cli(repository_scope)
    assert result.returncode == 2 and "docstring debt ledger" in result.stderr


@pytest.mark.parametrize("defect", ["missing_scope", "empty_scope", "syntax", "config"])
def test_real_ruff_measurement_refuses_incomplete_inspection(repository_scope: Path, defect: str) -> None:
    """Scope loss, broken source and invalid configuration cannot report zero debt."""
    if defect == "missing_scope":
        (repository_scope / "validation").rename(repository_scope / "saved_validation")
    elif defect == "empty_scope":
        source = repository_scope / "validation/control_benchmark_suite.py"
        source.rename(source.with_suffix(".saved"))
    elif defect == "syntax":
        source = repository_scope / "tools/check_test_quality_policy.py"
        source.write_text(source.read_text() + "\ndef broken(\n")
    else:
        (repository_scope / "pyproject.toml").write_text("[tool.ruff\n")
    with pytest.raises(RuntimeError):
        gate.measure(repository_scope)
    result = _cli(repository_scope, "--update")
    assert result.returncode == 2 and "FAIL:" in result.stderr


def test_cli_resolves_relative_root(repository_scope: Path) -> None:
    """Repository selection is relative to the caller, independent of script location."""
    result = subprocess.run(
        [sys.executable, str(gate.ROOT / "tools/check_docstring_debt.py"), "--repo", "."],
        cwd=repository_scope,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_update_reports_actual_filesystem_write_failure(repository_scope: Path) -> None:
    """A valid non-increasing comparison cannot report success when its ledger is read-only."""
    ledger = repository_scope / "tools" / gate.LEDGER.name
    before = ledger.read_bytes()
    mode = ledger.stat().st_mode
    ledger.chmod(0o400)
    try:
        result = _cli(repository_scope, "--update")
        assert result.returncode == 2 and "could not write" in result.stderr
        assert ledger.read_bytes() == before
    finally:
        ledger.chmod(mode)


def test_probe_reports_missing_interpreter(repository_scope: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real process boundary refuses execution when the selected interpreter is absent."""
    interpreter = repository_scope / "absent-interpreter"
    assert not interpreter.exists()
    monkeypatch.setattr(sys, "executable", str(interpreter))
    with pytest.raises(RuntimeError, match="could not run ruff"):
        gate.measure(repository_scope)


def test_api_resolves_relative_repository(repository_scope: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The public Python entry point resolves its relative scope before setting subprocess cwd."""
    expected = gate.measure(repository_scope)
    monkeypatch.chdir(repository_scope)
    assert gate.measure(Path(".")) == expected


def test_native_documentation_example_executes() -> None:
    """Execute the native measurement example against maintained repository paths."""
    result = doctest.testmod(gate, raise_on_error=True)
    assert result.failed == 0 and result.attempted == 2


@pytest.mark.parametrize("fault", ["silence", "json", "object", "code"])
def test_public_measure_refuses_corrupted_real_ruff_response(
    repository_scope: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """Corrupt an executed Ruff response at the capture boundary and require refusal.

    The real tool still inspects a maintained owner. Fault injection establishes
    the gate's response-validation contract, not a claim that Ruff emits these
    malformed responses during normal operation.
    """
    _remove_guard_docstring(repository_scope)
    real_run = subprocess.run

    def capture_with_fault(
        argv: list[str], *, capture_output: bool, text: bool, cwd: Path, check: bool
    ) -> subprocess.CompletedProcess[str]:
        """Execute the actual subprocess before corrupting its captured protocol bytes."""
        result = real_run(argv, capture_output=capture_output, text=text, cwd=cwd, check=check)
        assert result.returncode == 1
        diagnostics = json.loads(result.stdout)
        assert any(finding["code"] == "D103" for finding in diagnostics)
        if fault == "silence":
            result.stdout = ""
        elif fault == "json":
            result.stdout += "trailing corruption"
        elif fault == "object":
            result.stdout = json.dumps({"diagnostics": diagnostics})
        else:
            diagnostics[0].pop("code")
            result.stdout = json.dumps(diagnostics)
        return result

    monkeypatch.setattr(subprocess, "run", capture_with_fault)
    with pytest.raises(RuntimeError, match="ruff"):
        gate.measure(repository_scope)
