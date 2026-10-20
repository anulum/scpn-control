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
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools import check_docstring_debt as gate
from tools import preflight
from tools.docstring_debt_ledger import SOURCE_FINGERPRINT_PROTOCOL, measure_owner_sources, measurement_profile


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
    owners = gate.measure_owners(repo)
    payload: dict[str, object] = {
        "total": total,
        "scope": list(gate.SCOPE),
        "per_file": owners,
        "profile": measurement_profile(repo),
        "source_fingerprints": measure_owner_sources(repo, owners),
        "source_fingerprint_protocol": SOURCE_FINGERPRINT_PROTOCOL,
    }
    if counts is not None:
        payload["by_code"] = counts
    path = repo / "tools" / gate.LEDGER.name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _cli(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Launch the actual script with its selected repository and a bounded runtime."""
    command = [sys.executable]
    coverage_rc = os.environ.get("SCPN_DOCSTRING_DEBT_COVERAGE_RC")
    if coverage_rc:
        command += ["-m", "coverage", "run", "--parallel-mode", f"--rcfile={coverage_rc}"]
    command += [str(gate.ROOT / "tools/check_docstring_debt.py"), "--repo", str(repo), *args]
    return subprocess.run(
        command,
        cwd=gate.ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _remove_guard_docstring(
    repo: Path, owner: str = "tools/check_test_quality_policy.py", symbol: str = "collect_violations"
) -> Path:
    """Remove one actual public callable contract without reformatting other source."""
    source = repo / owner
    original = source.read_text()
    tree = ast.parse(original)
    collector = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == symbol)
    assert ast.get_docstring(collector)
    contract = collector.body[0]
    assert contract.end_lineno is not None
    lines = original.splitlines(keepends=True)
    del lines[contract.lineno - 1 : contract.end_lineno]
    source.write_text("".join(lines), encoding="utf-8")
    return source


def _document_historical_test(source: Path) -> str:
    """Repair one real copied test contract while preserving all other source text."""
    original = source.read_text()
    function = next(
        node
        for node in ast.parse(original).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_") and not ast.get_docstring(node)
    )
    first = function.body[0]
    lines = original.splitlines(keepends=True)
    lines.insert(
        first.lineno - 1, " " * first.col_offset + '"""Exercise the maintained electromagnetic regression case."""\n'
    )
    source.write_text("".join(lines), encoding="utf-8")
    return function.name


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


@pytest.mark.parametrize(
    "fault", ["silence", "json", "object", "code", "filename", "foreign_filename", "integer", "nesting"]
)
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
        argv: list[str], *, capture_output: bool, text: bool, cwd: Path, check: bool, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        """Execute the actual subprocess before corrupting its captured protocol bytes."""
        result = real_run(argv, capture_output=capture_output, text=text, cwd=cwd, check=check, timeout=timeout)
        assert result.returncode == 1
        diagnostics = json.loads(result.stdout)
        assert any(finding["code"] == "D103" for finding in diagnostics)
        if fault == "silence":
            result.stdout = ""
        elif fault == "json":
            result.stdout += "trailing corruption"
        elif fault == "object":
            result.stdout = json.dumps({"diagnostics": diagnostics})
        elif fault == "code":
            diagnostics[0].pop("code")
            result.stdout = json.dumps(diagnostics)
        elif fault == "filename":
            diagnostics[0].pop("filename")
            result.stdout = json.dumps(diagnostics)
        elif fault == "foreign_filename":
            diagnostics[0]["filename"] = str(repository_scope.parent / "outside.py")
            result.stdout = json.dumps(diagnostics)
        elif fault == "integer":
            result.stdout = "[" + "9" * 5000 + "]"
        else:
            result.stdout = "[" * 20000 + "0" + "]" * 20000
        return result

    monkeypatch.setattr(subprocess, "run", capture_with_fault)
    with pytest.raises(RuntimeError, match="ruff"):
        gate.measure(repository_scope)


def test_repository_loop_is_an_authored_api_and_cli_refusal(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Refuse a real repository link loop before inspection, with no traceback or writes."""
    loop = tmp_path / "repository-loop"
    loop.symlink_to(loop.name, target_is_directory=True)
    assert gate.main(["--repo", str(loop)]) == 2
    captured = capsys.readouterr()
    assert not captured.out and captured.err.startswith("FAIL:")
    source_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(source_root / "tools/check_docstring_debt.py"), "--repo", str(loop)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2 and not result.stdout
    assert result.stderr.startswith("FAIL:") and "Traceback" not in result.stderr


@pytest.mark.parametrize("update", [False, True])
def test_missing_root_refuses_without_creating_a_ledger(tmp_path: Path, update: bool) -> None:
    """A missing debt scope refuses through the actual API and CLI, including update mode."""
    missing = tmp_path / "absent-repository"
    arguments = ["--repo", str(missing), *(["--update"] if update else [])]
    assert gate.main(arguments) == 2
    result = subprocess.run(
        [sys.executable, str(gate.ROOT / "tools/check_docstring_debt.py"), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2 and "debt scope missing" in result.stderr
    assert not result.stdout and "Traceback" not in result.stderr
    assert not missing.exists()


@pytest.mark.parametrize("update", [False, True])
@pytest.mark.parametrize("new_owner", [False, True])
def test_real_cli_refuses_flat_total_owner_exchange(repository_scope: Path, update: bool, new_owner: bool) -> None:
    """Repairing a historical test cannot fund undocumented code in another owner."""
    historical = repository_scope / "tests/test_gk_electromagnetic.py"
    shutil.copy2(gate.ROOT / "tests/test_gk_electromagnetic.py", historical)
    counts = gate.measure(repository_scope)
    assert counts["D103"] > 0
    ledger = _write_ledger(repository_scope, sum(counts.values()), counts)
    before = ledger.read_bytes()
    _document_historical_test(historical)
    owner = "tools/new_guard.py" if new_owner else "tools/check_test_quality_policy.py"
    if new_owner:
        shutil.copy2(gate.ROOT / "tools/check_test_quality_policy.py", repository_scope / owner)
    _remove_guard_docstring(repository_scope, owner)
    assert sum(gate.measure(repository_scope).values()) == sum(counts.values())
    result = _cli(repository_scope, *(["--update"] if update else []))
    assert result.returncode == 1 and "owner/rule" in result.stdout and owner in result.stdout
    assert ledger.read_bytes() == before


@pytest.mark.parametrize("update", [False, True])
def test_real_cli_refuses_different_rule_within_same_owner(repository_scope: Path, update: bool) -> None:
    """A missing-contract repair cannot buy a new punctuation defect in that file."""
    source = _remove_guard_docstring(repository_scope)
    counts = gate.measure(repository_scope)
    ledger = _write_ledger(repository_scope, sum(counts.values()), counts)
    before = ledger.read_bytes()
    original = (gate.ROOT / "tools/check_test_quality_policy.py").read_text()
    tree = ast.parse(original)
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "collect_violations"
    )
    contract = function.body[0]
    lines = original.splitlines(keepends=True)
    assert lines[contract.lineno - 1].endswith(".\n")
    lines[contract.lineno - 1] = lines[contract.lineno - 1][:-2] + "\n"
    source.write_text("".join(lines), encoding="utf-8")
    after = gate.measure(repository_scope)
    assert after.get("D400", 0) > counts.get("D400", 0)
    assert sum(after.values()) == sum(counts.values())
    result = _cli(repository_scope, *(["--update"] if update else []))
    assert result.returncode == 1 and "owner/rule" in result.stdout
    assert ledger.read_bytes() == before


@pytest.mark.parametrize("defect", ["configuration", "tool", "aggregate_only"])
def test_real_cli_refuses_unqualified_profile_or_missing_owner_bounds(repository_scope: Path, defect: str) -> None:
    """Changed selection/tool declarations need qualification, not an automatic baseline reset."""
    ledger = repository_scope / "tools" / gate.LEDGER.name
    payload = json.loads(ledger.read_text())
    if defect == "configuration":
        config = repository_scope / "pyproject.toml"
        config.write_text(config.read_text() + "\n# changed profile bytes\n")
    elif defect == "tool":
        payload["profile"]["ruff_version"] = "0.0.1"
        ledger.write_text(json.dumps(payload))
    else:
        payload.pop("per_file")
        ledger.write_text(json.dumps(payload))
    before = ledger.read_bytes()
    result = _cli(repository_scope, "--update")
    assert result.returncode == 2 and "FAIL:" in result.stderr and "Traceback" not in result.stderr
    assert ledger.read_bytes() == before


@pytest.mark.parametrize("update", [False, True])
@pytest.mark.parametrize("exchange", [False, True])
def test_changed_file_cannot_retain_or_exchange_individual_old_cases(
    repository_scope: Path, update: bool, exchange: bool
) -> None:
    """A smaller or flat same-rule count cannot clear a changed file's remaining debt."""
    owner = "tests/test_gk_electromagnetic.py"
    source = repository_scope / owner
    shutil.copy2(gate.ROOT / owner, source)
    previously_documented = _document_historical_test(source)
    counts = gate.measure(repository_scope)
    ledger = _write_ledger(repository_scope, sum(counts.values()), counts)
    before = ledger.read_bytes()
    assert _cli(repository_scope).returncode == 0
    _document_historical_test(source)
    if exchange:
        _remove_guard_docstring(repository_scope, owner, previously_documented)
    after = gate.measure(repository_scope)
    assert sum(after.values()) == sum(counts.values()) - (0 if exchange else 1)
    result = _cli(repository_scope, *(["--update"] if update else []))
    assert result.returncode == 1 and "changed owners" in result.stdout and owner in result.stdout
    assert ledger.read_bytes() == before


@pytest.mark.parametrize("admitted", [False, True])
def test_registered_preflight_command_propagates_default_owner_gate(
    repository_scope: Path, admitted: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    """The real registered command admits valid default input and propagates native rejection."""
    for name in ["tools/check_docstring_debt.py", "tools/docstring_debt_ledger.py"]:
        shutil.copy2(gate.ROOT / name, repository_scope / name)
    if not admitted:
        _remove_guard_docstring(repository_scope)
    name, command, _cwd = next(row for row in preflight.GATES if row[0] == "docstring-debt-ratchet")
    assert preflight.run_gate(name, command, repository_scope) is admitted
    output = capsys.readouterr().out
    assert name in output and ("PASS" if admitted else "FAIL") in output
