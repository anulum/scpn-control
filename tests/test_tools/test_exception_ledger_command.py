# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real coverage-ledger policy and command contracts.

"""Exercise byte-identical ledger commands over actual TOML/Python/CI inputs."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _repository(tmp_path: Path) -> Path:
    """Copy the maintained guard, policy and physical CI graph for native IO.

    Parameters
    ----------
    tmp_path : Path
        Pytest-owned physical directory on the configured task storage.

    Returns
    -------
    Path
        Repository fixture; source guard and CI declarations are byte-identical.
    """
    root = tmp_path / "coverage-exception-root"
    names = [
        "tools/__init__.py",
        "tools/coverage_exception_ledger.py",
        "tools/coverage_exception_policy.toml",
        "tools/ci_workflow_inventory.py",
        "tools/ci_workflow_policy.json",
        "pyproject.toml",
    ]
    names.extend(path.relative_to(ROOT).as_posix() for path in (ROOT / ".github/workflows").glob("ci*.yml"))
    for name in names:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    for name in ("src/scpn_control/core", "tests", "validation"):
        (root / name).mkdir(parents=True)
    return root


def _run(root: Path, *args: str, api: bool = False) -> subprocess.CompletedProcess[str]:
    """Execute the physical CLI or public builder in a fresh native process.

    Parameters
    ----------
    root : Path
        Fixture repository; native constants resolve from its copied script.
    *args : str
        CLI arguments; ignored for the no-argument public builder.
    api : bool, optional
        Import build_ledger through a real child-process driver when true.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Actual exit status and decoded standard streams; no provider injection.
    """
    entry = root / "tools/coverage_exception_ledger.py"
    if api:
        entry = root / "public_builder_probe.py"
        entry.write_text(
            "import json\nfrom tools.coverage_exception_ledger import build_ledger\n"
            "print(json.dumps(build_ledger(), sort_keys=True))\n",
            encoding="utf-8",
        )
    argv = [sys.executable]
    config = os.environ.get("SCPN_EXCEPTION_LEDGER_COVERAGE_RC")
    if config:
        argv += ["-m", "coverage", "run", "--parallel-mode", "--rcfile=" + config]
    argv += [str(entry)] + ([] if api else list(args))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(root))
    return subprocess.run(argv, cwd=root.parent, env=env, capture_output=True, text=True, timeout=60)


def _replace(root: Path, key: str, value: object, *, fallback: bool = False) -> None:
    """Write a native policy value without replacing production providers.

    Parameters
    ----------
    root : Path
        Physical fixture repository.
    key : str
        Existing policy key or fallback-rule field.
    value : object
        JSON-compatible TOML literal; None removes the key.
    fallback : bool, optional
        Select the last rule table instead of the policy header.
    """
    path = root / "tools/coverage_exception_policy.toml"
    head, tail = path.read_text(encoding="utf-8").rsplit("[[rules]]", 1)
    text = tail if fallback else head
    replacement = "" if value is None else key + " = " + json.dumps(value)
    text, count = re.subn(r"(?m)^" + re.escape(key) + r" = .*?$", replacement, text, count=1)
    assert count == 1
    path.write_text((head + "[[rules]]" + text) if fallback else (text + "[[rules]]" + tail), encoding="utf-8")


def _seal(root: Path) -> dict[str, Any]:
    """Declare the actual fixture inventory through its public builder.

    Parameters
    ----------
    root : Path
        Physical repository with valid policy and inputs.

    Returns
    -------
    dict[str, Any]
        Actual ledger; its count/digest are written into the fixture policy.
    """
    result = _run(root, api=True)
    assert result.returncode == 0, result.stderr
    ledger: dict[str, Any] = json.loads(result.stdout)
    _replace(root, "expected_total", ledger["entry_count"])
    _replace(root, "expected_sha256", ledger["entry_sha256"])
    return ledger


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema", "scpn-control.coverage-exception-policy.v999"),
        ("schema", None),
        ("expected_total", True),
        ("expected_total", -1),
        ("expected_total", 1.0),
        ("expected_total", "1"),
        ("expected_total", None),
        ("expected_sha256", 7),
        ("expected_sha256", "A" * 64),
        ("expected_sha256", "a" * 63),
        ("expected_sha256", None),
        ("last_review", 7),
        ("last_review", "not-a-date"),
        ("last_review", "2026-02-30"),
        ("last_review", None),
    ],
)
def test_policy_header_refuses_malformed_ownership_values(tmp_path: Path, key: str, value: object) -> None:
    """Malformed policy metadata fails both public inspection and real CLI.

    Parameters
    ----------
    tmp_path : Path
        Physical input directory.
    key : str
        Header declaration under test.
    value : object
        Invalid value or removal of the required field.
    """
    root = _repository(tmp_path)
    _replace(root, key, value)
    for api in (False, True):
        result = _run(root, "--print-summary", api=api)
        assert result.returncode == 1
        assert not result.stdout
        assert "ValueError" in result.stderr
        assert not (root / "tools/coverage_exception_ledger.json").exists()


@pytest.mark.parametrize(
    "key", ["id", "pattern", "external_dependency", "execution_lane", "status", "removal_condition"]
)
@pytest.mark.parametrize("value", [7, "", "   ", None])
def test_rule_required_metadata_is_never_coerced(tmp_path: Path, key: str, value: object) -> None:
    """Numeric, blank and absent rule labels cannot fabricate owned entries.

    Parameters
    ----------
    tmp_path : Path
        Physical input directory.
    key : str
        Required fallback-rule field.
    value : object
        Invalid metadata value.
    """
    root = _repository(tmp_path)
    _replace(root, key, value, fallback=True)
    result = _run(root, "--print-summary")
    assert result.returncode == 1
    assert "must be a nonempty string" in result.stderr
    assert not result.stdout


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("workflow_evidence", 7, "must be a string"),
        ("id", "rust", "duplicate"),
        ("status", "executed-without-evidence", "unsupported"),
        ("pattern", "[", "unterminated"),
    ],
)
def test_rule_identity_status_and_regex_are_native_refusals(
    tmp_path: Path, key: str, value: object, error: str
) -> None:
    """Reject duplicate classifications, unsupported statuses and invalid regex.

    Parameters
    ----------
    tmp_path : Path
        Physical input directory.
    key, error : str
        Invalid field and actual diagnostic substring.
    value : object
        Native TOML value for the malformed declaration.
    """
    root = _repository(tmp_path)
    _replace(root, key, value, fallback=True)
    result = _run(root, "--print-summary")
    assert result.returncode == 1 and error in result.stderr


@pytest.mark.parametrize("rules", ["", "rules = []\n", 'rules = "invalid"\n', "rules = [7]\n"])
def test_rule_collection_requires_native_tables(tmp_path: Path, rules: str) -> None:
    """Missing, empty, scalar and non-table rule collections refuse.

    Parameters
    ----------
    tmp_path : Path
        Physical input directory.
    rules : str
        Replacement native TOML rule declaration.
    """
    root = _repository(tmp_path)
    path = root / "tools/coverage_exception_policy.toml"
    path.write_text(path.read_text().split("[[rules]]", 1)[0] + rules)
    result = _run(root, "--print-summary")
    assert result.returncode == 1 and "ValueError" in result.stderr


@pytest.mark.parametrize("value", ['"2026-09-23"', '"2026-02-30"', "2026-09-23", '"invalid"'])
def test_rule_review_date_is_an_exact_calendar_string(tmp_path: Path, value: str) -> None:
    """Per-rule review overrides require strings and real calendar dates.

    Parameters
    ----------
    tmp_path : Path
        Physical input directory.
    value : str
        Native TOML expression, including a native date object.
    """
    root = _repository(tmp_path)
    path = root / "tools/coverage_exception_policy.toml"
    path.write_text(path.read_text() + "\nlast_review = " + value + "\n")
    result = _run(root, "--print-summary")
    assert result.returncode == (0 if value == '"2026-09-23"' else 1)


def test_real_scanner_extracts_all_declared_exception_families(tmp_path: Path) -> None:
    """Read real Python/TOML text through the public builder and CLI seal.

    Parameters
    ----------
    tmp_path : Path
        Physical repository directory.
    """
    root = _repository(tmp_path)
    source = root / "src/scpn_control/core/rust_backend.py"
    source.write_text("if False:  # pragma: no cover - defensive decoder invariant\n    pass\n")
    test = root / "tests/test_boris_pyo3_bridge.py"
    test.write_text(
        "import pytest\n"
        "pytest.mark.skipif(False, reason='optional Rust backend path')\n"
        "pytest.mark.skipif(False, 'optional JAX backend unavailable')\n"
        "pytest.mark.skipif(False)\n"
        "pytest.skip('compiler unavailable')\n"
        "pytest.skip()\n"
        "pytest.skip(reason=runtime_reason())\n"
        "pytest.mark.xfail(condition())\n"
        "pytest.mark.xfail()\n"
        "pytest.mark.xfail(False, reason='diagnosed decoder boundary')\n"
        "pytest.mark.skip(reason='outside the three-name inventory')\n"
        "(lambda: None)()\n"
    )
    (root / "validation/check.py").write_text("import pytest\npytest.skip(reason='facility data unavailable')\n")
    ledger = _seal(root)
    entries = ledger["entries"]
    pragma = next(entry for entry in entries if entry["kind"] == "pragma-no-cover")
    assert pragma["owner"] == "python/core" and pragma["classification"] == "rust"
    assert pragma["line"] == 1 and pragma["reason"] == "defensive decoder invariant"
    assert ledger["counts"] == {
        "coverage-exclude-pattern": 11,
        "pragma-no-cover": 1,
        "pytest-runtime-skip": 4,
        "pytest-skipif": 3,
        "pytest-xfail": 3,
    }
    assert any(entry["reason"] == "unspecified by call site" for entry in entries)
    assert any(entry["reason"] == "dynamic expression: runtime_reason()" for entry in entries)
    assert any(entry["owner"] == "validation/check" for entry in entries)
    assert all("unexpected pass" in entry["removal_condition"] for entry in entries if entry["kind"] == "pytest-xfail")
    assert _run(root).returncode == 0
    output = root / "tools/coverage_exception_ledger.json"
    rendered = json.dumps(ledger, indent=2, sort_keys=True) + "\n"
    assert output.read_text() == rendered
    assert _run(root, "--check").returncode == 0
    before = output.read_bytes()
    assert _run(root, "--print-summary", "--check").returncode == 0
    assert output.read_bytes() == before


@pytest.mark.parametrize("mode", ["missing", "stale", "invalid-utf8", "count", "digest"])
def test_checked_output_and_inventory_seal_refuse_real_drift(tmp_path: Path, mode: str) -> None:
    """Stale bytes and either changed seal field preserve output on refusal.

    Parameters
    ----------
    tmp_path : Path
        Physical repository directory.
    mode : str
        Missing/stale/undecodable output or independently changed seal field.
    """
    root = _repository(tmp_path)
    _seal(root)
    output = root / "tools/coverage_exception_ledger.json"
    if mode == "stale":
        output.write_text("{}\n")
    elif mode == "invalid-utf8":
        output.write_bytes(b"\xff")
    elif mode == "count":
        _replace(root, "expected_total", 999)
    elif mode == "digest":
        _replace(root, "expected_sha256", "0" * 64)
    original = output.read_bytes() if output.exists() else None
    result = _run(root, "--check")
    assert result.returncode == 1
    assert (output.read_bytes() if output.exists() else None) == original
    assert _run(root, "--print-summary").returncode == 0


@pytest.mark.parametrize(
    "fault",
    [
        "unreasoned",
        "unclassified",
        "workflow",
        "rust-owner",
        "syntax",
        "policy-missing",
        "toml",
        "source-utf8",
        "write-directory",
    ],
)
def test_real_input_and_output_faults_remain_native(tmp_path: Path, fault: str) -> None:
    """Exercise real parser, workflow, source and filesystem failures.

    Parameters
    ----------
    tmp_path : Path
        Physical repository directory.
    fault : str
        Native bad input or blocked direct output path.
    """
    root = _repository(tmp_path)
    path = root / "tools/coverage_exception_policy.toml"
    if fault == "unreasoned":
        (root / "src/scpn_control/core/probe.py").write_text("if False:  # pragma: no cover --#]–—\n    pass\n")
    elif fault == "unclassified":
        _replace(root, "pattern", "DOES-NOT-MATCH", fallback=True)
    elif fault == "workflow":
        _replace(root, "workflow_evidence", "absent-required-physical-job", fallback=True)
    elif fault == "rust-owner":
        (root / "tests/test_missing_rust_owner.py").write_text("import pytest\npytest.skip('optional Rust backend')\n")
    elif fault == "syntax":
        (root / "validation/bad.py").write_text("def invalid(\n")
    elif fault == "policy-missing":
        path.unlink()
    elif fault == "toml":
        path.write_text("schema = [\n")
    elif fault == "source-utf8":
        (root / "src/scpn_control/core/bad.py").write_bytes(b"\xff")
    elif fault == "write-directory":
        _seal(root)
        (root / "tools/coverage_exception_ledger.json").mkdir()
    result = _run(root)
    assert result.returncode == 1 and result.stderr


def test_empty_inventory_and_argparse_use_real_entry_points(tmp_path: Path) -> None:
    """A valid empty inventory writes/checks; help and errors precede reads.

    Parameters
    ----------
    tmp_path : Path
        Physical repository directory.
    """
    root = _repository(tmp_path)
    (root / "pyproject.toml").write_text("[tool.coverage.report]\nexclude_lines = []\n")
    ledger = _seal(root)
    assert ledger["entry_count"] == 0 and ledger["entries"] == [] and ledger["counts"] == {}
    assert _run(root).returncode == 0
    assert _run(root, "--check").returncode == 0
    (root / "tools/coverage_exception_policy.toml").unlink()
    assert _run(root, "--help").returncode == 0
    assert _run(root, "--unknown").returncode == 2
