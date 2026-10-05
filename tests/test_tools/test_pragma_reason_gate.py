# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Coverage pragma checker tests.

"""Tests for the coverage-pragma reason gate."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "check_coverage_pragmas.py"


def _load_tool() -> ModuleType:
    """Load the actual repository guard with native dataclass registration.

    Returns
    -------
    ModuleType
        Executed guard module, without replacing its implementation or globals.
    """
    spec = importlib.util.spec_from_file_location("check_coverage_pragmas", TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def pragma_tool() -> ModuleType:
    """Load the checker from the checked-out repository.

    Returns
    -------
    ModuleType
        Executed maintained implementation for public API assertions.
    """
    return _load_tool()


def test_reason_parser_accepts_hyphen_colon_and_dash_reasons(pragma_tool: ModuleType) -> None:
    """Reasoned coverage pragmas may use common separator styles.

    Parameters
    ----------
    pragma_tool : ModuleType
        Actual guard for the retained private-parser characterisation.
    """
    assert pragma_tool._is_reasoned(" - optional dependency path") is True
    assert pragma_tool._is_reasoned(": defensive guard") is True
    assert pragma_tool._is_reasoned(" — unreachable numerical branch") is True
    assert pragma_tool._is_reasoned("") is False
    assert pragma_tool._is_reasoned(")") is False


def test_checker_reports_bare_pragmas(tmp_path: Path, pragma_tool: ModuleType) -> None:
    """Bare pragmas are returned with path, line, and source text.

    Parameters
    ----------
    tmp_path : Path
        Physical UTF-8 source allocation outside the checkout.
    pragma_tool : ModuleType
        Actual public diagnostic producer.
    """
    source = tmp_path / "example.py"
    source.write_text(
        "\n".join(
            [
                "def optional_path():  # pragma: no cover - optional dependency path",
                "    return 1",
                "def bare_path():  # pragma: no cover",
                "    return 2",
            ]
        ),
        encoding="utf-8",
    )

    violations = pragma_tool.find_unreasoned_pragmas([source])

    assert len(violations) == 1
    assert violations[0].line == 3
    assert violations[0].text == "def bare_path():  # pragma: no cover"


def test_live_source_tree_has_no_unreasoned_pragmas(pragma_tool: ModuleType) -> None:
    """Every source coverage exclusion must carry a one-line reason.

    Parameters
    ----------
    pragma_tool : ModuleType
        Public guard scanning the actual maintained source tree.
    """
    violations = pragma_tool.find_unreasoned_pragmas([REPO_ROOT / "src" / "scpn_control"])

    assert violations == []


def test_main_fails_closed_when_unreasoned_pragmas_are_reported(
    pragma_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI must fail if a future bare pragma enters the source tree.

    Parameters
    ----------
    pragma_tool : ModuleType
        Retained legacy CLI subject.
    monkeypatch : pytest.MonkeyPatch
        Substitutes a diagnostic in this legacy unit characterisation only.
    capsys : pytest.CaptureFixture[str]
        Captures the retained summary and diagnostic assertion.
    """
    violation = pragma_tool.CoveragePragmaViolation(
        path="src/scpn_control/example.py",
        line=10,
        text="if optional:  # pragma: no cover",
    )
    monkeypatch.setattr(pragma_tool, "find_unreasoned_pragmas", lambda _paths: [violation])

    assert pragma_tool.main([]) == 1
    assert "src/scpn_control/example.py:10" in capsys.readouterr().out


def _cli_tree(tmp_path: Path, text: str = "value = 1\n") -> Path:
    """Build a physical repository for the byte-identical maintained CLI.

    Parameters
    ----------
    tmp_path : Path
        Isolated pytest allocation, outside the actual checkout.
    text : str, optional
        UTF-8 contents for the default source file.

    Returns
    -------
    Path
        Fixture root containing the actual guard and source directory.
    """
    root = tmp_path / "pragma-policy"
    tools = root / "tools"
    source = root / "src" / "scpn_control"
    tools.mkdir(parents=True)
    source.mkdir(parents=True)
    (tools / TOOL_PATH.name).write_bytes(TOOL_PATH.read_bytes())
    (source / "example.py").write_text(text, encoding="utf-8")
    assert (tools / TOOL_PATH.name).read_bytes() == TOOL_PATH.read_bytes()
    return root


def _cli(root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    """Execute the actual CLI from outside its repository root.

    Parameters
    ----------
    root : Path
        Physical repository from :func:`_cli_tree`.
    args : list[str]
        Literal CLI tokens. No shell or source mocking is used.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Native process exit and captured UTF-8 output. A supplied coverage
        configuration records the child separately without excluding branches.
    """
    argv = [sys.executable]
    coverage_rc = os.environ.get("SCPN_COVERAGE_PRAGMA_COVERAGE_RC")
    if coverage_rc:
        argv += ["-m", "coverage", "run", "--rcfile=" + coverage_rc, "--parallel-mode"]
    argv += [str(root / "tools" / TOOL_PATH.name), *args]
    return subprocess.run(argv, cwd=root.parent, capture_output=True, text=True, check=False, timeout=30)


@pytest.mark.parametrize(
    ("tail", "reasoned"),
    [
        ("", False),
        (" \t", False),
        (")", False),
        (" -:;.,#) ]–—", False),
        (" — : ;", False),
        (" —\t;", False),
        ("—\u2003:–", False),
        (" : – ; —\t#", False),
        (" - optional dependency path", True),
        (": defensive guard", True),
        (" — unreachable numerical branch", True),
        (" —\t: explanation", True),
        (" reason without a separator", True),
    ],
)
def test_real_marker_reason_suffix(tmp_path: Path, tail: str, reasoned: bool) -> None:
    """The public CLI refuses separator-only suffixes and retains stated reasons.

    Parameters
    ----------
    tmp_path : Path
        Isolated physical repository allocation.
    tail : str
        Same-line reason text, including mixed punctuation and whitespace.
    reasoned : bool
        Expected lexical reason-presence result, not justification acceptance.
    """
    root = _cli_tree(tmp_path, "value = 1  # pragma: no cover" + tail + "\n")
    result = _cli(root, ["--json"])
    assert result.returncode == (0 if reasoned else 1), result.stdout + result.stderr
    findings = json.loads(result.stdout)["unreasoned"]
    if reasoned:
        assert findings == []
    else:
        assert findings == [
            {"path": "src/scpn_control/example.py", "line": 1, "text": "value = 1  # pragma: no cover" + tail.rstrip()}
        ]
    assert result.stderr == ""


def test_real_public_enumeration_and_duplicate_diagnostics(tmp_path: Path, pragma_tool: ModuleType) -> None:
    """Public enumeration preserves ordering, duplicate requests and file filtering.

    Parameters
    ----------
    tmp_path : Path
        External directory with Python files and a directory ending in .py.
    pragma_tool : ModuleType
        Actual repository implementation.
    """
    source = tmp_path / "sources"
    source.mkdir()
    first = source / "a.py"
    second = source / "b.py"
    first.write_text("\n  if optional:  # pragma: no cover\n", encoding="utf-8")
    second.write_text("value = 2\n", encoding="utf-8")
    (source / "README.md").write_text("not a selected source\n", encoding="utf-8")
    (source / "not_a_file.py").mkdir()
    assert pragma_tool.iter_python_files([second, source, first]) == [first, first, second, second]
    findings = pragma_tool.find_unreasoned_pragmas([second, source, first])
    assert len(findings) == 2
    for finding in findings:
        assert finding.path == first.resolve().as_posix()
        assert finding.line == 2
        assert finding.text == "if optional:  # pragma: no cover"
        with pytest.raises(FrozenInstanceError):
            finding.line = 9
    empty = tmp_path / "empty"
    empty.mkdir()
    assert pragma_tool.iter_python_files([]) == []
    assert pragma_tool.find_unreasoned_pragmas([empty]) == []


def test_real_public_relative_api_and_lexical_scope(
    tmp_path: Path, pragma_tool: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relative API inputs use the caller directory and match strings lexically.

    Parameters
    ----------
    tmp_path : Path
        Caller directory with a literal string containing an exclusion marker.
    pragma_tool : ModuleType
        Actual guard module; its repository root is not replaced.
    monkeypatch : pytest.MonkeyPatch
        Changes the process working directory only.
    """
    source = tmp_path / "relative.py"
    source.write_text('value = 1\n"pragma: no cover\n# PRAGMA: no cover\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert pragma_tool.iter_python_files([Path("relative.py")]) == [Path("relative.py")]
    findings = pragma_tool.find_unreasoned_pragmas([Path("relative.py")])
    assert [(x.path, x.line, x.text) for x in findings] == [(source.resolve().as_posix(), 2, '"pragma: no cover')]


@pytest.mark.parametrize("mode", ["missing", "non_python"])
def test_real_public_invalid_request(tmp_path: Path, pragma_tool: ModuleType, mode: str) -> None:
    """Invalid explicit paths raise before an empty result can imply inspection.

    Parameters
    ----------
    tmp_path : Path
        Isolated requested-path allocation.
    pragma_tool : ModuleType
        Actual public API owner.
    mode : str
        Missing-path or existing unsupported-file witness.
    """
    path = tmp_path / ("missing.py" if mode == "missing" else "README.md")
    if mode == "non_python":
        path.write_text("not Python\n", encoding="utf-8")
    expected = FileNotFoundError if mode == "missing" else ValueError
    with pytest.raises(expected, match="Coverage pragma scan"):
        pragma_tool.iter_python_files([path])
    with pytest.raises(expected, match="Coverage pragma scan"):
        pragma_tool.find_unreasoned_pragmas([path])


def test_real_public_native_constructor_and_decode_errors(tmp_path: Path, pragma_tool: ModuleType) -> None:
    """Public dataclass initialisation and UTF-8 reading retain native errors.

    Parameters
    ----------
    tmp_path : Path
        Allocation for an undecodable Python file.
    pragma_tool : ModuleType
        Actual guard and diagnostic type.
    """
    with pytest.raises(TypeError):
        pragma_tool.CoveragePragmaViolation()
    with pytest.raises(TypeError):
        pragma_tool.CoveragePragmaViolation(path="x", line=1, text="", extra=True)
    record = pragma_tool.CoveragePragmaViolation("x", -1, "")
    assert record.line == -1
    source = tmp_path / "invalid.py"
    source.write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        pragma_tool.find_unreasoned_pragmas([source])


@pytest.mark.parametrize("mode", ["missing_explicit", "missing_default", "non_python", "invalid_utf8"])
def test_real_cli_native_input_failures(tmp_path: Path, mode: str) -> None:
    """Actual CLI input errors fail before JSON or human success is emitted.

    Parameters
    ----------
    tmp_path : Path
        Physical fixture repository allocation.
    mode : str
        Missing request/default directory, unsupported file or undecodable file.
    """
    root = _cli_tree(tmp_path)
    args = ["--json"]
    expected = "FileNotFoundError"
    if mode == "missing_explicit":
        args += ["absent.py"]
    elif mode == "missing_default":
        source = root / "src/scpn_control"
        (source / "example.py").unlink()
        source.rmdir()
    elif mode == "non_python":
        (root / "README.md").write_text("not Python\n", encoding="utf-8")
        args += ["README.md"]
        expected = "ValueError"
    else:
        (root / "src/scpn_control/example.py").write_bytes(b"\xff")
        expected = "UnicodeDecodeError"
    result = _cli(root, args)
    assert result.returncode == 1
    assert result.stdout == ""
    assert expected in result.stderr


@pytest.mark.parametrize("json_output", [False, True])
def test_real_cli_clean_absolute_file(tmp_path: Path, json_output: bool) -> None:
    """An actual absolute-file scan produces the documented clean output.

    Parameters
    ----------
    tmp_path : Path
        Physical fixture repository allocation.
    json_output : bool
        Selects the human or JSON output contract.
    """
    root = _cli_tree(tmp_path)
    source = root / "src/scpn_control/example.py"
    args = [str(source)]
    if json_output:
        args.append("--json")
    result = _cli(root, args)
    assert result.returncode == 0 and result.stderr == ""
    if json_output:
        assert json.loads(result.stdout) == {"unreasoned": []}
    else:
        assert result.stdout == "Coverage pragma reason guard passed: 0 unreasoned pragmas.\n"


def test_real_cli_relative_human_findings(tmp_path: Path) -> None:
    """Relative CLI paths resolve against the script and print every diagnostic.

    Parameters
    ----------
    tmp_path : Path
        Physical repository, executed from its parent directory.
    """
    root = _cli_tree(tmp_path, "# pragma: no cover\nvalue = 1\n# pragma: no cover — :\n")
    result = _cli(root, ["src/scpn_control/example.py"])
    assert result.returncode == 1 and result.stderr == ""
    assert "FAILED: 2 unreasoned pragma(s)." in result.stdout
    assert "src/scpn_control/example.py:1: # pragma: no cover" in result.stdout
    assert "src/scpn_control/example.py:3: # pragma: no cover — :" in result.stdout


@pytest.mark.parametrize(("argument", "status"), [("--help", 0), ("--unknown", 2)])
def test_real_cli_argparse_status(tmp_path: Path, argument: str, status: int) -> None:
    """Native argparse help and invalid-option status remain observable.

    Parameters
    ----------
    tmp_path : Path
        Physical byte-identical CLI allocation.
    argument : str
        Help request or unknown option token.
    status : int
        Expected native argparse process exit.
    """
    result = _cli(_cli_tree(tmp_path), [argument])
    assert result.returncode == status
    assert "usage:" in result.stdout + result.stderr
