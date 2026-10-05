# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Module documentation discovery contract tests.
"""Exercise module discovery and reference checks with actual copied source/docs."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools import check_docs_coverage as coverage_tool

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def copied_documented_tree(tmp_path: Path) -> Path:
    """Copy the real guard, API reference and two referenced source owners.

    This declared partial source tree exercises catalog mechanics without
    importing, rendering or claiming semantic acceptance of the listed models.
    """
    for name in (
        "tools/check_docs_coverage.py",
        "docs/api.md",
        "src/scpn_control/__init__.py",
        "src/scpn_control/core/eqdsk.py",
        "src/scpn_control/phase/kuramoto.py",
    ):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    return tmp_path


def _cli(repo: Path) -> subprocess.CompletedProcess[str]:
    """Run the copied production script from an independent cwd without imports."""
    return subprocess.run(
        [sys.executable, str(ROOT / "tools/check_docs_coverage.py"), "--repo", str(repo)],
        cwd=repo / "docs",
        capture_output=True,
        text=True,
        check=False,
    )


def test_actual_copied_tree_accepts_existing_object_directives(copied_documented_tree: Path) -> None:
    """Existing class/function directives represent the two actual source modules."""
    result = _cli(copied_documented_tree)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "Documentation coverage OK: 2 Python modules represented and documented.\n"


@pytest.mark.parametrize("shape", ["absent", "file", "empty"])
def test_cli_refuses_missing_or_empty_source_scope(copied_documented_tree: Path, shape: str) -> None:
    """An absent/file/empty package tree cannot certify documentation coverage."""
    source = copied_documented_tree / "src/scpn_control"
    shutil.rmtree(source)
    if shape == "file":
        source.write_text("scpn_control")
    elif shape == "empty":
        source.mkdir()
    result = _cli(copied_documented_tree)
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "Documentation coverage refused: source scope must contain Python modules.\n"


@pytest.mark.parametrize("corruption", ["syntax", "encoding", "api-directory"])
def test_cli_handles_actual_source_or_reference_inspection_failure(
    copied_documented_tree: Path,
    corruption: str,
) -> None:
    """Actual unreadable/undecodable/unparseable carriers produce fixed refusals."""
    source = copied_documented_tree / "src/scpn_control/core/eqdsk.py"
    if corruption == "syntax":
        source.write_bytes(source.read_bytes() + b"\nif (\n")
    elif corruption == "encoding":
        source.write_bytes(source.read_bytes() + b"\xff")
    else:
        reference = copied_documented_tree / "docs/api.md"
        reference.unlink()
        reference.mkdir()
    result = _cli(copied_documented_tree)
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "Documentation coverage refused: could not inspect UTF-8 source and API reference.\n"


def test_default_script_root_works_independently_of_cwd(copied_documented_tree: Path) -> None:
    """The no-flag hook command uses the copied script's repository root."""
    result = subprocess.run(
        [sys.executable, str(copied_documented_tree / "tools/check_docs_coverage.py")],
        cwd=copied_documented_tree / "docs",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0 and "2 Python modules" in result.stdout


def test_object_prefix_discovery_does_not_claim_symbol_resolution(copied_documented_tree: Path) -> None:
    """Existing file prefixes count even with unknown attributes; nonmatching syntax does not."""
    text = (ROOT / "docs/api.md").read_text(encoding="utf-8")
    real = "::: scpn_control.core.eqdsk.GEqdsk"
    assert real in text
    fragment = "\n".join(
        [
            real,
            real + ".unknown_attribute",
            " " + real,
            ":::\tscpn_control.phase.kuramoto.order_parameter",
            "::: other_package.core.eqdsk",
            "::: scpn_control.absent_source.attribute",
        ]
    )
    assert coverage_tool.api_module_directives(fragment, copied_documented_tree) == {"scpn_control.core.eqdsk"}
    (copied_documented_tree / "src/scpn_control/absent_source.py").mkdir()
    assert coverage_tool.api_module_directives(fragment, copied_documented_tree) == {"scpn_control.core.eqdsk"}


def test_source_inventory_includes_private_files_but_not_packages(copied_documented_tree: Path) -> None:
    """Copy an actual private owner; discovery includes it and excludes __init__ and .py directories."""
    target = copied_documented_tree / "src/scpn_control/core/_validators.py"
    shutil.copy2(ROOT / "src/scpn_control/core/_validators.py", target)
    (copied_documented_tree / "src/scpn_control/core/directory.py").mkdir()
    names = [
        coverage_tool.module_name(p, copied_documented_tree)
        for p in coverage_tool.iter_public_modules(copied_documented_tree)
    ]
    assert names == ["scpn_control.core._validators", "scpn_control.core.eqdsk", "scpn_control.phase.kuramoto"]
    assert coverage_tool.modules_missing_docstrings([target], copied_documented_tree) == []


def test_direct_cli_reports_reference_then_module_doc_gaps(
    copied_documented_tree: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Removing the actual module doc and its directives emits both ordered lists without writes."""
    import ast

    repo = copied_documented_tree
    source = repo / "src/scpn_control/core/eqdsk.py"
    text = source.read_text(encoding="utf-8")
    module_doc = ast.parse(text).body[0]
    assert isinstance(module_doc, ast.Expr) and isinstance(module_doc.value, ast.Constant)
    lines = text.splitlines(keepends=True)
    source.write_text("".join(lines[: module_doc.lineno - 1] + lines[module_doc.end_lineno :]))
    reference = repo / "docs/api.md"
    reference.write_text(
        "\n".join(
            line for line in reference.read_text().splitlines() if not line.startswith("::: scpn_control.core.eqdsk")
        )
    )
    before = source.read_bytes(), reference.read_bytes()
    assert coverage_tool.modules_missing_docstrings([source, source], repo) == [
        "scpn_control.core.eqdsk",
        "scpn_control.core.eqdsk",
    ]
    argv = ["--repo", str(repo)]
    assert coverage_tool.main(argv) == 1
    assert argv == ["--repo", str(repo)]
    result = capsys.readouterr()
    assert result.out == ""
    assert result.err == (
        "Modules missing from docs/api.md:\n  - scpn_control.core.eqdsk\n"
        "Modules missing module docstrings:\n  - scpn_control.core.eqdsk\n"
    )
    assert (source.read_bytes(), reference.read_bytes()) == before


def test_public_inspection_errors_propagate(copied_documented_tree: Path) -> None:
    """Library callers receive scope/path/parse errors rather than a CLI status."""
    source = copied_documented_tree / "src/scpn_control/core/eqdsk.py"
    with pytest.raises(ValueError):
        coverage_tool.module_name(copied_documented_tree / "docs/api.md", copied_documented_tree)
    source.write_bytes(source.read_bytes() + b"\nif (\n")
    with pytest.raises(SyntaxError):
        coverage_tool.modules_missing_docstrings([source], copied_documented_tree)
    source.unlink()
    with pytest.raises(OSError):
        coverage_tool.modules_missing_docstrings([source], copied_documented_tree)
    with pytest.raises(ValueError):
        coverage_tool.iter_public_modules(copied_documented_tree / "docs")


def test_native_example_reads_actual_source_module() -> None:
    """Run the owning native API example against the real EQDSK source."""
    import doctest

    result = doctest.testmod(coverage_tool)
    assert result.attempted == 1 and result.failed == 0


def test_cli_rejects_unknown_arguments(copied_documented_tree: Path) -> None:
    """Malformed arguments are argparse status two, before any source inspection."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/check_docs_coverage.py"), "--invalid"],
        cwd=copied_documented_tree,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2 and "unrecognized arguments: --invalid" in result.stderr


def test_canonical_reference_covers_actual_discovered_modules() -> None:
    """Whole-task closeout requires API directives for every currently selected source file."""
    represented = coverage_tool.api_module_directives(coverage_tool.API_DOC.read_text(encoding="utf-8"))
    missing = [
        coverage_tool.module_name(path)
        for path in coverage_tool.iter_public_modules()
        if coverage_tool.module_name(path) not in represented
    ]
    assert not missing, missing


@pytest.mark.parametrize("gap", ["reference", "docstring"])
def test_cli_distinguishes_independent_reference_and_docstring_failures(
    copied_documented_tree: Path,
    gap: str,
) -> None:
    """Changing only one actual carrier emits only its corresponding gap list."""
    import ast

    repo = copied_documented_tree
    if gap == "reference":
        p = repo / "docs/api.md"
        p.write_text(
            "\n".join(line for line in p.read_text().splitlines() if not line.startswith("::: scpn_control.core.eqdsk"))
        )
        expected = "Modules missing from docs/api.md:\n  - scpn_control.core.eqdsk\n"
    else:
        p = repo / "src/scpn_control/core/eqdsk.py"
        lines = p.read_text().splitlines(keepends=True)
        doc = ast.parse("".join(lines)).body[0]
        assert isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant)
        p.write_text("".join(lines[: doc.lineno - 1] + lines[doc.end_lineno :]))
        expected = "Modules missing module docstrings:\n  - scpn_control.core.eqdsk\n"
    result = _cli(repo)
    assert result.returncode == 1 and result.stderr == expected and result.stdout == ""
