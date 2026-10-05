# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Runtime-wiring checker tests.

"""Tests for the source-module runtime wiring guard."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "check_runtime_wiring.py"


def _load_tool() -> ModuleType:
    """Load the actual checker without importing any inspected production modules."""
    spec = importlib.util.spec_from_file_location("check_runtime_wiring", TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def wiring_tool() -> ModuleType:
    """Load the wiring checker from the checked-out repository."""
    return _load_tool()


@pytest.fixture
def copied_wiring_checkout(tmp_path: Path) -> Path:
    """Copy actual EQDSK sources and their importer into a bounded inspection checkout."""
    for name in (
        "tools/check_runtime_wiring.py",
        "src/scpn_control/__init__.py",
        "src/scpn_control/core/__init__.py",
        "src/scpn_control/core/eqdsk.py",
        "tests/test_eqdsk.py",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / name, path)
    return tmp_path


def _cli(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the actual current script with an explicit checkout and capture its process result."""
    return subprocess.run(
        [sys.executable, str(TOOL_PATH), "--repo", str(repo), *args],
        cwd=repo.parent,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize("kind", ["empty", "missing", "file"])
def test_actual_cli_refuses_empty_source_scope(tmp_path: Path, kind: str) -> None:
    """An empty, missing or regular-file checkout cannot certify static source references."""
    repo = tmp_path / "checkout"
    if kind == "empty":
        repo.mkdir()
    elif kind == "file":
        repo.write_bytes(b"preserve existing checkout placeholder")
    result = _cli(repo, "--json")
    assert result.returncode == 1
    assert result.stdout == "" and "Wiring inspection refused:" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("kind", ["syntax", "utf8"])
def test_actual_cli_refuses_uninspectable_importer(copied_wiring_checkout: Path, kind: str) -> None:
    """Real syntax and UTF-8 corruption in the copied importer refuse a positive report."""
    source = copied_wiring_checkout / "tests/test_eqdsk.py"
    original = source.read_bytes()
    source.write_bytes(original + (b"\n]" if kind == "syntax" else b"\xff"))
    before = source.read_bytes()
    result = _cli(copied_wiring_checkout, "--json")
    assert result.returncode == 1 and result.stdout == ""
    assert "Wiring inspection refused:" in result.stderr and "Traceback" not in result.stderr
    assert source.read_bytes() == before


def test_relative_imports_resolve_from_package_init(wiring_tool: Any) -> None:
    """Package ``__init__`` relative imports must stay inside that package."""
    resolved = wiring_tool._resolve_relative(
        "scpn_control.studio",
        1,
        "adapters",
        is_package=True,
    )

    assert resolved == "scpn_control.studio.adapters"


def test_relative_imports_resolve_from_normal_module(wiring_tool: Any) -> None:
    """Normal module relative imports resolve from the containing package."""
    same_package = wiring_tool._resolve_relative(
        "scpn_control.control.nmpc_controller",
        1,
        "realtime_efit",
        is_package=False,
    )
    parent_package = wiring_tool._resolve_relative(
        "scpn_control.control.nmpc_controller",
        2,
        "core",
        is_package=False,
    )

    assert same_package == "scpn_control.control.realtime_efit"
    assert parent_package == "scpn_control.core"


def test_live_source_tree_has_no_orphan_modules(wiring_tool: Any) -> None:
    """Every source module is referenced from a package, test, tool, or pipeline file."""
    orphans, total = wiring_tool.find_orphans()

    assert total >= 150
    assert orphans == []


def test_main_fails_closed_when_orphans_are_reported(
    wiring_tool: Any,
    copied_wiring_checkout: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Removing the real copied EQDSK importer makes the public command report its source owner."""
    (copied_wiring_checkout / "tests/test_eqdsk.py").unlink()
    argv_before = sys.argv[:]

    assert wiring_tool.main(["--repo", str(copied_wiring_checkout)]) == 1
    assert "scpn_control.core.eqdsk" in capsys.readouterr().out
    assert sys.argv == argv_before


@pytest.mark.parametrize("json_output", [False, True])
def test_actual_cli_accepts_referenced_eqdsk_source(copied_wiring_checkout: Path, json_output: bool) -> None:
    """The copied actual source/test import graph produces a bounded static positive report."""
    result = _cli(copied_wiring_checkout, *(["--json"] if json_output else []))
    assert result.returncode == 0 and result.stderr == ""
    if json_output:
        assert json.loads(result.stdout) == {"total_modules": 3, "orphans": []}
    else:
        assert "Static wiring check: 3 source modules" in result.stdout
        assert "non-exempt source module has a static repository import reference" in result.stdout


def test_actual_cli_json_reports_the_real_removed_importer(copied_wiring_checkout: Path) -> None:
    """Machine-readable refusal reports the actual copied EQDSK owner after its importer disappears."""
    (copied_wiring_checkout / "tests/test_eqdsk.py").unlink()
    result = _cli(copied_wiring_checkout, "--json")
    assert result.returncode == 1 and result.stderr == ""
    assert json.loads(result.stdout) == {"total_modules": 3, "orphans": ["scpn_control.core.eqdsk"]}


def test_default_script_scope_is_independent_of_process_cwd(copied_wiring_checkout: Path) -> None:
    """The actual copied script defaults to its own checkout even when invoked outside it."""
    result = subprocess.run(
        [sys.executable, str(copied_wiring_checkout / "tools/check_runtime_wiring.py"), "--json"],
        cwd=copied_wiring_checkout.parent,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0 and result.stderr == ""
    assert json.loads(result.stdout) == {"total_modules": 3, "orphans": []}


def test_api_refuses_uninspectable_source_without_writes(copied_wiring_checkout: Path, wiring_tool: Any) -> None:
    """The public API propagates a genuine source decode error while leaving source bytes intact."""
    source = copied_wiring_checkout / "src/scpn_control/core/eqdsk.py"
    source.write_bytes(source.read_bytes() + b"\xff")
    before = source.read_bytes()
    with pytest.raises(UnicodeError):
        wiring_tool.find_orphans(copied_wiring_checkout)
    assert source.read_bytes() == before


def test_native_static_reference_example_executes(wiring_tool: Any) -> None:
    """The native public example inspects the actual canonical checkout, without importing its modules."""
    import doctest

    result = doctest.testmod(wiring_tool, raise_on_error=True)
    assert result.failed == 0 and result.attempted == 1


@pytest.mark.parametrize("import_kind", ["conditional", "dynamic"])
def test_static_reference_report_does_not_attest_import_execution(
    copied_wiring_checkout: Path, import_kind: str
) -> None:
    """A real importer moved into a false branch counts; its equivalent dynamic import is not discovered."""
    path = copied_wiring_checkout / "tests/test_eqdsk.py"
    text = path.read_text(encoding="utf-8")
    original = "from scpn_control.core.eqdsk import GEqdsk, read_geqdsk, write_geqdsk"
    assert text.count(original) == 1
    replacement = (
        "if False:\n    " + original
        if import_kind == "conditional"
        else "import importlib\nimportlib.import_module('scpn_control.core.eqdsk')"
    )
    path.write_text(text.replace(original, replacement), encoding="utf-8")
    result = _cli(copied_wiring_checkout, "--json")
    expected = [] if import_kind == "conditional" else ["scpn_control.core.eqdsk"]
    assert result.returncode == (0 if import_kind == "conditional" else 1)
    assert result.stderr == "" and json.loads(result.stdout) == {"total_modules": 3, "orphans": expected}


def test_static_inspection_needs_no_inspected_runtime_dependencies(copied_wiring_checkout: Path) -> None:
    """Actual -S execution scans NumPy-dependent EQDSK source without requiring or importing NumPy."""
    result = subprocess.run(
        [sys.executable, "-S", str(copied_wiring_checkout / "tools/check_runtime_wiring.py"), "--json"],
        cwd=copied_wiring_checkout.parent,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0 and result.stderr == ""
    assert json.loads(result.stdout) == {"total_modules": 3, "orphans": []}
