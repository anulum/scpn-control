# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Static API declaration inventory tests.

"""Public inventory tests over actual copied Python and native source families."""

from __future__ import annotations

import ast
import doctest
import shutil
from pathlib import Path

import pytest

from tools import api_contract_inventory as inventory

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def copied_api_checkout(tmp_path: Path) -> Path:
    """Copy the actual selected production graph, never import or fabricate owners."""
    for relative, glob in (
        ("src/scpn_control", "*.py"),
        ("scpn-control-rs/crates", "*.rs"),
        ("studio-web/src", "*.ts*"),
    ):
        for source in (ROOT / relative).rglob(glob):
            if "target" in source.relative_to(ROOT).parts:
                continue
            target = tmp_path / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for relative in (
        "src/scpn_control/core/solver.h",
        "lean/SCPNControl/PulsedFSM.lean",
        "docs/api.md",
        "tools/api_contract_registry.toml",
        "pyproject.toml",
        "Makefile",
        "studio-web/typedoc.json",
        "mkdocs.yml",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    return tmp_path


def replace_root_literal(repo: Path, name: str, expression: str) -> None:
    """Replace one actual root assignment's value while preserving all other source."""
    path = repo / "src/scpn_control/__init__.py"
    source = path.read_text()
    lines = source.splitlines(keepends=True)
    node = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    )
    value = node.value
    assert value.end_lineno is not None and value.end_col_offset is not None
    start = sum(map(len, lines[: value.lineno - 1])) + value.col_offset
    end = sum(map(len, lines[: value.end_lineno - 1])) + value.end_col_offset
    path.write_text(source[:start] + expression + source[end:])


def test_actual_inventory_retains_original_name_contract(copied_api_checkout: Path) -> None:
    """The split preserves all five family values on the maintained actual graph."""
    result = inventory.build_inventory(copied_api_checkout)
    assert result == inventory.build_inventory()
    assert result["python"]["classifications"]["stable-root-owner"] == 49
    assert sum(result["python"]["classifications"].values()) == result["python"]["candidate_count"]
    result["c"]["symbols"].clear()
    assert inventory.build_inventory(copied_api_checkout)["c"]["symbols"][0] == "scpn_solver_create_v1"


@pytest.mark.parametrize(
    "name,expression,message",
    [
        ("__all__", "'WrongExportShape'", "literal list and owner table"),
        ("__all__", "[5]", "unique identifier strings"),
        ("__all__", "['FusionKernel', 'FusionKernel']", "unique identifier strings"),
        ("__all__", "['bad-name']", "unique identifier strings"),
        ("__all__", "list()", "literal containers"),
        ("_EXPORT_MODULES", "[]", "literal list and owner table"),
        ("_EXPORT_MODULES", "{1: 'scpn_control.core'}", "identifier names and module strings"),
        ("_EXPORT_MODULES", "{'bad-name': 'scpn_control.core'}", "identifier names and module strings"),
        ("_EXPORT_MODULES", "{'FusionKernel': 5}", "identifier names and module strings"),
        ("_EXPORT_MODULES", "{'FusionKernel': 'scpn_control..core'}", "dotted module identifiers"),
    ],
)
def test_actual_root_export_shape_refusal(copied_api_checkout: Path, name: str, expression: str, message: str) -> None:
    """Malformed actual package export declarations refuse a successful inventory."""
    replace_root_literal(copied_api_checkout, name, expression)
    with pytest.raises(inventory.ApiContractInspectionError, match=message):
        inventory.build_inventory(copied_api_checkout)


@pytest.mark.parametrize(
    "family,glob", [("src/scpn_control", "*.py"), ("scpn-control-rs/crates", "*.rs"), ("studio-web/src", "*.ts*")]
)
def test_actual_empty_family_refuses(copied_api_checkout: Path, family: str, glob: str) -> None:
    """Preserved but unselected actual files cannot certify an empty source family."""
    for path in (copied_api_checkout / family).rglob(glob):
        path.rename(path.with_name(path.stem + ".held"))
    with pytest.raises(inventory.ApiContractInspectionError, match="empty or missing"):
        inventory.build_inventory(copied_api_checkout)


@pytest.mark.parametrize(
    "relative,payload",
    [
        ("docs/api.md", b"\xff"),
        ("src/scpn_control/core/fusion_kernel.py", b"]"),
        ("src/scpn_control/core/fusion_kernel.py", b"\x00"),
        ("src/scpn_control/core/solver.h", b"\xff"),
        ("lean/SCPNControl/PulsedFSM.lean", b"\xff"),
        ("studio-web/src/domain.ts", b"\xff"),
    ],
)
def test_actual_unreadable_or_malformed_source_refuses(
    copied_api_checkout: Path, relative: str, payload: bytes
) -> None:
    """Real graph I/O or parsing failures never return a partial successful report."""
    (copied_api_checkout / relative).write_bytes(payload)
    with pytest.raises(inventory.ApiContractInspectionError):
        inventory.build_inventory(copied_api_checkout)


def test_root_owner_prefix_remains_lexical(copied_api_checkout: Path) -> None:
    """Aggregator-prefix ownership does not imply exact module or import resolution."""
    before = inventory.build_inventory(copied_api_checkout)["python"]
    replace_root_literal(copied_api_checkout, "_EXPORT_MODULES", "{}")
    after = inventory.build_inventory(copied_api_checkout)["python"]
    assert after["candidate_sha256"] == before["candidate_sha256"]
    assert after["stable_export_sha256"] == before["stable_export_sha256"]
    assert after["classifications"]["stable-root-owner"] == 0


def test_rust_target_files_are_unselected(copied_api_checkout: Path) -> None:
    """An actual Rust source copy under a target directory cannot change exports."""
    before = inventory.build_inventory(copied_api_checkout)
    source = next((copied_api_checkout / "scpn-control-rs/crates").rglob("*.rs"))
    target = copied_api_checkout / "scpn-control-rs/crates/target/copied.rs"
    target.parent.mkdir()
    shutil.copy2(source, target)
    assert inventory.build_inventory(copied_api_checkout) == before


def test_actual_native_inventory_examples() -> None:
    """Native doctests inspect the maintained real graph without optional imports."""
    result = doctest.testmod(inventory, raise_on_error=True)
    assert result.attempted == 3 and result.failed == 0


def test_actual_missing_literal_binding_refuses(copied_api_checkout: Path) -> None:
    """An assignment to an export-list element is not a declared root container."""
    path = copied_api_checkout / "src/scpn_control/__init__.py"
    path.write_text(path.read_text().replace("__all__ = [", "__all__[0] = ["))
    with pytest.raises(inventory.ApiContractInspectionError, match="literal list and owner table"):
        inventory.build_inventory(copied_api_checkout)


def test_actual_source_family_enumeration_refusal(copied_api_checkout: Path) -> None:
    """A real excessive symlink target refuses source enumeration without OS details."""
    family = copied_api_checkout / "scpn-control-rs/crates"
    family.rename(family.with_name("crates.held"))
    family.symlink_to("a" * 256)
    with pytest.raises(inventory.ApiContractInspectionError, match="^could not enumerate an API source family$"):
        inventory.build_inventory(copied_api_checkout)
