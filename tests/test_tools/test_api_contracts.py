# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Cross-language API inventory and contract tests.

"""Tests for deterministic cross-language API ownership classification."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools import check_api_contracts

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/check_api_contracts.py"


def test_api_inventory_has_disjoint_complete_python_classification() -> None:
    """Every Python candidate belongs to exactly one ownership class."""
    inventory = check_api_contracts.build_inventory()
    python = inventory["python"]

    assert sum(python["classifications"].values()) == python["candidate_count"]
    assert python["stable_export_count"] == 51
    assert len(inventory["c"]["symbols"]) == 10
    assert len(inventory["lean"]["symbols"]) == 9


def test_committed_api_contract_registry_is_current() -> None:
    """The canonical worktree registry matches its actual cross-language declarations."""
    assert check_api_contracts.main([]) == 0


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


def cli(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the actual stdlib-only source CLI with cwd-relative operator paths."""
    return subprocess.run(
        [sys.executable, "-S", str(TOOL), "--repo", str(repo), *args],
        cwd=repo.parent,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_actual_candidate_match_and_inventory_print(
    copied_api_checkout: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Actual copied canonical declarations and registry match without rebasing a fixture."""
    registry = copied_api_checkout / "tools/api_contract_registry.toml"
    assert check_api_contracts.check_contracts(copied_api_checkout, registry) == []
    assert check_api_contracts.main(["--repo", str(copied_api_checkout)]) == 0
    assert "declarations match" in capsys.readouterr().out
    registry.rename(registry.with_suffix(".held"))
    result = cli(copied_api_checkout, "--print-inventory", "--registry", "missing.toml")
    assert result.returncode == 0 and result.stderr == ""
    assert json.loads(result.stdout) == check_api_contracts.build_inventory(copied_api_checkout)
    assert check_api_contracts.main(["--repo", str(copied_api_checkout), "--print-inventory"]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(result.stdout)


def test_actual_renderer_fragment_drift_and_registry_path(copied_api_checkout: Path) -> None:
    """The real CLI checks cwd-relative registry paths and refuses removed actual fragments."""
    registry = copied_api_checkout / "tools/api_contract_registry.toml"
    target = copied_api_checkout / "studio-web/typedoc.json"
    target.write_text(target.read_text().replace("requiredToBeDocumented", "removedRequiredDeclaration"))
    result = cli(copied_api_checkout, "--registry", str(registry.relative_to(copied_api_checkout.parent)))
    assert result.returncode == 1 and result.stderr == ""
    assert "studio-web/typedoc.json lacks required renderer contract: requiredToBeDocumented" in result.stdout
    assert check_api_contracts.check_contracts(copied_api_checkout, registry) == [
        "studio-web/typedoc.json lacks required renderer contract: requiredToBeDocumented"
    ]


@pytest.mark.parametrize("payload", [b"inventory=3", b"schema =", b"\xff"])
def test_actual_cli_malformed_registry_refuses(copied_api_checkout: Path, payload: bytes) -> None:
    """Malformed actual registry inputs produce status2 and authored refusal text."""
    (copied_api_checkout / "tools/api_contract_registry.toml").write_bytes(payload)
    result = cli(copied_api_checkout)
    assert result.returncode == 2 and result.stderr == ""
    assert result.stdout.startswith("API contract inspection refused: ")
    assert "Traceback" not in result.stdout and "KeyError" not in result.stdout
    assert check_api_contracts.main(["--repo", str(copied_api_checkout)]) == result.returncode


@pytest.mark.parametrize(
    "before,after",
    [
        ("scpn-control.api-contract-registry.v1", "unsupported.schema"),
        (r"^python_candidate_count = .*$", "python_candidate_count = true"),
        (r"^python_candidate_count = .*$", "python_candidate_count = -1"),
        (r"^python_candidate_count = .*$", 'python_candidate_count = "many"'),
        (r"^python_candidate_count = .*$", "python_candidate_count = 1"),
        (r"^python_stable_export_count = .*$", "python_stable_export_count = false"),
        (
            r"^python_candidate_sha256 = .*$",
            "python_candidate_sha256 = 3",
        ),
        (
            r"^rust_export_sha256 = .*$",
            'rust_export_sha256 = "bad"',
        ),
        ("c_symbols = [", "c_symbols = [5,"),
        ('"scpn_solver_create_v1",', '"bad-name",'),
        ('"scpn_solver_create_v1",', '"scpn_solver_create_v1", "scpn_solver_create_v1",'),
        ("[inventory.python_classifications]", "[inventory.wrong_classifications]"),
        ("stable-root-owner = 49", "other-owner = 49"),
        ('"pyproject.toml" =', '"../pyproject.toml" ='),
        ('"pyproject.toml" =', '"/pyproject.toml" ='),
        ('["requiredToBeDocumented"]', "[]"),
        ('["requiredToBeDocumented"]', "[3]"),
        ('["requiredToBeDocumented"]', '[" "]'),
        ('["requiredToBeDocumented"]', '"requiredToBeDocumented"'),
    ],
)
def test_actual_registry_field_shape_refusal(copied_api_checkout: Path, before: str, after: str) -> None:
    """Mutating a required real field's type or scope cannot certify the registry."""
    registry = copied_api_checkout / "tools/api_contract_registry.toml"
    source = registry.read_text()
    if before.startswith("^"):
        source, replacements = re.subn(before, after, source, flags=re.M)
        assert replacements == 1
    else:
        assert before in source
        source = source.replace(before, after)
    registry.write_text(source)
    with pytest.raises(check_api_contracts.ApiContractInspectionError):
        check_api_contracts.check_contracts(copied_api_checkout)


def test_actual_missing_renderer_and_registry_refuse(copied_api_checkout: Path) -> None:
    """Real input disappearance produces inspection refusal rather than policy success."""
    renderer = copied_api_checkout / "studio-web/typedoc.json"
    renderer.rename(renderer.with_suffix(".held"))
    result = cli(copied_api_checkout)
    assert result.returncode == 2 and "required renderer input" in result.stdout
    registry = copied_api_checkout / "tools/api_contract_registry.toml"
    registry.rename(registry.with_suffix(".held"))
    result = cli(copied_api_checkout)
    assert result.returncode == 2 and "valid UTF-8 API registry" in result.stdout


def test_actual_cli_root_resolution_refusal(tmp_path: Path) -> None:
    """A real looping operator root yields status2 without OS/interpreter details."""
    root = tmp_path / "loop"
    root.symlink_to(root.name)
    result = cli(root)
    assert result.returncode == 2 and result.stderr == ""
    assert result.stdout == "API contract inspection refused: could not resolve the API repository root\n"
    assert check_api_contracts.main(["--repo", str(root)]) == result.returncode


def test_actual_cli_help_and_usage(tmp_path: Path) -> None:
    """Argparse's actual stdlib entrypoint retains help0 and malformed-argument2."""
    assert cli(tmp_path, "--help").returncode == 0
    result = cli(tmp_path, "--unknown")
    assert result.returncode == 2 and "unrecognized arguments" in result.stderr


@pytest.mark.parametrize("kind", ["symbols-string", "renderer-empty", "renderer-utf8"])
def test_actual_required_registry_and_renderer_shapes(copied_api_checkout: Path, kind: str) -> None:
    """Actual typed inputs refuse scalar symbol lists, empty contracts and decoding failure."""
    registry = copied_api_checkout / "tools/api_contract_registry.toml"
    source = registry.read_text()
    if kind == "symbols-string":
        source = re.sub(r"c_symbols = \[.*?\]", 'c_symbols = "scpn_solver_create_v1"', source, flags=re.S)
    elif kind == "renderer-empty":
        source = source.split("[renderer_contracts]")[0] + "[renderer_contracts]\n"
    else:
        (copied_api_checkout / "studio-web/typedoc.json").write_bytes(b"\xff")
    registry.write_text(source)
    with pytest.raises(check_api_contracts.ApiContractInspectionError):
        check_api_contracts.check_contracts(copied_api_checkout)
    assert check_api_contracts.main(["--repo", str(copied_api_checkout)]) == 2


def test_actual_native_declaration_change_refuses_stale_registry(copied_api_checkout: Path) -> None:
    """Changing a real versioned C prototype refuses the unchanged copied registry."""
    header = copied_api_checkout / "src/scpn_control/core/solver.h"
    text = header.read_text()
    assert "scpn_solver_create_v1" in text
    header.write_text(text.replace("scpn_solver_create_v1", "scpn_solver_create_v2"))
    errors = check_api_contracts.check_contracts(copied_api_checkout)
    assert len(errors) == 1 and errors[0].startswith("API declaration inventory differs from the registry")
    assert check_api_contracts.main(["--repo", str(copied_api_checkout)]) == 1
    result = cli(copied_api_checkout)
    assert result.returncode == 1 and result.stderr == ""
    assert "scpn_solver_create_v2" in result.stdout
