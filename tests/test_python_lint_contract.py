# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Python lint-scope consistency tests.
"""Exercise the real lint-scope checker over live and malformed surfaces."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import check_python_lint_contract as gate


def _write_contract_surfaces(root: Path) -> None:
    """Write the smallest surface set satisfying the production contract."""
    for contract in gate.CONTRACTS:
        path = root / contract.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(contract.required), encoding="utf-8")


def test_live_repository_contract_passes() -> None:
    """Require the checked-in repository surfaces to agree."""
    assert gate.lint_contract_errors(gate.ROOT) == []


def test_missing_surface_is_rejected(tmp_path: Path) -> None:
    """Reject a repository that omits any governed surface."""
    errors = gate.lint_contract_errors(tmp_path)
    assert len(errors) == len(gate.CONTRACTS)
    assert errors[0].startswith("missing contract surface:")


@pytest.mark.parametrize("contract_index", range(len(gate.CONTRACTS)))
def test_missing_required_fragment_is_rejected(tmp_path: Path, contract_index: int) -> None:
    """Reject removal of a required scope fragment from each surface."""
    _write_contract_surfaces(tmp_path)
    contract = gate.CONTRACTS[contract_index]
    path = tmp_path / contract.path
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(contract.required[0], ""), encoding="utf-8")
    assert any("missing required fragment" in error for error in gate.lint_contract_errors(tmp_path))


@pytest.mark.parametrize(
    "contract",
    [contract for contract in gate.CONTRACTS if contract.forbidden],
)
def test_forbidden_broad_scope_is_rejected(tmp_path: Path, contract: gate.SurfaceContract) -> None:
    """Reject the historical broad test-lint forms explicitly."""
    _write_contract_surfaces(tmp_path)
    path = tmp_path / contract.path
    with path.open("a", encoding="utf-8") as stream:
        stream.write(contract.forbidden[0])
    assert any("forbidden broad lint scope" in error for error in gate.lint_contract_errors(tmp_path))


def test_generated_ledger_spellcheck_exclusion_is_required(tmp_path: Path) -> None:
    """Reject a spellcheck hook that can rewrite deterministic ledger IDs."""
    _write_contract_surfaces(tmp_path)
    path = tmp_path / ".pre-commit-config.yaml"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(gate.TYPO_LEDGER_EXCLUSION, ""), encoding="utf-8")
    assert any("coverage_exception_ledger" in error for error in gate.lint_contract_errors(tmp_path))


def test_cli_reports_success_and_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Expose deterministic CLI status and diagnostics for automation."""
    _write_contract_surfaces(tmp_path)
    assert gate.main(["--repo", str(tmp_path)]) == 0
    assert "PASS:" in capsys.readouterr().out
    (tmp_path / "Makefile").unlink()
    assert gate.main(["--repo", str(tmp_path)]) == 1
    assert "FAIL:" in capsys.readouterr().out
