# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Public-API docstring coverage gate tests

"""Regression tests for the docstring-coverage gate and per-module debt ratchet."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import run_docstring_gate as rdg

_DIAGNOSTICS_SAMPLE: list[dict[str, object]] = [
    {"filename": "/abs/repo/src/scpn_control/core/locked_mode.py", "code": "D102"},
    {"filename": "/abs/repo/src/scpn_control/core/locked_mode.py", "code": "D101"},
    {"filename": "src/scpn_control/control/density_controller.py", "code": "D103"},
]


def test_file_to_module_strips_absolute_src_prefix() -> None:
    """An absolute path is reduced to its dotted import path below ``src``."""

    path = "/abs/repo/src/scpn_control/core/elm_model.py"
    assert rdg.file_to_module(path) == "scpn_control.core.elm_model"


def test_file_to_module_strips_relative_src_prefix() -> None:
    """A ``src``-relative path maps to its dotted import path."""

    assert rdg.file_to_module("src/scpn_control/scpn/artifact.py") == "scpn_control.scpn.artifact"


def test_file_to_module_normalises_backslashes() -> None:
    """Windows-style separators are normalised before conversion."""

    assert rdg.file_to_module("src\\scpn_control\\core\\marfe.py") == "scpn_control.core.marfe"


def test_file_to_module_uses_last_src_marker() -> None:
    """A spurious earlier ``src/`` segment does not shadow the package root."""

    path = "/home/src/checkout/src/scpn_control/core/pedestal.py"
    assert rdg.file_to_module(path) == "scpn_control.core.pedestal"


def test_parse_module_counts_groups_per_module() -> None:
    """Diagnostics are grouped by their owning module."""

    counts = rdg.parse_module_counts(_DIAGNOSTICS_SAMPLE)

    assert counts == {
        "scpn_control.core.locked_mode": 2,
        "scpn_control.control.density_controller": 1,
    }


def test_parse_module_counts_empty() -> None:
    """No diagnostics yields an empty mapping (full coverage)."""

    assert rdg.parse_module_counts([]) == {}


def test_parse_module_counts_rejects_missing_filename() -> None:
    """A diagnostic without a string filename is a hard failure."""

    with pytest.raises(ValueError):
        rdg.parse_module_counts([{"code": "D102"}])


def test_ledger_round_trip() -> None:
    """A ledger survives serialisation and deserialisation unchanged."""

    ledger = rdg.DocstringDebtLedger(total=5, per_module={"a.b": 3, "c.d": 2})
    restored = rdg.DocstringDebtLedger.from_dict(ledger.to_dict())

    assert restored == ledger


def test_ledger_to_dict_sorts_modules_and_records_rules() -> None:
    """Serialised counts are key-sorted and the rule set is recorded."""

    ledger = rdg.DocstringDebtLedger(total=2, per_module={"z.z": 1, "a.a": 1})
    payload = ledger.to_dict()

    assert list(payload["per_module"]) == ["a.a", "z.z"]
    assert payload["rules"] == list(rdg.COVERAGE_RULES)


def test_ledger_from_dict_rejects_wrong_schema() -> None:
    """An unrecognised schema marker is refused."""

    with pytest.raises(ValueError):
        rdg.DocstringDebtLedger.from_dict({"schema": "other", "total": 0, "per_module": {}})


def test_ledger_from_dict_rejects_non_object_per_module() -> None:
    """A non-object ``per_module`` payload is refused."""

    with pytest.raises(ValueError):
        rdg.DocstringDebtLedger.from_dict({"schema": rdg.LEDGER_SCHEMA, "total": 0, "per_module": []})


def test_ledger_from_dict_rejects_non_integer_total() -> None:
    """A non-integer ``total`` payload is refused."""

    with pytest.raises(ValueError):
        rdg.DocstringDebtLedger.from_dict({"schema": rdg.LEDGER_SCHEMA, "total": "many", "per_module": {}})


def _ledger(total: int, per_module: dict[str, int]) -> rdg.DocstringDebtLedger:
    return rdg.DocstringDebtLedger(total=total, per_module=per_module)


def test_ratchet_clean_when_unchanged() -> None:
    """Identical current and baseline debt holds the ratchet."""

    result = rdg.evaluate_ratchet(5, {"a.b": 3, "c.d": 2}, _ledger(5, {"a.b": 3, "c.d": 2}))

    assert result.ok
    assert not result.regressions
    assert not result.improvements
    assert result.total_delta == 0


def test_ratchet_detects_module_regression_even_with_flat_total() -> None:
    """A module rising while another falls is still a regression."""

    result = rdg.evaluate_ratchet(5, {"a.b": 4, "c.d": 1}, _ledger(5, {"a.b": 3, "c.d": 2}))

    assert not result.ok
    assert result.regressions == {"a.b": (3, 4)}
    assert result.improvements == {"c.d": (2, 1)}
    assert result.total_delta == 0


def test_ratchet_detects_new_module_with_debt() -> None:
    """A module absent from the baseline that now has debt regresses."""

    result = rdg.evaluate_ratchet(7, {"a.b": 3, "new.mod": 2}, _ledger(5, {"a.b": 3, "c.d": 2}))

    assert not result.ok
    assert result.regressions == {"new.mod": (0, 2)}
    assert result.total_delta == 2


def test_ratchet_passes_on_overall_improvement() -> None:
    """Falling total with no per-module regression holds the ratchet."""

    result = rdg.evaluate_ratchet(3, {"a.b": 3}, _ledger(5, {"a.b": 3, "c.d": 2}))

    assert result.ok
    assert result.improvements == {"c.d": (2, 0)}
    assert result.total_delta == -2


@pytest.fixture
def patched_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the ruff probe deterministic for ``main`` tests."""

    monkeypatch.setattr(rdg, "run_ruff_probe", lambda: list(_DIAGNOSTICS_SAMPLE))


def _point_ledger_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    ledger_path = tmp_path / "docstring_debt.json"
    monkeypatch.setattr(rdg, "LEDGER_PATH", ledger_path)
    return ledger_path


def test_main_passes_when_debt_within_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_probe: None
) -> None:
    """Current debt equal to the committed baseline passes."""

    ledger_path = _point_ledger_at(tmp_path, monkeypatch)
    rdg.write_ledger(
        _ledger(3, {"scpn_control.core.locked_mode": 2, "scpn_control.control.density_controller": 1}),
        ledger_path,
    )

    assert rdg.main([]) == 0


def test_main_fails_on_regression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_probe: None) -> None:
    """Debt above the baseline fails the ratchet."""

    ledger_path = _point_ledger_at(tmp_path, monkeypatch)
    rdg.write_ledger(_ledger(1, {}), ledger_path)

    assert rdg.main([]) == 1


def test_main_update_baseline_creates_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_probe: None
) -> None:
    """``--update-baseline`` writes a ledger reflecting the current probe."""

    ledger_path = _point_ledger_at(tmp_path, monkeypatch)

    assert rdg.main(["--update-baseline"]) == 0

    payload = json.loads(ledger_path.read_text())
    assert payload["schema"] == rdg.LEDGER_SCHEMA
    assert payload["total"] == 3
    assert payload["per_module"]["scpn_control.core.locked_mode"] == 2


def test_main_update_baseline_refuses_increase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_probe: None
) -> None:
    """Raising the recorded total needs the explicit increase flag."""

    ledger_path = _point_ledger_at(tmp_path, monkeypatch)
    rdg.write_ledger(_ledger(1, {}), ledger_path)

    assert rdg.main(["--update-baseline"]) == 3
    assert json.loads(ledger_path.read_text())["total"] == 1


def test_main_update_baseline_allows_increase_with_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_probe: None
) -> None:
    """The increase flag permits recording higher debt deliberately."""

    ledger_path = _point_ledger_at(tmp_path, monkeypatch)
    rdg.write_ledger(_ledger(1, {}), ledger_path)

    assert rdg.main(["--update-baseline", "--allow-baseline-increase"]) == 0
    assert json.loads(ledger_path.read_text())["total"] == 3


def test_main_fails_on_unparseable_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ruff probe that cannot run is a hard failure, not a silent pass."""

    def _explode() -> list[dict[str, object]]:
        raise RuntimeError("ruff produced no JSON output")

    monkeypatch.setattr(rdg, "run_ruff_probe", _explode)
    _point_ledger_at(tmp_path, monkeypatch)

    assert rdg.main([]) == 2


def test_committed_ledger_matches_schema() -> None:
    """The committed ledger loads cleanly under the current schema."""

    ledger = rdg.load_ledger()

    assert ledger.total == sum(ledger.per_module.values())
