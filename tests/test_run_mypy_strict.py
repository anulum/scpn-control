# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Strict mypy gate and debt-ratchet tests

"""Regression tests for the strict-mypy gate and per-module debt ratchet."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import run_mypy_strict as rms

_PROBE_SAMPLE = (
    "src/scpn_control/control/nmpc_controller.py:2032: error: Missing type arguments "
    'for generic type "ndarray"  [type-arg]\n'
    "src/scpn_control/control/nmpc_controller.py:2040: error: Returning Any  [no-any-return]\n"
    "src/scpn_control/core/current_drive.py:11: error: Function is missing a type "
    "annotation  [no-untyped-def]\n"
    "src/scpn_control/core/current_drive.py:12: note: this is only a note, not an error\n"
    "Found 3 errors in 2 files (checked 152 source files)\n"
)


def test_file_to_module_strips_src_prefix_and_suffix() -> None:
    """A ``src``-relative path maps to its dotted import path."""

    assert rms.file_to_module("src/scpn_control/core/current_drive.py") == "scpn_control.core.current_drive"


def test_file_to_module_normalises_backslashes() -> None:
    """Windows-style separators are normalised before conversion."""

    assert rms.file_to_module("src\\scpn_control\\scpn\\observation.py") == "scpn_control.scpn.observation"


def test_file_to_module_without_src_prefix() -> None:
    """A path that is not under ``src`` keeps its leading segments."""

    assert rms.file_to_module("tools/run_mypy_strict.py") == "tools.run_mypy_strict"


def test_parse_module_error_counts_groups_and_ignores_notes() -> None:
    """Error lines are grouped per module and note lines are excluded."""

    counts = rms.parse_module_error_counts(_PROBE_SAMPLE)

    assert counts == {
        "scpn_control.control.nmpc_controller": 2,
        "scpn_control.core.current_drive": 1,
    }


def test_parse_total_errors_reads_summary() -> None:
    """The ``Found N errors`` summary is parsed as the authoritative total."""

    assert rms.parse_total_errors(_PROBE_SAMPLE) == 3


def test_parse_total_errors_handles_success() -> None:
    """A success summary yields zero errors."""

    assert rms.parse_total_errors("Success: no issues found in 152 source files\n") == 0


def test_parse_total_errors_falls_back_to_line_count() -> None:
    """Without a summary, the parsed error lines are counted."""

    text = "src/scpn_control/core/x.py:1: error: boom  [misc]\n"
    assert rms.parse_total_errors(text) == 1


def test_parse_total_errors_rejects_unrecognised_output() -> None:
    """Output with neither summary nor error lines is a hard failure."""

    with pytest.raises(ValueError):
        rms.parse_total_errors("mypy: error: cannot find module to check\n")


def test_ledger_round_trip() -> None:
    """A ledger survives serialisation and deserialisation unchanged."""

    ledger = rms.StrictDebtLedger(mypy_version="mypy 1.20.0", total=5, per_module={"a.b": 3, "c.d": 2})
    restored = rms.StrictDebtLedger.from_dict(ledger.to_dict())

    assert restored == ledger


def test_ledger_to_dict_sorts_modules() -> None:
    """Serialised per-module counts are key-sorted for stable diffs."""

    ledger = rms.StrictDebtLedger(mypy_version="v", total=2, per_module={"z.z": 1, "a.a": 1})
    assert list(ledger.to_dict()["per_module"]) == ["a.a", "z.z"]


def test_ledger_from_dict_rejects_wrong_schema() -> None:
    """An unrecognised schema marker is refused."""

    with pytest.raises(ValueError):
        rms.StrictDebtLedger.from_dict({"schema": "other", "total": 0, "per_module": {}})


def test_ledger_from_dict_rejects_non_object_per_module() -> None:
    """A non-object ``per_module`` payload is refused."""

    with pytest.raises(ValueError):
        rms.StrictDebtLedger.from_dict({"schema": rms.LEDGER_SCHEMA, "total": 0, "per_module": []})


def _ledger(total: int, per_module: dict[str, int]) -> rms.StrictDebtLedger:
    return rms.StrictDebtLedger(mypy_version="v", total=total, per_module=per_module)


def test_ratchet_clean_when_unchanged() -> None:
    """Identical current and baseline debt holds the ratchet."""

    result = rms.evaluate_ratchet(5, {"a.b": 3, "c.d": 2}, _ledger(5, {"a.b": 3, "c.d": 2}))

    assert result.ok
    assert not result.regressions
    assert not result.improvements
    assert result.total_delta == 0


def test_ratchet_detects_module_regression_even_with_flat_total() -> None:
    """A module rising while another falls is still a regression."""

    result = rms.evaluate_ratchet(5, {"a.b": 4, "c.d": 1}, _ledger(5, {"a.b": 3, "c.d": 2}))

    assert not result.ok
    assert result.regressions == {"a.b": (3, 4)}
    assert result.improvements == {"c.d": (2, 1)}
    assert result.total_delta == 0


def test_ratchet_detects_new_module_with_errors() -> None:
    """A module absent from the baseline that now has errors regresses."""

    result = rms.evaluate_ratchet(7, {"a.b": 3, "new.mod": 2}, _ledger(5, {"a.b": 3, "c.d": 2}))

    assert not result.ok
    assert result.regressions == {"new.mod": (0, 2)}
    assert result.total_delta == 2


def test_ratchet_passes_on_overall_improvement() -> None:
    """Falling total with no per-module regression holds the ratchet."""

    result = rms.evaluate_ratchet(3, {"a.b": 3}, _ledger(5, {"a.b": 3, "c.d": 2}))

    assert result.ok
    assert result.improvements == {"c.d": (2, 0)}
    assert result.total_delta == -2


def test_ratchet_fails_when_total_rises_without_module_regression() -> None:
    """A total above baseline fails even if no single module regressed."""

    # Construction is contrived: total claims to rise while modules look flat.
    result = rms.evaluate_ratchet(6, {"a.b": 3, "c.d": 2}, _ledger(5, {"a.b": 3, "c.d": 2}))

    assert not result.ok
    assert not result.regressions
    assert result.total_delta == 1


@pytest.fixture
def patched_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the configured gate pass and the strict probe deterministic."""

    monkeypatch.setattr(rms, "run_configured_mypy", lambda: (0, "Success: no issues found\n"))
    monkeypatch.setattr(rms, "run_strict_probe", lambda: _PROBE_SAMPLE)
    monkeypatch.setattr(rms, "_mypy_version", lambda: "mypy 1.20.0")


def _point_ledger_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    ledger_path = tmp_path / "mypy_strict_debt.json"
    monkeypatch.setattr(rms, "LEDGER_PATH", ledger_path)
    return ledger_path


def test_main_fails_closed_when_configured_gate_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing configured gate returns its code and skips the probe."""

    monkeypatch.setattr(rms, "run_configured_mypy", lambda: (2, "boom\n"))

    def _explode() -> str:
        raise AssertionError("strict probe must not run when the configured gate fails")

    monkeypatch.setattr(rms, "run_strict_probe", _explode)

    assert rms.main([]) == 2


def test_main_passes_when_debt_within_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_probe: None
) -> None:
    """Current debt equal to the committed baseline passes."""

    ledger_path = _point_ledger_at(tmp_path, monkeypatch)
    rms.write_ledger(
        rms.StrictDebtLedger(
            mypy_version="v",
            total=3,
            per_module={"scpn_control.control.nmpc_controller": 2, "scpn_control.core.current_drive": 1},
        ),
        ledger_path,
    )

    assert rms.main([]) == 0


def test_main_fails_on_regression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_probe: None) -> None:
    """Debt above the baseline fails the ratchet."""

    ledger_path = _point_ledger_at(tmp_path, monkeypatch)
    rms.write_ledger(rms.StrictDebtLedger(mypy_version="v", total=1, per_module={}), ledger_path)

    assert rms.main([]) == 1


def test_main_update_baseline_creates_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_probe: None
) -> None:
    """``--update-baseline`` writes a ledger reflecting the current probe."""

    ledger_path = _point_ledger_at(tmp_path, monkeypatch)

    assert rms.main(["--update-baseline"]) == 0

    payload = json.loads(ledger_path.read_text())
    assert payload["schema"] == rms.LEDGER_SCHEMA
    assert payload["total"] == 3
    assert payload["per_module"]["scpn_control.control.nmpc_controller"] == 2


def test_main_update_baseline_refuses_increase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_probe: None
) -> None:
    """Raising the recorded total needs the explicit increase flag."""

    ledger_path = _point_ledger_at(tmp_path, monkeypatch)
    rms.write_ledger(rms.StrictDebtLedger(mypy_version="v", total=1, per_module={}), ledger_path)

    assert rms.main(["--update-baseline"]) == 3
    # The existing ledger must be left untouched on refusal.
    assert json.loads(ledger_path.read_text())["total"] == 1


def test_main_update_baseline_allows_increase_with_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_probe: None
) -> None:
    """The increase flag permits recording higher debt deliberately."""

    ledger_path = _point_ledger_at(tmp_path, monkeypatch)
    rms.write_ledger(rms.StrictDebtLedger(mypy_version="v", total=1, per_module={}), ledger_path)

    assert rms.main(["--update-baseline", "--allow-baseline-increase"]) == 0
    assert json.loads(ledger_path.read_text())["total"] == 3


def test_main_fails_on_unparsable_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unparsable strict probe is a hard failure, not a silent pass."""

    monkeypatch.setattr(rms, "run_configured_mypy", lambda: (0, "Success: no issues found\n"))
    monkeypatch.setattr(rms, "run_strict_probe", lambda: "mypy: fatal internal error\n")
    _point_ledger_at(tmp_path, monkeypatch)

    assert rms.main([]) == 2


def test_main_skip_configured_gate_runs_only_ratchet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--skip-configured-gate`` bypasses the gate and runs the ratchet."""

    def _explode() -> tuple[int, str]:
        raise AssertionError("configured gate must not run when skipped")

    monkeypatch.setattr(rms, "run_configured_mypy", _explode)
    monkeypatch.setattr(rms, "run_strict_probe", lambda: _PROBE_SAMPLE)
    ledger_path = _point_ledger_at(tmp_path, monkeypatch)
    rms.write_ledger(
        rms.StrictDebtLedger(
            mypy_version="v",
            total=3,
            per_module={"scpn_control.control.nmpc_controller": 2, "scpn_control.core.current_drive": 1},
        ),
        ledger_path,
    )

    assert rms.main(["--skip-configured-gate"]) == 0
