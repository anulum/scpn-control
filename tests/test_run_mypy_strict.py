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
import subprocess
import sys
from pathlib import Path

import pytest
from _pytest.capture import CaptureFixture

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
    per_module = ledger.to_dict()["per_module"]

    assert isinstance(per_module, dict)
    assert list(per_module) == ["a.a", "z.z"]


def test_ledger_from_dict_rejects_wrong_schema() -> None:
    """An unrecognised schema marker is refused."""
    with pytest.raises(ValueError):
        rms.StrictDebtLedger.from_dict({"schema": "other", "total": 0, "per_module": {}})


def test_ledger_from_dict_rejects_non_object_per_module() -> None:
    """A non-object ``per_module`` payload is refused."""
    with pytest.raises(ValueError):
        rms.StrictDebtLedger.from_dict({"schema": rms.LEDGER_SCHEMA, "total": 0, "per_module": []})


def test_ledger_from_dict_rejects_non_integer_total() -> None:
    """A non-integer ``total`` payload is refused."""
    with pytest.raises(ValueError):
        rms.StrictDebtLedger.from_dict({"schema": rms.LEDGER_SCHEMA, "total": "0", "per_module": {}})


def _ledger(total: int, per_module: dict[str, int]) -> rms.StrictDebtLedger:
    """Construct the explicit baseline snapshot used by pure ratchet comparisons."""
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
    """Redirect retained unit tests to a temporary ledger without changing the committed baseline."""
    ledger_path = tmp_path / "mypy_strict_debt.json"
    monkeypatch.setattr(rms, "LEDGER_PATH", ledger_path)
    return ledger_path


def test_mypy_version_reads_subprocess_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The version helper returns stripped mypy version output."""
    calls: list[list[str]] = []

    def _fake_run(
        cmd: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        """Return the declared command response for this retained subprocess-wrapper unit test."""
        calls.append(cmd)
        assert cwd == rms.REPO_ROOT
        assert capture_output is True
        assert text is True
        return subprocess.CompletedProcess(cmd, 0, stdout="mypy 1.20.0\n", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    assert rms._mypy_version() == "mypy 1.20.0"
    assert calls == [[sys.executable, "-m", "mypy", "--version"]]


def test_mypy_version_falls_back_to_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty version output is reported as unknown."""

    def _fake_run(
        cmd: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        """Return the declared command response for this retained subprocess-wrapper unit test."""
        return subprocess.CompletedProcess(cmd, 0, stdout="\n", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    assert rms._mypy_version() == "unknown"


def test_run_configured_mypy_returns_combined_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """The configured gate wrapper preserves return code and combined streams."""
    calls: list[list[str]] = []

    def _fake_run(
        cmd: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        """Return the declared command response for this retained subprocess-wrapper unit test."""
        calls.append(cmd)
        assert cwd == rms.REPO_ROOT
        assert capture_output is True
        assert text is True
        return subprocess.CompletedProcess(cmd, 7, stdout="out\n", stderr="err\n")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    assert rms.run_configured_mypy() == (7, "out\nerr\n")
    assert calls == [[sys.executable, "-m", "mypy"]]


def test_run_strict_probe_returns_combined_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """The strict probe wrapper returns combined streams and ignores exit code."""
    calls: list[list[str]] = []

    def _fake_run(
        cmd: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        """Return the declared command response for this retained subprocess-wrapper unit test."""
        calls.append(cmd)
        assert cwd == rms.REPO_ROOT
        assert capture_output is True
        assert text is True
        return subprocess.CompletedProcess(cmd, 1, stdout="strict-out\n", stderr="strict-err\n")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    assert rms.run_strict_probe() == "strict-out\nstrict-err\n"
    assert calls == [[sys.executable, "-m", "mypy", "--strict", rms.SOURCE_TARGET]]


def test_main_fails_closed_when_configured_gate_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing configured gate returns its code and skips the probe."""
    monkeypatch.setattr(rms, "run_configured_mypy", lambda: (2, "boom\n"))

    def _explode() -> str:
        """Reject an unexpected probe invocation in this retained control-flow unit test."""
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


def test_main_reports_improvement_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patched_probe: None,
    capsys: CaptureFixture[str],
) -> None:
    """Debt reduction prints the baseline-tightening hint."""
    ledger_path = _point_ledger_at(tmp_path, monkeypatch)
    rms.write_ledger(
        rms.StrictDebtLedger(
            mypy_version="v",
            total=5,
            per_module={"scpn_control.control.nmpc_controller": 4, "scpn_control.core.current_drive": 1},
        ),
        ledger_path,
    )

    assert rms.main([]) == 0

    assert "debt fell by 2 since the baseline" in capsys.readouterr().out


def test_main_fails_on_regression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_probe: None) -> None:
    """Debt above the baseline fails the ratchet."""
    ledger_path = _point_ledger_at(tmp_path, monkeypatch)
    rms.write_ledger(rms.StrictDebtLedger(mypy_version="v", total=1, per_module={"historical.only": 1}), ledger_path)

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
    rms.write_ledger(rms.StrictDebtLedger(mypy_version="v", total=1, per_module={"historical.only": 1}), ledger_path)

    assert rms.main(["--update-baseline"]) == 3
    # The existing ledger must be left untouched on refusal.
    assert json.loads(ledger_path.read_text())["total"] == 1


def test_main_update_baseline_allows_increase_with_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_probe: None
) -> None:
    """The increase flag permits recording higher debt deliberately."""
    ledger_path = _point_ledger_at(tmp_path, monkeypatch)
    rms.write_ledger(rms.StrictDebtLedger(mypy_version="v", total=1, per_module={"historical.only": 1}), ledger_path)

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
        """Reject an unexpected probe invocation in this retained control-flow unit test."""
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


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "other"),
        ("total", True),
        ("total", -1),
        ("total", 0.5),
        ("total", "0"),
        ("total", None),
        ("per_module", []),
        ("per_module", {"x": True}),
        ("per_module", {"x": 1.9}),
        ("per_module", {"x": "1"}),
        ("per_module", {"x": -1}),
        ("per_module", {"x": None}),
        ("per_module", {"x": 1}),
        ("per_module", {"": 0}),
        ("per_module", {" ": 0}),
        ("per_module", {1: 0}),
        ("mypy_version", None),
        ("mypy_version", 1),
    ],
)
def test_public_ledger_refuses_invalid_counts_labels_and_version(field: str, value: object) -> None:
    """The JSON-value API rejects coercion and incoherent totals instead of silently admitting debt."""
    payload: dict[str, object] = {"schema": rms.LEDGER_SCHEMA, "total": 0, "per_module": {}}
    payload[field] = value
    with pytest.raises(ValueError):
        rms.StrictDebtLedger.from_dict(payload)


def test_public_legacy_ledger_retains_exact_counts_and_optional_version(tmp_path: Path) -> None:
    """Omitted version and unknown fields remain supported; serialization copies and sorts module counts."""
    payload: dict[str, object] = {
        "schema": rms.LEDGER_SCHEMA,
        "total": 2,
        "per_module": {"z": 0, "a": 2},
        "opaque": {"old": True},
    }
    ledger = rms.StrictDebtLedger.from_dict(payload)
    assert ledger.mypy_version == "" and ledger.per_module == {"z": 0, "a": 2}
    path = tmp_path / "ledger.json"
    rms.write_ledger(ledger, path)
    assert rms.load_ledger(path) == ledger
    assert path.read_bytes().endswith(b"\n")
    assert rms.StrictDebtLedger.from_dict({"schema": rms.LEDGER_SCHEMA, "total": 0}) == rms.StrictDebtLedger("", 0, {})


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        "[]",
        "null",
        '{"schema":"scpn-control.mypy-strict-debt.v1","total":0,"total":1,"per_module":{}}',
        '{"schema":"scpn-control.mypy-strict-debt.v1","total":1,"per_module":{"x":1,"x":0}}',
    ],
)
def test_public_persisted_ledger_refuses_invalid_json_and_duplicate_keys(tmp_path: Path, payload: str) -> None:
    """Actual file loading rejects malformed roots and duplicate keys before a dict can discard their history."""
    path = tmp_path / "ledger.json"
    path.write_text(payload, encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        rms.load_ledger(path)
    assert path.read_bytes() == before


@pytest.fixture
def actual_mypy_project(tmp_path: Path) -> Path:
    """Copy the exact gate/config into a typed subset project using the installed native Mypy provider.

    The small source and validation packages are declared fixtures, not a claim
    that the entire canonical package was checked by these subprocess tests.
    """
    (tmp_path / "tools").mkdir()
    (tmp_path / "src/scpn_control").mkdir(parents=True)
    (tmp_path / "validation").mkdir()
    for name in ["tools/run_mypy_strict.py", "pyproject.toml"]:
        (tmp_path / name).write_bytes((rms.REPO_ROOT / name).read_bytes())
    (tmp_path / "src/scpn_control/__init__.py").write_text(
        '"""Typed native Mypy fixture package."""\n'
        "def identity(value: int) -> int:\n"
        '    """Return a typed value unchanged."""\n'
        "    return value\n"
    )
    (tmp_path / "validation/__init__.py").write_text('"""Native validation target fixture."""\n')
    return tmp_path


def _actual_cli(root: Path, flags: list[str]) -> subprocess.CompletedProcess[str]:
    """Execute the byte-exact copied CLI with the actual current Python/Mypy runtime and a bounded timeout."""
    return subprocess.run(
        [sys.executable, str(root / "tools/run_mypy_strict.py"), *flags],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=40,
    )


@pytest.mark.parametrize("flags", [[], ["--update-baseline"], ["--update-baseline", "--allow-baseline-increase"]])
@pytest.mark.parametrize("fault", ["boolean", "float", "negative", "version", "duplicate", "root", "utf8"])
def test_actual_cli_refuses_invalid_ledger_without_overwrite(
    actual_mypy_project: Path, flags: list[str], fault: str
) -> None:
    """Real Mypy checks cannot admit a malformed ledger or bypass its validation with either update flag."""
    payload: dict[str, object] = {"schema": rms.LEDGER_SCHEMA, "total": 0, "per_module": {}}
    if fault == "boolean":
        payload["total"] = True
    elif fault == "float":
        payload.update(total=1, per_module={"x": 1.9})
    elif fault == "negative":
        payload["per_module"] = {"x": -1}
    elif fault == "version":
        payload["mypy_version"] = None
    data = json.dumps(payload).encode()
    if fault == "duplicate":
        data = b'{"schema":"scpn-control.mypy-strict-debt.v1","total":0,"total":1,"per_module":{}}'
    elif fault == "root":
        data = b"[]"
    elif fault == "utf8":
        data = b"\xff"
    path = actual_mypy_project / "tools/mypy_strict_debt.json"
    path.write_bytes(data)
    result = _actual_cli(actual_mypy_project, flags)
    assert result.returncode == 2, result.stdout + result.stderr
    assert result.stderr == "[mypy-strict] FAILED: strict-debt ledger could not be read.\n"
    assert path.read_bytes() == data


@pytest.mark.parametrize("baseline", [0, 2])
def test_actual_cli_clean_counts_retain_or_improve_baseline(actual_mypy_project: Path, baseline: int) -> None:
    """The default command runs both real checks and leaves a valid flat or improving historical ledger intact."""
    path = actual_mypy_project / "tools/mypy_strict_debt.json"
    rms.write_ledger(_ledger(baseline, {"historical.only": baseline} if baseline else {}), path)
    before = path.read_bytes()
    result = _actual_cli(actual_mypy_project, [])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "configured gate clean" in result.stdout
    assert f"within baseline (0 <= {baseline})" in result.stdout
    assert ("debt fell by 2" in result.stdout) is bool(baseline)
    assert path.read_bytes() == before


@pytest.mark.parametrize("existing", [False, True])
def test_actual_cli_update_clean_native_counts(actual_mypy_project: Path, existing: bool) -> None:
    """A real clean probe records zero debt and the installed provider version in a new or existing ledger."""
    path = actual_mypy_project / "tools/mypy_strict_debt.json"
    if existing:
        rms.write_ledger(_ledger(2, {"historical.only": 2}), path)
    result = _actual_cli(actual_mypy_project, ["--update-baseline"])
    assert result.returncode == 0, result.stdout + result.stderr
    saved = rms.load_ledger(path)
    assert saved.total == 0 and saved.per_module == {} and saved.mypy_version.startswith("mypy ")


@pytest.mark.parametrize(
    "case",
    ["configured_failure", "equal", "total_regression", "flat_regression", "increase_refused", "increase_allowed"],
)
def test_actual_cli_typing_errors_obey_configured_gate_and_ratchet(actual_mypy_project: Path, case: str) -> None:
    """Real type errors block the default gate; the explicit advisory mode still enforces per-module debt/update rules."""
    (actual_mypy_project / "src/scpn_control/__init__.py").write_text(
        '"""A real typed source with a return-contract error."""\n'
        "def invalid(value: int) -> int:\n"
        '    """Declare an integer result but return text for the native checker."""\n'
        '    return "wrong"\n'
    )
    baseline = (
        {"scpn_control.__init__": 1} if case == "equal" else {"historical.only": 1} if case == "flat_regression" else {}
    )
    path = actual_mypy_project / "tools/mypy_strict_debt.json"
    rms.write_ledger(_ledger(sum(baseline.values()), baseline), path)
    before = path.read_bytes()
    flags = (
        ["--update-baseline", "--allow-baseline-increase"]
        if case == "configured_failure"
        else ["--skip-configured-gate"]
    )
    if case.startswith("increase_"):
        flags.append("--update-baseline")
    if case == "increase_allowed":
        flags.append("--allow-baseline-increase")
    result = _actual_cli(actual_mypy_project, flags)
    expected = 0 if case in {"equal", "increase_allowed"} else 3 if case == "increase_refused" else 1
    assert result.returncode == expected, result.stdout + result.stderr
    if case == "increase_allowed":
        saved = rms.load_ledger(path)
        assert saved.total == 1 and saved.per_module == {"scpn_control.__init__": 1}
    else:
        assert path.read_bytes() == before
    if case == "configured_failure":
        assert "running strict probe" not in result.stdout


def test_actual_cli_refuses_fatal_native_probe(actual_mypy_project: Path) -> None:
    """A genuine Mypy syntax-error exit outside zero/one refuses the advisory probe with fixed caller text."""
    (actual_mypy_project / "src/scpn_control/__init__.py").write_text("def invalid(:\n")
    result = _actual_cli(actual_mypy_project, ["--skip-configured-gate"])
    assert result.returncode == 2
    assert result.stderr == "[mypy-strict] FAILED: strict mypy probe did not complete.\n"


@pytest.mark.parametrize("skip", [False, True])
def test_actual_python_entry_refuses_missing_native_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], skip: bool
) -> None:
    """An actual process launch failure maps to a static configured/probe refusal without exception text."""
    monkeypatch.setattr(sys, "executable", str(tmp_path / "absent-python"))
    assert rms.main(["--skip-configured-gate"] if skip else []) == 2
    label = "strict mypy probe did not complete" if skip else "configured mypy process could not be launched"
    assert capsys.readouterr().err == f"[mypy-strict] FAILED: {label}.\n"


def test_actual_python_entry_refuses_write_to_missing_parent(
    actual_mypy_project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A real clean Mypy probe precedes a native missing-parent write failure; no directory is created."""
    path = actual_mypy_project / "absent-parent/ledger.json"
    monkeypatch.setattr(rms, "REPO_ROOT", actual_mypy_project)
    monkeypatch.setattr(rms, "LEDGER_PATH", path)
    assert rms.main(["--update-baseline"]) == 2
    assert capsys.readouterr().err == "[mypy-strict] FAILED: strict-debt ledger could not be written.\n"
    assert not path.exists() and not path.parent.exists()


@pytest.mark.parametrize("variant", ["clean_no_summary", "error_no_summary", "error_columns"])
def test_actual_cli_observes_native_output_configuration(actual_mypy_project: Path, variant: str) -> None:
    """Real output variants exercise line-count fallback and refuse missing or unattributable summaries.

    Only the copied fixture config changes. The canonical Mypy configuration is
    preserved, and these cases are explicitly not its default output profile.
    """
    config = actual_mypy_project / "pyproject.toml"
    option = "show_column_numbers = true" if variant == "error_columns" else "error_summary = false"
    text = config.read_text().replace("[tool.mypy]\n", "[tool.mypy]\n" + option + "\n", 1)
    config.write_text(text)
    if variant != "clean_no_summary":
        (actual_mypy_project / "src/scpn_control/__init__.py").write_text(
            '"""A real type error under a declared output configuration."""\nx: int = "bad"\n'
        )
    path = actual_mypy_project / "tools/mypy_strict_debt.json"
    rms.write_ledger(_ledger(1, {"scpn_control.__init__": 1}), path)
    before = path.read_bytes()
    result = _actual_cli(actual_mypy_project, ["--skip-configured-gate"])
    assert result.returncode == (0 if variant == "error_no_summary" else 2), result.stdout + result.stderr
    if variant == "clean_no_summary":
        assert result.stderr == "[mypy-strict] FAILED: strict mypy output could not be counted.\n"
    elif variant == "error_columns":
        assert result.stderr == "[mypy-strict] FAILED: strict mypy total disagrees with module counts.\n"
    assert path.read_bytes() == before


def test_actual_cli_refuses_missing_comparison_ledger(actual_mypy_project: Path) -> None:
    """Native clean checks cannot create a missing baseline during an ordinary read-only comparison."""
    result = _actual_cli(actual_mypy_project, [])
    assert result.returncode == 2
    assert result.stderr == "[mypy-strict] FAILED: strict-debt ledger could not be read.\n"
    assert not (actual_mypy_project / "tools/mypy_strict_debt.json").exists()
