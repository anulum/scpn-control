# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — pnpm lock audit gate tests.

"""Tests for the pnpm lock audit gate and its expiring development-only exceptions.

The report fixture keeps the shape ``pnpm audit --json`` (pnpm 11.25.0) printed
for ``studio-web/pnpm-lock.yaml`` on 2026-11-17: one advisory keyed by its npm
id, with ``github_advisory_id``, ``module_name`` and one finding whose paths
start at the lock's direct dependencies.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from tools import pnpm_audit_gate as gate

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "pnpm_audit_gate.py"
GHSA = "GHSA-fc5q-39xc-vjpf"
GRANTED = date(2026, 11, 17)
EXPIRES = date(2026, 11, 24)


MANIFEST: dict[str, Any] = {
    "dependencies": {"react": "19.3.1", "react-dom": "19.3.1"},
    "devDependencies": {
        "@module-federation/vite": "1",
        "@vitejs/plugin-react": "5",
        "@vitest/coverage-v8": "5",
        "vite": "8",
        "vitest": "5",
    },
}

REPORT: dict[str, Any] = {
    "advisories": {
        "1247710": {
            "id": 1247710,
            "module_name": "picomatch",
            "severity": "high",
            "github_advisory_id": GHSA,
            "url": f"https://github.com/advisories/{GHSA}",
            "patched_versions_unpublished": True,
            "findings": [
                {
                    "version": "4.0.7",
                    "dev": True,
                    "optional": False,
                    "bundled": False,
                    "paths": [
                        ".>vite>tinyglobby>picomatch",
                        ".>vite>tinyglobby>fdir>picomatch",
                        ".>vite>picomatch",
                        ".>vitest>vite>picomatch",
                        ".>@vitest/coverage-v8>vitest>@vitest/mocker>vite>picomatch",
                        ".>@vitejs/plugin-react>vite>picomatch",
                        ".>@module-federation/vite>vite>picomatch",
                    ],
                }
            ],
        }
    },
    "metadata": {
        "vulnerabilities": {
            "info": 0,
            "low": 0,
            "moderate": 0,
            "high": 1,
            "critical": 0,
        },
        "dependencies": 4,
        "devDependencies": 160,
        "optionalDependencies": 61,
        "totalDependencies": 178,
    },
}

CLEAN_REPORT: dict[str, Any] = {
    "advisories": {},
    "metadata": copy.deepcopy(REPORT["metadata"]),
}


def _exception(**overrides: object) -> dict[str, object]:
    """Return a valid exception entry, with selected fields replaced."""
    item: dict[str, object] = {
        "advisory": GHSA,
        "package": "picomatch",
        "scope": "dev",
        "granted": GRANTED.isoformat(),
        "expires": EXPIRES.isoformat(),
        "reason": "no patched release; development dependencies only",
    }
    item.update(overrides)
    return item


def _active() -> tuple[Any, ...]:
    """Return the parsed default exception as the active set."""
    return (gate.parse_exception(_exception()),)


class _Runner:
    """Stand-in for ``subprocess.run`` that returns one fixed audit result."""

    def __init__(self, stdout: str, returncode: int) -> None:
        """Store the fixed result and an empty call log."""
        self.stdout = stdout
        self.returncode = returncode
        self.calls: list[dict[str, Any]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Record the call and return the fixed result."""
        self.calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(command, self.returncode, self.stdout, "")


def _runner(stdout: str, returncode: int) -> _Runner:
    """Return a runner that answers every call with one audit result."""
    return _Runner(stdout, returncode)


def _lock_dir(tmp_path: Path, manifest: object = MANIFEST, *, lock: bool = True) -> Path:
    """Create a lock directory with a package.json and, by default, a pnpm lock."""
    lock_dir = tmp_path / "web"
    lock_dir.mkdir()
    (lock_dir / "package.json").write_text(json.dumps(manifest), encoding="utf-8")
    if lock:
        (lock_dir / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n", encoding="utf-8")
    return lock_dir


def _exceptions_file(tmp_path: Path, items: list[dict[str, object]]) -> Path:
    """Write an exceptions file with the given entries."""
    path = tmp_path / "exceptions.json"
    path.write_text(json.dumps({"schema": 1, "exceptions": items}), encoding="utf-8")
    return path


# --- exceptions ------------------------------------------------------------


def test_parse_exception_accepts_a_seven_day_development_grant() -> None:
    """Parse exception accepts a seven day development grant."""
    item = gate.parse_exception(_exception())
    assert (item.advisory, item.package, item.granted, item.expires) == (
        GHSA,
        "picomatch",
        GRANTED,
        EXPIRES,
    )


@pytest.mark.parametrize(
    ("item", "message"),
    [
        ("not a mapping", "exactly the fields"),
        (
            {k: v for k, v in _exception().items() if k != "reason"},
            "exactly the fields",
        ),
        ({**_exception(), "extra": 1}, "exactly the fields"),
        (_exception(advisory="CVE-2026-1"), "not a GitHub advisory id"),
        (_exception(package=" "), "names its package and its reason"),
        (_exception(reason=""), "names its package and its reason"),
        (_exception(scope="prod"), "development dependencies only"),
        (_exception(expires="2026-11-17"), "not 0"),
        (_exception(expires="2026-11-25"), "not 8"),
        (_exception(expires="2026-11-16"), "not -1"),
    ],
)
def test_parse_exception_refuses_invalid_entries(item: object, message: str) -> None:
    """Parse exception refuses invalid entries."""
    with pytest.raises(gate.AuditGateError, match=message):
        gate.parse_exception(item)


def test_parse_exception_refuses_an_unreadable_date() -> None:
    """Parse exception refuses an unreadable date."""
    with pytest.raises(ValueError):
        gate.parse_exception(_exception(granted="17 November"))


def test_load_exceptions_reads_the_schema(tmp_path: Path) -> None:
    """Load exceptions reads the schema."""
    path = _exceptions_file(tmp_path, [_exception()])
    assert gate.load_exceptions(path) == _active()


@pytest.mark.parametrize(
    "payload",
    [[], {"schema": 2, "exceptions": []}, {"schema": 1, "exceptions": {}}],
)
def test_load_exceptions_refuses_another_shape(tmp_path: Path, payload: object) -> None:
    """Load exceptions refuses another shape."""
    path = tmp_path / "exceptions.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(gate.AuditGateError, match="schema: 1"):
        gate.load_exceptions(path)


def test_committed_exceptions_file_is_valid_and_bounded() -> None:
    """The committed exceptions file loads, and every entry it holds is bounded.

    An empty list is the normal state: no advisory in the lock needs an exception.
    """
    items = gate.load_exceptions(ROOT / ".github" / "npm-audit-exceptions.json")
    assert all(1 <= (item.expires - item.granted).days <= gate.MAX_EXCEPTION_DAYS for item in items)


# --- running the audit -----------------------------------------------------


def test_run_audit_runs_pnpm_in_the_lock_directory(tmp_path: Path) -> None:
    """Run audit runs pnpm in the lock directory."""
    runner = _runner(json.dumps(REPORT), 1)
    assert gate.run_audit(tmp_path, runner) == REPORT
    call = runner.calls[0]
    assert call["command"] == ["pnpm", "audit", "--json"]
    assert call["cwd"] == tmp_path
    assert call["check"] is False


def test_run_audit_accepts_a_clean_report(tmp_path: Path) -> None:
    """Run audit accepts a clean report."""
    assert gate.run_audit(tmp_path, _runner(json.dumps(CLEAN_REPORT), 0)) == CLEAN_REPORT


@pytest.mark.parametrize(
    ("stdout", "returncode", "message"),
    [
        ("ERR_PNPM_AUDIT_BAD_RESPONSE", 1, "printed no JSON report"),
        (json.dumps({"error": {"code": "ENOTFOUND"}}), 1, "pnpm audit failed"),
        (json.dumps(["advisories"]), 1, "pnpm audit failed"),
        (json.dumps({"advisories": {}}), 0, "did not return an advisories report"),
        (json.dumps({"advisories": [], "metadata": {}}), 0, "advisories report"),
        (json.dumps(REPORT), 0, "exit 0 disagrees"),
        (json.dumps(CLEAN_REPORT), 1, "exit 1 disagrees"),
        (json.dumps(REPORT), 2, "exit 2 disagrees"),
    ],
)
def test_run_audit_refuses_unusable_or_contradictory_reports(
    tmp_path: Path, stdout: str, returncode: int, message: str
) -> None:
    """Run audit refuses unusable or contradictory reports."""
    with pytest.raises(gate.AuditGateError, match=message):
        gate.run_audit(tmp_path, _runner(stdout, returncode))


# --- judging ---------------------------------------------------------------


def test_development_roots_exclude_production_groups() -> None:
    """Development roots exclude production groups."""
    manifest = {
        "dependencies": {"react": "1"},
        "devDependencies": {"vite": "8", "react": "1", "jsdom": "1"},
        "optionalDependencies": {"jsdom": "1"},
    }
    assert gate.development_roots(manifest) == frozenset({"vite"})
    assert gate.development_roots({}) == frozenset()


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ({"dependencies": ["react"]}, "dependencies is not a mapping"),
        ({"devDependencies": ["vite"]}, "devDependencies is not a mapping"),
    ],
)
def test_development_roots_refuse_malformed_groups(manifest: dict[str, Any], message: str) -> None:
    """Development roots refuse malformed groups."""
    with pytest.raises(gate.AuditGateError, match=message):
        gate.development_roots(manifest)


def test_judge_accepts_the_excepted_development_advisory() -> None:
    """Judge accepts the excepted development advisory."""
    failures, used = gate.judge(REPORT, MANIFEST, _active(), date(2026, 11, 23))
    assert failures == []
    assert used == {(GHSA, "picomatch")}


def test_judge_fails_on_the_day_of_expiry() -> None:
    """Judge fails on the day of expiry."""
    failures, used = gate.judge(REPORT, MANIFEST, _active(), EXPIRES)
    assert f"picomatch {GHSA}: the exception expired on 2026-11-24" in failures
    assert any("has no active exception" in failure for failure in failures)
    assert used == set()


def test_judge_fails_a_stale_exception_after_expiry_even_on_a_clean_lock() -> None:
    """Judge fails a stale exception after expiry even on a clean lock."""
    failures, used = gate.judge(CLEAN_REPORT, MANIFEST, _active(), date(2026, 12, 1))
    assert failures == [f"picomatch {GHSA}: the exception expired on 2026-11-24"]
    assert used == set()


def test_judge_names_the_package_of_each_expired_exception() -> None:
    """One advisory granted for two packages reports the package whose grant expired."""
    exceptions = (
        gate.parse_exception(_exception()),
        gate.parse_exception(_exception(package="micromatch", expires="2026-11-20")),
    )
    failures, used = gate.judge(REPORT, MANIFEST, exceptions, date(2026, 11, 21))
    assert failures == [f"micromatch {GHSA}: the exception expired on 2026-11-20"]
    assert used == {(GHSA, "picomatch")}


def test_judge_accepts_a_clean_lock_while_an_exception_is_still_active() -> None:
    """Judge accepts a clean lock while an exception is still active."""
    assert gate.judge(CLEAN_REPORT, MANIFEST, _active(), GRANTED) == ([], set())


def test_judge_fails_another_advisory() -> None:
    """Judge fails another advisory."""
    report = copy.deepcopy(REPORT)
    report["advisories"]["1247710"]["github_advisory_id"] = "GHSA-aaaa-bbbb-cccc"
    failures, used = gate.judge(report, MANIFEST, _active(), GRANTED)
    assert failures == ["picomatch GHSA-aaaa-bbbb-cccc (high) has no active exception"]
    assert used == set()


def test_judge_fails_the_same_advisory_in_another_package() -> None:
    """Judge fails the same advisory in another package."""
    report = copy.deepcopy(REPORT)
    report["advisories"]["1247710"]["module_name"] = "micromatch"
    failures, _ = gate.judge(report, MANIFEST, _active(), GRANTED)
    assert failures == [f"micromatch {GHSA} (high) has no active exception"]


def test_judge_labels_an_advisory_without_a_github_id_by_its_key() -> None:
    """Judge labels an advisory without a GitHub id by its key."""
    report = copy.deepcopy(REPORT)
    del report["advisories"]["1247710"]["github_advisory_id"]
    failures, _ = gate.judge(report, MANIFEST, _active(), GRANTED)
    assert failures == ["picomatch 1247710 (high) has no active exception"]


def test_judge_fails_an_install_pnpm_does_not_report_as_development_only() -> None:
    """Judge fails an install pnpm does not report as development only."""
    report = copy.deepcopy(REPORT)
    report["advisories"]["1247710"]["findings"][0]["dev"] = False
    failures, used = gate.judge(report, MANIFEST, _active(), GRANTED)
    assert failures == [f"picomatch {GHSA}@4.0.7: pnpm does not report a development-only install"]
    assert used == {(GHSA, "picomatch")}


@pytest.mark.parametrize(
    "path",
    [
        ".>react>vite>picomatch",
        ".>react-dom>picomatch",
        ".>unlisted>picomatch",
        "picomatch",
        "vite>picomatch",
        ".",
    ],
)
def test_judge_fails_a_path_that_does_not_start_at_a_development_dependency(
    path: str,
) -> None:
    """Judge fails a path that does not start at a development dependency."""
    report = copy.deepcopy(REPORT)
    report["advisories"]["1247710"]["findings"][0]["paths"].append(path)
    failures, _ = gate.judge(report, MANIFEST, _active(), GRANTED)
    assert failures == [f"picomatch {GHSA}@4.0.7: {path} does not start at a development dependency"]


def test_judge_fails_a_finding_without_paths() -> None:
    """Judge fails a finding without paths."""
    report = copy.deepcopy(REPORT)
    report["advisories"]["1247710"]["findings"][0]["paths"] = []
    failures, _ = gate.judge(report, MANIFEST, _active(), GRANTED)
    assert failures == [f"picomatch {GHSA}@4.0.7: the report names no dependency path"]


def test_judge_fails_an_advisory_without_findings() -> None:
    """Judge fails an advisory without findings."""
    report = copy.deepcopy(REPORT)
    report["advisories"]["1247710"]["findings"] = []
    failures, _ = gate.judge(report, MANIFEST, _active(), GRANTED)
    assert failures == [f"picomatch {GHSA}: the report names no installed version"]


def test_judge_fails_an_unreadable_advisory_entry() -> None:
    """Judge fails an unreadable advisory entry."""
    report = {"advisories": {"7": "picomatch"}, "metadata": {}}
    failures, used = gate.judge(report, MANIFEST, _active(), GRANTED)
    assert failures == ["advisory 7: unreadable entry"]
    assert used == set()


# --- lock directories and the command line ---------------------------------


def test_check_lock_refuses_a_directory_without_a_pnpm_lock(tmp_path: Path) -> None:
    """Check lock refuses a directory without a pnpm lock."""
    lock_dir = _lock_dir(tmp_path, lock=False)
    failures, used = gate._check_lock(lock_dir, _active(), GRANTED, _runner("", 0))
    assert failures == [f"{lock_dir} has no pnpm-lock.yaml"]
    assert used == set()


def test_check_lock_refuses_a_manifest_that_is_not_an_object(tmp_path: Path) -> None:
    """Check lock refuses a manifest that is not an object."""
    lock_dir = _lock_dir(tmp_path, manifest=["vite"])
    failures, _ = gate._check_lock(lock_dir, _active(), GRANTED, _runner("", 0))
    assert failures == [f"{lock_dir}/package.json is not an object"]


def test_check_lock_reports_a_missing_manifest(tmp_path: Path) -> None:
    """Check lock reports a missing manifest."""
    lock_dir = _lock_dir(tmp_path)
    (lock_dir / "package.json").unlink()
    failures, _ = gate._check_lock(lock_dir, _active(), GRANTED, _runner("", 0))
    assert len(failures) == 1 and "package.json" in failures[0]


def test_main_accepts_the_excepted_advisory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Main accepts the excepted advisory."""
    lock_dir = _lock_dir(tmp_path)
    exceptions = _exceptions_file(tmp_path, [_exception()])
    status = gate.main(
        ["--exceptions", str(exceptions), str(lock_dir)],
        runner=_runner(json.dumps(REPORT), 1),
        today=GRANTED,
    )
    out = capsys.readouterr()
    assert status == 0
    assert out.out == (
        f"{lock_dir}: picomatch {GHSA} accepted in development dependencies only; the exception expires on 2026-11-24\n"
    )
    assert out.err == ""


def test_main_reports_each_package_with_its_own_earliest_active_expiry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One advisory in two packages prints each package with the day its acceptance ends.

    picomatch holds two active grants (11-22 and 11-24) and one expired one
    (11-20); its acceptance ends on the earliest active expiry, 11-22.
    micromatch holds one grant to 11-23. The expired grant still fails the lock.
    """
    lock_dir = _lock_dir(tmp_path)
    report = copy.deepcopy(REPORT)
    second = copy.deepcopy(report["advisories"]["1247710"])
    second["module_name"] = "micromatch"
    second["findings"][0]["paths"] = [".>vite>micromatch"]
    report["advisories"]["1247711"] = second
    exceptions = _exceptions_file(
        tmp_path,
        [
            _exception(),
            _exception(expires="2026-11-22"),
            _exception(expires="2026-11-20"),
            _exception(package="micromatch", expires="2026-11-23"),
        ],
    )
    status = gate.main(
        ["--exceptions", str(exceptions), str(lock_dir)],
        runner=_runner(json.dumps(report), 1),
        today=date(2026, 11, 21),
    )
    out = capsys.readouterr()
    assert status == 1
    assert out.out == (
        f"{lock_dir}: micromatch {GHSA} accepted in development dependencies only;"
        " the exception expires on 2026-11-23\n"
        f"{lock_dir}: picomatch {GHSA} accepted in development dependencies only;"
        " the exception expires on 2026-11-22\n"
    )
    assert out.err == f"{lock_dir}: picomatch {GHSA}: the exception expired on 2026-11-20\n"


def test_main_reports_a_clean_lock(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Main reports a clean lock."""
    lock_dir = _lock_dir(tmp_path)
    exceptions = _exceptions_file(tmp_path, [])
    status = gate.main(
        ["--exceptions", str(exceptions), str(lock_dir)],
        runner=_runner(json.dumps(CLEAN_REPORT), 0),
    )
    assert status == 0
    assert capsys.readouterr().out == f"{lock_dir}: found 0 vulnerabilities\n"


def test_main_fails_without_an_exception(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Main fails without an exception."""
    lock_dir = _lock_dir(tmp_path)
    exceptions = _exceptions_file(tmp_path, [])
    status = gate.main(
        ["--exceptions", str(exceptions), str(lock_dir)],
        runner=_runner(json.dumps(REPORT), 1),
        today=GRANTED,
    )
    out = capsys.readouterr()
    assert status == 1
    assert out.out == ""
    assert out.err == f"{lock_dir}: picomatch {GHSA} (high) has no active exception\n"


def test_main_fails_one_bad_lock_of_several(tmp_path: Path) -> None:
    """Main fails one bad lock of several."""
    good = _lock_dir(tmp_path)
    missing = tmp_path / "missing"
    missing.mkdir()
    exceptions = _exceptions_file(tmp_path, [_exception()])
    status = gate.main(
        ["--exceptions", str(exceptions), str(good), str(missing)],
        runner=_runner(json.dumps(REPORT), 1),
        today=GRANTED,
    )
    assert status == 1


@pytest.mark.parametrize("content", ["{", json.dumps({"schema": 1, "exceptions": [{}]})])
def test_main_stops_on_an_unusable_exceptions_file(
    tmp_path: Path, content: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Main stops on an unusable exceptions file."""
    exceptions = tmp_path / "exceptions.json"
    exceptions.write_text(content, encoding="utf-8")
    status = gate.main(["--exceptions", str(exceptions), str(tmp_path)], runner=_runner("", 0))
    assert status == 2
    assert capsys.readouterr().err.startswith("pnpm audit gate: ")


def test_script_entry_point_exits_two_on_a_missing_exceptions_file(
    tmp_path: Path,
) -> None:
    """Script entry point exits two on a missing exceptions file."""
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--exceptions",
            str(tmp_path / "absent.json"),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 2
    assert "pnpm audit gate:" in completed.stderr
