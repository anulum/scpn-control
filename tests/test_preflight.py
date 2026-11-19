# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Preflight Release Evidence Tests

"""Regression tests for the local preflight release-evidence contract."""

from __future__ import annotations

import importlib.util
import re
import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_PATH = REPO_ROOT / "tools" / "preflight.py"


def _load_preflight_module() -> ModuleType:
    """Execute the preflight script as a fresh module object."""
    spec = importlib.util.spec_from_file_location("scpn_control_preflight", PREFLIGHT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_runs_top_level_release_evidence_gate() -> None:
    """Local preflight must exercise release-evidence artifact admission."""
    preflight = _load_preflight_module()

    gates = {name: command for name, command, _cwd in preflight.GATES}

    assert gates["release-evidence"] == [
        preflight._PY,
        "-m",
        "scpn_control.cli",
        "validate-release-evidence",
    ]
    assert callable(preflight.run_release_evidence_gate)


def test_release_evidence_gate_is_not_skipped_by_no_tests_or_no_rust() -> None:
    """Evidence admission is not a test or Rust-only gate, so fast preflight keeps it."""
    preflight = _load_preflight_module()

    assert "release-evidence" not in preflight.TEST_GATES
    assert "release-evidence" not in preflight.RUST_GATES


def test_preflight_runs_joss_submission_gate() -> None:
    """Local preflight must keep the JOSS paper workflow reviewable."""
    preflight = _load_preflight_module()

    gates = {name: command for name, command, _cwd in preflight.GATES}

    assert gates["joss-submission"] == [preflight._PY, "tools/check_joss_submission.py"]
    assert "joss-submission" not in preflight.TEST_GATES
    assert "joss-submission" not in preflight.RUST_GATES


def test_preflight_runs_changelog_sync_gate() -> None:
    """Local preflight must reject rendered changelog drift."""
    preflight = _load_preflight_module()

    gates = {name: command for name, command, _cwd in preflight.GATES}

    assert gates["changelog-sync"] == [preflight._PY, "tools/check_changelog_sync.py"]
    assert "changelog-sync" not in preflight.TEST_GATES
    assert "changelog-sync" not in preflight.RUST_GATES


def test_preflight_runs_rust_toolchain_contract_gate() -> None:
    """Local preflight must reject floating or divergent Rust toolchains."""
    preflight = _load_preflight_module()

    gates = {name: command for name, command, _cwd in preflight.GATES}

    assert gates["rust-toolchain-contract"] == [preflight._PY, "tools/check_rust_toolchain_contract.py"]
    assert "rust-toolchain-contract" not in preflight.TEST_GATES
    assert "rust-toolchain-contract" not in preflight.RUST_GATES


def test_preflight_runs_runtime_wiring_gate() -> None:
    """Local preflight must fail closed on source modules with no repository wiring."""
    preflight = _load_preflight_module()

    gates = {name: command for name, command, _cwd in preflight.GATES}

    assert gates["runtime-wiring"] == [preflight._PY, "tools/check_runtime_wiring.py"]
    assert "runtime-wiring" not in preflight.TEST_GATES
    assert "runtime-wiring" not in preflight.RUST_GATES


def test_preflight_runs_coverage_pragma_gate() -> None:
    """Local preflight must reject unreasoned source coverage exclusions."""
    preflight = _load_preflight_module()

    gates = {name: command for name, command, _cwd in preflight.GATES}

    assert gates["coverage-pragmas"] == [preflight._PY, "tools/check_coverage_pragmas.py"]
    assert "coverage-pragmas" not in preflight.TEST_GATES
    assert "coverage-pragmas" not in preflight.RUST_GATES


def test_preflight_runs_docstring_quality_gate() -> None:
    """Local preflight must enforce public API docstring quality."""
    preflight = _load_preflight_module()

    gates = {name: command for name, command, _cwd in preflight.GATES}

    assert gates["docstring-quality"] == [preflight._PY, "tools/run_docstring_gate.py"]
    assert "docstring-quality" not in preflight.TEST_GATES
    assert "docstring-quality" not in preflight.RUST_GATES


def test_preflight_runs_public_claim_ledger_gate() -> None:
    """Local preflight must reject public-claim admission drift."""
    preflight = _load_preflight_module()

    gates = {name: command for name, command, _cwd in preflight.GATES}

    assert gates["public-claim-ledger"] == [preflight._PY, "tools/public_claim_ledger.py", "--check"]
    assert "public-claim-ledger" not in preflight.TEST_GATES
    assert "public-claim-ledger" not in preflight.RUST_GATES


def test_preflight_runs_studio_offline_sealing_gate() -> None:
    """Local preflight must keep Studio signing custody out of CI and deploy surfaces."""
    preflight = _load_preflight_module()

    gates = {name: command for name, command, _cwd in preflight.GATES}

    assert gates["studio-offline-sealing"] == [preflight._PY, "tools/check_studio_offline_sealing.py"]
    assert "studio-offline-sealing" not in preflight.TEST_GATES
    assert "studio-offline-sealing" not in preflight.RUST_GATES


def test_preflight_subprocess_env_prefers_source_tree() -> None:
    """Local preflight imports the checked-out source tree before installed packages."""
    preflight = _load_preflight_module()

    pythonpath = preflight._subprocess_env()["PYTHONPATH"].split(preflight.os.pathsep)

    assert pythonpath[0] == str(REPO_ROOT / "src")
    assert pythonpath[1] == str(REPO_ROOT)


def test_release_evidence_gate_generates_and_admits_temporary_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Local preflight validates generated release evidence before passing the gate."""
    preflight = _load_preflight_module()
    calls: list[list[str]] = []

    class DummyHandle:
        """File-handle stand-in that discards writes."""

        def __enter__(self) -> "DummyHandle":
            """Return this handle."""
            return self

        def __exit__(self, *_exc_info: object) -> Literal[False]:
            """Propagate any exception."""
            return False

        def write(self, _data: str) -> None:
            """Discard the data."""
            return None

    class DummyTemporaryDirectory:
        """Temporary-directory stand-in that yields the test directory."""

        def __init__(self, prefix: str) -> None:
            """Require the gate's directory prefix."""
            assert prefix == "scpn-control-release-evidence-"

        def __enter__(self) -> str:
            """Create and return the test directory."""
            tmp_path.mkdir(exist_ok=True)
            return str(tmp_path)

        def __exit__(self, *_exc_info: object) -> Literal[False]:
            """Keep the directory and propagate any exception."""
            return False

    class DummyCompletedProcess:
        """Successful process record."""

        returncode = 0
        stdout = ""
        stderr = ""

    def fake_open(_self: Path, *_args: object, **_kwargs: object) -> DummyHandle:
        """Return a discarding handle instead of opening the path."""
        return DummyHandle()

    def fake_run(cmd: list[str], **_kwargs: Any) -> DummyCompletedProcess:
        """Record the command and report success."""
        calls.append(cmd)
        return DummyCompletedProcess()

    monkeypatch.setattr(preflight.tempfile, "TemporaryDirectory", DummyTemporaryDirectory)
    monkeypatch.setattr(preflight.Path, "open", fake_open)
    monkeypatch.setattr(preflight.subprocess, "run", fake_run)

    assert preflight.run_release_evidence_gate() is True
    assert calls[0] == [preflight._PY, "-m", "scpn_control.cli", "validate", "--json-out"]
    assert calls[1][:4] == [preflight._PY, "-m", "scpn_control.cli", "validate-release-evidence"]
    assert calls[1][-1] == "--json-out"


def test_every_static_governance_command_is_a_preflight_gate() -> None:
    """A local "ALL CLEAR" covers every tool command of the static-governance workflow.

    The commit-message authorship check is the one exception: it reads the
    message files of the pushed range, which a local preflight does not know.
    """
    preflight = _load_preflight_module()
    workflow = (REPO_ROOT / ".github/workflows/ci-static-governance.yml").read_text(encoding="utf-8")
    commands = {
        tuple(match.group(1).split())
        for match in re.finditer(r"^\s*(?:run:\s*)?python (tools/[a-z_]+\.py(?: --[a-z-]+)*)\s*$", workflow, re.M)
    }
    gates = {tuple(command[1:]) for _name, command, _cwd in preflight.GATES if command[0] == preflight._PY}
    assert ("tools/capability_manifest.py", "--check") in commands
    assert ("tools/check_docs_coverage.py",) in commands
    assert commands - gates == set()


def test_subprocess_env_keeps_the_callers_pythonpath_last(monkeypatch: pytest.MonkeyPatch) -> None:
    """An existing PYTHONPATH follows the checkout's source and root entries."""
    preflight = _load_preflight_module()
    monkeypatch.setenv("PYTHONPATH", "/caller/site")
    assert preflight._subprocess_env()["PYTHONPATH"].split(preflight.os.pathsep) == [
        str(preflight.ROOT / "src"),
        str(preflight.ROOT),
        "/caller/site",
    ]


def _completed(code: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    """Return a finished process record with the given status and output."""
    return subprocess.CompletedProcess([], code, stdout, stderr)


def _lines(prefix: str) -> str:
    """Return twelve numbered output lines."""
    return "\n".join(f"{prefix}{index}" for index in range(12))


def test_run_gate_reports_pass_and_the_last_ten_lines_of_a_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A gate passes on status 0; a failure prints the last ten lines of each stream; cargo doc denies warnings."""
    preflight = _load_preflight_module()
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Record the call and return the scripted outcome for the command."""
        calls.append((cmd, kwargs))
        if cmd == ["ok"]:
            return _completed(0)
        if cmd == ["quiet"]:
            return _completed(1)
        return _completed(1, _lines("out"), _lines("err"))

    monkeypatch.setattr(preflight.subprocess, "run", fake_run)
    assert preflight.run_gate("first", ["ok"], None) is True
    assert preflight.run_gate("cargo doc", ["noisy"], REPO_ROOT / "scpn-control-rs") is False
    assert preflight.run_gate("silent", ["quiet"], None) is False
    output = capsys.readouterr().out
    assert "  PASS  first (" in output and "  FAIL  cargo doc (" in output and "  FAIL  silent (" in output
    assert "        out2\n" in output and "        out1\n" not in output
    assert "        err11\n" in output and "        err1\n" not in output
    assert calls[0][1]["cwd"] == preflight.ROOT
    assert calls[1][1]["cwd"] == REPO_ROOT / "scpn-control-rs"
    assert calls[1][1]["env"]["RUSTDOCFLAGS"] == "-D warnings -D missing_docs"
    assert "RUSTDOCFLAGS" not in calls[0][1]["env"] or calls[0][1]["env"]["RUSTDOCFLAGS"] == preflight.os.environ.get(
        "RUSTDOCFLAGS"
    )

    monkeypatch.setattr(preflight, "run_release_evidence_gate", lambda: True)
    assert preflight.run_gate("release-evidence", ["unused"], None) is True
    assert len(calls) == 3


@pytest.mark.parametrize(("generate", "admit"), [(1, 0), (0, 1)])
def test_release_evidence_gate_fails_on_either_step(
    generate: int, admit: int, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed report generation stops before admission; a failed admission fails the gate."""
    preflight = _load_preflight_module()
    statuses = [generate, admit]
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Return a failure on the scripted step."""
        calls.append(cmd)
        if len(calls) == 1:
            return _completed(statuses[0], "", "refused\n")
        return _completed(statuses[1], "partial admission\n", "")

    monkeypatch.setattr(preflight.subprocess, "run", fake_run)
    assert preflight.run_release_evidence_gate() is False
    assert len(calls) == (1 if generate else 2)
    output = capsys.readouterr().out
    assert "  FAIL  release-evidence (" in output
    assert ("        refused" if generate else "        partial admission") in output


def test_requirements_change_is_read_from_the_worktree_and_the_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """A requirements path in either the HEAD diff or the staged diff enables hash validation."""
    preflight = _load_preflight_module()
    outputs: dict[tuple[str, ...], str] = {}

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Return the scripted git output for the command."""
        return _completed(0, outputs.get(tuple(cmd), ""))

    monkeypatch.setattr(preflight.subprocess, "run", fake_run)
    outputs = {("git", "diff", "--name-only", "HEAD"): "src/a.py\n", ("git", "diff", "--cached", "--name-only"): ""}
    assert preflight._requirements_changed() is False
    outputs[("git", "diff", "--cached", "--name-only")] = "requirements/ci-test.txt\n"
    assert preflight._requirements_changed() is True


def test_hash_pin_check_refuses_missing_locks_and_failed_dry_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every declared lock must exist and pass a hash-pinned pip dry run; no package is installed."""
    preflight = _load_preflight_module()
    calls: list[list[str]] = []
    statuses: dict[str, subprocess.CompletedProcess[str]] = {}

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Record the dry run and return its scripted outcome."""
        calls.append(cmd)
        return statuses.get(Path(cmd[-1]).name, _completed(0))

    monkeypatch.setattr(preflight.subprocess, "run", fake_run)
    (tmp_path / "requirements").mkdir()
    for name in preflight.CI_REQUIREMENTS[:2]:
        (tmp_path / name).write_text("", encoding="utf-8")
    statuses = {"ci-deps.txt": _completed(1, "", "first\nhash mismatch"), "ci-test.txt": _completed(1)}
    assert preflight.run_hash_pin_check(tmp_path) is False
    output = capsys.readouterr().out
    assert "requirements/ci-deps.txt: hash mismatch" in output
    assert "requirements/ci-test.txt: unknown error" in output
    assert "requirements/ci-lint.txt: required hash lock is missing" in output
    assert all(cmd[1:7] == ["-m", "pip", "install", "--require-hashes", "--dry-run", "-r"] for cmd in calls)

    for name in preflight.CI_REQUIREMENTS:
        (tmp_path / name).write_text("", encoding="utf-8")
    statuses = {}
    assert preflight.run_hash_pin_check(tmp_path) is True
    assert "  PASS  hash-pin check (" in capsys.readouterr().out


def _stub_main(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    *,
    compat: bool = True,
    hashes: bool = True,
    changed: bool = False,
    failing: str | None = None,
) -> tuple[int, list[str], list[str]]:
    """Run ``main`` with stubbed checks; return its status, the gates run and the checks run."""
    gates: list[str] = []
    checks: list[str] = []
    monkeypatch.setattr(sys, "argv", ["preflight.py", *argv])

    def fake_compat() -> bool:
        """Record the compatibility check and return its scripted result."""
        checks.append("compat")
        return compat

    def fake_hashes() -> bool:
        """Record the hash-pin check and return its scripted result."""
        checks.append("hashes")
        return hashes

    monkeypatch.setattr(preflight, "run_python_compat_check", fake_compat)
    monkeypatch.setattr(preflight, "run_hash_pin_check", fake_hashes)
    monkeypatch.setattr(preflight, "_requirements_changed", lambda: changed)

    def fake_gate(name: str, _cmd: list[str], _cwd: Path | None) -> bool:
        """Record the gate and fail only the scripted one."""
        gates.append(name)
        return name != failing

    monkeypatch.setattr(preflight, "run_gate", fake_gate)
    return preflight.main(), gates, checks


def test_main_orders_checks_filters_gates_and_stops_at_the_first_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Compatibility runs first, hashes only when asked or changed, then gates in order until one fails."""
    preflight = _load_preflight_module()
    names = [name for name, _cmd, _cwd in preflight.GATES]

    status, gates, checks = _stub_main(preflight, monkeypatch, [])
    assert (status, gates, checks) == (0, names, ["compat"])
    assert f"preflight: {len(names) + 1} gates\n" in capsys.readouterr().out

    status, gates, checks = _stub_main(preflight, monkeypatch, ["--no-tests", "--no-rust", "--coverage"], changed=True)
    assert status == 0 and checks == ["compat", "hashes"]
    assert gates == [n for n in names if n not in preflight.TEST_GATES | preflight.RUST_GATES]
    output = capsys.readouterr().out
    assert "[coverage, no-rust, no-tests]" in output and "hash-pin validation enabled" in output
    assert "ALL CLEAR: ready to push" in output

    status, gates, _ = _stub_main(preflight, monkeypatch, ["--coverage"])
    assert status == 0 and "pytest (coverage)" in gates and "pytest" not in gates

    status, gates, checks = _stub_main(preflight, monkeypatch, ["--hashes"], compat=False)
    assert (status, gates, checks) == (1, [], ["compat"])
    assert "BLOCKED: python-compat" in capsys.readouterr().out

    status, gates, checks = _stub_main(preflight, monkeypatch, ["--hashes"], hashes=False)
    assert (status, gates, checks) == (1, [], ["compat", "hashes"])
    assert "BLOCKED: hash-pin check" in capsys.readouterr().out

    status, gates, _ = _stub_main(preflight, monkeypatch, [], failing=names[2])
    assert status == 1 and gates == names[:3]
    assert f"BLOCKED: {names[2]}" in capsys.readouterr().out


def test_module_reconfigures_a_non_utf8_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loading the module under a non-UTF-8 console switches both streams to UTF-8 with replacement."""

    class Console:
        """Stand-in stream that records reconfiguration."""

        encoding = "ascii"

        def __init__(self) -> None:
            """Start with no recorded settings."""
            self.settings: dict[str, str] = {}

        def reconfigure(self, **settings: str) -> None:
            """Record the requested settings."""
            self.settings = settings

    out, err = Console(), Console()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    _load_preflight_module()
    assert out.settings == err.settings == {"encoding": "utf-8", "errors": "replace"}


def test_script_entry_exits_with_the_status_of_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """Executed as a script with every process stubbed to succeed, preflight exits 0."""
    from tools import python_dependency_compatibility

    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: _completed(0))
    monkeypatch.setattr(python_dependency_compatibility, "python_compatibility_errors", lambda *_a, **_k: [])
    monkeypatch.setattr(sys, "argv", [str(PREFLIGHT_PATH)])
    monkeypatch.setattr(sys, "path", list(sys.path))
    with pytest.raises(SystemExit) as script_exit:
        runpy.run_path(str(PREFLIGHT_PATH), run_name="__main__")
    assert script_exit.value.code == 0
