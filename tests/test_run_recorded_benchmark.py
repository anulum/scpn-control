# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Recorded benchmark runner tests
"""Exercise the recorded benchmark command through its real subprocess CLI."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import NoReturn

import pytest

import tools.run_recorded_benchmark as runner

TOOL = Path(__file__).resolve().parents[1] / "tools" / "run_recorded_benchmark.py"
SOURCE_ROOT = TOOL.parents[1] / "src"


def _wrapper_command() -> list[str]:
    """Use the real script, with optional explicit raw child-process coverage."""
    command = [sys.executable]
    config = os.environ.get("SCPN_RECORDED_WRAPPER_COVERAGE_RC")
    if config:
        command.extend(["-m", "coverage", "run", "--parallel-mode", "--rcfile=" + config])
    return [*command, str(TOOL)]


def _environment() -> dict[str, str]:
    """Inherit the native process environment and select the actual source tree."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE_ROOT)
    return environment


def _run(
    repository: Path, campaign_id: str, generation: int, *, exit_code: int = 0
) -> subprocess.CompletedProcess[str]:
    """Run a real producer and capture its immutable wrapper campaign result."""
    output = repository / "reports" / "result.json"
    producer = (
        "from pathlib import Path; import sys; "
        f"p=Path({str(output)!r}); p.parent.mkdir(parents=True, exist_ok=True); "
        f"p.write_text('{{\"generation\":{generation}}}\\n', encoding='utf-8'); sys.exit({exit_code})"
    )
    return subprocess.run(
        [
            *_wrapper_command(),
            "--repository-root",
            str(repository),
            "--records-root",
            "benchmark-records",
            "--family",
            "cli-contract",
            "--campaign-id",
            campaign_id,
            "--artifact",
            "report=reports/result.json",
            "--",
            sys.executable,
            "-c",
            producer,
            "--steps",
            "5",
            "--warmup",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_environment(),
    )


def test_cli_preserves_two_successful_runs_and_selects_the_second(tmp_path: Path) -> None:
    """Two real producer processes create two immutable CLI campaign records."""
    repository = tmp_path / "repository"
    repository.mkdir()

    first = _run(repository, "first-cli-run", 1)
    second = _run(repository, "second-cli-run", 2)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    runs = repository / "benchmark-records" / "runs" / "cli-contract"
    assert {path.name for path in runs.iterdir()} == {"first-cli-run", "second-cli-run"}
    latest = json.loads((repository / "benchmark-records" / "latest" / "cli-contract.json").read_text(encoding="utf-8"))
    assert latest["campaign_id"] == "second-cli-run"
    assert json.loads((repository / "reports" / "result.json").read_text(encoding="utf-8")) == {"generation": 2}


def test_cli_retains_failed_output_without_advancing_latest(tmp_path: Path) -> None:
    """A nonzero producer result is sealed but cannot replace latest success."""
    repository = tmp_path / "repository"
    repository.mkdir()
    assert _run(repository, "success", 1).returncode == 0

    failed = _run(repository, "failure", 2, exit_code=7)

    assert failed.returncode == 7
    failed_manifest = json.loads(
        (repository / "benchmark-records" / "runs" / "cli-contract" / "failure" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    latest = json.loads((repository / "benchmark-records" / "latest" / "cli-contract.json").read_text(encoding="utf-8"))
    assert failed_manifest["status"] == "failed"
    assert failed_manifest["artifacts"]
    assert latest["campaign_id"] == "success"


def test_help_creates_no_repository_files(tmp_path: Path) -> None:
    """Help exits before reserving a campaign or touching benchmark custody."""
    repository = tmp_path / "repository"
    repository.mkdir()
    completed = subprocess.run(
        [*_wrapper_command(), "--repository-root", str(repository), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=_environment(),
    )

    assert completed.returncode == 0
    assert "immutable evidence campaign" in completed.stdout
    assert list(repository.iterdir()) == []


def test_direct_main_runs_real_process_and_merges_measurement_metadata(tmp_path: Path) -> None:
    """The imported CLI surface executes a real process with parsed metadata."""
    repository = tmp_path / "repository"
    repository.mkdir()
    producer = "from pathlib import Path; Path('report.json').write_text('result')"
    rc = runner.main(
        [
            "--repository-root",
            str(repository),
            "--records-root",
            str(repository / "records"),
            "--family",
            "direct-main",
            "--campaign-id",
            "direct-main-run",
            "--artifact",
            str("report=" + str(repository / "report.json")),
            "--measurement-json",
            '{"repeat_definition":"one process"}',
            "--",
            sys.executable,
            "-c",
            producer,
            "--steps=3",
            "--warmup",
            "dynamic",
            "--samples",
        ]
    )
    assert rc == 0
    manifest = json.loads(
        (repository / "records" / "runs" / "direct-main" / "direct-main-run" / "manifest.json").read_text()
    )
    assert manifest["measurement"] == {
        "repeat_definition": "one process",
        "samples": "",
        "steps": 3,
        "warmup": "dynamic",
    }


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--family", "invalid", "--artifact", "report=result.json"], "benchmark command is required"),
        (
            ["--family", "invalid", "--artifact", "report=result.json", "--measurement-json", "{", "--", "cmd"],
            "not valid JSON",
        ),
        (
            ["--family", "invalid", "--artifact", "report=result.json", "--measurement-json", "[]", "--", "cmd"],
            "must decode to an object",
        ),
        (["--family", "invalid", "--artifact", "missing-equals", "--", "cmd"], "ROLE=PATH"),
        (["--family", "invalid", "--artifact", "bad role=result.json", "--", "cmd"], "output role"),
    ],
)
def test_direct_main_rejects_invalid_cli_contract(
    arguments: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Malformed command, JSON, and artifact declarations fail before a run."""
    with pytest.raises(SystemExit):
        runner.main(arguments)
    assert message in capsys.readouterr().err


def test_direct_main_rejects_records_root_outside_repository(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Immutable custody cannot be redirected outside the governed repository."""
    repository = tmp_path / "repository"
    repository.mkdir()
    with pytest.raises(SystemExit):
        runner.main(
            [
                "--repository-root",
                str(repository),
                "--records-root",
                str(tmp_path / "outside"),
                "--family",
                "invalid",
                "--artifact",
                "report=result.json",
                "--",
                "cmd",
            ]
        )
    assert "must remain inside" in capsys.readouterr().err


@pytest.mark.parametrize("failure, expected", [(OSError("missing"), 127), (KeyboardInterrupt(), 130)])
def test_direct_main_seals_startup_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: BaseException, expected: int
) -> None:
    """Startup errors and interrupts are retained as failed immutable runs."""
    repository = tmp_path / f"repository-{expected}"
    repository.mkdir()

    def _raise(*_args: object, **_kwargs: object) -> NoReturn:
        """Retain the original injected startup-failure regression."""
        raise failure

    monkeypatch.setattr(subprocess, "run", _raise)
    rc = runner.main(
        [
            "--repository-root",
            str(repository),
            "--records-root",
            "records",
            "--family",
            "startup-failure",
            "--campaign-id",
            f"failure-{expected}",
            "--artifact",
            "report=result.json",
            "--",
            "missing-command",
        ]
    )
    assert rc == expected
    manifest = json.loads(
        (repository / "records" / "runs" / "startup-failure" / f"failure-{expected}" / "manifest.json").read_text()
    )
    assert manifest["status"] == "failed"


def test_cli_noop_refuses_stale_output_and_restores_original(tmp_path: Path) -> None:
    """A successful no-op command fails custody without destroying old output."""
    output = tmp_path / "reports/result.json"
    output.parent.mkdir()
    output.write_text("historical result", encoding="utf-8")
    result = subprocess.run(
        [
            *_wrapper_command(),
            "--repository-root",
            str(tmp_path),
            "--records-root",
            "benchmark-records",
            "--family",
            "noop",
            "--campaign-id",
            "noop-run",
            "--artifact",
            "report=reports/result.json",
            "--",
            sys.executable,
            "-c",
            "pass",
        ],
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stderr
    manifest = json.loads((tmp_path / "benchmark-records/runs/noop/noop-run/manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["exit_code"] == 0
    assert manifest["missing_output_roles"] == ["report"]
    assert output.read_text() == "historical result"
    assert not (tmp_path / "benchmark-records/latest/noop.json").exists()


def test_cli_disjoint_campaigns_complete_concurrently(tmp_path: Path) -> None:
    """Two live producers own disjoint destinations and publish intact records."""
    release = tmp_path / "release"
    processes: list[subprocess.Popen[str]] = []
    try:
        for name in ("first", "second"):
            output = tmp_path / f"{name}.json"
            ready = tmp_path / f"{name}.ready"
            producer = (
                "import time; from pathlib import Path; "
                f"Path({str(ready)!r}).touch(); deadline=time.monotonic()+10\n"
                f"while not Path({str(release)!r}).exists():\n"
                " assert time.monotonic()<deadline; time.sleep(0.01)\n"
                f"Path({str(output)!r}).write_text({name!r})\n"
            )
            processes.append(
                subprocess.Popen(
                    [
                        *_wrapper_command(),
                        "--repository-root",
                        str(tmp_path),
                        "--records-root",
                        f"records-{name}",
                        "--family",
                        "parallel",
                        "--campaign-id",
                        name,
                        "--artifact",
                        f"report={output}",
                        "--",
                        sys.executable,
                        "-c",
                        producer,
                    ],
                    env=_environment(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        deadline = time.monotonic() + 10
        while not all((tmp_path / f"{name}.ready").exists() for name in ("first", "second")):
            assert time.monotonic() < deadline
            assert all(process.poll() is None for process in processes)
            time.sleep(0.01)
        release.touch()
        for name, process in zip(("first", "second"), processes, strict=True):
            _, stderr = process.communicate(timeout=15)
            assert process.returncode == 0, stderr
            manifest = json.loads((tmp_path / f"records-{name}/runs/parallel/{name}/manifest.json").read_text())
            assert manifest["status"] == "succeeded"
            assert (tmp_path / manifest["artifacts"][0]["immutable_path"]).read_text() == name
    finally:
        release.touch(exist_ok=True)
        for process in processes:
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=15)


def test_cli_missing_executable_seals_failure_and_restores_legacy(tmp_path: Path) -> None:
    """A native launch error seals exit127 and releases custody without losing old bytes."""
    output = tmp_path / "report.json"
    output.write_bytes(b"historical result")
    completed = subprocess.run(
        [
            *_wrapper_command(),
            "--repository-root",
            str(tmp_path),
            "--records-root",
            "records",
            "--family",
            "native-launch",
            "--campaign-id",
            "missing-executable",
            "--artifact",
            "report=report.json",
            str(tmp_path / "absent-executable"),
        ],
        env=_environment(),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 127 and completed.stdout == ""
    assert "recorded benchmark could not start:" in completed.stderr
    manifest = json.loads((tmp_path / "records/runs/native-launch/missing-executable/manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["exit_code"] == 127
    assert manifest["missing_output_roles"] == ["report"] and manifest["artifacts"] == []
    assert output.read_bytes() == b"historical result"
    assert not (tmp_path / "records/latest/native-launch.json").exists()
    assert not list((tmp_path / "artifacts/benchmarks/output-leases").glob("*.json"))


def test_cli_preserves_native_argv_cwd_campaign_and_explicit_measurement(tmp_path: Path) -> None:
    """A real child observes exact arguments, its native cwd and the reserved environment."""
    repository = tmp_path / "repository with spaces"
    repository.mkdir()
    producer = (
        "import json,os,sys; from pathlib import Path; "
        "p=Path('reports/argument vector.json'); p.parent.mkdir(); "
        "p.write_text(json.dumps({'argv':sys.argv[1:],'cwd':os.getcwd(),"
        "'campaign':os.environ['SCPN_BENCHMARK_CAMPAIGN_ID'],"
        "'inherited':os.environ['SCPN_WRAPPER_TEST_VALUE']}),encoding='utf-8')"
    )
    arguments = [
        "literal spaces = $`",
        "--steps=3",
        "--repeats",
        "4",
        "--n-bench=8",
        "--samples=9",
        "--iterations=-2",
        "--warmup=x",
        "--steps=5",
    ]
    completed = subprocess.run(
        [
            *_wrapper_command(),
            "--repository-root",
            str(repository),
            "--records-root",
            "records",
            "--family",
            "native-argv",
            "--campaign-id",
            "argv-run",
            "--artifact",
            "report=reports/argument vector.json",
            "--measurement-json",
            '{"steps":"explicit override","operator":"process boundary"}',
            "--",
            sys.executable,
            "-c",
            producer,
            *arguments,
        ],
        env={**_environment(), "SCPN_WRAPPER_TEST_VALUE": "inherited value"},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    output = repository / "reports/argument vector.json"
    observed = json.loads(output.read_text(encoding="utf-8"))
    assert observed["argv"] == arguments
    assert Path(observed["cwd"]) == repository.resolve()
    assert observed["campaign"] == "argv-run" and observed["inherited"] == "inherited value"
    manifest = json.loads((repository / "records/runs/native-argv/argv-run/manifest.json").read_text())
    assert manifest["status"] == "succeeded" and manifest["production_claim_allowed"] is False
    assert manifest["measurement"] == {
        "steps": "explicit override",
        "operator": "process boundary",
        "repeats": 4,
        "samples": 9,
        "iterations": -2,
        "warmup": "x",
    }
    assert (repository / manifest["artifacts"][0]["immutable_path"]).read_bytes() == output.read_bytes()


def _send_native_interrupt(process: subprocess.Popen[str]) -> None:
    """Signal the wrapper through native SIGINT or its private Windows console."""
    if sys.platform != "win32":
        process.send_signal(signal.SIGINT)
        return
    sender = (
        "import ctypes,sys,time; from ctypes import wintypes; "
        "k=ctypes.WinDLL('kernel32',use_last_error=True); "
        "k.AttachConsole.argtypes=(wintypes.DWORD,); k.AttachConsole.restype=wintypes.BOOL; "
        "k.GenerateConsoleCtrlEvent.argtypes=(wintypes.DWORD,wintypes.DWORD); "
        "k.GenerateConsoleCtrlEvent.restype=wintypes.BOOL; "
        "k.SetConsoleCtrlHandler.argtypes=(ctypes.c_void_p,wintypes.BOOL); "
        "k.SetConsoleCtrlHandler.restype=wintypes.BOOL; "
        "k.FreeConsole(); assert k.AttachConsole(int(sys.argv[1])),ctypes.get_last_error(); "
        "assert k.SetConsoleCtrlHandler(None,True),ctypes.get_last_error(); "
        "assert k.GenerateConsoleCtrlEvent(0,0),ctypes.get_last_error(); "
        "time.sleep(0.2); k.FreeConsole()"
    )
    completed = subprocess.run(
        [sys.executable, "-c", sender, str(process.pid)],
        env=_environment(),
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr


def _assert_native_child_stopped(pid: int) -> None:
    """Check the actual child exited without sending a destructive liveness probe on Windows."""
    deadline = time.monotonic() + 5
    if sys.platform != "win32":
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            assert time.monotonic() < deadline, f"interrupted child {pid} still exists"
            time.sleep(0.01)
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        assert ctypes.get_last_error() == 87
        return
    try:
        code = wintypes.DWORD()
        while True:
            assert kernel.GetExitCodeProcess(handle, ctypes.byref(code))
            if code.value != 259:
                return
            assert time.monotonic() < deadline, f"interrupted child {pid} is still running"
            time.sleep(0.01)
    finally:
        assert kernel.CloseHandle(handle)


def test_cli_native_interrupt_stops_child_and_seals_failed_custody(tmp_path: Path) -> None:
    """A genuine OS interrupt stops the actual waiting command and restores absent output."""
    ready = tmp_path / "producer.ready"
    output = tmp_path / "report.json"
    output.write_bytes(b"historical result")
    producer = (
        "import os,time; from pathlib import Path; "
        "p=Path('producer.ready.partial'); p.write_text(str(os.getpid())); p.replace('producer.ready')\n"
        "while True: time.sleep(0.02)\n"
    )
    command = _wrapper_command()
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_CONSOLE
        bootstrap = (
            "import ctypes,runpy,sys; from ctypes import wintypes; "
            "k=ctypes.WinDLL('kernel32',use_last_error=True); "
            "k.SetConsoleCtrlHandler.argtypes=(ctypes.c_void_p,wintypes.BOOL); "
            "k.SetConsoleCtrlHandler.restype=wintypes.BOOL; "
            "assert k.SetConsoleCtrlHandler(None,False),ctypes.get_last_error(); "
            "args=sys.argv[1:]\n"
            "if args[0]=='-m':\n"
            " sys.argv=args[1:];runpy.run_module(args[1],run_name='__main__',alter_sys=True)\n"
            "else:\n"
            " sys.argv=args;runpy.run_path(args[0],run_name='__main__')\n"
        )
        command = [sys.executable, "-c", bootstrap, *command[1:]]
    else:
        flags = 0
    process = subprocess.Popen(
        [
            *command,
            "--repository-root",
            str(tmp_path),
            "--records-root",
            "records",
            "--family",
            "native-interrupt",
            "--campaign-id",
            "interrupted-command",
            "--artifact",
            "report=report.json",
            "--",
            sys.executable,
            "-c",
            producer,
        ],
        env=_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=flags,
    )
    try:
        deadline = time.monotonic() + 15
        while True:
            assert time.monotonic() < deadline and process.poll() is None
            try:
                child_pid = int(ready.read_text())
            except (FileNotFoundError, PermissionError):
                time.sleep(0.01)
            else:
                break
        _send_native_interrupt(process)
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 130 and stdout == "", stderr
        _assert_native_child_stopped(child_pid)
        manifest = json.loads(
            (tmp_path / "records/runs/native-interrupt/interrupted-command/manifest.json").read_text()
        )
        assert manifest["status"] == "failed" and manifest["exit_code"] == 130
        assert manifest["missing_output_roles"] == ["report"]
        assert output.read_bytes() == b"historical result"
        assert not (tmp_path / "records/latest/native-interrupt.json").exists()
        assert not list((tmp_path / "artifacts/benchmarks/output-leases").glob("*.json"))
    finally:
        if process.poll() is None:
            _send_native_interrupt(process)
        process.communicate(timeout=20)
