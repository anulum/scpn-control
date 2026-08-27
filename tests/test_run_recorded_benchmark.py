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
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import pytest

import tools.run_recorded_benchmark as runner

TOOL = Path(__file__).resolve().parents[1] / "tools" / "run_recorded_benchmark.py"
SOURCE_ROOT = TOOL.parents[1] / "src"


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE_ROOT)
    return environment


def _run(
    repository: Path, campaign_id: str, generation: int, *, exit_code: int = 0
) -> subprocess.CompletedProcess[str]:
    output = repository / "reports" / "result.json"
    producer = (
        "from pathlib import Path; import sys; "
        f"p=Path({str(output)!r}); p.parent.mkdir(parents=True, exist_ok=True); "
        f"p.write_text('{{\"generation\":{generation}}}\\n', encoding='utf-8'); sys.exit({exit_code})"
    )
    return subprocess.run(
        [
            sys.executable,
            str(TOOL),
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
        [sys.executable, str(TOOL), "--repository-root", str(repository), "--help"],
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
