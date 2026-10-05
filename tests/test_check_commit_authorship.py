# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Commit Authorship Guard Tests
"""Exercise the maintained authorship API and hook/CI command on real message files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tools.check_commit_authorship import REQUIRED_AUTHORSHIP_LINE, main, validate_commit_message_file

ROOT = Path(__file__).resolve().parents[1]


def _message(*lines: str) -> str:
    """Serialize literal commit-message lines with a final newline."""
    return "\n".join(lines) + "\n"


def test_commit_message_accepts_required_authorship_line(tmp_path: Path) -> None:
    """Accept one canonical authorship line in a real UTF-8 message file."""
    path = tmp_path / "COMMIT_EDITMSG"
    path.write_text(
        _message(
            "chore: update policy guard",
            "",
            REQUIRED_AUTHORSHIP_LINE,
        ),
        encoding="utf-8",
    )

    assert validate_commit_message_file(path) == []


def test_commit_message_rejects_missing_authorship_line(tmp_path: Path) -> None:
    """Report the required literal when the message omits its authorship."""
    path = tmp_path / "COMMIT_EDITMSG"
    path.write_text("chore: update policy guard\n", encoding="utf-8")

    violations = validate_commit_message_file(path)

    assert violations == [f"missing required authorship line: {REQUIRED_AUTHORSHIP_LINE}"]


def test_commit_message_rejects_legacy_git_coauthor_trailer(tmp_path: Path) -> None:
    """Reject the superseded project co-author trailer alongside canonical authorship."""
    path = tmp_path / "COMMIT_EDITMSG"
    path.write_text(
        _message(
            "chore: update policy guard",
            "",
            REQUIRED_AUTHORSHIP_LINE,
            "Co-Authored-By: Arcane Sapience <protoscience@anulum.li>",
        ),
        encoding="utf-8",
    )

    violations = validate_commit_message_file(path)

    assert violations == ["legacy Git co-author trailer is forbidden for new commits"]


def test_commit_message_rejects_duplicate_required_line(tmp_path: Path) -> None:
    """Require exactly one canonical line rather than accepting duplicated attribution."""
    path = tmp_path / "COMMIT_EDITMSG"
    path.write_text(
        _message(
            "chore: update policy guard",
            "",
            REQUIRED_AUTHORSHIP_LINE,
            REQUIRED_AUTHORSHIP_LINE,
        ),
        encoding="utf-8",
    )

    violations = validate_commit_message_file(path)

    assert violations == ["duplicate required authorship line"]


def test_commit_message_rejects_other_authorship_line(tmp_path: Path) -> None:
    """Reject competing authorship even when the required line is also present."""
    path = tmp_path / "COMMIT_EDITMSG"
    path.write_text(
        _message(
            "chore: update policy guard",
            "",
            REQUIRED_AUTHORSHIP_LINE,
            "Authored by Example Person (example@example.invalid)",
        ),
        encoding="utf-8",
    )

    violations = validate_commit_message_file(path)

    assert violations == ["unexpected authorship line: Authored by Example Person (example@example.invalid)"]


@pytest.mark.parametrize(
    "additional",
    [
        "  Authored by Example Person (example@example.invalid)",
        "\tAuthored by Example Person (example@example.invalid)",
        "authored by Example Person (example@example.invalid)",
        "Authored by",
        "Authored  by Example Person (example@example.invalid)",
        "Co-Authored-By: Example Person <example@example.invalid>",
        "  Co-Authored-By: Arcane Sapience <protoscience@anulum.li>",
        "co-authored-by: Example Person <example@example.invalid>",
    ],
)
def test_alternate_attribution_cannot_hide_beside_required_line(tmp_path: Path, additional: str) -> None:
    """Forbid alternate attribution independent of its indentation, case or email."""
    path = tmp_path / "COMMIT_EDITMSG"
    path.write_text(_message("fix: bounded guard", REQUIRED_AUTHORSHIP_LINE, additional), encoding="utf-8")
    assert validate_commit_message_file(path)


@pytest.mark.parametrize("padded", ["  " + REQUIRED_AUTHORSHIP_LINE, REQUIRED_AUTHORSHIP_LINE + "  "])
def test_required_authorship_is_a_literal_line(tmp_path: Path, padded: str) -> None:
    """The canonical rule requires exact text instead of silently trimming a substitute."""
    path = tmp_path / "COMMIT_EDITMSG"
    path.write_text(_message("fix: bounded guard", padded), encoding="utf-8")
    assert validate_commit_message_file(path)


def _cli(*paths: Path | str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    """Run the production script with the same argument transport as hook and CI."""
    return subprocess.run(
        [sys.executable, str(ROOT / "tools/check_commit_authorship.py"), *map(str, paths)],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize("ending", ["\n", "\r\n", ""])
def test_real_git_commit_message_passes_without_rewriting(tmp_path: Path, ending: str) -> None:
    """Read actual canonical Git history and preserve its message bytes through API/CLI checks."""
    message = subprocess.check_output(["git", "log", "-1", "--format=%B", "HEAD"], cwd=ROOT, text=True)
    assert REQUIRED_AUTHORSHIP_LINE in message.splitlines()
    text = ending.join(message.rstrip("\n").splitlines()) if ending else message.rstrip("\n")
    path = tmp_path / "actual git message.txt"
    path.write_bytes(text.encode("utf-8"))
    before = path.read_bytes()
    assert validate_commit_message_file(path) == []
    result = _cli(path.name, cwd=tmp_path)
    assert result.returncode == 0 and result.stdout == "" and result.stderr == ""
    assert path.read_bytes() == before


def test_cli_reports_each_selected_invalid_message(tmp_path: Path) -> None:
    """The CI multi-message invocation checks later files after the first refusal."""
    missing = tmp_path / "missing attribution.txt"
    competing = tmp_path / "competing attribution.txt"
    valid = tmp_path / "valid.txt"
    missing.write_text("fix: no attribution\n", encoding="utf-8")
    competing.write_text(_message(REQUIRED_AUTHORSHIP_LINE, "  Authored by Example"), encoding="utf-8")
    valid.write_text(REQUIRED_AUTHORSHIP_LINE, encoding="utf-8")
    result = _cli(missing, valid, competing)
    assert result.returncode == 1 and result.stdout == ""
    assert str(missing) in result.stderr and str(competing) in result.stderr
    assert (
        str(valid) not in result.stderr
        and "missing required" in result.stderr
        and "unexpected authorship" in result.stderr
    )


@pytest.mark.parametrize("defect", ["missing", "directory", "invalid_utf8"])
def test_cli_read_failure_refuses_and_continues(tmp_path: Path, defect: str) -> None:
    """Actual filesystem/decoding failures never retire later requested messages."""
    unreadable = tmp_path / "unreadable.txt"
    if defect == "directory":
        unreadable.mkdir()
    elif defect == "invalid_utf8":
        unreadable.write_bytes(b"\xff\xfe")
    invalid = tmp_path / "later_invalid.txt"
    invalid.write_text("no authorship\n", encoding="utf-8")
    result = _cli(unreadable, invalid)
    assert result.returncode == 1 and "could not read UTF-8 commit message" in result.stderr
    assert str(unreadable) in result.stderr and str(invalid) in result.stderr and "Traceback" not in result.stderr


def test_file_api_preserves_its_read_error(tmp_path: Path) -> None:
    """The reusable file API exposes its declared missing-file error to callers."""
    with pytest.raises(FileNotFoundError):
        validate_commit_message_file(tmp_path / "absent")


def test_direct_main_applies_the_same_file_contract(tmp_path: Path) -> None:
    """Exercise the public main API without changing process arguments."""
    path = tmp_path / "COMMIT_EDITMSG"
    path.write_text(REQUIRED_AUTHORSHIP_LINE, encoding="utf-8")
    argv_before = sys.argv[:]
    assert main([str(path)]) == 0
    assert sys.argv == argv_before


@pytest.mark.parametrize("args", [[], ["--unknown"]])
def test_cli_requires_valid_message_arguments(args: list[str]) -> None:
    """Missing paths and unknown flags are parser refusals rather than vacuous passes."""
    assert _cli(*args).returncode == 2


def test_native_authorship_example_executes() -> None:
    """Execute the source API example with its canonical literal, without Git writes."""
    import doctest

    from tools import check_commit_authorship

    result = doctest.testmod(check_commit_authorship, raise_on_error=True)
    assert result.failed == 0 and result.attempted == 1
