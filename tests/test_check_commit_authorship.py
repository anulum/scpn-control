# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Commit Authorship Guard Tests

from __future__ import annotations

from pathlib import Path

from tools.check_commit_authorship import REQUIRED_AUTHORSHIP_LINE, validate_commit_message_file


def _message(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def test_commit_message_accepts_required_authorship_line(tmp_path: Path) -> None:
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
    path = tmp_path / "COMMIT_EDITMSG"
    path.write_text("chore: update policy guard\n", encoding="utf-8")

    violations = validate_commit_message_file(path)

    assert violations == [f"missing required authorship line: {REQUIRED_AUTHORSHIP_LINE}"]


def test_commit_message_rejects_legacy_git_coauthor_trailer(tmp_path: Path) -> None:
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
