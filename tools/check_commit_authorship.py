# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Commit Authorship Guard

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REQUIRED_AUTHORSHIP_LINE = "Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)"
LEGACY_COAUTHOR_RE = re.compile(r"^Co-Authored-By:\s+.*protoscience@anulum\.li.*$", re.MULTILINE)
AUTHORSHIP_RE = re.compile(r"^Authored by .+$", re.MULTILINE)


def validate_commit_message(message: str) -> list[str]:
    """Return commit-message authorship policy violations."""

    violations: list[str] = []
    required_count = sum(line.strip() == REQUIRED_AUTHORSHIP_LINE for line in message.splitlines())
    if required_count == 0:
        violations.append(f"missing required authorship line: {REQUIRED_AUTHORSHIP_LINE}")
    if required_count > 1:
        violations.append("duplicate required authorship line")

    legacy = [match.group(0).strip() for match in LEGACY_COAUTHOR_RE.finditer(message)]
    if legacy:
        violations.append("legacy Git co-author trailer is forbidden for new commits")

    extra_authorship = [
        line.strip()
        for line in AUTHORSHIP_RE.findall(message)
        if line.strip() != REQUIRED_AUTHORSHIP_LINE
    ]
    if extra_authorship:
        violations.append(f"unexpected authorship line: {extra_authorship[0]}")

    return violations


def validate_commit_message_file(path: Path) -> list[str]:
    """Return policy violations for one commit-message file."""

    return validate_commit_message(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate SCPN-CONTROL commit authorship lines.")
    parser.add_argument("message_files", nargs="+", type=Path)
    args = parser.parse_args(argv)

    failed = False
    for message_file in args.message_files:
        violations = validate_commit_message_file(message_file)
        for violation in violations:
            print(f"{message_file}: {violation}", file=sys.stderr)
        failed = failed or bool(violations)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
