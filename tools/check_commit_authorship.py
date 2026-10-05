# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Commit Authorship Guard
"""Check the required forward-only authorship text through the hook/CI API.

Each supplied message must contain exactly one literal canonical authorship
line and no competing authorship or superseded Git co-author line. Indentation
and case cannot hide alternate attribution; canonical text remains exact.
The guard reads UTF-8 files, reports all selected messages and never rewrites
them or Git history. These are text checks; Git author/committer identity and
the separate Seat trailer discipline still require their own verification.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REQUIRED_AUTHORSHIP_LINE = "Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)"
LEGACY_COAUTHOR_RE = re.compile(r"^Co-Authored-By:.*$", re.IGNORECASE)
AUTHORSHIP_RE = re.compile(r"^Authored\s+by(?:\s.*)?$", re.IGNORECASE)


def validate_commit_message(message: str) -> list[str]:
    """Collect authorship violations in deterministic policy order.

    Parameters
    ----------
    message
        Decoded commit text. Splitlines accepts LF/CRLF and a missing final
        newline. The required line must match exactly, including its case and
        absence of surrounding whitespace. Other lines remain unconstrained.

    Returns
    -------
    list[str]
        Empty for a passing message; otherwise missing/duplicate canonical
        attribution, any co-author trailer and the first competing authorship,
        in that order. Alternate attribution is detected after stripping outer
        whitespace and ignoring case. No Git identity or Seat check is implied.

    Examples
    --------
    Validate the exact required line through the owning public API:

    >>> validate_commit_message(REQUIRED_AUTHORSHIP_LINE)
    []
    """
    violations: list[str] = []
    lines = message.splitlines()
    required_count = lines.count(REQUIRED_AUTHORSHIP_LINE)
    if required_count == 0:
        violations.append(f"missing required authorship line: {REQUIRED_AUTHORSHIP_LINE}")
    if required_count > 1:
        violations.append("duplicate required authorship line")

    legacy = [line for line in lines if LEGACY_COAUTHOR_RE.fullmatch(line.strip())]
    if legacy:
        violations.append("legacy Git co-author trailer is forbidden for new commits")

    extra_authorship = [
        line.strip() for line in lines if AUTHORSHIP_RE.fullmatch(line.strip()) and line != REQUIRED_AUTHORSHIP_LINE
    ]
    if extra_authorship:
        violations.append(f"unexpected authorship line: {extra_authorship[0]}")

    return violations


def validate_commit_message_file(path: Path) -> list[str]:
    """Read one UTF-8 message and apply the public authorship text contract.

    Relative paths resolve against the caller's working directory. The file
    is not modified. Filesystem errors and UnicodeDecodeError propagate;
    the CLI translates them into refusal diagnostics while processing the
    remaining selected files.
    """
    return validate_commit_message(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    """Check all message files supplied by the commit hook or CI command.

    Parameters
    ----------
    argv
        Arguments excluding the executable. None reads process arguments.
        One or more UTF-8 message paths are required and resolve against cwd.

    Returns
    -------
    int
        Zero when every selected message passes; one for attribution or read
        failures. Each diagnostic includes the input path on stderr. Continue
        after a refused file, leave every input unchanged and print nothing on
        success. Argument errors exit two through argparse.
    """
    parser = argparse.ArgumentParser(description="Validate SCPN-CONTROL commit authorship lines.")
    parser.add_argument("message_files", nargs="+", type=Path)
    args = parser.parse_args(argv)

    failed = False
    for message_file in args.message_files:
        try:
            violations = validate_commit_message_file(message_file)
        except (OSError, UnicodeError) as exc:
            violations = [f"could not read UTF-8 commit message: {exc}"]
        for violation in violations:
            print(f"{message_file}: {violation}", file=sys.stderr)
        failed = failed or bool(violations)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
