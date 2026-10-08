# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Maintained lifecycle corpus and public CLI fixtures.
"""Copy maintained evidence for real inventory filesystem tests."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tools.validation_report_freshness import DEFAULT_LIFECYCLE_REGISTRY, ROOT

AUDIT_AS_OF = datetime(2026, 9, 5, 13, 0, 1, tzinfo=UTC)


@dataclass(frozen=True)
class CopiedCorpus:
    """Owned copy of maintained report, refresh and registry bytes.

    Parameters
    ----------
    root : pathlib.Path
        Temporary repository root holding the copied validation directories.
    registry : pathlib.Path
        Byte-identical lifecycle registry within that root.
    """

    root: Path
    registry: Path

    @property
    def reports(self) -> Path:
        """Return the copied report directory.

        Returns
        -------
        pathlib.Path
            Directory whose contents are checked by the public loader.
        """
        return self.root / "validation/reports"

    def arguments(self) -> list[str]:
        """Return deterministic public CLI input arguments.

        Returns
        -------
        list of str
            Supported report-root, registry and timestamp options.
        """
        return [
            "--reports-root",
            str(self.reports),
            "--registry",
            str(self.registry),
            "--as-of",
            AUDIT_AS_OF.isoformat(),
        ]

    def snapshot(self) -> dict[str, str]:
        """Hash every copied input for before/after custody checks.

        Returns
        -------
        dict of str to str
            Relative paths and SHA-256 digests, including the registry.
        """
        return {
            path.relative_to(self.root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((self.root / "validation").rglob("*.json"))
        }


def copy_corpus(root: Path) -> CopiedCorpus:
    """Copy actual maintained evidence without changing declarations.

    Parameters
    ----------
    root : pathlib.Path
        New owned directory for the complete copied validation corpus.

    Returns
    -------
    CopiedCorpus
        Full report/refresh trees and their unchanged registry.
    """
    for directory in ("reports", "report_refreshes"):
        shutil.copytree(ROOT / "validation" / directory, root / "validation" / directory)
    registry = root / "validation/report_lifecycle_registry.json"
    shutil.copy2(DEFAULT_LIFECYCLE_REGISTRY, registry)
    return CopiedCorpus(root, registry)


def run_cli(corpus: CopiedCorpus, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Run the actual script from an unrelated working directory.

    Parameters
    ----------
    corpus : CopiedCorpus
        Unmodified owned input copy selected through public options.
    arguments : list of str
        Additional supported CLI options.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Actual exit status and captured UTF-8 stdout/stderr.
    """
    return subprocess.run(
        [sys.executable, str(ROOT / "tools/validation_report_freshness.py"), *corpus.arguments(), *arguments],
        cwd=corpus.root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
