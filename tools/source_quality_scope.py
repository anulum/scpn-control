# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Git-bound selection of changed sources and supporting carriers.
"""Enroll changed files independently of directory-specific linter defaults.

Selection is not a quality verdict. Each source still requires its native
checks, tests and applicable coverage; supporting carriers remain visible.
The caller supplies the trusted original commit. No history fallback, baseline
update, source execution or repository mutation is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.check_source_headers import classify, load_policy

ROOT = Path(__file__).resolve().parents[1]
_PROFILES = {
    ".py": "python",
    ".rs": "rust",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".mjs": "javascript",
    ".lean": "lean",
    ".sh": "shell",
    ".cpp": "cpp",
    ".h": "c-header",
    ".hpp": "cpp-header",
    ".html": "html",
    ".ipynb": "notebook",
}


@dataclass(frozen=True)
class SelectedFile:
    """Bind one canonical Git path to its profile and current complete bytes.

    Attributes
    ----------
    path : str
        Repository-relative POSIX spelling, retained without line splitting.
    profile : str
        Native source family or ``carrier``. Neither value certifies quality.
    sha256 : str
        SHA-256 of the complete current file, without newline normalisation.
    """

    path: str
    profile: str
    sha256: str


def _git(repo: Path, args: Sequence[str], timeout_s: float) -> bytes:
    """Run real Git with bounded time and refuse every unavailable result."""
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-C", str(repo), *args],
            capture_output=True,
            check=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Source selection requires successful Git {' '.join(args[:2])}.") from exc
    return result.stdout


def _paths(raw: bytes) -> set[str]:
    """Decode complete NUL-delimited records without losing unusual filenames."""
    if not raw:
        return set()
    if not raw.endswith(b"\0"):
        raise RuntimeError("Git source inventory is not NUL terminated.")
    try:
        return set(raw[:-1].decode("utf-8").split("\0"))
    except UnicodeError as exc:
        raise RuntimeError("Git source paths must be valid UTF-8.") from exc


def select_changed_files(repo: Path, base_revision: str, *, timeout_s: float = 30.0) -> tuple[SelectedFile, ...]:
    """Select committed, staged, unstaged and nonignored untracked changes.

    Parameters
    ----------
    repo : pathlib.Path
        Existing Git working-tree root. Subdirectories and bare repositories
        refuse. Selected symlinks, including symlinked parents, refuse.
    base_revision : str
        Caller-authenticated full lowercase 40-digit commit ID. It must exist
        locally and be an ancestor of HEAD. Missing shallow history refuses;
        this function never guesses an earlier baseline.
    timeout_s : float
        Finite positive timeout for each Git invocation, in seconds.

    Returns
    -------
    tuple of SelectedFile
        Sorted unique complete-file selections, including supporting carriers.
        Deleted paths are excluded because no current source remains. Renames
        enroll their destination. A clean tree at the baseline returns empty.

    Raises
    ------
    RuntimeError
        Git, history, conflicts, paths, readable regular files or format
        classification are unavailable. Unsupported new formats refuse.

    Notes
    -----
    Existing header exemptions classify formats only; they never waive quality
    gates. Their policy bytes must match the original revision. This selector
    does not prove reviewer authenticity, native tool
    execution, historical-debt acceptance or concurrent filesystem stability.
    Consumers must recheck hashes before using a selection as evidence.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", base_revision):
        raise RuntimeError("Source selection requires a full original commit ID.")
    if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise RuntimeError("The Git timeout must be finite and positive.")
    repo = repo.resolve()
    try:
        top = _git(repo, ["rev-parse", "--show-toplevel"], timeout_s).decode("utf-8").rstrip("\n")
        if Path(top).resolve() != repo:
            raise RuntimeError("Source selection must run at the Git working-tree root.")
        if _git(repo, ["cat-file", "-t", base_revision], timeout_s) != b"commit\n":
            raise RuntimeError("The original source revision must be a commit object.")
        _git(repo, ["merge-base", "--is-ancestor", base_revision, "HEAD"], timeout_s)
        if _git(repo, ["ls-files", "--unmerged", "-z"], timeout_s):
            raise RuntimeError("Unmerged source paths require conflict resolution before selection.")
        changed = _paths(
            _git(
                repo,
                [
                    "diff",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--name-only",
                    "--diff-filter=ACMRT",
                    "-z",
                    base_revision,
                    "--",
                ],
                timeout_s,
            )
        )
        changed |= _paths(_git(repo, ["ls-files", "--others", "--exclude-standard", "-z"], timeout_s))
        policy_path = repo / "tools/source_header_policy.toml"
        original_policy = _git(repo, ["show", f"{base_revision}:tools/source_header_policy.toml"], timeout_s)
        if policy_path.read_bytes() != original_policy:
            raise RuntimeError("Format policy changes require an explicitly reviewed source-profile update.")
        policy = load_policy(policy_path)
        selected = []
        for name in sorted(changed):
            relative = PurePosixPath(name)
            if relative.is_absolute() or relative.as_posix() != name or ".." in relative.parts or "\\" in name:
                raise RuntimeError("Source selection requires canonical relative Git paths.")
            path = repo / name
            if any(parent.is_symlink() for parent in (path, *path.parents) if parent != repo):
                raise RuntimeError(f"Selected source has a symlinked path: {name!r}.")
            if not path.is_file():
                raise RuntimeError(f"Selected source is not a regular file: {name!r}.")
            profile = "shell" if path.name == "pre-push" else _PROFILES.get(path.suffix.casefold())
            if profile is None:
                if classify(Path(name), policy)[0] == "unclassified":
                    raise RuntimeError(f"Selected file requires an explicit format/profile: {name!r}.")
                profile = "carrier"
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            selected.append(SelectedFile(name, profile, digest))
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError("Cannot read the complete source selection or its format policy.") from exc
    return tuple(selected)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the complete selection as JSON, returning two on refused measurement.

    Parameters
    ----------
    argv : sequence of str or None
        CLI arguments; None uses process arguments. ``--base`` is mandatory.

    Returns
    -------
    int
        Zero for completed selection, two for unavailable input. Native quality
        acceptance is deliberately not reported by this selection command.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--base", required=True)
    args = parser.parse_args(argv)
    try:
        selected = select_changed_files(args.repo, args.base)
    except RuntimeError as exc:
        print(f"Source selection refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"base_revision": args.base, "paths": [asdict(item) for item in selected]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
