# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Complete-file native checks for automatically selected Python.
"""Check changed Python in every root using the original repository profile.

This supplements ordinary repository gates. It does not qualify other source
families, runtime tests, coverage, benchmarks or independent review.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import os
import re
import subprocess
import sys
import tokenize
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.docstring_debt_ledger import read_owner_ledger
from tools.run_docstring_gate import all_definition_findings
from tools.source_quality_scope import _git, select_changed_files

ROOT = Path(__file__).resolve().parents[1]
_WAIVER = re.compile(r"#\s*(?:type:\s*ignore|mypy:|fmt:\s*(?:off|skip))", re.IGNORECASE)


def changed_python_errors(repo: Path, base_revision: str, *, timeout_s: float = 120.0) -> list[str]:
    """Run complete-file documentation, Ruff, format and strict mypy checks.

    Parameters
    ----------
    repo : pathlib.Path
        Git working-tree root containing the original ``pyproject.toml`` and
        format policy. Directory-specific linter defaults do not select files.
    base_revision : str
        Trusted original full commit ID, validated by the Git selector.
    timeout_s : float
        Finite positive limit per native command. Missing tools and timeouts
        fail closed; native nonzero statuses remain failed checks.

    Returns
    -------
    list of str
        Diagnostics from all selected complete Python files. Empty means this
        Python-only static profile passed, or no changed Python was selected.
        Other languages and supporting carriers remain outside this verdict.

    Raises
    ------
    RuntimeError
        Selection, original configuration, source bytes, syntax or native
        process execution are unavailable. No baseline is updated.

    Notes
    -----
    Ruff uses the explicit original root configuration and ignores all noqa
    suppressions. Type-ignore, mypy and format-off/skip comment directives in
    selected files refuse. Existing declared third-party typing conventions
    remain part of the original profile; this does not claim waiver-free
    historical typing or certify a candidate's tests or coverage.
    """
    if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise RuntimeError("The native-check timeout must be finite and positive.")
    repo = repo.resolve()
    scope = select_changed_files(repo, base_revision)
    selected = [item for item in scope if item.profile == "python"]
    if not selected:
        return []
    errors: list[str] = []
    try:
        config = repo / "pyproject.toml"
        if config.read_bytes() != _git(repo, ["show", f"{base_revision}:pyproject.toml"], timeout_s):
            raise RuntimeError("Changed Python requires an explicitly reviewed original native configuration.")
        paths = []
        for item in selected:
            path = repo / item.path
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != item.sha256:
                raise RuntimeError(f"Selected Python changed during measurement: {item.path!r}.")
            for token in tokenize.generate_tokens(io.StringIO(raw.decode("utf-8")).readline):
                if token.type == tokenize.COMMENT and _WAIVER.match(token.string):
                    errors.append(f"{item.path}:{token.start[0]}: static-check waiver requires explicit review")
            paths.append(path)
        errors.extend(all_definition_findings(paths))
        names = [str(path) for path in paths]
        commands = (
            ("Ruff", ["ruff", "check", "--config", str(config), "--ignore-noqa", "--no-respect-gitignore", *names]),
            ("format", ["ruff", "format", "--check", "--config", str(config), "--no-respect-gitignore", *names]),
            ("mypy", ["mypy", "--strict", "--explicit-package-bases", "--config-file", str(config), *names]),
        )
        native_env = os.environ.copy()
        native_env["MYPYPATH"] = os.pathsep.join(str(repo / directory) for directory in ("src", "tests", "."))
        interpreter = [sys.executable, "-B"]
        if sys.flags.isolated:
            interpreter.append("-I")
        if sys.flags.no_site:
            interpreter.append("-S")
        for label, args in commands:
            result = subprocess.run(
                [*interpreter, "-m", *args],
                cwd=repo,
                env=native_env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            if result.returncode:
                errors.append(f"{label} failed ({result.returncode}):\n{result.stdout}{result.stderr}")
        for item in selected:
            if hashlib.sha256((repo / item.path).read_bytes()).hexdigest() != item.sha256:
                raise RuntimeError(f"Selected Python changed during native checks: {item.path!r}.")
        if select_changed_files(repo, base_revision) != scope:
            raise RuntimeError("Changed-source selection shifted during native checks.")
    except (OSError, UnicodeError, SyntaxError, tokenize.TokenError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Changed-Python checks require readable sources and completed native tools.") from exc
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the Python-only profile with a required original revision.

    Parameters
    ----------
    argv : sequence of str or None
        CLI arguments. Without ``--base``, use the mandatory documentation
        ledger's original commit; unavailable ledger/history refuses.

    Returns
    -------
    int
        Zero for this static profile, one for quality findings, two for refused
        measurement. No whole-repository quality verdict is emitted.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--base")
    args = parser.parse_args(argv)
    try:
        base = args.base or read_owner_ledger(args.repo / "tools/docstring_debt_ceiling.json").origin_revision
        errors = changed_python_errors(args.repo, base)
    except RuntimeError as exc:
        print(f"Changed-Python checks refused: {exc}", file=sys.stderr)
        return 2
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Changed-Python static profile passed; runtime and other native profiles remain separate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
