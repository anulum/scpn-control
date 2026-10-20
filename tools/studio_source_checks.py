# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Native enrollment checks for changed Studio TypeScript and JavaScript.
"""Require compiler enrollment and real native checks for changed Studio sources.

This static profile does not qualify browser behaviour, tests, coverage, build
consumption or other native languages. JavaScript/TypeScript outside Studio
requires an explicitly qualified profile rather than silently passing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.docstring_debt_ledger import read_owner_ledger
from tools.source_quality_scope import select_changed_files

ROOT = Path(__file__).resolve().parents[1]
_TOOL_VERSIONS = {"node": "v22.23.2", "pnpm": "11.25.0"}
_CONFIGURATION = {
    "eslint.config.mjs": "f993ca10ef869a3e3dd7369a9f4c4d57bef49589ae6580290d64e4e1ee6dd954",
    "tsconfig.json": "974ccb2c633c405c0de9ea42d26e39f7de1d172b8f0edc1eeb2f4936bd556781",
    "vitest.config.ts": "125caad472ee561776f7b333318f6fb156c2b749500e944a971c7cd71357d277",
    "package.json": "da19393e25539d17e7dd4fd15f268f78e5d2664fcd526e6c7e71eded218be43f",
    "pnpm-lock.yaml": "2ea29df26110469875ead45dedcc64290c26b7c9ea31871f532b83785f538559",
    ".prettierrc.json": "ebe733191f06a29fdd34a4683327d9a36c008ed8da564d8e7def1c700d7463ff",
}
_NO_WAIVERS = json.dumps(
    [
        "error",
        {
            "terms": ["@ts-ignore", "@ts-expect-error", "@ts-nocheck", "prettier-ignore"],
            "location": "anywhere",
        },
    ]
)


def _native(cwd: Path, argv: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
    """Execute the actual installed tool without shell expansion or retries."""
    try:
        return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False, timeout=timeout_s)
    except (OSError, UnicodeError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"Studio checks require completed native {argv[0]} execution.") from error


def _configuration(studio: Path) -> None:
    """Bind qualified configurations and every declared direct package version."""
    for name, digest in _CONFIGURATION.items():
        if hashlib.sha256((studio / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Studio {name} requires explicit native-profile requalification.")
    package = json.loads((studio / "package.json").read_text(encoding="utf-8"))
    dependencies = package["dependencies"] | package["devDependencies"]
    for name, version in dependencies.items():
        installed = json.loads((studio / "node_modules" / name / "package.json").read_text(encoding="utf-8"))
        if not isinstance(installed, dict) or installed.get("name") != name or installed.get("version") != version:
            raise RuntimeError(f"Studio package identity differs from its qualified exact pin: {name}.")


def changed_studio_errors(repo: Path, base_revision: str, *, timeout_s: float = 120.0) -> list[str]:
    """Enroll changed complete Studio files in compiler, ESLint and Prettier checks.

    Parameters
    ----------
    repo : pathlib.Path
        Real Git working-tree root with a frozen-installed ``studio-web``.
    base_revision : str
        Trusted original full commit ID, validated by the Git selector.
    timeout_s : float
        Finite positive limit per native command; no retries or installation.

    Returns
    -------
    list of str
        Native findings. Empty means the selected Studio static profile passed,
        or no changed JavaScript/TypeScript was selected. Other source families
        and supporting carriers are outside this verdict.

    Raises
    ------
    RuntimeError
        Sources, original history, qualified configuration/package identities,
        compiler enrollment or completed native processes are unavailable.
        Missing tools, source aliases, new unsupported roots and changed final
        inputs refuse instead of yielding a partial pass.

    Notes
    -----
    Node 22.23.2 and pnpm 11.25.0 match the existing CI pins. Configuration
    fingerprints bind the locally qualified profile, not an independent review
    or transitive installed-byte integrity. Updating them requires fresh native
    evidence. Root compiler options, actual program membership and explicit
    ESLint/Prettier file arguments prevent directory defaults hiding new code.
    Comment-based type/format suppressions and ESLint inline directives cannot
    silence the selected rules. Runtime tests and coverage remain separate.
    """
    if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise RuntimeError("The Studio-check timeout must be finite and positive.")
    repo = repo.resolve()
    scope = select_changed_files(repo, base_revision)
    selected = [item for item in scope if item.profile in {"typescript", "javascript"}]
    if not selected:
        return []
    if any(not item.path.startswith("studio-web/") for item in selected):
        raise RuntimeError("Changed JavaScript/TypeScript outside Studio requires its own qualified native profile.")
    studio = repo / "studio-web"
    try:
        _configuration(studio)
        for executable, expected in _TOOL_VERSIONS.items():
            argv = [executable, "--version"]
            result = _native(studio, argv, timeout_s)
            if result.returncode or result.stdout.strip() != expected:
                raise RuntimeError(f"Studio requires qualified {argv[0]} {expected}.")
        names = [str(repo / item.path) for item in selected]
        compiler = [
            "node",
            "node_modules/typescript/bin/tsc",
            "--project",
            "tsconfig.json",
            "--noEmit",
            "--strict",
            "--allowJs",
            "--checkJs",
        ]
        program = _native(studio, [*compiler, "--listFilesOnly"], timeout_s)
        if program.returncode:
            raise RuntimeError(f"Studio compiler enrollment failed:\n{program.stdout}{program.stderr}")
        members = {str(Path(line).resolve()) for line in program.stdout.splitlines() if line.strip()}
        missing = set(names) - members
        if missing:
            raise RuntimeError(f"Changed Studio sources absent from the actual compiler program: {sorted(missing)!r}.")
        commands = (
            ("TypeScript", compiler),
            (
                "ESLint",
                [
                    "node",
                    "node_modules/eslint/bin/eslint.js",
                    "--config",
                    "eslint.config.mjs",
                    "--no-config-lookup",
                    "--no-ignore",
                    "--no-inline-config",
                    "--max-warnings",
                    "0",
                    "--rule",
                    "no-warning-comments: " + _NO_WAIVERS,
                    *names,
                ],
            ),
            (
                "Prettier",
                [
                    "node",
                    "node_modules/prettier/bin/prettier.cjs",
                    "--check",
                    "--config",
                    ".prettierrc.json",
                    "--ignore-path",
                    os.devnull,
                    *names,
                ],
            ),
        )
        errors = []
        for label, argv in commands:
            result = _native(studio, argv, timeout_s)
            if result.returncode:
                errors.append(f"{label} failed ({result.returncode}):\n{result.stdout}{result.stderr}")
        _configuration(studio)
        if select_changed_files(repo, base_revision) != scope:
            raise RuntimeError("Studio source selection changed during native checks.")
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        raise RuntimeError(
            "Studio checks require readable qualified configurations, sources and package metadata."
        ) from error
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Studio-only static profile with original-history enforcement.

    Parameters
    ----------
    argv : sequence of str or None
        CLI arguments; omitted ``--base`` uses the required owner ledger origin.

    Returns
    -------
    int
        Zero for this static profile, one for native findings, two for refused
        measurement. No other-language or whole-repository verdict is emitted.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--base")
    args = parser.parse_args(argv)
    try:
        base = args.base or read_owner_ledger(args.repo / "tools/docstring_debt_ceiling.json").origin_revision
        errors = changed_studio_errors(args.repo, base)
    except RuntimeError as error:
        print(f"Changed-Studio checks refused: {error}", file=sys.stderr)
        return 2
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Changed-Studio static profile passed; runtime and other native profiles remain separate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
