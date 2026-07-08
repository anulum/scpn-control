# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Studio offline sealing guard.
"""Validate that Studio evidence sealing remains keeper-offline.

The CONTROL repository may emit float-free claim JSON for the Studio Hub, but it
must not wire Hub/Studio publication-signing keys into CI, production deploy
surfaces, or tracked artifacts. This guard scans tracked policy-bearing files for
signing/sealing key references and private-key blocks while allowing unrelated
secrets such as coverage upload tokens and future deploy-only SSH credentials.
"""

from __future__ import annotations

import re
import subprocess  # noqa: S404
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SECRET_REFERENCE_RE = re.compile(r"\bsecrets\.([A-Za-z_][A-Za-z0-9_]*)\b")
ENV_ASSIGNMENT_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", re.MULTILINE)
PRIVATE_KEY_BLOCK_RE = re.compile(
    r"-----BEGIN (?:OPENSSH PRIVATE|RSA PRIVATE|EC PRIVATE|DSA PRIVATE|PRIVATE|ENCRYPTED PRIVATE) KEY-----"
)

POLICY_FILE_PREFIXES: tuple[str, ...] = (
    ".github/workflows/",
    "docs/",
    "src/scpn_control/studio/",
    "studio-web/",
    "tools/",
)
POLICY_FILE_NAMES: frozenset[str] = frozenset({"CHANGELOG.md", "Makefile", "README.md", "pyproject.toml"})
FORBIDDEN_PATH_SUFFIXES: tuple[str, ...] = (".key", ".pem", ".p8", ".pkcs8", ".jwk")


def tracked_files(root: Path = ROOT) -> list[str]:
    """Return tracked repository paths from git.

    Parameters
    ----------
    root
        Repository root for the ``git ls-files`` command.

    Returns
    -------
    list[str]
        Sorted tracked paths relative to ``root``.
    """
    result = subprocess.run(  # noqa: S603
        ["git", "ls-files"],
        cwd=root,
        check=True,
        capture_output=True,
        encoding="utf-8",
    )
    return sorted(path for path in result.stdout.splitlines() if path)


def is_policy_file(path: str) -> bool:
    """Return whether ``path`` is a tracked surface that can wire sealing keys.

    Parameters
    ----------
    path
        Repository-relative path.

    Returns
    -------
    bool
        ``True`` for workflow, Studio, documentation, or tool surfaces.
    """
    return path in POLICY_FILE_NAMES or any(path.startswith(prefix) for prefix in POLICY_FILE_PREFIXES)


def is_forbidden_studio_secret_name(name: str) -> bool:
    """Return whether a secret/env name suggests Studio signing custody.

    Deploy-only credentials are intentionally allowed here because the Studio Web
    deploy TODO uses SSH transport; publication signing and transparency-log
    sealing keys are the forbidden class for this guard.
    """
    upper = name.upper()
    if "DEPLOY" in upper:
        return False
    custody_scope = any(term in upper for term in ("STUDIO", "HUB", "PUBLICATION", "TRANSPARENCY", "SEAL"))
    custody_key = any(term in upper for term in ("SIGNING", "SEALING", "PRIVATE", "SEAL"))
    material = any(term in upper for term in ("KEY", "SECRET", "JWK"))
    return custody_scope and custody_key and material


def is_forbidden_key_path(path: str) -> bool:
    """Return whether a tracked path looks like an offline sealing private key.

    Parameters
    ----------
    path
        Repository-relative path.

    Returns
    -------
    bool
        ``True`` for signing/sealing key-like filenames unrelated to deploy.
    """
    normalized = path.lower()
    if "deploy" in normalized:
        return False
    if not normalized.endswith(FORBIDDEN_PATH_SUFFIXES):
        return False
    return any(term in normalized for term in ("signing", "sealing", "publication-seal", "transparency"))


def validate_secret_references(path: str, text: str) -> list[str]:
    """Find forbidden CI secret or environment names in a text file.

    Parameters
    ----------
    path
        Repository-relative path used in diagnostics.
    text
        File text to inspect.

    Returns
    -------
    list[str]
        Human-readable violations found in ``text``.
    """
    violations: list[str] = []
    for match in SECRET_REFERENCE_RE.finditer(text):
        name = match.group(1)
        if is_forbidden_studio_secret_name(name):
            violations.append(f"{path}: forbidden Studio sealing secret reference: secrets.{name}")
    if path.startswith(".github/workflows/"):
        for match in ENV_ASSIGNMENT_RE.finditer(text):
            name = match.group(1)
            if is_forbidden_studio_secret_name(name):
                violations.append(f"{path}: forbidden Studio sealing environment name: {name}")
    return violations


def validate_policy_files(paths: Iterable[str], root: Path = ROOT) -> list[str]:
    """Validate tracked policy-bearing files for offline-sealing violations.

    Parameters
    ----------
    paths
        Repository-relative tracked paths to inspect.
    root
        Repository root containing ``paths``.

    Returns
    -------
    list[str]
        Human-readable violations. An empty list means the guard passes.
    """
    violations: list[str] = []
    for path in paths:
        if is_forbidden_key_path(path):
            violations.append(f"{path}: tracked Studio sealing key-like path is forbidden")
        if not is_policy_file(path):
            continue
        try:
            text = (root / path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if PRIVATE_KEY_BLOCK_RE.search(text):
            violations.append(f"{path}: private-key block is forbidden on Studio sealing policy surfaces")
        violations.extend(validate_secret_references(path, text))
    return violations


def main() -> int:
    """Run the Studio offline-sealing guard."""
    try:
        violations = validate_policy_files(tracked_files())
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"FAIL: {exc}")
        return 1
    if violations:
        print("FAIL: Studio evidence sealing must remain keeper-offline")
        for violation in violations:
            print(f"  - {violation}")
        return 1
    print("PASS: Studio evidence sealing remains offline; no signing key material is wired")
    return 0


if __name__ == "__main__":
    sys.exit(main())
