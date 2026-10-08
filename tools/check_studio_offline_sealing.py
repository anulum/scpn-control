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

import os
import re
import subprocess  # noqa: S404
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SECRET_REFERENCE_RE = re.compile(r"\bsecrets(?:\s*\.\s*|\s*\[\s*['\"])([A-Za-z_][A-Za-z0-9_]*)")
SECRET_INDEX_RE = re.compile(r"\bsecrets\s*\[")
LITERAL_SECRET_INDEX_RE = re.compile(r"\[\s*(['\"])[A-Za-z_][A-Za-z0-9_]*\1\s*\]")
ENV_ASSIGNMENT_RE = re.compile(r"(?:^|[\s{,])['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?\s*:", re.MULTILINE)
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
TEXT_POLICY_SUFFIXES: frozenset[str] = frozenset(
    {
        ".cjs",
        ".css",
        ".html",
        ".ini",
        ".js",
        ".json",
        ".jsx",
        ".license",
        ".md",
        ".mjs",
        ".pub",
        ".py",
        ".rst",
        ".scss",
        ".sh",
        ".svg",
        ".tex",
        ".text",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
)
TEXT_POLICY_NAMES: frozenset[str] = frozenset({"CNAME", "Makefile", ".gitignore"})


def tracked_files(root: Path = ROOT) -> list[str]:
    """Return tracked repository paths from git.

    Parameters
    ----------
    root : pathlib.Path, optional
        Repository root for the ``git ls-files -z`` command.

    Returns
    -------
    list of str
        Sorted exact index paths relative to ``root``, decoded with the
        filesystem codec, including whitespace and non-ASCII names.

    Raises
    ------
    OSError, ValueError
        Git cannot be started or the supplied root cannot be inspected.
    subprocess.CalledProcessError
        Git refuses the selected index or repository.

    Notes
    -----
    Index and subsequent worktree reads are sequential observations, not an
    atomic snapshot or a history/untracked-file scan.
    """
    result = subprocess.run(  # noqa: S603
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return sorted(os.fsdecode(path) for path in result.stdout.split(b"\0") if path)


def is_policy_file(path: str) -> bool:
    """Return whether ``path`` is a tracked surface that can wire sealing keys.

    Parameters
    ----------
    path : str
        Repository-relative path.

    Returns
    -------
    bool
        ``True`` for workflow, Studio, documentation, or tool surfaces.
    """
    return path in POLICY_FILE_NAMES or any(path.startswith(prefix) for prefix in POLICY_FILE_PREFIXES)


def is_forbidden_studio_secret_name(name: str) -> bool:
    """Return whether a secret/env name suggests Studio signing custody.

    Parameters
    ----------
    name : str
        Secret or environment name classified case-insensitively.

    Returns
    -------
    bool
        True for scoped signing/sealing private-key-like names. A deploy label
        exempts transport-only names, including deploy-private names, but does
        not override explicit signing or sealing markers.

    Notes
    -----
    This classifies names; it does not inspect a remote secret's value or prove
    its operational role, key possession or offline custody.
    """
    upper = name.upper()
    if "DEPLOY" in upper and not any(term in upper for term in ("SIGNING", "SEAL")):
        return False
    custody_scope = any(term in upper for term in ("STUDIO", "HUB", "PUBLICATION", "TRANSPARENCY", "SEAL"))
    custody_key = any(term in upper for term in ("SIGNING", "SEALING", "PRIVATE", "SEAL"))
    material = any(term in upper for term in ("KEY", "SECRET", "JWK"))
    return custody_scope and custody_key and material


def is_forbidden_key_path(path: str) -> bool:
    """Return whether a tracked path looks like an offline sealing private key.

    Parameters
    ----------
    path : str
        Repository-relative path.

    Returns
    -------
    bool
        ``True`` for key-like suffixes with explicit signing/sealing path
        markers, including paths below deploy directories. Plain deploy-only
        paths without those markers remain allowed.
    """
    normalized = path.lower()
    if not normalized.endswith(FORBIDDEN_PATH_SUFFIXES):
        return False
    return any(term in normalized for term in ("signing", "sealing", "publication-seal", "transparency"))


def validate_secret_references(path: str, text: str) -> list[str]:
    """Find forbidden CI secret or environment names in a text file.

    Parameters
    ----------
    path : str
        Repository-relative path used in diagnostics.
    text : str
        Decoded text to inspect; a leading UTF-8 byte-order mark is accepted.
        Literal dot and quoted-index secret-name references are recognised.
        Workflow key-shaped tokens include quoted names and flow mappings.

    Returns
    -------
    list of str
        Human-readable violations found in ``text``.

    Notes
    -----
    Literal names are scanned, including a prefix in malformed expressions.
    Dynamic or malformed index expressions refuse because their secret name
    cannot be classified; full workflow semantics are not evaluated.
    """
    text = text.removeprefix("\ufeff")
    violations: list[str] = []
    for match in SECRET_REFERENCE_RE.finditer(text):
        name = match.group(1)
        if is_forbidden_studio_secret_name(name):
            violations.append(f"{path}: forbidden Studio sealing secret reference: secrets.{name}")
    for match in SECRET_INDEX_RE.finditer(text):
        if LITERAL_SECRET_INDEX_RE.match(text, match.end() - 1) is None:
            violations.append(f"{path}: Studio secret index must name a literal key")
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
    paths : iterable of str
        Repository-relative tracked paths to inspect.
    root : pathlib.Path, optional
        Repository root containing ``paths``.

    Returns
    -------
    list of str
        Authored contextual violations, including undecodable known text
        policy files. The CLI withholds these details and prints a count.

    Raises
    ------
    OSError, ValueError
        A selected policy file cannot be read or its path is invalid.

    Notes
    -----
    Already forbidden key-like paths are not opened. Undecodable known text
    formats, workflow paths and maintained text basenames refuse; other opaque
    binary formats retain their previous skip behaviour. No lossy decoding,
    full configuration parser or cryptographic custody attestation is provided.
    """
    violations: list[str] = []
    for path in paths:
        if is_forbidden_key_path(path):
            violations.append(f"{path}: tracked Studio sealing key-like path is forbidden")
            continue
        if not is_policy_file(path):
            continue
        try:
            text = (root / path).read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            if (
                path.startswith(".github/workflows/")
                or Path(path).suffix.lower() in TEXT_POLICY_SUFFIXES
                or Path(path).name in TEXT_POLICY_NAMES
            ):
                violations.append(f"{path}: textual Studio policy file must be valid UTF-8")
            continue
        if PRIVATE_KEY_BLOCK_RE.search(text):
            violations.append(f"{path}: private-key block is forbidden on Studio sealing policy surfaces")
        violations.extend(validate_secret_references(path, text))
    return violations


def main() -> int:
    """Run the indexed-file guard through its no-argument public CLI.

    Returns
    -------
    int
        Zero for the lexical policy checks or one for a policy/refusal or
        caught native input failure. Detailed findings remain withheld.

    Notes
    -----
    Native IO/Git/decode/path failures use fixed stdout without interpreter
    text, paths or secret-adjacent details. This does not sign, deploy, inspect
    remote secrets or independently certify keeper custody.
    """
    try:
        violations = validate_policy_files(tracked_files())
    except (OSError, ValueError, subprocess.CalledProcessError):
        print("FAIL: Studio sealing policy files could not be inspected")
        return 1
    if violations:
        print("FAIL: Studio evidence sealing must remain keeper-offline")
        print(
            f"FAIL: {len(violations)} policy violation(s) detected; secret-adjacent details are intentionally withheld"
        )
        return 1
    print("PASS: Studio offline-sealing lexical checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
