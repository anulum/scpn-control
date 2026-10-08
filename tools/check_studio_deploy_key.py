# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Studio deploy public-key guard.
"""Inspect local Studio public-key structure, Git filenames and deploy markers.

Defaults follow the resolved script and distributed CI policy. The public key
must contain the declared Ed25519 SSH wire structure and fixed comment; this
does not pin its fingerprint, authenticate a server or prove possession of a
private key. Filename exclusions inspect Git index names, never key contents.
Workflow markers are literal substrings, not executable YAML analysis. Inputs
are read sequentially; this guard performs no deployment or network operation.
"""

from __future__ import annotations

import base64
import subprocess  # noqa: S404
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.ci_workflow_inventory import workflow_path_for_job

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_KEY = ROOT / "studio-web" / "deploy" / "scpn-control-studio-ci-deploy.pub"
WORKFLOW = workflow_path_for_job("studio-web")
EXPECTED_COMMENT = "scpn-control-studio-ci-deploy-2026-07-08"
EXPECTED_DEPLOY_MARKERS = (
    "Configure Studio deploy SSH",
    "Deploy Studio remote",
    "if: github.event_name == 'push' && github.ref == 'refs/heads/main'",
    "${{ secrets.SCPN_CONTROL_STUDIO_DEPLOY_KEY }}",
    "${{ secrets.SCPN_CONTROL_STUDIO_KNOWN_HOSTS }}",
    "test -f dist/remoteEntry.js",
    "test -f dist/manifest.json",
    "test -f dist/studio-feed.json",
    "rsync -az --delete",
    "dist/ deploy@www.anulum.org:",
    "StrictHostKeyChecking=yes",
)


def parse_public_key(line: str) -> tuple[str, bytes, str]:
    """Parse one Ed25519 public-key line and validate its SSH wire structure.

    Parameters
    ----------
    line : str
        Three whitespace-separated fields: algorithm, ASCII base64 body and
        the fixed deployment comment. Other authorized_keys syntax is unsupported.

    Returns
    -------
    tuple[str, bytes, str]
        Key type, complete decoded SSH blob, and key comment. The blob must
        contain exactly the algorithm string and a 32-byte key, without trailing
        data (RFC 8709 section 4); no signature or curve-point check is performed.

    Raises
    ------
    ValueError
        Field count/type/comment, base64 or binary SSH structure is invalid.
    """
    parts = line.strip().split()
    if len(parts) != 3:
        msg = "public key must contain type, base64 body, and comment"
        raise ValueError(msg)
    key_type, encoded_body, comment = parts
    if key_type != "ssh-ed25519":
        msg = "public key must be ssh-ed25519"
        raise ValueError(msg)
    if comment != EXPECTED_COMMENT:
        msg = f"public key comment must be {EXPECTED_COMMENT!r}"
        raise ValueError(msg)
    try:
        decoded = base64.b64decode(encoded_body.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        msg = "public key body must be valid base64"
        raise ValueError(msg) from exc
    prefix = b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20"
    if len(decoded) != 51 or not decoded.startswith(prefix):
        raise ValueError("public key body must encode one 32-byte ssh-ed25519 key")
    return key_type, decoded, comment


def tracked_files(root: Path = ROOT) -> list[str]:
    """Return tracked repository paths from git.

    Parameters
    ----------
    root : pathlib.Path
        Working directory for Git. Explicit relative paths use caller cwd;
        a subdirectory can restrict enumeration. Git's environment selects its index.

    Returns
    -------
    list[str]
        Sorted NUL-framed index names decoded as UTF-8 with surrogate escapes,
        retaining whitespace and non-ASCII filenames. No content is opened.

    Raises
    ------
    OSError, subprocess.CalledProcessError
        Git cannot be launched or the selected working directory/index is invalid.
    """
    result = subprocess.run(  # noqa: S603
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return sorted(path.decode("utf-8", errors="surrogateescape") for path in result.stdout.split(b"\0") if path)


def validate_tracked_files(paths: list[str]) -> None:
    """Refuse declared key-like basenames without inspecting file contents.

    Parameters
    ----------
    paths : list of str
        Names to inspect using native Path basenames, lowercased before matching
        declared names or .pem/.key suffixes. This is not a general secret scanner;
        other extensions, backup suffixes and trailing whitespace are not classified.

    Raises
    ------
    ValueError
        A forbidden name is present; its supplied name appears in the API error.
    """
    forbidden_names = {
        "id_ed25519",
        "id_rsa",
        "id_ecdsa",
        "scpn-control-studio-ci-deploy_ed25519",
    }
    for path in paths:
        name = Path(path).name.lower()
        if name in forbidden_names or name.endswith(".pem") or name.endswith(".key"):
            msg = f"private key-like tracked path is forbidden: {path}"
            raise ValueError(msg)


def validate_public_key(path: Path = PUBLIC_KEY) -> None:
    """Validate the tracked Studio deploy public key file.

    Parameters
    ----------
    path : pathlib.Path
        UTF-8 carrier, relative to caller cwd when supplied. The script-relative
        default does not change with cwd. This API does not verify Git membership.

    Raises
    ------
    OSError, UnicodeError, ValueError
        The file cannot be read/decoded or its public-key declaration is invalid.
    """
    parse_public_key(path.read_text(encoding="utf-8"))


def validate_deploy_workflow(path: Path = WORKFLOW) -> None:
    """Require literal Studio deployment markers in the selected local workflow.

    Parameters
    ----------
    path : pathlib.Path
        UTF-8 workflow carrier. Comments and non-executable text can satisfy
        the marker check; no complete YAML, build, permission or remote audit occurs.

    Raises
    ------
    OSError, UnicodeError, ValueError
        Reading/decoding fails or a required marker is missing.
    """
    text = path.read_text(encoding="utf-8")
    missing = [marker for marker in EXPECTED_DEPLOY_MARKERS if marker not in text]
    if missing:
        msg = f"Studio deploy workflow missing marker: {missing[0]}"
        raise ValueError(msg)


def main() -> int:
    """Print one fixed line and return zero/pass or one/inspection failure.

    The no-argument API uses the script-relative key and declared workflow,
    then the current Git index. Caught read/decode/Git/validation errors produce
    a fixed stdout refusal instead of exception text. No files are changed and
    no credentials, signing operations, SSH connection or deployment are used.
    A pass establishes only the documented local declaration checks.

    Returns
    -------
    int
        Zero when all local inspections pass, otherwise one for caught failures.
    """
    try:
        validate_public_key()
        validate_tracked_files(tracked_files())
        validate_deploy_workflow()
    except (OSError, subprocess.CalledProcessError, ValueError):
        print("FAIL: Studio deploy inputs could not be inspected")
        return 1
    print("PASS: Studio deploy key and CI deploy workflow are valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
