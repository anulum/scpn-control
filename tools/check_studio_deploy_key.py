# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Studio deploy public-key guard.
"""Validate the tracked Studio deploy public key and private-key exclusion."""

from __future__ import annotations

import base64
import subprocess  # noqa: S404
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_KEY = ROOT / "studio-web" / "deploy" / "scpn-control-studio-ci-deploy.pub"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
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
    """Parse and validate a single OpenSSH public-key line.

    Parameters
    ----------
    line
        Public-key line from ``studio-web/deploy``.

    Returns
    -------
    tuple[str, bytes, str]
        Key type, decoded key body, and key comment.
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
    return key_type, decoded, comment


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


def validate_tracked_files(paths: list[str]) -> None:
    """Fail if tracked files include private deploy-key material.

    Parameters
    ----------
    paths
        Tracked repository paths to inspect.
    """
    forbidden_names = {
        "id_ed25519",
        "id_rsa",
        "scpn-control-studio-ci-deploy_ed25519",
    }
    for path in paths:
        name = Path(path).name
        if name in forbidden_names or name.endswith(".pem") or name.endswith(".key"):
            msg = f"private key-like tracked path is forbidden: {path}"
            raise ValueError(msg)


def validate_public_key(path: Path = PUBLIC_KEY) -> None:
    """Validate the tracked Studio deploy public key file.

    Parameters
    ----------
    path
        Public-key path to validate.
    """
    parse_public_key(path.read_text(encoding="utf-8"))


def validate_deploy_workflow(path: Path = WORKFLOW) -> None:
    """Validate the Studio Web CI deploy workflow contract.

    Parameters
    ----------
    path
        GitHub Actions workflow that builds and deploys the Studio Web remote.
    """
    text = path.read_text(encoding="utf-8")
    missing = [marker for marker in EXPECTED_DEPLOY_MARKERS if marker not in text]
    if missing:
        msg = f"Studio deploy workflow missing marker: {missing[0]}"
        raise ValueError(msg)


def main() -> int:
    """Run the Studio deploy-key guard."""
    try:
        validate_public_key()
        validate_tracked_files(tracked_files())
        validate_deploy_workflow()
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print("PASS: Studio deploy key and CI deploy workflow are valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
