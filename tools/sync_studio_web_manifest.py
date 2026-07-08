# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — deployed Studio Web manifest sync guard.
"""Sync the generated Studio manifest to the deployed Studio Web public artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "docs" / "_generated" / "studio_manifest.json"
WEB_MANIFEST = ROOT / "studio-web" / "public" / "manifest.json"


def read_manifest(path: Path) -> tuple[str, dict[str, Any]]:
    """Return the manifest text and parsed JSON payload from ``path``.

    Parameters
    ----------
    path
        Manifest file to read.

    Returns
    -------
    tuple[str, dict[str, Any]]
        The exact file text and decoded object payload.
    """
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        msg = f"{path} must contain a JSON object"
        raise ValueError(msg)
    return text, payload


def validate_deployed_contract(payload: dict[str, Any]) -> None:
    """Fail closed if ``payload`` is not the deployed CONTROL Studio manifest.

    Parameters
    ----------
    payload
        Parsed schema-A Studio manifest payload.
    """
    ui_module = payload.get("ui_module")
    if not isinstance(ui_module, dict):
        msg = "ui_module must be present"
        raise ValueError(msg)
    expected = {
        "studio": "scpn-control",
        "remote_entry": "https://www.anulum.org/studios/scpn-control/remoteEntry.js",
        "exposes": ["./Panel"],
        "federation": "module-federation-2",
    }
    if payload.get("studio") != expected["studio"]:
        msg = "studio must be scpn-control"
        raise ValueError(msg)
    for key in ("remote_entry", "exposes", "federation"):
        if ui_module.get(key) != expected[key]:
            msg = f"ui_module.{key} must match the deployed Studio contract"
            raise ValueError(msg)


def sync_manifest(*, check: bool = False) -> int:
    """Synchronize or check the deployed Studio Web ``manifest.json`` artifact.

    Parameters
    ----------
    check
        When true, report drift without writing.

    Returns
    -------
    int
        Process-style exit code: ``0`` for success, ``1`` for drift or invalid
        manifests.
    """
    try:
        source_text, source_payload = read_manifest(SOURCE_MANIFEST)
        validate_deployed_contract(source_payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"{SOURCE_MANIFEST} is invalid: {exc}")
        return 1

    if check:
        try:
            web_text, web_payload = read_manifest(WEB_MANIFEST)
            validate_deployed_contract(web_payload)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"{WEB_MANIFEST} is invalid or missing: {exc}")
            return 1
        if web_text != source_text:
            print(f"{WEB_MANIFEST} is stale; run `python tools/sync_studio_web_manifest.py`.")
            return 1
        return 0

    WEB_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    WEB_MANIFEST.write_text(source_text, encoding="utf-8")
    print(f"wrote {WEB_MANIFEST}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface for the Studio Web manifest sync guard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if studio-web/public/manifest.json differs from the generated Studio manifest.",
    )
    args = parser.parse_args(argv)
    return sync_manifest(check=args.check)


if __name__ == "__main__":
    sys.exit(main())
