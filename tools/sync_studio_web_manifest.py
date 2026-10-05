# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — deployed Studio Web manifest sync guard.
"""Copy or check a local Studio manifest without altering its UTF-8 bytes.

Validate the declared CONTROL id and three UI metadata fields, then copy the
source exactly, including newline style and the environment-specific version
stamp. Check mode requires exact bytes, unlike the producer's semantic parity
check. This does not validate the complete SDK schema, compatibility range,
content digest, verb execution, hosted URL or federation/deployment admission.
The CLI uses only the standard library and never contacts a remote service.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "docs" / "_generated" / "studio_manifest.json"
WEB_MANIFEST = ROOT / "studio-web" / "public" / "manifest.json"


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Decode each JSON object while refusing duplicate keys at every depth."""
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("Studio Web manifest contains a duplicate JSON key")
        payload[key] = value
    return payload


def _reject_nonfinite(token: str) -> None:
    """Refuse nonstandard JSON NaN/Infinity tokens, including unused metadata."""
    raise ValueError("Studio Web manifest contains a nonfinite JSON token")


def _finite_float(token: str) -> float:
    """Decode a floating JSON token and refuse overflow to a nonfinite value."""
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("Studio Web manifest floating token overflows")
    return value


def read_manifest(path: Path) -> tuple[str, dict[str, Any]]:
    """Return the manifest text and parsed JSON payload from ``path``.

    Parameters
    ----------
    path
        UTF-8 file to read. Relative paths resolve from the caller's cwd.

    Returns
    -------
    tuple[str, dict[str, Any]]
        The exact decoded text without newline translation and its object
        payload. Re-encoding the text as UTF-8 preserves the original bytes.
        Reject duplicate keys and nonfinite/overflowing numbers at any depth.
        The source is not modified; this does not validate the SDK schema.

    Raises
    ------
    OSError
        The file cannot be read.
    UnicodeError
        The file is not UTF-8.
    ValueError
        JSON is malformed, ambiguous, nonfinite or has a non-object root.

    Examples
    --------
    Inspect the actual generated local artifact and its declared UI contract:

    >>> text, payload = read_manifest(SOURCE_MANIFEST)
    >>> payload['studio'], text.endswith(chr(10))
    ('scpn-control', True)
    >>> validate_deployed_contract(payload)
    """
    text = path.read_bytes().decode("utf-8")
    payload = json.loads(
        text, object_pairs_hook=_unique_keys, parse_constant=_reject_nonfinite, parse_float=_finite_float
    )
    if not isinstance(payload, dict):
        msg = f"{path} must contain a JSON object"
        raise ValueError(msg)
    return text, payload


def validate_deployed_contract(payload: dict[str, Any]) -> None:
    """Check the declared CONTROL id and fixed local federation UI metadata.

    Parameters
    ----------
    payload
        Decoded object; require studio scpn-control and a ui_module object
        with the fixed remote_entry, exposes and federation values. Extra
        fields are ignored. This historical API name does not imply deployed
        availability, full schema/digest/SDK validation or verb execution.

    Raises
    ------
    ValueError
        An expected field is absent or differs. Nothing is written/imported
        and no remote URL is requested; success returns None.
    """
    ui_module = payload.get("ui_module")
    if not isinstance(ui_module, dict):
        msg = "ui_module must be present"
        raise ValueError(msg)
    expected = {
        "studio": "scpn-control",
        "remote_entry": "https://anulum.github.io/scpn-control/studios/scpn-control/remoteEntry.js",
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


def sync_manifest(*, check: bool = False, source: Path | None = None, destination: Path | None = None) -> int:
    """Copy exact validated source bytes or check a selected local web artifact.

    Parameters
    ----------
    check
        When true, validate both carriers and report byte drift without writes.
    source, destination
        Caller paths, relative to cwd. None uses the corresponding script-root
        constant at invocation time; no process argv or constants are mutated.

    Returns
    -------
    int
        Zero for equality or a successful copy; one for missing, unreadable,
        invalid or stale carriers and destination write errors. Source/check
        refusals and stale/success messages use stdout; write failures use a
        fixed stderr refusal. Parent directories are created for copying.
        Writes overwrite directly, without atomicity or fsync guarantees.
        Preserve newline style, key order, whitespace and studio_version;
        this is a byte-copy gate, not the producer's semantic parity gate.
    """
    source_path = SOURCE_MANIFEST if source is None else source
    web_path = WEB_MANIFEST if destination is None else destination
    try:
        source_text, source_payload = read_manifest(source_path)
        validate_deployed_contract(source_payload)
    except (OSError, ValueError):
        print(f"{source_path} is invalid: source manifest could not be read or validated.")
        return 1

    if check:
        try:
            web_text, web_payload = read_manifest(web_path)
            validate_deployed_contract(web_payload)
        except (OSError, ValueError):
            print(f"{web_path} is invalid or missing: web manifest could not be read or validated.")
            return 1
        if web_text != source_text:
            print(f"{web_path} is stale; copy the selected source manifest again.")
            return 1
        return 0

    try:
        web_path.parent.mkdir(parents=True, exist_ok=True)
        web_path.write_bytes(source_text.encode("utf-8"))
    except OSError:
        print("Studio Web manifest refused: destination could not be written.", file=sys.stderr)
        return 1
    print(f"wrote {web_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the stdlib-only local copy/check CLI with explicit paths and intact process argv.

    --source and --destination select caller paths relative to cwd; omitted
    options retain script-root defaults. Return sync_manifest's zero/one
    status; argparse errors exit two and help exits zero. No remote publication
    or producer-dependency import is performed. Check mode never writes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if studio-web/public/manifest.json differs from the generated Studio manifest.",
    )
    parser.add_argument("--source", type=Path, help="Local generated source manifest.")
    parser.add_argument("--destination", type=Path, help="Local Studio Web manifest to write or check.")
    args = parser.parse_args(argv)
    return sync_manifest(check=args.check, source=args.source, destination=args.destination)


if __name__ == "__main__":
    sys.exit(main())
