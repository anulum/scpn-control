# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — studio schema-A CapabilityManifest emitter
"""Emit (or check) the SCPN-CONTROL schema-A studio CapabilityManifest artifact.

This is the federation-gate counterpart the SCPN-STUDIO keeper consumes with
``validate_studio_manifest`` — the schema-A manifest carrying ``contract_era`` +
``evidence_types`` + ``verbs`` + ``content_digest``. It is distinct from
``docs/_generated/capability_manifest.json`` (the repo-inventory manifest); this one is
the canonical product of :func:`scpn_control.studio.manifest.build_manifest`.

``--check`` compares decoded objects after removing only studio_version. It is
insensitive to JSON whitespace/key order, not ambiguous keys or nonfinite tokens.
This checks declared metadata parity, not federation admission, SDK compatibility,
deployed URL availability or successful verb execution. --artifact selects a local
file; the default stays in this script's repository. No external publication occurs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_ARTIFACT = Path(__file__).resolve().parents[1] / "docs" / "_generated" / "studio_manifest.json"


def render() -> str:
    """Serialize the actual CONTROL/Studio SDK producer as sorted finite Unicode JSON.

    Return indented JSON with a trailing LF. The installed distribution version
    (or source sentinel) is included, so bytes can differ across environments.
    Import producer dependencies only on invocation; ImportError/producer errors
    propagate. This does not execute declared verbs, validate SDK compatibility,
    check remote UI resources or write files. Nonfinite JSON values raise ValueError.

    Examples
    --------
    Inspect the actual producer's declared Studio id and contract era:

    >>> declared = json.loads(render())
    >>> declared['studio'], declared['contract_era']
    ('scpn-control', 'v1')
    """
    from scpn_control.studio.manifest import build_manifest

    payload = build_manifest().to_dict()
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject repeated keys at every JSON object depth, even when values match."""
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("Studio manifest contains a duplicate JSON key")
        result[name] = value
    return result


def _reject_nonfinite(token: str) -> None:
    """Reject JSON's nonstandard NaN/Infinity tokens before stamp removal or comparison."""
    raise ValueError("Studio manifest contains a nonfinite JSON token")


def _finite_float(token: str) -> float:
    """Decode a JSON floating token and refuse overflow into a nonfinite Python value."""
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("Studio manifest floating token overflows")
    return value


def _manifest_object(text: str) -> dict[str, Any]:
    """Decode one JSON object with unique keys and standard finite numeric tokens.

    File decoding occurs at the caller. Reject non-object roots; syntax and
    duplicate/nonfinite errors propagate as ValueError. This validates decoding,
    not the SDK schema or its declared compatibility/admission policies.
    """
    payload = json.loads(
        text, object_pairs_hook=_unique_keys, parse_constant=_reject_nonfinite, parse_float=_finite_float
    )
    if not isinstance(payload, dict):
        raise ValueError("Studio manifest root must be a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    """Generate or compare a selected local manifest and return a process status.

    --artifact selects a path relative to cwd; default to this script repository's
    generated file. --check reads and compares without writes; no flags write
    UTF-8 output after rendering and create missing parent directories. Existing
    content is overwritten directly, without an atomic-file or transaction promise.
    Only studio_version is excluded from semantic object comparison. Return zero
    for equality/write success, one for missing/stale/invalid/unreadable artifacts,
    missing producer dependencies or write errors. Inspection failures print a fixed
    stderr refusal; missing/stale/write diagnostics use stdout. Argument errors
    exit two. Process argv stays intact; help requires no producer imports.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the committed artifact differs from the producer (no write).",
    )
    parser.add_argument("--artifact", type=Path, default=_ARTIFACT, help="Local generated artifact path.")
    args = parser.parse_args(argv)
    artifact = args.artifact
    try:
        rendered = render()
        if args.check:
            if not artifact.exists():
                print(f"{artifact} is missing; run `python tools/emit_studio_manifest.py`.")
                return 1
            # Stamp differences do not change the declared contract.
            committed = _manifest_object(artifact.read_text(encoding="utf-8"))
            produced = _manifest_object(rendered)
            committed.pop("studio_version", None)
            produced.pop("studio_version", None)
            if committed != produced:
                print(f"{artifact} is stale; run `python tools/emit_studio_manifest.py`.")
                return 1
            return 0

        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(rendered, encoding="utf-8")
        print(f"wrote {artifact}")
        return 0
    except (ImportError, OSError, UnicodeError, ValueError):
        print("Studio manifest refused: producer or artifact could not be inspected or written.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
