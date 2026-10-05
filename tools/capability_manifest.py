# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Capability manifest command.
"""Expose the source-inventory API and its local generation/check command.

Use --check to compare normalised UTF-8 text in the two generated files and README block without
writing. The no-flag command publishes configured local outputs after complete
input/render validation. All paths follow the configured working-directory
root. Source declaration counts do not establish executable feature readiness.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.capability_manifest_inventory import CONFIG_PATH as CONFIG_PATH
from tools.capability_manifest_inventory import ManifestError as ManifestError
from tools.capability_manifest_inventory import build_manifest as build_manifest
from tools.capability_manifest_inventory import validate_manifest as validate_manifest
from tools.capability_manifest_rendering import README_END as README_END
from tools.capability_manifest_rendering import README_START as README_START
from tools.capability_manifest_rendering import check_outputs as check_outputs
from tools.capability_manifest_rendering import extract_readme_block as extract_readme_block
from tools.capability_manifest_rendering import render_json as render_json
from tools.capability_manifest_rendering import render_markdown as render_markdown
from tools.capability_manifest_rendering import write_outputs as write_outputs

__all__ = [
    "CONFIG_PATH",
    "README_START",
    "README_END",
    "ManifestError",
    "build_manifest",
    "validate_manifest",
    "render_json",
    "render_markdown",
    "extract_readme_block",
    "write_outputs",
    "check_outputs",
    "main",
]


def main(argv: list[str] | None = None) -> int:
    """Run the actual inventory check or publication against the caller's cwd.

    Parameters
    ----------
    argv
        CLI arguments excluding the executable; None reads process arguments.
        --check selects text comparison. No flags select configured publication.

    Returns
    -------
    int
        Zero for exact snapshots or successful local output generation; one for
        drift, invalid inventory/configuration, read/decode or write errors.
        Failures print to stderr; success prints nothing. Unknown flags exit two
        via argparse. No source code is imported for inventory discovery.
    """
    parser = argparse.ArgumentParser(description="Build or check the SCPN-CONTROL capability manifest.")
    parser.add_argument("--check", action="store_true", help="fail if generated manifest surfaces are stale")
    args = parser.parse_args(argv)
    try:
        if args.check:
            failures = check_outputs(Path.cwd())
            for failure in failures:
                print(failure, file=sys.stderr)
            return 1 if failures else 0
        write_outputs(Path.cwd())
    except (ManifestError, OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        print(f"Capability inventory refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
