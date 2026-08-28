#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Deprecated static-mu validator compatibility entrypoint.

"""Compatibility entrypoint for static structured-mu reference validation."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

from validation.validate_static_mu_analysis_reference import (
    ROOT,
    validate_static_mu_analysis_reference,
)
from validation.validate_static_mu_analysis_reference import (
    main as _canonical_main,
)


def validate_mu_synthesis_reference(
    artifact_root: str | Path,
    *,
    require_reference_artifacts: bool = False,
) -> dict[str, Any]:
    """Forward the deprecated function to static mu-analysis validation."""
    warnings.warn(
        "validate_mu_synthesis_reference is deprecated and will be removed in "
        "0.25.0; use validate_static_mu_analysis_reference.",
        DeprecationWarning,
        stacklevel=2,
    )
    return validate_static_mu_analysis_reference(
        artifact_root,
        require_reference_artifacts=require_reference_artifacts,
    )


def main(argv: list[str] | None = None) -> int:
    """Forward the deprecated command-line entrypoint."""
    print(
        "DEPRECATED: validate_mu_synthesis_reference.py will be removed in "
        "0.25.0; use validate_static_mu_analysis_reference.py.",
        file=sys.stderr,
    )
    return _canonical_main(argv)


__all__ = ["ROOT", "main", "validate_mu_synthesis_reference"]


if __name__ == "__main__":
    raise SystemExit(main())
