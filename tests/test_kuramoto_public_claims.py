# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Kuramoto public-claim boundary tests.
"""Regression tests for Kuramoto public performance-claim wording."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
KURAMOTO_SOURCE: Final = ROOT / "src" / "scpn_control" / "phase" / "kuramoto.py"


def test_kuramoto_source_avoids_unqualified_latency_claims() -> None:
    """Kuramoto source comments and docstrings must not claim fixed latency."""
    source = KURAMOTO_SOURCE.read_text(encoding="utf-8")

    forbidden_patterns = (
        re.compile(r"\bsub[- ]?ms\b", re.IGNORECASE),
        re.compile(r"\bsub[- ]?millisecond\b", re.IGNORECASE),
        re.compile(r"\bunder\s+1\s*ms\b", re.IGNORECASE),
        re.compile(r"\b<\s*1\s*ms\b", re.IGNORECASE),
    )

    for pattern in forbidden_patterns:
        assert pattern.search(source) is None
