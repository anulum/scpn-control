# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Dashboard GK state
# © 1998–2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────
"""Streamlit-independent gyrokinetic display helpers."""

from __future__ import annotations

from collections.abc import Iterable


def dominant_mode_from_types(mode_types: Iterable[str]) -> str:
    """Return the most frequent non-stable GK mode, tie-broken by first sighting."""
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for index, raw_mode in enumerate(mode_types):
        mode = raw_mode.strip()
        if not mode:
            raise ValueError("mode types must not contain blank mode names.")
        if mode == "stable":
            continue
        counts[mode] = counts.get(mode, 0) + 1
        first_seen.setdefault(mode, index)

    if not counts:
        return "stable"
    return min(counts, key=lambda mode: (-counts[mode], first_seen[mode]))


__all__ = ["dominant_mode_from_types"]
