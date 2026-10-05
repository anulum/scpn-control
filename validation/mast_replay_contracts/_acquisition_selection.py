# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST acquisition decimal shot selection
"""Expand bounded decimal shot selections before acquisition I/O."""

from __future__ import annotations

from validation.mast_replay_contracts._inputs import shot_identity

MAX_REQUESTED_SHOTS = 100_000


def parse_shots(text: str) -> list[int]:
    """Expand a nonempty selection of unique positive int64 shot identities.

    Parameters
    ----------
    text : str
        Comma/whitespace-separated decimal identities or inclusive ``lo-hi``
        ranges. Ascending ranges and the supplied token order are preserved.

    Returns
    -------
    list of int
        Between one and 100000 unique identities, each at most int64 maximum.

    Raises
    ------
    ValueError
        Input is blank, non-decimal, descending, duplicated or outside either
        the shot domain or the selection-size limit. No I/O is performed.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("shots must be a nonempty decimal selection")
    out: list[int] = []
    for token in text.replace(",", " ").split():
        if "-" in token:
            lo, hi = token.split("-", 1)
            if not lo.isascii() or not lo.isdecimal() or not hi.isascii() or not hi.isdecimal():
                raise ValueError("shots must use decimal identities and ascending ranges")
            first, last = shot_identity(int(lo)), shot_identity(int(hi))
            if last < first:
                raise ValueError("shot ranges must be ascending")
            if len(out) + last - first + 1 > MAX_REQUESTED_SHOTS:
                raise ValueError("shots exceed the 100000 identity limit")
            out.extend(range(first, last + 1))
        else:
            if not token.isascii() or not token.isdecimal():
                raise ValueError("shots must use decimal identities and ascending ranges")
            if len(out) == MAX_REQUESTED_SHOTS:
                raise ValueError("shots exceed the 100000 identity limit")
            out.append(shot_identity(int(token)))
    if not out or len(set(out)) != len(out):
        raise ValueError("shots must be nonempty and unique")
    return out
