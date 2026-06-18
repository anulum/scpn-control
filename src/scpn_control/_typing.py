# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Shared numpy typing aliases for the strict-mypy migration
"""Canonical numpy array-type aliases for the staged ``strict = true`` migration.

The package is migrating module by module to fully strict mypy
(``disallow_any_generics``), which forbids bare ``numpy.ndarray`` annotations.
Several ``phase`` modules already define ``FloatArray = NDArray[np.float64]``
locally; this module promotes that idiom to a single shared definition and adds
the broader input alias the transport-solver modules need.

Conventions
-----------
``FloatArray``
    Use for arrays the code *constructs, returns, or stores* (attributes, return
    types, locals). Every public entry point normalises with
    ``numpy.asarray(..., dtype=float)``, so float64 is the precise dtype these
    positions carry.

``AnyFloatArray``
    Use for *public input parameters* that may legitimately receive an array of
    any floating precision — most importantly the output of ``numpy.gradient``
    and arrays produced by neighbouring modules — so callers need not pre-cast.
    A ``FloatArray`` is assignable to ``AnyFloatArray`` but not the reverse, which
    is why the distinction matters at module boundaries.

Shape typing
------------
``numpy.typing.NDArray`` is ``ndarray[tuple[int, ...], dtype[T]]`` (any number of
dimensions), whereas ``numpy.zeros(n)`` with a scalar argument infers the narrower
``ndarray[tuple[int], dtype[float64]]`` (exactly one dimension). Assigning an
any-dimensional value into an attribute whose type mypy inferred as one
dimensional fails under numpy 2.x shape typing. **Always annotate array
attributes and dataclass fields explicitly** with ``FloatArray`` (rather than
relying on inference from ``numpy.zeros``/``numpy.empty``) so stored arrays keep
the any-dimensional shape type and accept reconstructed values.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
AnyFloatArray: TypeAlias = npt.NDArray[np.floating[Any]]

__all__ = ["AnyFloatArray", "FloatArray"]
