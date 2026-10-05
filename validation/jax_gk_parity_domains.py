# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — JAX GK parity declared numeric/mode/hash domains

"""Preserve declared mode/growth/JSON hash rules without solver execution or authenticated source provenance."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _validate_case_acceptance(
    path: Path,
    case_acceptance: dict[str, Any],
    native_mode_types: list[str],
    jax_mode_types: list[str],
    gamma_max: float,
    errors: list[dict[str, object]],
) -> None:
    """Check a nonempty decoded acceptance object after outer shape validation.

    Required modes are stripped ordered strings and must occur in both
    spectra. Optional null/missing growth bound is allowed; a finite declared
    bound must not be exceeded by the larger native/JAX growth scalar.
    """
    required_mode_types = _string_list(case_acceptance.get("required_mode_types"))
    if not required_mode_types:
        errors.append(
            {
                "path": _display_path(path),
                "field": "case_acceptance.required_mode_types",
                "error": "required mode types missing",
            }
        )
    for mode_type in required_mode_types:
        if mode_type not in native_mode_types or mode_type not in jax_mode_types:
            errors.append(
                {
                    "path": _display_path(path),
                    "field": "case_acceptance.required_mode_types",
                    "error": f"required mode absent from native/JAX spectra: {mode_type}",
                }
            )
    gamma_bound = case_acceptance.get("max_gamma_max_cs_over_a")
    if gamma_bound is not None:
        if not _is_finite_number(gamma_bound):
            errors.append(
                {
                    "path": _display_path(path),
                    "field": "case_acceptance.max_gamma_max_cs_over_a",
                    "error": "gamma bound must be finite numeric or null",
                }
            )
        elif gamma_max > float(gamma_bound):
            errors.append(
                {
                    "path": _display_path(path),
                    "field": "case_acceptance.max_gamma_max_cs_over_a",
                    "error": "growth exceeds case bound",
                }
            )


def _string_list(value: object) -> list[str]:
    """Strip a complete string list or return empty on any invalid member.

    Ordered duplicate labels are retained. Nonlist, empty list, nonstring or
    blank elements cannot describe an admitted mode spectrum.
    """
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            return []
        out.append(item.strip())
    return out


def _is_finite_number(value: object) -> bool:
    """Accept nonboolean numeric values with a finite float representation.

    Very large JSON integers are valid JSON but cannot represent an admitted
    scalar or growth bound; conversion overflow returns false.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _display_path(path: Path) -> str:
    """Display resolved checkout paths relatively and other paths lexically.

    Resolution follows symlinks without enforcing containment. A ValueError,
    OSError or RuntimeError, falls back to the supplied path spelling.
    """
    try:
        return str(path.resolve(strict=False).relative_to(ROOT))
    except (ValueError, OSError, RuntimeError):
        return str(path)


def _is_sha256_hex(value: object) -> bool:
    """Recognize exactly 64 ASCII hex characters without normalizing case.

    A syntactically accepted uppercase digest still fails case-sensitive
    equality against a computed lowercase canonical SHA-256 string.
    """
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)


def _sha256_json(payload: dict[str, Any], *, include_payload_field: bool = False) -> str:
    """Hash sorted compact ASCII JSON with the defined top-level exclusions.

    Normal mode omits payload_sha256 and report_payload_sha256. Include mode
    retains every nested input key. This binds declarations, not original
    encoded bytes or authenticated provenance.
    """
    digest_payload = (
        dict(payload)
        if include_payload_field
        else {k: v for k, v in payload.items() if k not in {"payload_sha256", "report_payload_sha256"}}
    )
    encoded = json.dumps(digest_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class _ParityDeclarationRefusal(ValueError):
    """Carry only an authored duplicate/nonfinite/nonzero-underflow decoder finding."""


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Preserve decoded key order and refuse duplicates without exposing member names."""
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise _ParityDeclarationRefusal("JAX GK parity declaration contains duplicate JSON keys")
        out[key] = value
    return out


def _reject_json_constant(token: str) -> None:
    """Refuse nonstandard NaN and signed infinity decoder extensions at every depth."""
    raise _ParityDeclarationRefusal("JAX GK parity declaration contains non-finite JSON numbers")


def _finite_json_float(token: str) -> float:
    """Refuse decimal overflow and nonzero binary64 underflow before altering declared values."""
    value = float(token)
    if not math.isfinite(value):
        raise _ParityDeclarationRefusal("JAX GK parity declaration contains non-finite JSON numbers")
    if value == 0.0 and any(char in "123456789" for char in token.lower().split("e", 1)[0]):
        raise _ParityDeclarationRefusal("JAX GK parity declaration contains underflowed JSON numbers")
    return value
