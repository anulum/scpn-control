# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — External GK interface author contracts

"""Inspect original declared interface identities, lexical references and scalar units; authenticate no source bytes."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

from validation.reference_uri import external_executable_path_error

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

ROOT = Path(__file__).resolve().parents[1]

_ALLOWED_CODES = {"TGLF", "GENE", "GS2", "CGYRO", "QuaLiKiz"}
_ALLOWED_SOURCES = {"real_executable", "documented_public_reference"}
_SCHEMA_VERSION = "scpn-control.gk-interface-artifact.v1"
_REQUIRED_STR_FIELDS = (
    "interface_code",
    "source",
    "code_version",
    "run_id",
    "executed_at",
    "input_deck_uri",
    "output_artifact_uri",
    "parsed_output_uri",
    "input_deck_sha256",
    "output_artifact_sha256",
    "parsed_output_sha256",
    "payload_sha256",
    "parser_version",
    "units",
)
_REQUIRED_NUMERIC_FIELDS = (
    "chi_i_m2_s",
    "chi_e_m2_s",
    "D_e_m2_s",
    "gamma_max_cs_over_a",
    "omega_r_cs_over_a",
    "k_y_rho_s_at_max",
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_ARTIFACT_URI_FIELDS = ("input_deck_uri", "output_artifact_uri", "parsed_output_uri")
_SHA256_FIELDS = (
    "input_deck_sha256",
    "output_artifact_sha256",
    "parsed_output_sha256",
    "payload_sha256",
)
_REQUIRED_UNIT_TOKENS = ("m^2/s", "c_s/a", "k_y*rho_s")
_REPORT_SCHEMA = "scpn-control.gk-interface-artifact-report.v2"
_BLOCKED_REASON = "Requires persisted real-executable or documented public-reference GK interface artefacts."


def _validate_artifact(
    path: Path,
    raw_payload: bytes,
    payload: object,
    errors: list[dict[str, object]],
) -> dict[str, object] | None:
    """Preserve declared provenance/scalar/unit rules; bind actual captured bytes without authenticating source artifacts."""
    if not isinstance(payload, dict):
        errors.append({"path": _portable_path(path), "field": "root", "error": "artefact root must be an object"})
        return None
    if payload.get("schema_version") != _SCHEMA_VERSION:
        errors.append(
            {
                "path": _portable_path(path),
                "field": "schema_version",
                "error": f"schema_version must be '{_SCHEMA_VERSION}'",
            }
        )
    for field in _REQUIRED_STR_FIELDS:
        if not isinstance(payload.get(field), str) or not str(payload.get(field)).strip():
            errors.append({"path": _portable_path(path), "field": field, "error": "field must be a non-empty string"})
    for field in _SHA256_FIELDS:
        value = payload.get(field)
        if isinstance(value, str) and not _SHA256_RE.fullmatch(value):
            errors.append({"path": _portable_path(path), "field": field, "error": "field must be a SHA-256 hex digest"})
    for field in _ARTIFACT_URI_FIELDS:
        error = _artifact_uri_error(payload.get(field))
        if error is not None:
            errors.append({"path": _portable_path(path), "field": field, "error": error})
    units = payload.get("units")
    if isinstance(units, str) and not all(token in units for token in _REQUIRED_UNIT_TOKENS):
        errors.append(
            {
                "path": _portable_path(path),
                "field": "units",
                "error": "units must declare m^2/s transport, c_s/a frequencies, and k_y*rho_s wavenumber",
            }
        )
    if isinstance(payload.get("payload_sha256"), str) and _SHA256_RE.fullmatch(payload["payload_sha256"]):
        expected = canonical_artifact_sha256(payload)
        observed = str(payload["payload_sha256"])
        if not hmac.compare_digest(observed.lower(), expected):
            errors.append(
                {"path": _portable_path(path), "field": "payload_sha256", "error": "canonical payload digest mismatch"}
            )
    for field in _REQUIRED_NUMERIC_FIELDS:
        if not _is_finite_number(payload.get(field)):
            errors.append({"path": _portable_path(path), "field": field, "error": "field must be finite numeric"})
    if not isinstance(payload.get("interface_code"), str) or payload["interface_code"] not in _ALLOWED_CODES:
        errors.append(
            {"path": _portable_path(path), "field": "interface_code", "error": "unsupported external GK interface code"}
        )
    if not isinstance(payload.get("source"), str) or payload["source"] not in _ALLOWED_SOURCES:
        errors.append(
            {
                "path": _portable_path(path),
                "field": "source",
                "error": "source must be real_executable or documented_public_reference",
            }
        )

    source = payload.get("source")
    if source == "real_executable":
        binary_path_error = external_executable_path_error(payload.get("binary_path"))
        if binary_path_error is not None:
            errors.append({"path": _portable_path(path), "field": "binary_path", "error": binary_path_error})
    if source == "documented_public_reference" and not _has_public_reference(payload):
        errors.append(
            {
                "path": _portable_path(path),
                "field": "reference",
                "error": "documented public reference artefacts require reference_url or reference_doi",
            }
        )
    if any(error["path"] == _portable_path(path) for error in errors):
        return None

    chi_i = float(payload["chi_i_m2_s"])
    chi_e = float(payload["chi_e_m2_s"])
    d_e = float(payload["D_e_m2_s"])
    gamma = float(payload["gamma_max_cs_over_a"])
    ky = float(payload["k_y_rho_s_at_max"])
    if chi_i < 0.0:
        errors.append(
            {"path": _portable_path(path), "field": "chi_i_m2_s", "error": "transport coefficient must be non-negative"}
        )
    if chi_e < 0.0:
        errors.append(
            {"path": _portable_path(path), "field": "chi_e_m2_s", "error": "transport coefficient must be non-negative"}
        )
    if d_e < 0.0:
        errors.append(
            {"path": _portable_path(path), "field": "D_e_m2_s", "error": "transport coefficient must be non-negative"}
        )
    if gamma < 0.0:
        errors.append(
            {
                "path": _portable_path(path),
                "field": "gamma_max_cs_over_a",
                "error": "dominant growth rate must be non-negative",
            }
        )
    if ky <= 0.0:
        errors.append(
            {"path": _portable_path(path), "field": "k_y_rho_s_at_max", "error": "dominant wavenumber must be positive"}
        )
    if any(error["path"] == _portable_path(path) for error in errors):
        return None

    return {
        "path": _portable_path(path),
        "interface_code": str(payload["interface_code"]),
        "source": str(payload["source"]),
        "run_id": str(payload["run_id"]),
        "code_version": str(payload["code_version"]),
        "artifact_file_sha256": hashlib.sha256(raw_payload).hexdigest(),
        "payload_sha256": str(payload["payload_sha256"]).lower(),
    }


def _has_public_reference(payload: dict[str, object]) -> bool:
    """Require original nonblank URL or DOI declaration; fetch no reference and authenticate no publication."""
    for field in ("reference_url", "reference_doi"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return True
    return False


def canonical_artifact_sha256(payload: dict[str, object]) -> str:
    """Hash a shallow copy excluding only payload_sha256 using original sorted compact ASCII JSON.

    Preserve author body hash algorithm and Python json serialization behavior;
    stored declarations separately refuse nonfinite numbers before hashing. This
    authenticates no referenced file, executable, source provenance or parser run.

    Examples
    --------
    >>> canonical_artifact_sha256({})
    '44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a'
    >>> canonical_artifact_sha256({"payload_sha256": "author text"}) == canonical_artifact_sha256({})
    True
    """
    canonical_payload = dict(payload)
    canonical_payload.pop("payload_sha256", None)
    encoded = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_uri_error(value: object) -> str | None:
    """Preserve original nonblank/NUL/traversal/absolute/prefix lexical rules without fetching or authenticating a URI."""
    if not isinstance(value, str) or not value.strip():
        return "artefact URI must be a non-empty string"
    ref = value.strip()
    if "\x00" in ref:
        return "artefact URI must not contain NUL bytes"
    if ref.startswith(("http://", "https://", "doi:", "s3://", "gs://")):
        return None
    path = Path(ref)
    if path.is_absolute():
        return "artefact URI must be relative or an admitted external reference URI"
    if any(part == ".." for part in path.parts):
        return "artefact URI must not contain traversal"
    return None


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    """Require finite representable nonboolean numeric fields; huge integers are findings instead of conversion crashes."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _portable_path(path: Path) -> str:
    """Preserve repo-relative/outside lexical paths; failed filesystem resolution uses lexical fallback."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except (ValueError, OSError, RuntimeError):
        return path.as_posix()
