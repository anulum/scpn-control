# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Linear GK cross-code evidence validator

"""Validate declared external/native GK scalars; no binary or physical provenance authentication."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from validation.reference_uri import external_executable_path_error

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

_SCHEMA_VERSION = "scpn-control.gk-crosscode.v1"
_ALLOWED_EXTERNAL_CODES = {"TGLF", "GENE", "GS2", "CGYRO", "GYRO", "QuaLiKiz"}
_REQUIRED_STR_FIELDS = (
    "schema_version",
    "case",
    "external_code",
    "source",
    "binary_path",
    "code_version",
    "run_id",
    "executed_at",
    "units",
    "input_deck_sha256",
    "external_output_sha256",
    "native_input_sha256",
    "payload_sha256",
)
_REQUIRED_FLOAT_FIELDS = (
    "gamma_max_cs_over_a",
    "omega_r_cs_over_a",
    "k_y_rho_s_at_max",
    "native_gamma_max_cs_over_a",
    "native_omega_r_cs_over_a",
    "native_k_y_rho_s_at_max",
)
_MAX_GAMMA_RELATIVE_ERROR = 0.20
_MAX_OMEGA_RELATIVE_ERROR = 0.30
_MAX_KY_ABSOLUTE_ERROR = 0.10
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


def _validate_evidence_payload(
    path: Path, payload: object, errors: list[dict[str, object]]
) -> dict[str, object] | None:
    """Check author-declared identities, lexical binary path, units, digests and scalar discrepancies.

    Six finite nonboolean representable values retain original growth>=0 and
    wavenumber>0 requirements; signed frequencies are allowed. Comparisons use
    original 1e-12 denominator floors and inclusive20% gamma/30% frequency/0.10
    wavenumber limits. Data/binary existence, execution, source hashes and physical
    provenance are not authenticated; accepted entries count declarations.
    """
    if not isinstance(payload, dict):
        errors.append({"path": str(path), "field": "root", "error": "evidence report root must be an object"})
        return None
    if payload.get("schema_version") != _SCHEMA_VERSION:
        errors.append(
            {"path": str(path), "field": "schema_version", "error": f"schema_version must be {_SCHEMA_VERSION!r}"}
        )
    for field in _REQUIRED_STR_FIELDS:
        if not isinstance(payload.get(field), str) or not str(payload.get(field)).strip():
            errors.append({"path": str(path), "field": field, "error": "field must be a non-empty string"})
    for field in _REQUIRED_FLOAT_FIELDS:
        value = payload.get(field)
        if not _is_finite_number(value):
            errors.append({"path": str(path), "field": field, "error": "field must be finite numeric"})
    if payload.get("source") != "real_binary":
        errors.append({"path": str(path), "field": "source", "error": "source must be real_binary"})
    if not isinstance(payload.get("external_code"), str) or payload["external_code"] not in _ALLOWED_EXTERNAL_CODES:
        errors.append({"path": str(path), "field": "external_code", "error": "unsupported external GK code"})
    if payload.get("units") != "c_s/a":
        errors.append({"path": str(path), "field": "units", "error": "units must be c_s/a"})
    for field in ("input_deck_sha256", "external_output_sha256", "native_input_sha256", "payload_sha256"):
        if not _is_sha256_hex(payload.get(field)):
            errors.append({"path": str(path), "field": field, "error": "field must be SHA-256 hex"})
    if (
        _is_sha256_hex(payload.get("payload_sha256"))
        and _sha256_json(payload) != str(payload["payload_sha256"]).lower()
    ):
        errors.append({"path": str(path), "field": "payload_sha256", "error": "payload digest mismatch"})
    binary_path_error = external_executable_path_error(payload.get("binary_path"))
    if binary_path_error is not None:
        errors.append({"path": str(path), "field": "binary_path", "error": binary_path_error})
    if any(error["path"] == str(path) for error in errors):
        return None

    gamma = float(payload["gamma_max_cs_over_a"])
    native_gamma = float(payload["native_gamma_max_cs_over_a"])
    omega = float(payload["omega_r_cs_over_a"])
    native_omega = float(payload["native_omega_r_cs_over_a"])
    ky = float(payload["k_y_rho_s_at_max"])
    native_ky = float(payload["native_k_y_rho_s_at_max"])
    if gamma < 0.0 or native_gamma < 0.0:
        errors.append({"path": str(path), "field": "gamma_max_cs_over_a", "error": "growth rates must be non-negative"})
    if ky <= 0.0 or native_ky <= 0.0:
        errors.append(
            {"path": str(path), "field": "k_y_rho_s_at_max", "error": "dominant wavenumbers must be positive"}
        )
    if any(error["path"] == str(path) for error in errors):
        return None
    gamma_error = abs(native_gamma - gamma) / max(abs(gamma), 1e-12)
    omega_error = abs(native_omega - omega) / max(abs(omega), 1e-12)
    ky_error = abs(native_ky - ky)

    if gamma_error > _MAX_GAMMA_RELATIVE_ERROR:
        errors.append({"path": str(path), "field": "gamma_max_cs_over_a", "error": "native/external gamma mismatch"})
    if omega_error > _MAX_OMEGA_RELATIVE_ERROR:
        errors.append({"path": str(path), "field": "omega_r_cs_over_a", "error": "native/external omega mismatch"})
    if ky_error > _MAX_KY_ABSOLUTE_ERROR:
        errors.append({"path": str(path), "field": "k_y_rho_s_at_max", "error": "native/external k_y mismatch"})
    if any(error["path"] == str(path) for error in errors):
        return None

    return {
        "path": str(path),
        "case": str(payload["case"]),
        "external_code": str(payload["external_code"]),
        "run_id": str(payload["run_id"]),
        "gamma_relative_error": gamma_error,
        "omega_relative_error": omega_error,
        "k_y_absolute_error": ky_error,
        "payload_sha256": str(payload["payload_sha256"]).lower(),
    }


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    """Require a finite representable nonboolean scalar; conversion overflow is a field refusal."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _is_sha256_hex(value: object) -> bool:
    """Require exactly64 ASCII hexadecimal characters without trimming signs, spaces or Unicode digits."""
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _sha256_json(payload: dict[str, Any]) -> str:
    """Hash original compact sorted ASCII JSON excluding payload_sha256 without mutation.

    The public reader has already required an object and refused nonfinite JSON.
    The original dict filtering and serialisation algorithm remain unchanged.
    It proves author-supplied consistency, not binary/reference authentication.
    """
    digest_payload = {key: value for key, value in payload.items() if key != "payload_sha256"}
    encoded = json.dumps(digest_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
