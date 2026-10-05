# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — JAX GK parity author artifact contracts

"""Preserve original local native/JAX declaration comparisons; backend parity remains distinct from external validation."""

from __future__ import annotations

from pathlib import Path

from validation.jax_gk_parity_domains import (
    _display_path,
    _is_finite_number,
    _is_sha256_hex,
    _sha256_json,
    _string_list,
    _validate_case_acceptance,
)

_SCHEMA_VERSION = "scpn-control.jax-gk-parity.v1"
_ALLOWED_CASES = {"cyclone_base_case", "tem_kinetic_electron", "electromagnetic_kbm", "stable_mode"}
_ALLOWED_BACKENDS = {"cpu", "gpu", "tpu"}
_REQUIRED_STR_FIELDS = (
    "schema_version",
    "case",
    "backend",
    "jax_version",
    "jaxlib_version",
    "platform",
    "device_kind",
    "dtype",
    "executed_at",
    "solver_contract",
    "normalisation",
    "evidence_boundary",
    "solver_kwargs_sha256",
    "case_parameters_sha256",
    "payload_sha256",
)
_REQUIRED_FLOAT_FIELDS = (
    "native_gamma_max_cs_over_a",
    "jax_gamma_max_cs_over_a",
    "native_omega_r_cs_over_a",
    "jax_omega_r_cs_over_a",
    "gamma_relative_tolerance",
    "omega_absolute_tolerance",
)


def _validate_artifact(path: Path, payload: object, errors: list[dict[str, object]]) -> dict[str, object] | None:
    """Append domain findings and return one accepted summary or None.

    Input is a decoded JSON value. All shape/digest checks precede numeric
    conversion, then drift, ordered spectra and declared acceptance checks.
    Existing findings for this path prohibit admission; other paths do not.
    Metadata presence does not authenticate the device or physical source.
    """
    display_path = _display_path(path)
    if not isinstance(payload, dict):
        errors.append({"path": display_path, "field": "root", "error": "artifact root must be an object"})
        return None
    for field in _REQUIRED_STR_FIELDS:
        if not isinstance(payload.get(field), str) or not str(payload.get(field)).strip():
            errors.append({"path": display_path, "field": field, "error": "field must be a non-empty string"})
    if payload.get("schema_version") != _SCHEMA_VERSION:
        errors.append(
            {
                "path": _display_path(path),
                "field": "schema_version",
                "error": f"schema_version must be {_SCHEMA_VERSION!r}",
            }
        )
    if not isinstance(payload.get("x64_enabled"), bool):
        errors.append({"path": display_path, "field": "x64_enabled", "error": "field must be boolean"})
    if not isinstance(payload.get("external_validation_required"), bool) or not payload.get(
        "external_validation_required"
    ):
        errors.append(
            {
                "path": _display_path(path),
                "field": "external_validation_required",
                "error": "JAX GK parity artifacts must keep external validation required",
            }
        )
    if not isinstance(payload.get("admitted_for_control"), bool) or payload.get("admitted_for_control"):
        errors.append(
            {
                "path": _display_path(path),
                "field": "admitted_for_control",
                "error": "JAX GK parity artifacts are not control-admission evidence",
            }
        )
    for field in _REQUIRED_FLOAT_FIELDS:
        value = payload.get(field)
        if not _is_finite_number(value):
            errors.append({"path": _display_path(path), "field": field, "error": "field must be finite numeric"})
    if not isinstance(payload.get("case"), str) or payload["case"] not in _ALLOWED_CASES:
        errors.append({"path": _display_path(path), "field": "case", "error": "unsupported JAX GK parity case"})
    if not isinstance(payload.get("backend"), str) or payload["backend"] not in _ALLOWED_BACKENDS:
        errors.append({"path": _display_path(path), "field": "backend", "error": "backend must be cpu, gpu, or tpu"})
    if payload.get("solver_contract") != "native_linear_gk_local_dispersion":
        errors.append({"path": _display_path(path), "field": "solver_contract", "error": "unsupported solver contract"})
    if payload.get("normalisation") != "c_s_over_a":
        errors.append(
            {"path": _display_path(path), "field": "normalisation", "error": "normalisation must be c_s_over_a"}
        )
    if payload.get("evidence_boundary") != "backend_parity_only":
        errors.append(
            {"path": _display_path(path), "field": "evidence_boundary", "error": "unsupported evidence boundary"}
        )
    if not _is_sha256_hex(payload.get("solver_kwargs_sha256")):
        errors.append(
            {"path": _display_path(path), "field": "solver_kwargs_sha256", "error": "field must be SHA-256 hex"}
        )
    if not _is_sha256_hex(payload.get("case_parameters_sha256")):
        errors.append(
            {"path": _display_path(path), "field": "case_parameters_sha256", "error": "field must be SHA-256 hex"}
        )
    if not _is_sha256_hex(payload.get("payload_sha256")):
        errors.append({"path": _display_path(path), "field": "payload_sha256", "error": "field must be SHA-256 hex"})
    elif _sha256_json(payload) != payload.get("payload_sha256"):
        errors.append({"path": _display_path(path), "field": "payload_sha256", "error": "payload digest mismatch"})
    if not isinstance(payload.get("solver_kwargs"), dict) or not payload["solver_kwargs"]:
        errors.append(
            {"path": _display_path(path), "field": "solver_kwargs", "error": "solver_kwargs must be a non-empty object"}
        )
    elif _sha256_json(payload["solver_kwargs"], include_payload_field=True) != payload.get("solver_kwargs_sha256"):
        errors.append(
            {"path": _display_path(path), "field": "solver_kwargs_sha256", "error": "solver kwargs digest mismatch"}
        )
    if not isinstance(payload.get("case_parameters"), dict) or not payload["case_parameters"]:
        errors.append(
            {
                "path": _display_path(path),
                "field": "case_parameters",
                "error": "case_parameters must be a non-empty object",
            }
        )
    elif _sha256_json(payload["case_parameters"], include_payload_field=True) != payload.get("case_parameters_sha256"):
        errors.append(
            {"path": _display_path(path), "field": "case_parameters_sha256", "error": "case parameters digest mismatch"}
        )
    if not isinstance(payload.get("case_acceptance"), dict) or not payload["case_acceptance"]:
        errors.append(
            {
                "path": _display_path(path),
                "field": "case_acceptance",
                "error": "case_acceptance must be a non-empty object",
            }
        )
    native_mode_types = _string_list(payload.get("native_mode_types"))
    jax_mode_types = _string_list(payload.get("jax_mode_types"))
    if not native_mode_types:
        errors.append(
            {
                "path": _display_path(path),
                "field": "native_mode_types",
                "error": "mode spectrum must be a non-empty string list",
            }
        )
    if not jax_mode_types:
        errors.append(
            {
                "path": _display_path(path),
                "field": "jax_mode_types",
                "error": "mode spectrum must be a non-empty string list",
            }
        )
    if (
        not isinstance(payload.get("native_dominant_mode_type"), str)
        or not str(payload.get("native_dominant_mode_type")).strip()
    ):
        errors.append(
            {
                "path": _display_path(path),
                "field": "native_dominant_mode_type",
                "error": "field must be a non-empty string",
            }
        )
    if (
        not isinstance(payload.get("jax_dominant_mode_type"), str)
        or not str(payload.get("jax_dominant_mode_type")).strip()
    ):
        errors.append(
            {
                "path": _display_path(path),
                "field": "jax_dominant_mode_type",
                "error": "field must be a non-empty string",
            }
        )
    if any(error["path"] == display_path for error in errors):
        return None

    native_gamma = float(payload["native_gamma_max_cs_over_a"])
    jax_gamma = float(payload["jax_gamma_max_cs_over_a"])
    native_omega = float(payload["native_omega_r_cs_over_a"])
    jax_omega = float(payload["jax_omega_r_cs_over_a"])
    gamma_tolerance = float(payload["gamma_relative_tolerance"])
    omega_tolerance = float(payload["omega_absolute_tolerance"])
    if native_gamma < 0.0 or jax_gamma < 0.0:
        errors.append(
            {"path": display_path, "field": "gamma_max_cs_over_a", "error": "growth rates must be non-negative"}
        )
    if gamma_tolerance <= 0.0:
        errors.append(
            {"path": _display_path(path), "field": "gamma_relative_tolerance", "error": "tolerance must be positive"}
        )
    if omega_tolerance <= 0.0:
        errors.append(
            {"path": _display_path(path), "field": "omega_absolute_tolerance", "error": "tolerance must be positive"}
        )

    gamma_error = abs(jax_gamma - native_gamma) / max(abs(native_gamma), 1e-12)
    omega_error = abs(jax_omega - native_omega)
    if gamma_error > gamma_tolerance:
        errors.append(
            {"path": _display_path(path), "field": "gamma_max_cs_over_a", "error": "JAX/native gamma mismatch"}
        )
    if omega_error > omega_tolerance:
        errors.append({"path": _display_path(path), "field": "omega_r_cs_over_a", "error": "JAX/native omega mismatch"})
    native_dominant = str(payload["native_dominant_mode_type"]).strip()
    jax_dominant = str(payload["jax_dominant_mode_type"]).strip()
    if native_mode_types != jax_mode_types:
        errors.append(
            {"path": _display_path(path), "field": "mode_types", "error": "JAX/native mode spectrum mismatch"}
        )
    if native_dominant != jax_dominant:
        errors.append(
            {"path": _display_path(path), "field": "dominant_mode_type", "error": "JAX/native dominant mode mismatch"}
        )
    if native_dominant not in native_mode_types or jax_dominant not in jax_mode_types:
        errors.append(
            {"path": _display_path(path), "field": "dominant_mode_type", "error": "dominant mode missing from spectrum"}
        )
    _validate_case_acceptance(
        path,
        payload["case_acceptance"],
        native_mode_types,
        jax_mode_types,
        max(native_gamma, jax_gamma),
        errors,
    )
    if any(error["path"] == display_path for error in errors):
        return None

    return {
        "path": display_path,
        "case": str(payload["case"]),
        "backend": str(payload["backend"]),
        "dtype": str(payload["dtype"]),
        "x64_enabled": bool(payload["x64_enabled"]),
        "gamma_relative_error": gamma_error,
        "omega_absolute_error": omega_error,
        "payload_sha256": str(payload["payload_sha256"]),
        "evidence_boundary": str(payload["evidence_boundary"]),
        "native_dominant_mode_type": native_dominant,
        "jax_dominant_mode_type": jax_dominant,
    }
