#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disruption declaration contracts

"""Check original disruption identity, signal, inventory and declared error domains."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

_ALLOWED_SOURCES = {"documented_public_reference", "measured_disruption_campaign", "external_benchmark"}
_ALLOWED_EXTERNAL_CODES = {"JOREK", "M3D-C1", "NIMROD", "TSC"}
_REQUIRED_STR_FIELDS = (
    "source",
    "model_id",
    "model_version",
    "reference_dataset_id",
    "reference_artifact_sha256",
    "executed_at",
)
_REQUIRED_SIGNAL_FIELDS = (
    "sample_count",
    "sample_period_s",
    "pre_disruption_duration_s",
    "current_quench_duration_ms",
    "thermal_quench_duration_ms",
)
_REQUIRED_MITIGATION_FIELDS = (
    "neon_quantity_mol",
    "argon_quantity_mol",
    "xenon_quantity_mol",
    "total_impurity_mol",
    "mitigation_strength",
    "tbr_reference",
)
_REQUIRED_UNITS = {
    "time": "s",
    "quench_time": "ms",
    "current": "MA",
    "energy": "MJ",
    "impurity_inventory": "mol",
    "risk": "1",
    "tbr": "1",
}
_MAXIMUM_ERROR_METRICS = (
    "risk_after_abs_error",
    "detection_lead_time_abs_error_ms",
    "halo_current_relative_error",
    "runaway_beam_relative_error",
    "tbr_abs_error",
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _validate_artifact(path: Path, payload: object, errors: list[dict[str, object]]) -> dict[str, object] | None:
    """Check original schema1.0, format-only reference SHA, source presence, signal/inventory units and declared bounds; return identity only without mitigation physics or producer authentication."""
    if not isinstance(payload, dict):
        errors.append({"path": str(path), "field": "root", "error": "artifact root must be an object"})
        return None
    if payload.get("schema_version") != "1.0":
        errors.append({"path": str(path), "field": "schema_version", "error": "schema_version must be '1.0'"})
    for field in _REQUIRED_STR_FIELDS:
        if not isinstance(payload.get(field), str) or not str(payload.get(field)).strip():
            errors.append({"path": str(path), "field": field, "error": "field must be a non-empty string"})
    digest = payload.get("reference_artifact_sha256")
    if isinstance(digest, str) and not _SHA256_RE.fullmatch(digest):
        errors.append(
            {"path": str(path), "field": "reference_artifact_sha256", "error": "field must be a SHA-256 hex digest"}
        )
    source = payload.get("source")
    if not isinstance(source, str) or source not in _ALLOWED_SOURCES:
        errors.append(
            {
                "path": str(path),
                "field": "source",
                "error": "source must be documented_public_reference, measured_disruption_campaign, or external_benchmark",
            }
        )
    _validate_source_provenance(path, payload, errors)
    if not _valid_signal_window(payload.get("signal_window")):
        errors.append(
            {
                "path": str(path),
                "field": "signal_window",
                "error": "signal_window must declare finite disruption timing metadata",
            }
        )
    if not _valid_mitigation_metadata(payload.get("mitigation_metadata")):
        errors.append(
            {
                "path": str(path),
                "field": "mitigation_metadata",
                "error": "mitigation_metadata must declare finite mitigation inventory and TBR metadata",
            }
        )
    if not _valid_units(payload.get("units")):
        errors.append({"path": str(path), "field": "units", "error": "units must declare disruption reference units"})
    count = payload.get("reference_case_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        errors.append({"path": str(path), "field": "reference_case_count", "error": "field must be a positive integer"})
    _validate_metric_block(path, payload.get("metrics"), payload.get("tolerances"), errors)
    if any(error["path"] == str(path) for error in errors):
        return None
    return {
        "path": str(path),
        "source": str(payload["source"]),
        "model_id": str(payload["model_id"]),
        "model_version": str(payload["model_version"]),
        "reference_dataset_id": str(payload["reference_dataset_id"]),
        "reference_case_count": int(payload["reference_case_count"]),
    }


def _validate_source_provenance(path: Path, payload: dict[str, object], errors: list[dict[str, object]]) -> None:
    """Require public URL/DOI presence, measured shot/diagnostic presence or original named external code/URI presence, without parsing or fetching references."""
    source = payload.get("source")
    if source == "documented_public_reference" and not _has_public_reference(payload):
        errors.append(
            {
                "path": str(path),
                "field": "reference",
                "error": "documented public reference artifacts require reference_url or reference_doi",
            }
        )
    if source == "measured_disruption_campaign":
        if not _has_nonempty_str(payload, "shot_id"):
            errors.append(
                {"path": str(path), "field": "shot_id", "error": "measured disruption campaigns require shot_id"}
            )
        if not _has_nonempty_str(payload, "diagnostic_uri"):
            errors.append(
                {
                    "path": str(path),
                    "field": "diagnostic_uri",
                    "error": "measured disruption campaigns require diagnostic_uri",
                }
            )
    if source == "external_benchmark":
        external_code = payload.get("external_code")
        if not isinstance(external_code, str) or external_code not in _ALLOWED_EXTERNAL_CODES:
            errors.append(
                {
                    "path": str(path),
                    "field": "external_code",
                    "error": "external_code must be JOREK, M3D-C1, NIMROD, or TSC",
                }
            )
        if not _has_nonempty_str(payload, "reference_artifact_uri"):
            errors.append(
                {
                    "path": str(path),
                    "field": "reference_artifact_uri",
                    "error": "external benchmarks require reference_artifact_uri",
                }
            )


def _validate_metric_block(path: Path, metrics: object, tolerances: object, errors: list[dict[str, object]]) -> None:
    """Compare five nonnegative finite declared mitigation errors to positive finite bounds with equality admitted, without recomputing risk, lead time, halo, runaway or TBR."""
    if not isinstance(metrics, dict):
        errors.append({"path": str(path), "field": "metrics", "error": "metrics must be an object"})
        return
    if not isinstance(tolerances, dict):
        errors.append({"path": str(path), "field": "tolerances", "error": "tolerances must be an object"})
        return
    for field in _MAXIMUM_ERROR_METRICS:
        metric = metrics.get(field)
        tolerance = tolerances.get(field)
        if not _is_nonnegative_finite(metric):
            errors.append({"path": str(path), "field": field, "error": "metric must be finite and non-negative"})
            continue
        if not _is_positive_finite(tolerance):
            errors.append({"path": str(path), "field": field, "error": "tolerance must be finite and positive"})
            continue
        if float(metric) > float(tolerance):
            errors.append({"path": str(path), "field": field, "error": "metric exceeds declared tolerance"})


def _valid_signal_window(value: object) -> bool:
    """Require uncapped nonboolean integer sample_count>=8 and four positive finite timing values; impose no cross-duration relation."""
    if not isinstance(value, dict):
        return False
    sample_count = value.get("sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 8:
        return False
    return all(_is_positive_finite(value.get(field)) for field in _REQUIRED_SIGNAL_FIELDS if field != "sample_count")


def _valid_mitigation_metadata(value: object) -> bool:
    """Require six nonnegative finite inventory/strength/TBR values, with strength<=1; impose no inventory sum or positive TBR requirement."""
    if not isinstance(value, dict):
        return False
    if not all(_is_nonnegative_finite(value.get(field)) for field in _REQUIRED_MITIGATION_FIELDS):
        return False
    mitigation_strength = value.get("mitigation_strength")
    return _is_nonnegative_finite(mitigation_strength) and float(mitigation_strength) <= 1.0


def _valid_units(value: object) -> bool:
    """Require original s, ms, MA, MJ, mol and dimensionless risk/TBR labels, admitting extras without conversion."""
    return isinstance(value, dict) and all(value.get(field) == unit for field, unit in _REQUIRED_UNITS.items())


def _has_public_reference(payload: dict[str, object]) -> bool:
    """Require nonblank URL or DOI presence without URI policy or citation verification."""
    return any(_has_nonempty_str(payload, field) for field in ("reference_url", "reference_doi"))


def _has_nonempty_str(payload: dict[str, object], field: str) -> bool:
    """Accept nonblank strings while retaining unparsed original declaration values."""
    value = payload.get(field)
    return isinstance(value, str) and bool(value.strip())


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    """Admit nonboolean int/float only when binary64 conversion is finite, refusing overflow."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _is_nonnegative_finite(value: object) -> TypeGuard[int | float]:
    """Require admitted finite numbers greater than or equal to zero."""
    return _is_finite_number(value) and float(value) >= 0.0


def _is_positive_finite(value: object) -> TypeGuard[int | float]:
    """Require admitted finite numbers strictly greater than zero."""
    return _is_finite_number(value) and float(value) > 0.0
