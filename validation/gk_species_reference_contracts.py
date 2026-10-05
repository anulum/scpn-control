# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK species report identity and consistency contracts

"""GK species report identity and consistency contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

REPORT_SCHEMA_VERSION = "scpn-control.gk-species-reference.v3"

EXPECTED_UNITS = {
    "mass_kg": "kg",
    "thermal_speed_m_per_s": "m/s",
    "larmor_radius_per_tesla_m": "m*T",
    "nu_D_s^-1": "s^-1",
    "nu_E_s^-1": "s^-1",
    "omega_star_density": "dimensionless",
    "omega_star_temperature": "dimensionless",
    "omega_star_pressure": "dimensionless",
}

FULL_FIDELITY_BLOCKERS = (
    "field_particle_momentum_conservation_evidence",
    "external_fokker_planck_reference",
)

_REQUIRED_CASES = {
    "deuterium_cbc_main_ion",
    "kinetic_electron_cbc",
    "carbon_impurity_edge",
    "hot_deuterium_extreme_temperature",
}

_REQUIRED_HEADER_FIELDS = (
    "spdx_license_id",
    "commercial_license",
    "concepts_copyright",
    "code_copyright",
    "orcid",
    "contact",
    "file",
)

_EXPECTED_FIELDS = (
    "mass_kg",
    "thermal_speed_m_per_s",
    "larmor_radius_per_tesla_m",
    "nu_D_s^-1",
    "nu_E_s^-1",
    "omega_star_density",
    "omega_star_temperature",
    "omega_star_pressure",
)

_ABS_TOLERANCE = 1.0e-12

_REL_TOLERANCE = 1.0e-10


def _canonical_json(value: Any) -> str:
    """Serialise evidence deterministically for SHA-256 binding."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_payload(value: Any) -> str:
    """Return the canonical SHA-256 digest for a JSON-compatible payload."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _payload_without_digest(report: dict[str, Any]) -> dict[str, Any]:
    """Return a report copy with the self-referential digest field removed."""
    payload = dict(report)
    payload.pop("payload_sha256", None)
    return payload


def _portable_reference_path(path: Path) -> str:
    """Return a stable in-repository path for persisted reports."""
    try:
        return f"<repo-root>/{path.resolve().relative_to(ROOT).as_posix()}"
    except (OSError, ValueError, RuntimeError):
        return str(path)


def verify_payload_digest(report: dict[str, Any]) -> bool:
    """Return true when a persisted GK species report matches its digest."""
    digest = report.get("payload_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        return False
    try:
        return digest == _sha256_payload(_payload_without_digest(report))
    except (TypeError, ValueError, OverflowError):
        return False


def _finalise_report(report: dict[str, Any], *, reference_sha256: str | None) -> dict[str, Any]:
    """Bind single-read reference bytes and original bounded status/tolerances/payload digest."""
    status = report.get("status")
    report.update(
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "reference_sha256": reference_sha256,
            "bounded_operator_reference_admitted": status == "pass",
            "full_fidelity_claim_admitted": False,
            "blocked_reasons": list(FULL_FIDELITY_BLOCKERS),
            "claim_status": (
                "bounded GK species and test-particle collision reference admitted; "
                "full collision-operator claim remains blocked"
            ),
            "tolerances": {"absolute": _ABS_TOLERANCE, "relative": _REL_TOLERANCE},
        }
    )
    report["payload_sha256"] = _sha256_payload(report)
    return report


def _validate_header(path: Path, payload: dict[str, Any], errors: list[dict[str, object]]) -> None:
    """Require original nonblank header declarations and input schema1.0."""
    for field in _REQUIRED_HEADER_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(
                {
                    "path": str(path),
                    "field": field,
                    "error": "reference file requires canonical header metadata",
                }
            )
    if payload.get("schema_version") != "1.0":
        errors.append(
            {
                "path": str(path),
                "field": "schema_version",
                "error": "schema_version must be '1.0'",
            }
        )
