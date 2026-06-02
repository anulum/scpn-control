# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Project: SCPN Control
# Description: Lean formal-verification report contracts.
"""Lean 4 formal-verification evidence contracts for SCPN controllers.

This module does not execute Lean. It defines the repository admission format
for machine-checked Lean artefacts produced by a reproducible Lean/Lake
toolchain. Safety-critical controller artefacts can reference these reports
through ``FormalVerificationEvidence``; the artifact loader then verifies that
the report and manifest agree byte-for-byte on the bounded proof claim.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

LEAN_FORMAL_REPORT_SCHEMA_VERSION = "scpn-control.lean4-formal-report.v1"
LEAN_REQUIRED_PROVED_CONTRACTS = frozenset({"pid.actuator_saturation", "snn.marking_bounds"})
LEAN_REQUIRED_CONTRACT_MODULE_PREFIXES = {
    "pid.actuator_saturation": "ScpnControl.PID",
    "snn.marking_bounds": "ScpnControl.SNN",
}
LEAN_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_'.]*(?:\.[A-Za-z_][A-Za-z0-9_'.]*)*$")
LEAN_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
SAFETY_CASE_ID_RE = re.compile(r"^[A-Z0-9][A-Z0-9_.:-]*$")


class LeanFormalVerificationError(ValueError):
    """Raised when a Lean formal-verification report is not admissible."""


@dataclass(frozen=True)
class LeanFormalVerificationReport:
    """Machine-checkable Lean 4 report metadata for bounded controller claims."""

    status: str
    solver: str
    lean_version: str
    checked_specs: list[str]
    artifact_sha256: str
    proof_source_sha256: str
    lakefile_sha256: str
    theorem_names: list[str]
    theorem_modules: list[str]
    proved_contracts: list[str]
    module_paths: list[str]
    safety_case_ids: list[str]
    claim_boundary: str
    proof_assumptions: list[str]


def _is_sha256_hex(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _canonical_payload_sha256(payload: dict[str, Any]) -> str:
    payload_without_digest = dict(payload)
    payload_without_digest.pop("payload_sha256", None)
    encoded = json.dumps(
        payload_without_digest,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_assumption_sha256(proof_assumptions: list[str]) -> str:
    """Compute a canonical digest for bounded proof assumptions."""
    encoded = json.dumps(
        proof_assumptions,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_non_empty_string_list(
    value: object,
    field_name: str,
    *,
    pattern: re.Pattern[str] | None = None,
) -> list[str]:
    """Validate a non-empty, duplicate-free string list."""
    if not isinstance(value, list) or not value:
        raise LeanFormalVerificationError(f"{field_name} must be a non-empty list")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item:
            raise LeanFormalVerificationError(f"{field_name} must contain non-empty strings")
        if pattern is not None and pattern.fullmatch(item) is None:
            raise LeanFormalVerificationError(f"{field_name} contains invalid identifier")
        if item in seen:
            raise LeanFormalVerificationError(f"{field_name} must not contain duplicates")
        seen.add(item)
        result.append(item)
    return result


def validate_safe_relative_path_list(value: object, field_name: str) -> list[str]:
    """Validate report path lists without accepting traversal or URI syntax."""
    paths = validate_non_empty_string_list(value, field_name)
    for path in paths:
        if "\\" in path or "://" in path or path.startswith(("file:", "/", "~")):
            raise LeanFormalVerificationError(f"{field_name} must contain safe relative paths")
        rel = PurePosixPath(path)
        if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
            raise LeanFormalVerificationError(f"{field_name} must contain safe relative paths")
    return paths


def validate_required_contract_theorem_coverage(
    *,
    proved_contracts: list[str],
    theorem_names: list[str],
    theorem_modules: list[str],
) -> None:
    """Validate that required contracts are covered by matching Lean theorem namespaces."""
    for contract, module_prefix in LEAN_REQUIRED_CONTRACT_MODULE_PREFIXES.items():
        if contract not in proved_contracts:
            continue
        if module_prefix not in theorem_modules:
            raise LeanFormalVerificationError(f"{contract} requires theorem_modules to include {module_prefix}")
        theorem_prefix = f"{module_prefix}."
        if not any(theorem.startswith(theorem_prefix) for theorem in theorem_names):
            raise LeanFormalVerificationError(f"{contract} requires theorem_names under {module_prefix}")


def validate_bounded_proof_assumptions(value: object) -> list[str]:
    """Validate explicit bounded proof assumptions for Lean evidence."""
    assumptions = validate_non_empty_string_list(value, "proof_assumptions")
    for assumption in assumptions:
        lowered = assumption.lower()
        if "bounded" not in lowered or "unbounded" in lowered:
            raise LeanFormalVerificationError("proof_assumptions must state bounded assumptions")
        if "certified" in lowered or "certification" in lowered:
            raise LeanFormalVerificationError("proof_assumptions must not claim certification")
    return assumptions


def build_lean_formal_report_payload(report: LeanFormalVerificationReport) -> dict[str, Any]:
    """Build a canonical Lean formal-verification report payload."""
    payload: dict[str, Any] = {
        "schema_version": LEAN_FORMAL_REPORT_SCHEMA_VERSION,
        "status": report.status,
        "backend": "lean4",
        "solver": report.solver,
        "lean_version": report.lean_version,
        "checked_specs": report.checked_specs,
        "artifact_sha256": report.artifact_sha256,
        "proof_source_sha256": report.proof_source_sha256,
        "lakefile_sha256": report.lakefile_sha256,
        "theorem_names": report.theorem_names,
        "theorem_modules": report.theorem_modules,
        "proved_contracts": report.proved_contracts,
        "module_paths": report.module_paths,
        "safety_case_ids": report.safety_case_ids,
        "claim_boundary": report.claim_boundary,
        "proof_assumptions": report.proof_assumptions,
        "assumption_sha256": compute_assumption_sha256(report.proof_assumptions),
    }
    validate_lean_formal_report_payload(payload)
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    return payload


def validate_lean_formal_report_payload(payload: object) -> None:
    """Validate a Lean formal-verification report payload."""
    if not isinstance(payload, dict):
        raise LeanFormalVerificationError("Lean 4 report must be an object")
    if payload.get("schema_version") != LEAN_FORMAL_REPORT_SCHEMA_VERSION:
        raise LeanFormalVerificationError("Lean 4 report schema_version is invalid")
    if payload.get("backend") != "lean4":
        raise LeanFormalVerificationError("Lean 4 report backend must be lean4")
    status = payload.get("status")
    if status not in {"pass", "fail", "blocked"}:
        raise LeanFormalVerificationError("Lean 4 report status is invalid")
    for field in ("solver", "lean_version", "claim_boundary"):
        value = payload.get(field)
        if not isinstance(value, str) or not value:
            raise LeanFormalVerificationError(f"Lean 4 report {field} is invalid")
    boundary = str(payload["claim_boundary"]).lower()
    if "bounded" not in boundary or "unbounded" in boundary:
        raise LeanFormalVerificationError("Lean 4 report claim_boundary must state a bounded proof boundary")
    for field in ("artifact_sha256", "proof_source_sha256", "lakefile_sha256"):
        value = payload.get(field)
        if not isinstance(value, str) or not _is_sha256_hex(value):
            raise LeanFormalVerificationError(f"Lean 4 report {field} is invalid")
    proof_assumptions = validate_bounded_proof_assumptions(payload.get("proof_assumptions"))
    assumption_sha256 = payload.get("assumption_sha256")
    if not isinstance(assumption_sha256, str) or not _is_sha256_hex(assumption_sha256):
        raise LeanFormalVerificationError("Lean 4 report assumption_sha256 is invalid")
    if assumption_sha256.lower() != compute_assumption_sha256(proof_assumptions):
        raise LeanFormalVerificationError("Lean 4 report assumption_sha256 does not match proof_assumptions")
    checked_specs = validate_non_empty_string_list(payload.get("checked_specs"), "checked_specs")
    theorem_names = validate_non_empty_string_list(
        payload.get("theorem_names"),
        "theorem_names",
        pattern=LEAN_IDENTIFIER_RE,
    )
    theorem_modules = validate_non_empty_string_list(
        payload.get("theorem_modules"),
        "theorem_modules",
        pattern=LEAN_MODULE_RE,
    )
    proved_contracts = validate_non_empty_string_list(payload.get("proved_contracts"), "proved_contracts")
    module_paths = validate_safe_relative_path_list(payload.get("module_paths"), "module_paths")
    safety_case_ids = validate_non_empty_string_list(
        payload.get("safety_case_ids"),
        "safety_case_ids",
        pattern=SAFETY_CASE_ID_RE,
    )
    missing_contracts = sorted(LEAN_REQUIRED_PROVED_CONTRACTS.difference(proved_contracts))
    if missing_contracts:
        raise LeanFormalVerificationError(
            "Lean 4 report proved_contracts missing required contracts: " + ", ".join(missing_contracts)
        )
    missing_specs = sorted(set(proved_contracts).difference(checked_specs))
    if missing_specs:
        raise LeanFormalVerificationError(
            "Lean 4 report checked_specs missing proved contracts: " + ", ".join(missing_specs)
        )
    if len(theorem_modules) > len(theorem_names):
        raise LeanFormalVerificationError("Lean 4 report theorem_modules cannot exceed theorem_names")
    validate_required_contract_theorem_coverage(
        proved_contracts=proved_contracts,
        theorem_names=theorem_names,
        theorem_modules=theorem_modules,
    )
    if len(module_paths) < len(theorem_modules):
        raise LeanFormalVerificationError("Lean 4 report module_paths must cover theorem_modules")
    if len(safety_case_ids) < len(proved_contracts):
        raise LeanFormalVerificationError("Lean 4 report safety_case_ids must cover proved_contracts")
    payload_digest = payload.get("payload_sha256")
    if payload_digest is not None:
        if not isinstance(payload_digest, str) or not _is_sha256_hex(payload_digest):
            raise LeanFormalVerificationError("Lean 4 report payload_sha256 is invalid")
        if payload_digest.lower() != _canonical_payload_sha256(payload):
            raise LeanFormalVerificationError("Lean 4 report payload_sha256 does not match payload")


def write_lean_formal_report(report: LeanFormalVerificationReport, path: str | Path) -> dict[str, Any]:
    """Write a canonical Lean formal-verification report and return the payload."""
    payload = build_lean_formal_report_payload(report)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_lean_formal_report(path: str | Path) -> dict[str, Any]:
    """Load and validate a Lean formal-verification report payload."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_lean_formal_report_payload(payload)
    return cast(dict[str, Any], payload)
