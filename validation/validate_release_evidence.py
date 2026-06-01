# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Release Evidence Validation
"""Validate top-level release evidence reports before publication."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RELEASE_EVIDENCE_SCHEMA_VERSION = "scpn-control.release-evidence-admission.v1"
REQUIRED_GATES = ("data_manifests", "jax_gk_parity", "physics_traceability")
REQUIRED_JAX_CASES = frozenset({"cyclone_base_case", "tem_kinetic_electron", "stable_mode"})
REQUIRED_JAX_BACKENDS = frozenset({"cpu", "gpu"})


@dataclass(frozen=True)
class ReleaseEvidenceAdmission:
    """Strict validation result for a top-level release evidence report."""

    status: str
    errors: tuple[str, ...]
    report_sha256: str | None
    admitted_gates: tuple[str, ...]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_report(path: str | Path) -> tuple[dict[str, Any], str]:
    blob = Path(path).read_bytes()
    payload = json.loads(blob.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(payload, dict):
        raise ValueError("release evidence report root must be a JSON object")
    return payload, hashlib.sha256(blob).hexdigest()


def _positive_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _non_negative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _require_pass_gate(payload: dict[str, Any], gate: str, errors: list[str]) -> dict[str, Any]:
    section = payload.get(gate)
    if not isinstance(section, dict):
        errors.append(f"{gate} must be an object")
        return {}
    status = section.get("status")
    if status != "pass":
        errors.append(f"{gate}.status must be 'pass', got {status!r}")
    return section


def validate_release_evidence(path: str | Path) -> ReleaseEvidenceAdmission:
    """Validate the JSON report emitted by ``scpn-control validate --json-out``."""

    errors: list[str] = []
    try:
        payload, report_sha256 = _load_report(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return ReleaseEvidenceAdmission(
            status="fail",
            errors=(str(exc),),
            report_sha256=None,
            admitted_gates=(),
        )

    if payload.get("status") != "pass":
        errors.append(f"status must be 'pass', got {payload.get('status')!r}")
    if payload.get("transport_solver_available") is not True:
        errors.append("transport_solver_available must be true")
    if payload.get("import_clean") is not True:
        errors.append("import_clean must be true")

    data_manifests = _require_pass_gate(payload, "data_manifests", errors)
    if data_manifests:
        if not _positive_int(data_manifests.get("total")):
            errors.append("data_manifests.total must be a positive integer")
        if not _non_negative_int(data_manifests.get("real")):
            errors.append("data_manifests.real must be a non-negative integer")
        if not _non_negative_int(data_manifests.get("synthetic")):
            errors.append("data_manifests.synthetic must be a non-negative integer")
        artifact_coverage = data_manifests.get("artifact_coverage")
        if not isinstance(artifact_coverage, dict):
            errors.append("data_manifests.artifact_coverage must be an object")
        elif artifact_coverage.get("expected") != artifact_coverage.get("covered"):
            errors.append("data_manifests.artifact_coverage must cover every expected artifact")

    parity = _require_pass_gate(payload, "jax_gk_parity", errors)
    if parity:
        if not _positive_int(parity.get("parity_artifacts")):
            errors.append("jax_gk_parity.parity_artifacts must be a positive integer")
        cases = set(parity.get("required_cases", ()))
        backends = set(parity.get("required_backends", ()))
        if not REQUIRED_JAX_CASES.issubset(cases):
            errors.append("jax_gk_parity.required_cases must include the release CPU/GPU campaign cases")
        if not REQUIRED_JAX_BACKENDS.issubset(backends):
            errors.append("jax_gk_parity.required_backends must include cpu and gpu")
        entries = parity.get("entries")
        if not isinstance(entries, list):
            errors.append("jax_gk_parity.entries must be a list")
        else:
            observed_pairs = {(entry.get("case"), entry.get("backend")) for entry in entries if isinstance(entry, dict)}
            missing_pairs = {
                (case, backend)
                for case in REQUIRED_JAX_CASES
                for backend in REQUIRED_JAX_BACKENDS
                if (case, backend) not in observed_pairs
            }
            if missing_pairs:
                errors.append("jax_gk_parity.entries must include every required case/backend pair")

    traceability = _require_pass_gate(payload, "physics_traceability", errors)
    if traceability:
        if not _positive_int(traceability.get("total")):
            errors.append("physics_traceability.total must be a positive integer")
        if not _non_negative_int(traceability.get("open_fidelity_gaps")):
            errors.append("physics_traceability.open_fidelity_gaps must be a non-negative integer")
        if not _non_negative_int(traceability.get("public_claim_blocked")):
            errors.append("physics_traceability.public_claim_blocked must be a non-negative integer")
        if traceability.get("public_claim_blocked", 0) < traceability.get("open_fidelity_gaps", 0):
            errors.append("physics_traceability must block every open fidelity gap from public claims")

    admitted = tuple(
        gate for gate in REQUIRED_GATES if isinstance(payload.get(gate), dict) and payload[gate].get("status") == "pass"
    )
    return ReleaseEvidenceAdmission(
        status="pass" if not errors else "fail",
        errors=tuple(errors),
        report_sha256=report_sha256,
        admitted_gates=admitted,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate top-level SCPN-CONTROL release evidence")
    parser.add_argument("report", type=Path)
    parser.add_argument("--json-out", action="store_true")
    args = parser.parse_args(argv)

    result = validate_release_evidence(args.report)
    payload = {
        "schema_version": RELEASE_EVIDENCE_SCHEMA_VERSION,
        "status": result.status,
        "errors": list(result.errors),
        "report_sha256": result.report_sha256,
        "admitted_gates": list(result.admitted_gates),
    }
    if args.json_out:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Release evidence: {result.status}")
        if result.report_sha256 is not None:
            print(f"Report SHA-256: {result.report_sha256}")
        for error in result.errors:
            print(f"ERROR {error}")
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
