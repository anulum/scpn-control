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
REQUIRED_GATES = (
    "data_manifests",
    "jax_gk_parity",
    "physics_traceability",
    "multi_shot_campaign",
    "runtime_admission",
    "native_formal_certificate",
)
REQUIRED_JAX_CASES = frozenset({"cyclone_base_case", "tem_kinetic_electron", "stable_mode"})
REQUIRED_JAX_BACKENDS = frozenset({"cpu", "gpu"})
NATIVE_FORMAL_EVIDENCE_CLASSES = frozenset({"local_regression", "production_benchmark"})
RUNTIME_ADMISSION_EVIDENCE_CLASSES = frozenset({"local_regression", "production_benchmark"})


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


def _sha256_hex(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


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

    multi_shot = _require_pass_gate(payload, "multi_shot_campaign", errors)
    if multi_shot:
        admitted_surfaces = multi_shot.get("admitted_surfaces")
        if not isinstance(admitted_surfaces, list):
            errors.append("multi_shot_campaign.admitted_surfaces must be a list")
        elif set(admitted_surfaces) != {"python", "pyo3", "rust"}:
            errors.append("multi_shot_campaign.admitted_surfaces must include python, pyo3, and rust")
        if multi_shot.get("pyo3_status") != "ok":
            errors.append("multi_shot_campaign.pyo3_status must be 'ok'")
        if not _sha256_hex(multi_shot.get("python_report_sha256")):
            errors.append("multi_shot_campaign.python_report_sha256 must be a SHA-256 hex digest")
        if not _sha256_hex(multi_shot.get("rust_report_sha256")):
            errors.append("multi_shot_campaign.rust_report_sha256 must be a SHA-256 hex digest")
        if not _sha256_hex(multi_shot.get("python_payload_sha256")):
            errors.append("multi_shot_campaign.python_payload_sha256 must be a SHA-256 hex digest")
        if not _sha256_hex(multi_shot.get("rust_payload_sha256")):
            errors.append("multi_shot_campaign.rust_payload_sha256 must be a SHA-256 hex digest")
        if not _positive_int(multi_shot.get("minimum_digest_count")):
            errors.append("multi_shot_campaign.minimum_digest_count must be a positive integer")
        production_claim_allowed = multi_shot.get("production_claim_allowed")
        if not isinstance(production_claim_allowed, bool):
            errors.append("multi_shot_campaign.production_claim_allowed must be a boolean")
        multi_shot_errors = multi_shot.get("errors")
        if multi_shot_errors != []:
            errors.append("multi_shot_campaign.errors must be empty")

    runtime_admission = _require_pass_gate(payload, "runtime_admission", errors)
    if runtime_admission:
        if not _sha256_hex(runtime_admission.get("report_sha256")):
            errors.append("runtime_admission.report_sha256 must be a SHA-256 hex digest")
        if not _sha256_hex(runtime_admission.get("payload_sha256")):
            errors.append("runtime_admission.payload_sha256 must be a SHA-256 hex digest")
        evidence_class = runtime_admission.get("benchmark_evidence_class")
        if evidence_class not in RUNTIME_ADMISSION_EVIDENCE_CLASSES:
            errors.append("runtime_admission.benchmark_evidence_class must be a recognised evidence class")
        production_claim_allowed = runtime_admission.get("production_claim_allowed")
        if not isinstance(production_claim_allowed, bool):
            errors.append("runtime_admission.production_claim_allowed must be a boolean")
        elif evidence_class == "local_regression" and production_claim_allowed:
            errors.append("local runtime admission evidence must not allow production benchmark claims")
        elif evidence_class == "production_benchmark" and production_claim_allowed is not True:
            errors.append("production runtime admission evidence must allow production benchmark claims")
        admission_status = runtime_admission.get("admission_status")
        if admission_status not in {"pass", "fail"}:
            errors.append("runtime_admission.admission_status must be 'pass' or 'fail'")
        elif evidence_class == "production_benchmark" and admission_status != "pass":
            errors.append("production runtime admission evidence must pass strict runtime admission")
        if not _non_negative_int(runtime_admission.get("admission_error_count")):
            errors.append("runtime_admission.admission_error_count must be a non-negative integer")
        elif evidence_class == "production_benchmark" and runtime_admission.get("admission_error_count") != 0:
            errors.append("production runtime admission evidence must not carry admission errors")
        if not _positive_int(runtime_admission.get("samples")):
            errors.append("runtime_admission.samples must be a positive integer")
        runtime_errors = runtime_admission.get("errors")
        if runtime_errors != []:
            errors.append("runtime_admission.errors must be empty")

    native_formal = _require_pass_gate(payload, "native_formal_certificate", errors)
    if native_formal:
        admitted_cases = native_formal.get("admitted_cases")
        if not isinstance(admitted_cases, list) or not admitted_cases:
            errors.append("native_formal_certificate.admitted_cases must be a non-empty list")
        elif not all(isinstance(case, str) and ":aot_certificate:" in case for case in admitted_cases):
            errors.append("native_formal_certificate.admitted_cases must contain AOT certificate case labels")
        if not _sha256_hex(native_formal.get("certificate_assumption_sha256")):
            errors.append("native_formal_certificate.certificate_assumption_sha256 must be a SHA-256 hex digest")
        if not _sha256_hex(native_formal.get("report_sha256")):
            errors.append("native_formal_certificate.report_sha256 must be a SHA-256 hex digest")
        evidence_class = native_formal.get("benchmark_evidence_class")
        if evidence_class not in NATIVE_FORMAL_EVIDENCE_CLASSES:
            errors.append("native_formal_certificate.benchmark_evidence_class must be a recognised evidence class")
        production_claim_allowed = native_formal.get("production_claim_allowed")
        if not isinstance(production_claim_allowed, bool):
            errors.append("native_formal_certificate.production_claim_allowed must be a boolean")
        elif evidence_class == "local_regression" and production_claim_allowed:
            errors.append("local native formal evidence must not allow production benchmark claims")
        native_errors = native_formal.get("errors")
        if native_errors != []:
            errors.append("native_formal_certificate.errors must be empty")

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
