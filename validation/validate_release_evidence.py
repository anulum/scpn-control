# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Release Evidence Validation
"""Check declarations in a top-level release report before publication.

This reader hashes the exact input bytes and checks mandatory gate summaries.
It does not reopen referenced artifacts, recompute their digests or metrics,
authenticate a producer, run solvers, or grant facility/control readiness.
Even a passing declaration may describe local regression with failed realtime
admission; its production claim flag must remain false. Extra fields are
accepted, but duplicate keys and nonfinite floating tokens at any depth fail.

Run ``python validation/validate_release_evidence.py REPORT --json-out`` or the
registered ``scpn-control validate-release-evidence REPORT --json-out``. The
standalone reader uses only the standard library and prints rather than writes
reports. Paths are caller-relative and symlinks are followed; no size/depth
budget or filesystem containment is supplied by this reader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

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
    """Frozen result of declaration checks on one local report.

    Attributes
    ----------
    status : str
        ``pass`` exactly when no declaration or read findings were collected.
    errors : tuple[str, ...]
        Findings in deterministic gate/check order; no exception traceback.
    report_sha256 : str or None
        SHA-256 of exact input bytes after successful JSON-object decoding;
        ``None`` on read/decode failure. This does not authenticate those bytes.
    admitted_gates : tuple[str, ...]
        Required gates declaring ``status=pass``, in the required gate order.
        A gate can appear here while field findings make overall status fail.
    """

    status: str
    errors: tuple[str, ...]
    report_sha256: str | None
    admitted_gates: tuple[str, ...]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Refuse shadowed JSON keys in every object before field inspection."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json_float(token: str) -> float:
    """Refuse nonfinite constants and overflowing exponents at any JSON depth."""
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"nonfinite JSON number: {token}")
    return value


def _load_report(path: str | Path) -> tuple[dict[str, Any], str]:
    """Read UTF-8 JSON once, refuse invalid objects, and hash the same exact bytes."""
    blob = Path(path).read_bytes()
    payload = json.loads(
        blob.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_float=_parse_json_float,
        parse_constant=_parse_json_float,
    )
    if not isinstance(payload, dict):
        raise ValueError("release evidence report root must be a JSON object")
    return payload, hashlib.sha256(blob).hexdigest()


def _positive_int(value: object) -> bool:
    """Recognise a positive JSON integer while refusing booleans and floats."""
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _non_negative_int(value: object) -> bool:
    """Recognise a nonnegative JSON integer while refusing booleans and floats."""
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _sha256_hex(value: object) -> bool:
    """Check lowercase SHA-256 spelling without opening or hashing any referenced object."""
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _string_list(value: object) -> TypeGuard[list[str]]:
    """Recognise a JSON list containing only strings before set comparisons."""
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _require_pass_gate(payload: dict[str, Any], gate: str, errors: list[str]) -> dict[str, Any]:
    """Collect object/status findings and keep a valid object for remaining checks."""
    section = payload.get(gate)
    if not isinstance(section, dict):
        errors.append(f"{gate} must be an object")
        return {}
    status = section.get("status")
    if status != "pass":
        errors.append(f"{gate}.status must be 'pass', got {status!r}")
    return section


def validate_release_evidence(path: str | Path) -> ReleaseEvidenceAdmission:
    """Inspect mandatory declarations in a ``scpn-control validate`` report.

    Parameters
    ----------
    path : str or pathlib.Path
        Local UTF-8 JSON file, resolved by the caller's working directory.

    Returns
    -------
    ReleaseEvidenceAdmission
        Structured read/JSON/type/domain findings and exact report-byte digest.
        Missing or malformed required sections fail; unrelated extra fields are
        not validated beyond duplicate/nonfinite JSON rejection. Manifest
        coverage counts must be nonnegative integers, equal, and missing empty.
        Declared JAX lists must cover all three cases on both CPU and GPU.
        Local runtime ``admission_status=fail`` is permitted with no production
        claim; production runtime summaries must declare pass and zero errors.
        Digests and AOT case labels are syntax, not proof/artifact verification.

    Examples
    --------
    Invalid bytes produce a refusal without relying on a reference corpus.

    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as directory:
    ...     report = Path(directory) / "invalid.json"
    ...     _ = report.write_text("[]", encoding="utf-8")
    ...     result = validate_release_evidence(report)
    >>> result.status, result.report_sha256, result.admitted_gates
    ('fail', None, ())
    """
    errors: list[str] = []
    try:
        payload, report_sha256 = _load_report(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
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
        else:
            expected = artifact_coverage.get("expected")
            covered = artifact_coverage.get("covered")
            if not _non_negative_int(expected) or not _non_negative_int(covered):
                errors.append("data_manifests.artifact_coverage counts must be non-negative integers")
            elif expected != covered:
                errors.append("data_manifests.artifact_coverage must cover every expected artifact")
            if artifact_coverage.get("missing") != []:
                errors.append("data_manifests.artifact_coverage.missing must be empty")

    parity = _require_pass_gate(payload, "jax_gk_parity", errors)
    if parity:
        if not _positive_int(parity.get("parity_artifacts")):
            errors.append("jax_gk_parity.parity_artifacts must be a positive integer")
        cases = parity.get("required_cases")
        backends = parity.get("required_backends")
        if not _string_list(cases) or not REQUIRED_JAX_CASES.issubset(cases):
            errors.append("jax_gk_parity.required_cases must include the release CPU/GPU campaign cases")
        if not _string_list(backends) or not REQUIRED_JAX_BACKENDS.issubset(backends):
            errors.append("jax_gk_parity.required_backends must include cpu and gpu")
        entries = parity.get("entries")
        if not isinstance(entries, list):
            errors.append("jax_gk_parity.entries must be a list")
        elif not all(
            isinstance(entry, dict) and isinstance(entry.get("case"), str) and isinstance(entry.get("backend"), str)
            for entry in entries
        ):
            errors.append("jax_gk_parity.entries must contain objects with string case and backend")
        else:
            observed_pairs = {(entry["case"], entry["backend"]) for entry in entries}
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
        if (
            _non_negative_int(traceability.get("public_claim_blocked"))
            and _non_negative_int(traceability.get("open_fidelity_gaps"))
            and traceability["public_claim_blocked"] < traceability["open_fidelity_gaps"]
        ):
            errors.append("physics_traceability must block every open fidelity gap from public claims")

    multi_shot = _require_pass_gate(payload, "multi_shot_campaign", errors)
    if multi_shot:
        admitted_surfaces = multi_shot.get("admitted_surfaces")
        if not _string_list(admitted_surfaces):
            errors.append("multi_shot_campaign.admitted_surfaces must be a list of strings")
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
        if not isinstance(evidence_class, str) or evidence_class not in RUNTIME_ADMISSION_EVIDENCE_CLASSES:
            errors.append("runtime_admission.benchmark_evidence_class must be a recognised evidence class")
        production_claim_allowed = runtime_admission.get("production_claim_allowed")
        if not isinstance(production_claim_allowed, bool):
            errors.append("runtime_admission.production_claim_allowed must be a boolean")
        elif evidence_class == "local_regression" and production_claim_allowed:
            errors.append("local runtime admission evidence must not allow production benchmark claims")
        elif evidence_class == "production_benchmark" and production_claim_allowed is not True:
            errors.append("production runtime admission evidence must allow production benchmark claims")
        admission_status = runtime_admission.get("admission_status")
        if not isinstance(admission_status, str) or admission_status not in {"pass", "fail"}:
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
        if not isinstance(evidence_class, str) or evidence_class not in NATIVE_FORMAL_EVIDENCE_CLASSES:
            errors.append("native_formal_certificate.benchmark_evidence_class must be a recognised evidence class")
        production_claim_allowed = native_formal.get("production_claim_allowed")
        if not isinstance(production_claim_allowed, bool):
            errors.append("native_formal_certificate.production_claim_allowed must be a boolean")
        elif evidence_class == "local_regression" and production_claim_allowed:
            errors.append("local native formal evidence must not allow production benchmark claims")
        elif evidence_class == "production_benchmark" and production_claim_allowed is not True:
            errors.append("production native formal evidence must allow production benchmark claims")
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
    """Print admission JSON or text; return zero for pass and one for findings.

    Parameters
    ----------
    argv : list[str] or None
        Argparse arguments: required local report and optional ``--json-out``.
        ``None`` reads process arguments. Parser errors retain SystemExit(2).

    Returns
    -------
    int
        Admission exit code. Findings and byte digest are printed to stdout;
        no output artifact is written and no upstream validation is executed.
    """
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
