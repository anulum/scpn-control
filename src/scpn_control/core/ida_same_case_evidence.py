# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — IDA same-case evidence admission
"""Validate FUSION-owned IDA same-case evidence at the CONTROL boundary.

The validator distinguishes an authentic, internally consistent benchmark
artifact from evidence that is sufficient for control admission.  The v2
FUSION report records an integration-observed DIII-D-like case, so a valid
artifact remains blocked even when every byte and threshold projection checks
out.  No Grad-Shafranov mathematics is duplicated here.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FUSION_SCHEMA_VERSION = "scpn-fusion.ida-same-case-evidence.v2"
CONTROL_SCHEMA_VERSION = "scpn-control.ida-same-case-admission.v1"
BENCHMARK_ID = "DIII-D-IDA-FB-JAX-B"
FUSION_BLOCKED_STATUS = "blocked_same_case_evidence"
CONTROL_BLOCKED_STATUS = "blocked_external_same_case_evidence"
SOLVER_ID = "scpn_fusion.core.jax_free_boundary_predictive.solve_predictive_equilibrium_diff"
CLAIM_FIELDS = (
    "control_admission",
    "facility_validation",
    "pcs_deployment",
    "scientific_validation",
    "safety_admission",
)
THRESHOLDS: dict[str, float] = {
    "coil_gradient_relative_error_max": 5.0e-2,
    "gradient_smoothness_ratio_max": 2.5e-1,
    "latency_p95_ms_max": 20.0,
    "profile_gradient_relative_error_max": 1.0e-2,
    "psi_n_rmse_max": 5.0e-2,
    "relative_current_error_max": 5.0e-2,
    "relative_nonlinear_residual_rms_max": 5.0e-2,
}
SOURCE_PATHS = {
    "benchmark": "validation/benchmark_ida_same_case.py",
    "freegs_public_case_runner": ("validation/benchmark_freegs_public_example_reconstruction.py"),
    "profile_basis": "src/scpn_fusion/core/jax_profile_basis.py",
    "solver": "src/scpn_fusion/core/jax_free_boundary_predictive.py",
    "compiled_forward": "src/scpn_fusion/core/jax_predictive_forward_compiled.py",
    "o_point": "src/scpn_fusion/core/jax_o_point.py",
    "x_point": "src/scpn_fusion/core/jax_x_point.py",
}
REQUIRED_FUSION_BLOCKERS = {
    "collaborator_solver_reference_not_bound",
    "facility_validation_not_bound",
    "pcs_and_safety_programmes_not_bound",
    "statistically_held_out_case_missing",
}
_TOP_LEVEL_FIELDS = {
    "benchmark_id",
    "blockers",
    "case_role_contract",
    "cases",
    "claim_boundary",
    "environment",
    "generated_at",
    "payload_sha256",
    "schema_version",
    "solver_contract",
    "source_artifacts",
    "status",
    "thresholds",
}
_THRESHOLD_RESULT_FIELDS = {
    "gradient_audit",
    "latency",
    "psi_n_rmse",
    "relative_current_error",
    "relative_nonlinear_residual_rms",
}
_SHA256_LENGTHS = frozenset({64})
_GIT_OID_LENGTHS = frozenset({40, 64})


class IDASameCaseEvidenceError(ValueError):
    """Raised when IDA same-case evidence is malformed or inconsistent."""


@dataclass(frozen=True, slots=True)
class IDASameCaseAdmission:
    """Bounded CONTROL interpretation of one valid FUSION evidence artifact."""

    schema_version: str
    status: str
    artifact_valid: bool
    source_verified: bool
    admitted: bool
    benchmark_id: str
    source_payload_sha256: str
    source_commit: str
    evaluation_case_id: str
    evaluation_metrics: dict[str, float]
    threshold_results: dict[str, bool]
    blockers: tuple[str, ...]
    claim_boundary: tuple[tuple[str, bool], ...]

    def as_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible admission record."""
        return {
            "admitted": self.admitted,
            "artifact_valid": self.artifact_valid,
            "benchmark_id": self.benchmark_id,
            "blockers": list(self.blockers),
            "claim_boundary": dict(self.claim_boundary),
            "evaluation_case_id": self.evaluation_case_id,
            "evaluation_metrics": dict(self.evaluation_metrics),
            "schema_version": self.schema_version,
            "source_commit": self.source_commit,
            "source_payload_sha256": self.source_payload_sha256,
            "source_verified": self.source_verified,
            "status": self.status,
            "threshold_results": dict(self.threshold_results),
        }


def _duplicate_guard(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise IDASameCaseEvidenceError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def load_ida_same_case_report(path: str | Path) -> dict[str, Any]:
    """Load one JSON object while rejecting duplicate keys."""
    try:
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=_duplicate_guard)
    except (OSError, json.JSONDecodeError) as exc:
        raise IDASameCaseEvidenceError(f"unable to load IDA evidence: {exc}") from exc
    if not isinstance(payload, dict):
        raise IDASameCaseEvidenceError("IDA evidence root must be an object")
    return payload


def _canonical_json(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise IDASameCaseEvidenceError("IDA evidence must be canonical finite JSON") from exc


def _payload_sha256(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned["payload_sha256"] = ""
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _require_digest(
    value: object,
    *,
    field: str,
    lengths: frozenset[int] = _SHA256_LENGTHS,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise IDASameCaseEvidenceError(f"{field} is not a lowercase digest")
    return value


def _require_dict(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise IDASameCaseEvidenceError(f"{field} must be an object")
    return value


def _require_finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IDASameCaseEvidenceError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise IDASameCaseEvidenceError(f"{field} must be a finite number")
    return result


def _solver_contract() -> dict[str, Any]:
    return {
        "differentiated_inputs": [
            "coil_current_a",
            "pprime_coefficients_pa_per_wb",
            "ffprime_coefficients_t2_m2_per_wb",
        ],
        "reference_baseline": {
            "grid_shape": [129, 129],
            "latency_ms": 20.0,
            "picard_iterations": 10,
        },
        "conditioned_inputs": ["ip_target_a"],
        "solver_id": SOLVER_ID,
        "units": {
            "coil_current": "A",
            "ffprime": "T^2 m^2 / Wb",
            "pprime": "Pa / Wb",
            "psi": "Wb",
            "r": "m",
            "z": "m",
        },
    }


def _validate_source_contract(payload: dict[str, Any]) -> tuple[str, dict[str, str]]:
    source_artifacts = _require_dict(
        payload.get("source_artifacts"),
        field="source_artifacts",
    )
    if set(source_artifacts) != {*SOURCE_PATHS, "repository"}:
        raise IDASameCaseEvidenceError("source_artifacts do not match the IDA v2 contract")
    digests: dict[str, str] = {}
    for name, expected_path in SOURCE_PATHS.items():
        artifact = _require_dict(
            source_artifacts.get(name),
            field=f"source_artifacts.{name}",
        )
        if set(artifact) != {"path", "sha256"} or artifact.get("path") != expected_path:
            raise IDASameCaseEvidenceError(f"source_artifacts.{name} does not match the IDA v2 contract")
        digests[expected_path] = _require_digest(
            artifact.get("sha256"),
            field=f"source_artifacts.{name}.sha256",
        )
    repository = _require_dict(
        source_artifacts.get("repository"),
        field="source_artifacts.repository",
    )
    if set(repository) != {"git_commit", "path"} or repository.get("path") != ".":
        raise IDASameCaseEvidenceError("source_artifacts.repository does not match the IDA v2 contract")
    source_commit = _require_digest(
        repository.get("git_commit"),
        field="source_artifacts.repository.git_commit",
        lengths=_GIT_OID_LENGTHS,
    )
    return source_commit, digests


def _validate_gradient_audit(case: dict[str, Any], *, field: str) -> bool:
    audit = _require_dict(case.get("gradient_audit"), field=f"{field}.gradient_audit")
    if set(audit) != {"all_finite", "cotangent_sha256", "rows"}:
        raise IDASameCaseEvidenceError(f"{field}.gradient_audit fields do not match the IDA v2 contract")
    if audit.get("all_finite") is not True:
        return False
    _require_digest(
        audit.get("cotangent_sha256"),
        field=f"{field}.gradient_audit.cotangent_sha256",
    )
    rows = audit.get("rows")
    expected_inputs = [
        "coil_current_a",
        "pprime_coefficients_pa_per_wb",
        "ffprime_coefficients_t2_m2_per_wb",
    ]
    if not isinstance(rows, list) or len(rows) != len(expected_inputs):
        raise IDASameCaseEvidenceError(f"{field}.gradient_audit must contain three rows")
    all_passed = True
    for index, (row_value, expected_input) in enumerate(zip(rows, expected_inputs, strict=True)):
        row = _require_dict(
            row_value,
            field=f"{field}.gradient_audit.rows[{index}]",
        )
        if row.get("input") != expected_input or not isinstance(row.get("passed"), bool):
            raise IDASameCaseEvidenceError(f"{field}.gradient_audit.rows[{index}] identity is invalid")
        relative_error = _require_finite_float(
            row.get("relative_error"),
            field=f"{field}.gradient_audit.rows[{index}].relative_error",
        )
        relative_limit = _require_finite_float(
            row.get("relative_error_limit"),
            field=f"{field}.gradient_audit.rows[{index}].relative_error_limit",
        )
        smoothness = _require_finite_float(
            row.get("smoothness_ratio"),
            field=f"{field}.gradient_audit.rows[{index}].smoothness_ratio",
        )
        smoothness_limit = _require_finite_float(
            row.get("smoothness_ratio_limit"),
            field=f"{field}.gradient_audit.rows[{index}].smoothness_ratio_limit",
        )
        expected_relative_limit = (
            THRESHOLDS["coil_gradient_relative_error_max"]
            if expected_input == "coil_current_a"
            else THRESHOLDS["profile_gradient_relative_error_max"]
        )
        projected = bool(
            relative_limit == expected_relative_limit
            and smoothness_limit == THRESHOLDS["gradient_smoothness_ratio_max"]
            and relative_error <= relative_limit
            and smoothness <= smoothness_limit
        )
        if row["passed"] is not projected:
            raise IDASameCaseEvidenceError(f"{field}.gradient_audit.rows[{index}].passed is inconsistent")
        all_passed = all_passed and projected
    return all_passed


def _validate_case(case_value: object, *, role: str, index: int) -> dict[str, Any]:
    field = f"cases[{index}]"
    case = _require_dict(case_value, field=field)
    if case.get("role") != role or case.get("admitted") is not False:
        raise IDASameCaseEvidenceError(f"{field} role/admission contract is invalid")
    digests = _require_dict(case.get("digests"), field=f"{field}.digests")
    if not digests:
        raise IDASameCaseEvidenceError(f"{field}.digests must not be empty")
    for name, digest in digests.items():
        if not isinstance(name, str):
            raise IDASameCaseEvidenceError(f"{field}.digests keys must be strings")
        _require_digest(digest, field=f"{field}.digests.{name}")
    public_example = _require_dict(
        case.get("public_example"),
        field=f"{field}.public_example",
    )
    if set(public_example) != {"path", "sha256"} or not isinstance(public_example.get("path"), str):
        raise IDASameCaseEvidenceError(f"{field}.public_example is invalid")
    _require_digest(
        public_example.get("sha256"),
        field=f"{field}.public_example.sha256",
    )
    metrics = _require_dict(case.get("metrics"), field=f"{field}.metrics")
    psi_nrmse = _require_finite_float(
        metrics.get("psi_n_rmse"),
        field=f"{field}.metrics.psi_n_rmse",
    )
    current_error = _require_finite_float(
        metrics.get("relative_current_error"),
        field=f"{field}.metrics.relative_current_error",
    )
    residual = _require_finite_float(
        metrics.get("relative_nonlinear_residual_rms"),
        field=f"{field}.metrics.relative_nonlinear_residual_rms",
    )
    latency = _require_dict(case.get("latency"), field=f"{field}.latency")
    p95_ms = _require_finite_float(
        latency.get("p95_ms"),
        field=f"{field}.latency.p95_ms",
    )
    gradient_passed = _validate_gradient_audit(case, field=field)
    projected = {
        "gradient_audit": gradient_passed,
        "latency": p95_ms <= THRESHOLDS["latency_p95_ms_max"],
        "psi_n_rmse": psi_nrmse <= THRESHOLDS["psi_n_rmse_max"],
        "relative_current_error": (current_error <= THRESHOLDS["relative_current_error_max"]),
        "relative_nonlinear_residual_rms": (residual <= THRESHOLDS["relative_nonlinear_residual_rms_max"]),
    }
    threshold_results = _require_dict(
        case.get("threshold_results"),
        field=f"{field}.threshold_results",
    )
    if set(threshold_results) != _THRESHOLD_RESULT_FIELDS:
        raise IDASameCaseEvidenceError(f"{field}.threshold_results fields do not match the IDA v2 contract")
    if threshold_results != projected:
        raise IDASameCaseEvidenceError(f"{field}.threshold_results are inconsistent with measured values")
    if role == "evaluation_candidate" and case.get("grid_shape") != [129, 129]:
        raise IDASameCaseEvidenceError("evaluation_candidate grid must match the 129x129 reference contract")
    return case


def _verify_source_commit(
    fusion_root: Path,
    *,
    source_commit: str,
    source_digests: dict[str, str],
) -> None:
    if not (fusion_root / ".git").exists():
        raise IDASameCaseEvidenceError("fusion_root is not a Git working tree")
    for path, expected_digest in source_digests.items():
        try:
            completed = subprocess.run(
                ["git", "show", f"{source_commit}:{path}"],
                cwd=fusion_root,
                check=True,
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise IDASameCaseEvidenceError(f"unable to resolve {path} at source commit {source_commit}") from exc
        digest = hashlib.sha256(completed.stdout).hexdigest()
        if digest != expected_digest:
            raise IDASameCaseEvidenceError(f"source digest mismatch for {path} at {source_commit}")


def validate_ida_same_case_evidence(
    report_path: str | Path,
    *,
    fusion_root: str | Path | None = None,
) -> IDASameCaseAdmission:
    """Validate one FUSION report and return fail-closed CONTROL admission."""
    payload = load_ida_same_case_report(report_path)
    if set(payload) != _TOP_LEVEL_FIELDS:
        raise IDASameCaseEvidenceError("IDA evidence top-level fields do not match the v2 schema")
    if payload.get("schema_version") != FUSION_SCHEMA_VERSION:
        raise IDASameCaseEvidenceError("unsupported IDA evidence schema")
    if payload.get("benchmark_id") != BENCHMARK_ID:
        raise IDASameCaseEvidenceError("unexpected IDA benchmark_id")
    digest = _require_digest(payload.get("payload_sha256"), field="payload_sha256")
    if digest != _payload_sha256(payload):
        raise IDASameCaseEvidenceError("payload_sha256 does not match report content")
    if payload.get("thresholds") != THRESHOLDS:
        raise IDASameCaseEvidenceError("thresholds do not match the frozen v2 contract")
    if payload.get("solver_contract") != _solver_contract():
        raise IDASameCaseEvidenceError("solver_contract does not match the frozen v2 contract")
    if payload.get("claim_boundary") != {field: False for field in CLAIM_FIELDS}:
        raise IDASameCaseEvidenceError("claim_boundary must keep every promotion claim false")
    if payload.get("status") != FUSION_BLOCKED_STATUS:
        raise IDASameCaseEvidenceError("v2 integration evidence must remain blocked")
    blockers_value = payload.get("blockers")
    if (
        not isinstance(blockers_value, list)
        or not blockers_value
        or any(not isinstance(item, str) or not item for item in blockers_value)
    ):
        raise IDASameCaseEvidenceError("blockers must be sorted unique non-empty strings")
    if blockers_value != sorted(set(blockers_value)):
        raise IDASameCaseEvidenceError("blockers must be sorted unique non-empty strings")
    blockers = set(blockers_value)
    if not REQUIRED_FUSION_BLOCKERS.issubset(blockers):
        raise IDASameCaseEvidenceError("required FUSION claim-boundary blockers are missing")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 2:
        raise IDASameCaseEvidenceError("IDA evidence must contain exactly two cases")
    development = _validate_case(cases[0], role="development", index=0)
    evaluation = _validate_case(
        cases[1],
        role="evaluation_candidate",
        index=1,
    )
    role_contract = _require_dict(
        payload.get("case_role_contract"),
        field="case_role_contract",
    )
    if (
        role_contract.get("development_case_id") != development.get("case_id")
        or role_contract.get("evaluation_case_id") != evaluation.get("case_id")
        or role_contract.get("evaluation_previously_observed_during_integration") is not True
        or role_contract.get("statistically_held_out") is not False
    ):
        raise IDASameCaseEvidenceError("case_role_contract must preserve the integration-observed boundary")
    selection_lock = _require_dict(
        role_contract.get("selection_lock"),
        field="case_role_contract.selection_lock",
    )
    if (
        selection_lock.get("case_id") != evaluation.get("case_id")
        or selection_lock.get("created_before_execution") is not False
        or selection_lock.get("valid") is not False
        or "execution_preceding_selection_lock_missing" not in blockers
    ):
        raise IDASameCaseEvidenceError("selection lock must preserve the integration-observed blocker")
    environment = _require_dict(
        payload.get("environment"),
        field="environment",
    )
    if environment.get("x64_enabled") is not True:
        raise IDASameCaseEvidenceError("IDA same-case evidence must use JAX FP64")
    if environment.get("isolated_host") is not False or "isolated_latency_evidence_missing" not in blockers:
        raise IDASameCaseEvidenceError("non-isolated timing must retain its admission blocker")
    evaluation_thresholds = _require_dict(
        evaluation.get("threshold_results"),
        field="cases[1].threshold_results",
    )
    for name, passed in evaluation_thresholds.items():
        threshold_blocker = f"evaluation_threshold_failed:{name}"
        if (passed is False) != (threshold_blocker in blockers):
            raise IDASameCaseEvidenceError(f"threshold blocker projection is inconsistent for {name}")
    source_commit, source_digests = _validate_source_contract(payload)
    source_verified = fusion_root is not None
    if fusion_root is not None:
        _verify_source_commit(
            Path(fusion_root),
            source_commit=source_commit,
            source_digests=source_digests,
        )
    control_blockers = blockers | {"control_admission_not_granted"}
    if not source_verified:
        control_blockers.add("upstream_source_tree_not_verified")
    metrics = _require_dict(
        evaluation.get("metrics"),
        field="cases[1].metrics",
    )
    evaluation_metrics = {
        name: _require_finite_float(metrics.get(name), field=f"cases[1].metrics.{name}")
        for name in (
            "psi_n_rmse",
            "relative_current_error",
            "relative_nonlinear_residual_rms",
        )
    }
    latency = _require_dict(
        evaluation.get("latency"),
        field="cases[1].latency",
    )
    evaluation_metrics["latency_p95_ms"] = _require_finite_float(
        latency.get("p95_ms"),
        field="cases[1].latency.p95_ms",
    )
    threshold_results = _require_dict(
        evaluation.get("threshold_results"),
        field="cases[1].threshold_results",
    )
    return IDASameCaseAdmission(
        schema_version=CONTROL_SCHEMA_VERSION,
        status=CONTROL_BLOCKED_STATUS,
        artifact_valid=True,
        source_verified=source_verified,
        admitted=False,
        benchmark_id=BENCHMARK_ID,
        source_payload_sha256=digest,
        source_commit=source_commit,
        evaluation_case_id=str(evaluation["case_id"]),
        evaluation_metrics=evaluation_metrics,
        threshold_results={name: bool(threshold_results[name]) for name in sorted(_THRESHOLD_RESULT_FIELDS)},
        blockers=tuple(sorted(control_blockers)),
        claim_boundary=tuple((field, False) for field in CLAIM_FIELDS),
    )
