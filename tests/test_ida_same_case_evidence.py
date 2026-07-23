# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — IDA same-case evidence admission tests
"""Tests for the public FUSION-to-CONTROL IDA evidence boundary."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

import scpn_control.core.ida_same_case_evidence as evidence
from scpn_control.cli import main


def _digest(character: str = "a") -> str:
    return character * 64


def _gradient_row(name: str, *, passed: bool) -> dict[str, Any]:
    relative_limit = (
        evidence.THRESHOLDS["coil_gradient_relative_error_max"]
        if name == "coil_current_a"
        else evidence.THRESHOLDS["profile_gradient_relative_error_max"]
    )
    return {
        "autodiff": 1.0,
        "epsilon": 0.01,
        "finite_difference": 1.0,
        "index": 0,
        "input": name,
        "passed": passed,
        "relative_error": 0.0 if passed else relative_limit * 2.0,
        "relative_error_limit": relative_limit,
        "smoothness_ratio": 0.0,
        "smoothness_ratio_limit": evidence.THRESHOLDS["gradient_smoothness_ratio_max"],
    }


def _case(role: str) -> dict[str, Any]:
    evaluation = role == "evaluation_candidate"
    gradient_passed = not evaluation
    metrics = {
        "candidate_axis_wb": -0.6,
        "candidate_boundary_wb": -0.1,
        "candidate_current_a": -2.3e6,
        "psi_max_abs_error_wb": 1.4,
        "psi_rmse_wb": 1.1,
        "psi_n_rmse": 2.0 if evaluation else 0.01,
        "raw_psi_span_nrmse": 0.2 if evaluation else 0.01,
        "reference_axis_wb": -0.66,
        "reference_boundary_wb": -0.12,
        "reference_current_a": -1.5e6,
        "relative_current_error": 0.5 if evaluation else 0.01,
        "relative_nonlinear_residual_rms": 0.03 if evaluation else 0.01,
    }
    p95_ms = 2100.0 if evaluation else 10.0
    threshold_results = {
        "gradient_audit": gradient_passed,
        "latency": not evaluation,
        "psi_n_rmse": not evaluation,
        "relative_current_error": not evaluation,
        "relative_nonlinear_residual_rms": True,
    }
    return {
        "admitted": False,
        "case_id": f"{role}-case",
        "digests": {"candidate_psi_sha256": _digest("b")},
        "freegs_version": "0.8.2",
        "gradient_audit": {
            "all_finite": True,
            "cotangent_sha256": _digest("c"),
            "rows": [
                _gradient_row("coil_current_a", passed=gradient_passed),
                _gradient_row(
                    "pprime_coefficients_pa_per_wb",
                    passed=gradient_passed,
                ),
                _gradient_row(
                    "ffprime_coefficients_t2_m2_per_wb",
                    passed=gradient_passed,
                ),
            ],
        },
        "grid_shape": [129, 129] if evaluation else [65, 65],
        "input_contract": {},
        "latency": {"p95_ms": p95_ms},
        "machine_class": "DIIID" if evaluation else "TestTokamak",
        "metrics": metrics,
        "public_example": {
            "path": f"data/{role}.py",
            "sha256": _digest("d"),
        },
        "reference_mask_point_count": 100,
        "role": role,
        "threshold_results": threshold_results,
    }


def _report() -> dict[str, Any]:
    development = _case("development")
    evaluation = _case("evaluation_candidate")
    payload: dict[str, Any] = {
        "benchmark_id": evidence.BENCHMARK_ID,
        "blockers": sorted(
            evidence.REQUIRED_FUSION_BLOCKERS
            | {
                "evaluation_threshold_failed:gradient_audit",
                "evaluation_threshold_failed:latency",
                "evaluation_threshold_failed:psi_n_rmse",
                "evaluation_threshold_failed:relative_current_error",
                "execution_preceding_selection_lock_missing",
                "isolated_latency_evidence_missing",
            }
        ),
        "case_role_contract": {
            "development_case_id": development["case_id"],
            "evaluation_case_id": evaluation["case_id"],
            "evaluation_previously_observed_during_integration": True,
            "selection_lock": {
                "case_id": evaluation["case_id"],
                "created_before_execution": False,
                "path": None,
                "sha256": None,
                "valid": False,
            },
            "statistically_held_out": False,
        },
        "cases": [development, evaluation],
        "claim_boundary": {field: False for field in evidence.CLAIM_FIELDS},
        "environment": {
            "isolated_host": False,
            "x64_enabled": True,
        },
        "generated_at": "2026-07-23T03:31:48Z",
        "payload_sha256": "",
        "schema_version": evidence.FUSION_SCHEMA_VERSION,
        "solver_contract": evidence._solver_contract(),
        "source_artifacts": {
            name: {"path": path, "sha256": _digest("e")} for name, path in evidence.SOURCE_PATHS.items()
        }
        | {
            "repository": {
                "git_commit": "f" * 40,
                "path": ".",
            }
        },
        "status": evidence.FUSION_BLOCKED_STATUS,
        "thresholds": dict(evidence.THRESHOLDS),
    }
    _seal(payload)
    return payload


def _seal(payload: dict[str, Any]) -> None:
    payload["payload_sha256"] = evidence._payload_sha256(payload)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_valid_report_is_authentic_but_never_admitted(tmp_path: Path) -> None:
    """Accept the artifact contract while retaining every CONTROL blocker."""
    path = tmp_path / "report.json"
    _write_report(path, _report())

    admission = evidence.validate_ida_same_case_evidence(path)
    payload = admission.as_dict()

    assert admission.artifact_valid is True
    assert admission.source_verified is False
    assert admission.admitted is False
    assert admission.status == evidence.CONTROL_BLOCKED_STATUS
    assert admission.evaluation_metrics["psi_n_rmse"] == 2.0
    assert admission.threshold_results["relative_nonlinear_residual_rms"] is True
    assert "upstream_source_tree_not_verified" in admission.blockers
    assert payload["claim_boundary"] == {field: False for field in evidence.CLAIM_FIELDS}


def test_source_commit_bytes_are_verified_in_git(
    tmp_path: Path,
) -> None:
    """Resolve every bound FUSION source from the exact recorded commit."""
    fusion_root = tmp_path / "fusion"
    fusion_root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=fusion_root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=fusion_root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Evidence Test"],
        cwd=fusion_root,
        check=True,
    )
    payload = _report()
    for index, (name, relative_path) in enumerate(
        evidence.SOURCE_PATHS.items(),
        start=1,
    ):
        source = fusion_root / relative_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"source-{index}\n".encode())
        payload["source_artifacts"][name]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    subprocess.run(["git", "add", "."], cwd=fusion_root, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "fixture"],
        cwd=fusion_root,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=fusion_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload["source_artifacts"]["repository"]["git_commit"] = commit
    _seal(payload)
    path = tmp_path / "report.json"
    _write_report(path, payload)

    admission = evidence.validate_ida_same_case_evidence(
        path,
        fusion_root=fusion_root,
    )
    assert admission.source_verified is True
    assert "upstream_source_tree_not_verified" not in admission.blockers

    first_path = next(iter(evidence.SOURCE_PATHS.values()))
    payload["source_artifacts"]["benchmark"]["sha256"] = _digest("0")
    _seal(payload)
    _write_report(path, payload)
    with pytest.raises(
        evidence.IDASameCaseEvidenceError,
        match=f"source digest mismatch for {first_path}",
    ):
        evidence.validate_ida_same_case_evidence(
            path,
            fusion_root=fusion_root,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update(schema_version="bad"), "unsupported"),
        (lambda payload: payload.update(benchmark_id="bad"), "benchmark_id"),
        (lambda payload: payload.update(status="accepted"), "must remain blocked"),
        (lambda payload: payload.update(thresholds={}), "thresholds"),
        (lambda payload: payload.update(solver_contract={}), "solver_contract"),
        (lambda payload: payload.update(cases=[]), "exactly two"),
        (
            lambda payload: payload["claim_boundary"].update(control_admission=True),
            "claim_boundary",
        ),
        (
            lambda payload: payload["case_role_contract"].update(statistically_held_out=True),
            "integration-observed",
        ),
    ],
)
def test_schema_and_claim_drift_is_rejected(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    """Reject schema drift and every attempted promotion."""
    payload = _report()
    mutate(payload)
    _seal(payload)
    path = tmp_path / "report.json"
    _write_report(path, payload)
    with pytest.raises(evidence.IDASameCaseEvidenceError, match=message):
        evidence.validate_ida_same_case_evidence(path)


def test_tamper_nonfinite_and_threshold_projection_are_rejected(
    tmp_path: Path,
) -> None:
    """Reject digest tamper, non-finite metrics, and forged gate booleans."""
    path = tmp_path / "report.json"
    payload = _report()
    _write_report(path, payload)
    payload["generated_at"] = "tampered"
    _write_report(path, payload)
    with pytest.raises(evidence.IDASameCaseEvidenceError, match="payload_sha256"):
        evidence.validate_ida_same_case_evidence(path)

    payload = _report()
    payload["cases"][1]["metrics"]["psi_n_rmse"] = float("inf")
    payload["payload_sha256"] = "a" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(evidence.IDASameCaseEvidenceError, match="finite JSON"):
        evidence.validate_ida_same_case_evidence(path)

    payload = _report()
    payload["cases"][1]["threshold_results"]["latency"] = True
    _seal(payload)
    _write_report(path, payload)
    with pytest.raises(evidence.IDASameCaseEvidenceError, match="inconsistent"):
        evidence.validate_ida_same_case_evidence(path)


def test_loader_rejects_duplicate_nonobject_and_missing_files(
    tmp_path: Path,
) -> None:
    """Reject ambiguous JSON roots and unreadable report paths."""
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(evidence.IDASameCaseEvidenceError, match="duplicate"):
        evidence.load_ida_same_case_report(duplicate)

    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    with pytest.raises(evidence.IDASameCaseEvidenceError, match="root"):
        evidence.load_ida_same_case_report(array)

    with pytest.raises(evidence.IDASameCaseEvidenceError, match="unable to load"):
        evidence.load_ida_same_case_report(tmp_path / "missing.json")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "lowercase digest"),
        ("a", "lowercase digest"),
        ("g" * 64, "lowercase digest"),
    ],
)
def test_digest_guard_rejects_type_length_and_alphabet(
    value: object,
    message: str,
) -> None:
    """Exercise each digest-format failure independently."""
    with pytest.raises(evidence.IDASameCaseEvidenceError, match=message):
        evidence._require_digest(value, field="digest")


def test_low_level_object_and_number_guards() -> None:
    """Reject non-object, boolean, nonnumeric, and non-finite values."""
    with pytest.raises(evidence.IDASameCaseEvidenceError, match="object"):
        evidence._require_dict([], field="row")
    for value in (True, "1.0", float("inf")):
        with pytest.raises(evidence.IDASameCaseEvidenceError, match="finite"):
            evidence._require_finite_float(value, field="metric")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["source_artifacts"].pop("x_point"),
            "source_artifacts do not match",
        ),
        (
            lambda payload: payload["source_artifacts"]["solver"].update(path="wrong.py"),
            r"source_artifacts\.solver",
        ),
        (
            lambda payload: payload["source_artifacts"]["repository"].update(path="wrong"),
            r"source_artifacts\.repository",
        ),
    ],
)
def test_source_contract_shape_drift_is_rejected(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], object],
    message: str,
) -> None:
    """Reject missing, moved, or malformed source bindings."""
    payload = _report()
    mutate(payload)
    _seal(payload)
    path = tmp_path / "report.json"
    _write_report(path, payload)
    with pytest.raises(evidence.IDASameCaseEvidenceError, match=message):
        evidence.validate_ida_same_case_evidence(path)


def test_gradient_audit_shape_identity_and_projection_guards() -> None:
    """Reject malformed gradient rows and forged per-row pass states."""
    case = _case("development")
    case["gradient_audit"]["unexpected"] = True
    with pytest.raises(evidence.IDASameCaseEvidenceError, match="fields"):
        evidence._validate_gradient_audit(case, field="case")

    case = _case("development")
    case["gradient_audit"]["all_finite"] = False
    assert evidence._validate_gradient_audit(case, field="case") is False

    case = _case("development")
    case["gradient_audit"]["rows"] = "bad"
    with pytest.raises(evidence.IDASameCaseEvidenceError, match="three rows"):
        evidence._validate_gradient_audit(case, field="case")

    case = _case("development")
    case["gradient_audit"]["rows"] = case["gradient_audit"]["rows"][:2]
    with pytest.raises(evidence.IDASameCaseEvidenceError, match="three rows"):
        evidence._validate_gradient_audit(case, field="case")

    case = _case("development")
    case["gradient_audit"]["rows"][0] = []
    with pytest.raises(evidence.IDASameCaseEvidenceError, match="object"):
        evidence._validate_gradient_audit(case, field="case")

    case = _case("development")
    case["gradient_audit"]["rows"][0]["input"] = "wrong"
    with pytest.raises(evidence.IDASameCaseEvidenceError, match="identity"):
        evidence._validate_gradient_audit(case, field="case")

    case = _case("development")
    case["gradient_audit"]["rows"][0]["passed"] = "yes"
    with pytest.raises(evidence.IDASameCaseEvidenceError, match="identity"):
        evidence._validate_gradient_audit(case, field="case")

    case = _case("development")
    case["gradient_audit"]["rows"][0]["passed"] = False
    with pytest.raises(evidence.IDASameCaseEvidenceError, match="inconsistent"):
        evidence._validate_gradient_audit(case, field="case")


def test_case_shape_and_projection_guards() -> None:
    """Reject malformed case identity, digests, public source, gates, and grid."""
    case = _case("development")
    case["role"] = "wrong"
    with pytest.raises(evidence.IDASameCaseEvidenceError, match="role/admission"):
        evidence._validate_case(case, role="development", index=0)

    case = _case("development")
    case["admitted"] = True
    with pytest.raises(evidence.IDASameCaseEvidenceError, match="role/admission"):
        evidence._validate_case(case, role="development", index=0)

    case = _case("development")
    case["digests"] = {}
    with pytest.raises(evidence.IDASameCaseEvidenceError, match="must not be empty"):
        evidence._validate_case(case, role="development", index=0)

    case = _case("development")
    case["digests"] = {1: _digest()}
    with pytest.raises(evidence.IDASameCaseEvidenceError, match="keys"):
        evidence._validate_case(case, role="development", index=0)

    case = _case("development")
    case["public_example"]["path"] = 1
    with pytest.raises(evidence.IDASameCaseEvidenceError, match="public_example"):
        evidence._validate_case(case, role="development", index=0)

    case = _case("development")
    case["threshold_results"].pop("latency")
    with pytest.raises(evidence.IDASameCaseEvidenceError, match="fields"):
        evidence._validate_case(case, role="development", index=0)

    case = _case("evaluation_candidate")
    case["grid_shape"] = [65, 65]
    with pytest.raises(evidence.IDASameCaseEvidenceError, match="129x129"):
        evidence._validate_case(
            case,
            role="evaluation_candidate",
            index=1,
        )


def test_source_verifier_rejects_nonrepo_and_missing_commit(tmp_path: Path) -> None:
    """Fail closed when the claimed Git tree or object cannot be resolved."""
    with pytest.raises(evidence.IDASameCaseEvidenceError, match="Git working tree"):
        evidence._verify_source_commit(
            tmp_path,
            source_commit="a" * 40,
            source_digests={"missing.py": _digest()},
        )

    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    with pytest.raises(evidence.IDASameCaseEvidenceError, match="unable to resolve"):
        evidence._verify_source_commit(
            repository,
            source_commit="a" * 40,
            source_digests={"missing.py": _digest()},
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.pop("generated_at"), "top-level fields"),
        (lambda payload: payload.update(blockers=None), "blockers must"),
        (
            lambda payload: payload.update(blockers=["z", "a"]),
            "blockers must",
        ),
        (
            lambda payload: payload.update(blockers=[""]),
            "blockers must",
        ),
        (
            lambda payload: payload.update(blockers=["valid", 1]),
            "blockers must",
        ),
        (
            lambda payload: payload.update(
                blockers=sorted(set(payload["blockers"]) - {"statistically_held_out_case_missing"})
            ),
            "required FUSION",
        ),
    ],
)
def test_top_level_and_blocker_guards(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], object],
    message: str,
) -> None:
    """Reject top-level drift and malformed or incomplete blocker sets."""
    payload = _report()
    mutate(payload)
    _seal(payload)
    path = tmp_path / "report.json"
    _write_report(path, payload)
    with pytest.raises(evidence.IDASameCaseEvidenceError, match=message):
        evidence.validate_ida_same_case_evidence(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["case_role_contract"]["selection_lock"].update(valid=True),
            "selection lock",
        ),
        (
            lambda payload: payload["environment"].update(x64_enabled=False),
            "JAX FP64",
        ),
        (
            lambda payload: payload["environment"].update(isolated_host=True),
            "non-isolated timing",
        ),
        (
            lambda payload: payload.update(
                blockers=[
                    blocker for blocker in payload["blockers"] if blocker != "evaluation_threshold_failed:latency"
                ]
            ),
            "threshold blocker projection",
        ),
    ],
)
def test_selection_environment_and_threshold_blockers_are_consistent(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], object],
    message: str,
) -> None:
    """Bind selection, FP64, isolation, and failed-threshold blocker semantics."""
    payload = _report()
    mutate(payload)
    _seal(payload)
    path = tmp_path / "report.json"
    _write_report(path, payload)
    with pytest.raises(evidence.IDASameCaseEvidenceError, match=message):
        evidence.validate_ida_same_case_evidence(path)


def test_public_cli_emits_blocked_json_and_exit_two(tmp_path: Path) -> None:
    """Expose the valid-but-blocked admission result through the root CLI."""
    path = tmp_path / "report.json"
    _write_report(path, _report())

    result = CliRunner().invoke(
        main,
        ["validate-ida-same-case", str(path), "--json-out"],
    )

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["artifact_valid"] is True
    assert payload["admitted"] is False
    assert payload["status"] == evidence.CONTROL_BLOCKED_STATUS
