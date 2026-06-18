# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable scenario validation tests

"""Behavioral tests for coupled differentiable scenario readiness validation."""

from __future__ import annotations

import json
from pathlib import Path

from validation.validate_differentiable_scenario import validate_differentiable_scenario_report


def _campaign_metadata() -> dict[str, object]:
    return {
        "schema_version": 1,
        "backend": "jax",
        "dtype": "float64",
        "n_rho": 16,
        "n_steps": 3,
        "equilibrium_param_count": 2,
        "flux_grid_shape": [13, 16],
        "dt": 8.0e-4,
        "gradient_tolerance": 5.0e-4,
        "jax_enable_x64": True,
        "equilibrium_params": [1.3, 0.7],
        "inputs_sha256": "a" * 64,
    }


def _audit() -> dict[str, object]:
    return {
        "loss": 0.01,
        "epsilon": 1.0e-5,
        "tolerance": 5.0e-4,
        "checked_param_indices": [0, 1],
        "checked_source_indices": [[0, 0, 1], [2, 1, 8], [1, 2, 14]],
        "param_max_abs_error": 1.0e-7,
        "source_max_abs_error": 2.0e-7,
        "passed": True,
    }


def _readiness() -> dict[str, object]:
    return {
        "schema_version": 1,
        "backend": "jax",
        "campaign_sha256": "b" * 64,
        "gradient_audit_sha256": "c" * 64,
        "gradient_tolerance": 5.0e-4,
        "audit_passed": True,
        "latency_p95_ms": 25.0,
        "traceability_passed": False,
        "claim_admissible": False,
        "blocked_reasons": ["physics_traceability"],
        "claim_status": "bounded coupled differentiable scenario gradient evidence only",
    }


def _benchmark_context() -> dict[str, object]:
    return {
        "command": "python validation/benchmark_differentiable_scenario.py",
        "isolation": "local_non_isolated_admission_smoke",
        "warmup_runs": 1,
        "timed_runs": 2,
        "durations_ms": [20.0, 25.0],
        "loadavg_before": [1.0, 1.0, 1.0],
        "loadavg_after": [1.1, 1.0, 1.0],
    }


def _report() -> dict[str, object]:
    return {
        "schema_version": "scpn-control.differentiable-scenario-readiness.v1",
        "status": "pass",
        "claim_status": "bounded coupled differentiable scenario evidence only; full-fidelity claim remains blocked",
        "campaign_metadata": _campaign_metadata(),
        "gradient_audit": _audit(),
        "readiness": _readiness(),
        "benchmark_context": _benchmark_context(),
    }


def test_differentiable_scenario_validator_accepts_bounded_report(tmp_path: Path) -> None:
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(_report()), encoding="utf-8")

    result = validate_differentiable_scenario_report(path)

    assert result["status"] == "pass"
    assert result["claim_admissible"] is False
    assert result["blocked_reasons"] == ["physics_traceability"]


def test_differentiable_scenario_validator_accepts_blocked_jax_report(tmp_path: Path) -> None:
    path = tmp_path / "blocked.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "scpn-control.differentiable-scenario-readiness.v1",
                "status": "blocked",
                "reason": "JAX is required for coupled differentiable scenario gradient evidence",
                "claim_status": "no coupled-scenario gradient claim; JAX backend unavailable",
            }
        ),
        encoding="utf-8",
    )

    assert validate_differentiable_scenario_report(path)["status"] == "pass"


def test_differentiable_scenario_validator_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version": 1, "schema_version": 2}', encoding="utf-8")

    result = validate_differentiable_scenario_report(path)

    assert result["status"] == "fail"
    assert result["errors"][0]["error"] == "duplicate JSON key: schema_version"


def test_differentiable_scenario_validator_rejects_promoted_claim(tmp_path: Path) -> None:
    payload = _report()
    readiness = payload["readiness"]
    assert isinstance(readiness, dict)
    readiness["claim_admissible"] = True
    readiness["blocked_reasons"] = []
    path = tmp_path / "promoted.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_differentiable_scenario_report(path)

    assert result["status"] == "fail"
    assert any(error["field"] == "readiness.claim_admissible" for error in result["errors"])
    assert any(error["field"] == "readiness.blocked_reasons" for error in result["errors"])


def test_differentiable_scenario_validator_rejects_inconsistent_audit_flag(tmp_path: Path) -> None:
    payload = _report()
    audit = payload["gradient_audit"]
    assert isinstance(audit, dict)
    audit["source_max_abs_error"] = 1.0
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_differentiable_scenario_report(path)

    assert result["status"] == "fail"
    assert any(error["field"] == "gradient_audit.passed" for error in result["errors"])


def test_repository_differentiable_scenario_report_is_admitted() -> None:
    root = Path(__file__).resolve().parents[1]
    result = validate_differentiable_scenario_report(
        root / "validation" / "reports" / "differentiable_scenario_readiness.json"
    )

    assert result["status"] == "pass"
    assert result["claim_admissible"] is False
