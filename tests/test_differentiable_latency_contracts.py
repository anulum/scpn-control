# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable latency declaration boundary tests.

"""Public reader refusals and internally consistent flags over real JAX observations."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from differentiable_latency_observation import observed_transport_reports as observed_transport_reports
from differentiable_latency_observation import write_derivative

from validation.validate_differentiable_transport_latency import validate_differentiable_transport_latency


def test_actual_audited_reports_remain_local_not_full_fidelity(
    observed_transport_reports: tuple[Path, Path, Path],
) -> None:
    """Actual audited latency is accepted while missing external evidence keeps readiness blocked."""
    one, rollout, readiness = observed_transport_reports
    result = validate_differentiable_transport_latency(one, rollout, readiness_report=readiness, require_admitted=True)
    assert result["status"] == "pass" and result["admitted_reports"] == 2 and result["blocked_reports"] == 0
    assert result["full_fidelity_ready"] is False and result["readiness_entry"]["status"] == "blocked"
    assert validate_differentiable_transport_latency(one)["status"] == "pass"


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("schema_version", True, "schema_version"),
        ("schema_version", 1.0, "schema_version"),
        ("backend", "other", "backend"),
        ("dtype", "float32", "dtype"),
        ("channel_count", 4.0, "channel_count"),
        ("claim_status", "promoted", "claim_status"),
        ("n_rho", True, "n_rho"),
        ("n_rho", 2, "n_rho"),
        ("warmup_runs", -1, "warmup_runs"),
        ("timed_runs", 0, "timed_runs"),
        ("p50_ms", -1.0, "p50_ms"),
        ("p95_ms", math.inf, "p95_ms"),
        ("p95_ms", 10**400, "p95_ms"),
        ("max_ms", True, "max_ms"),
        ("p50_ms", 1e15, "latency"),
        ("runtime_metadata", None, "runtime_metadata"),
        ("runtime_metadata.measured_at_unix_s", 0, "runtime_metadata.measured_at_unix_s"),
        ("runtime_metadata.python_version", "", "runtime_metadata.python_version"),
        ("runtime_metadata.machine", None, "runtime_metadata.machine"),
        ("runtime_metadata.processor", None, "runtime_metadata.processor"),
        ("runtime_metadata.jax_devices", None, "runtime_metadata.jax_devices"),
        ("runtime_metadata.jax_devices", [], "runtime_metadata.jax_devices"),
        ("runtime_metadata.jax_devices", [""], "runtime_metadata.jax_devices"),
        ("runtime_metadata.jax_enable_x64", "yes", "runtime_metadata.jax_enable_x64"),
        ("audit", None, "audit"),
        ("audit.passed", False, "audit.passed"),
        ("audit.tolerance", None, "audit.tolerance"),
        ("audit.epsilon", 0.0, "audit.epsilon"),
        ("audit.loss", "0", "audit.loss"),
        ("audit.loss", math.nan, "audit.loss"),
        ("audit.source_max_abs_error", None, "audit.source_max_abs_error"),
        ("audit.source_max_abs_error", 1e9, "audit.source_max_abs_error"),
        ("audit.chi_max_abs_error", None, "audit.chi_max_abs_error"),
        ("audit.chi_max_abs_error", 1e9, "audit.chi_max_abs_error"),
        ("audit.checked_indices", [], "audit.checked_indices"),
        ("audit.checked_indices", [[0]], "audit.checked_indices"),
        ("audit.checked_indices", [[True, 1]], "audit.checked_indices"),
        ("audit.checked_indices", [[0, 1], [0, 1]], "audit.checked_indices"),
        ("audit.checked_indices", [[4, 1]], "audit.checked_indices"),
        ("audit.checked_indices", [[0, 21]], "audit.checked_indices"),
    ],
)
def test_bad_one_step_is_fail_and_never_counted_as_admitted(
    observed_transport_reports: tuple[Path, Path, Path],
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    """A malformed derivative produces fail/count1 while the independent valid rollout remains pass."""
    one, rollout, readiness = observed_transport_reports
    changed = write_derivative(one, tmp_path / "authored-one.json", field, value)
    result = validate_differentiable_transport_latency(
        changed, rollout, readiness_report=readiness, require_admitted=True
    )
    assert result["status"] == "fail" and result["admitted_reports"] == 1
    assert result["entries"][0]["status"] == "fail" and result["entries"][1]["status"] == "pass"
    assert any(e["field"] == error for e in result["errors"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_steps", 0),
        ("audit.checked_indices", [[0, 0]]),
        ("audit.checked_indices", [[0, 0, 1], [0, 0, 1]]),
        ("audit.checked_indices", [[4, 0, 1]]),
        ("audit.checked_indices", [[0, 4, 1]]),
        ("audit.checked_indices", [[0, 0, 21]]),
    ],
)
def test_bad_rollout_is_not_admitted(
    observed_transport_reports: tuple[Path, Path, Path],
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    """Invalid declared rollout length/indices cannot count as accepted latency."""
    one, rollout, _ = observed_transport_reports
    changed = write_derivative(rollout, tmp_path / "authored-rollout.json", field, value)
    result = validate_differentiable_transport_latency(one, changed)
    assert result["status"] == "fail" and result["admitted_reports"] == 1
    assert result["entries"][1]["status"] == "fail"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("backend", "other"),
        ("n_rho", 2),
        ("rollout_steps", 0),
        ("campaign_sha256", "bad"),
        ("gradient_audit_sha256", "g" * 64),
        ("controller_formal_artifact_sha256", 4),
        ("external_reference_artifact_sha256", "bad"),
        ("channel_order", []),
        ("equilibrium_coupled", False),
        ("external_reference_admitted", None),
        ("full_fidelity_claim_admissible", "yes"),
        ("blocked_reasons", "reason"),
        ("blocked_reasons", []),
        ("claim_status", "promoted"),
    ],
)
def test_invalid_readiness_has_fail_status_and_false_flag(
    observed_transport_reports: tuple[Path, Path, Path],
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    """Malformed readiness is fail/False while independently valid local latency remains accepted."""
    one, rollout, readiness = observed_transport_reports
    changed = write_derivative(readiness, tmp_path / "authored-ready.json", field, value)
    result = validate_differentiable_transport_latency(one, rollout, readiness_report=changed)
    assert result["status"] == "fail" and result["admitted_reports"] == 2
    assert result["readiness_entry"]["status"] == "fail" and result["full_fidelity_ready"] is False


@pytest.mark.parametrize("valid", [True, False])
def test_declared_ready_flag_never_survives_its_errors(
    observed_transport_reports: tuple[Path, Path, Path],
    tmp_path: Path,
    valid: bool,
) -> None:
    """The schema-valid authored readiness branch is metadata only; any digest error clears its flag."""
    one, rollout, readiness = observed_transport_reports
    payload = json.loads(readiness.read_text())
    payload.update(
        full_fidelity_claim_admissible=True,
        blocked_reasons=[],
        claim_status="full-fidelity differentiable transport claim admitted",
        external_reference_admitted=True,
        external_reference_artifact_sha256="A" * 64,
        controller_formal_artifact_sha256="A" * 64,
    )
    payload["controller_formal_artifact_sha256"] = "A" * 64
    if not valid:
        payload["campaign_sha256"] = "invalid"
    changed = tmp_path / "authored-declared-ready.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    result = validate_differentiable_transport_latency(one, rollout, readiness_report=changed)
    assert result["status"] == ("pass" if valid else "fail")
    assert result["full_fidelity_ready"] is valid
    assert result["readiness_entry"]["status"] == ("pass" if valid else "fail")


def test_declared_ready_with_blocked_reasons_is_inconsistent(
    observed_transport_reports: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    """A true readiness declaration with residual blockers cannot retain ready=True."""
    one, rollout, readiness = observed_transport_reports
    changed = write_derivative(readiness, tmp_path / "inconsistent.json", "full_fidelity_claim_admissible", True)
    result = validate_differentiable_transport_latency(one, rollout, readiness_report=changed)
    assert result["status"] == "fail" and result["full_fidelity_ready"] is False


@pytest.mark.parametrize("field", ["one", "readiness"])
@pytest.mark.parametrize("content", [None, "[]", "{", '{"schema_version":1,"schema_version":1}'])
def test_native_report_read_refusals_become_json_diagnostics(
    observed_transport_reports: tuple[Path, Path, Path],
    tmp_path: Path,
    field: str,
    content: str | None,
) -> None:
    """Absent, non-object, malformed and duplicate JSON produces an authored fail entry."""
    one, rollout, readiness = observed_transport_reports
    bad = tmp_path / "bad.json"
    if content is not None:
        bad.write_text(content, encoding="utf-8")
    result = validate_differentiable_transport_latency(
        bad if field == "one" else one, rollout, readiness_report=bad if field == "readiness" else readiness
    )
    assert result["status"] == "fail" and any(e["field"] == "json" for e in result["errors"])
    assert result["full_fidelity_ready"] is False


@pytest.mark.parametrize("require", [False, True])
@pytest.mark.parametrize("valid", [False, True])
def test_authored_blocked_declarations_are_counted_only_when_valid(tmp_path: Path, require: bool, valid: bool) -> None:
    """No-backend declarations are reader input, not a mock of actual installed JAX availability."""
    path = tmp_path / "authored-blocked.json"
    payload = {
        "schema_version": 1,
        "status": "blocked",
        "reason": "JAX is required for authored reader-boundary declaration",
        "claim_status": "no latency claim; JAX gradient backend unavailable in this environment",
    }
    if not valid:
        payload["schema_version"] = True
        payload["reason"] = "invalid"
        payload["claim_status"] = "promoted"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = validate_differentiable_transport_latency(path, path, readiness_report=path, require_admitted=require)
    assert result["status"] == ("pass" if valid and not require else "fail")
    assert result["admitted_reports"] == 0 and result["blocked_reports"] == (2 if valid else 0)
    assert result["full_fidelity_ready"] is False
    assert result["entries"][0]["status"] == ("blocked" if valid else "fail")


def test_invalid_latency_clears_aggregate_declared_readiness(
    observed_transport_reports: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    """An individually valid readiness declaration cannot make a failing aggregate ready."""
    one, rollout, readiness = observed_transport_reports
    payload = json.loads(readiness.read_text())
    payload.update(
        full_fidelity_claim_admissible=True,
        blocked_reasons=[],
        claim_status="full-fidelity differentiable transport claim admitted",
        external_reference_admitted=True,
        external_reference_artifact_sha256="A" * 64,
        controller_formal_artifact_sha256="A" * 64,
    )
    ready = tmp_path / "authored-ready.json"
    ready.write_text(json.dumps(payload), encoding="utf-8")
    bad = write_derivative(one, tmp_path / "authored-bad-audit.json", "audit.passed", False)
    result = validate_differentiable_transport_latency(bad, rollout, readiness_report=ready)
    assert result["status"] == "fail" and result["full_fidelity_ready"] is False
    assert result["readiness_entry"]["status"] == "pass"
    assert result["readiness_entry"]["full_fidelity_ready"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("external_reference_admitted", False),
        ("external_reference_artifact_sha256", None),
        ("controller_formal_artifact_sha256", None),
    ],
)
def test_declared_readiness_requires_producer_prerequisites(
    observed_transport_reports: tuple[Path, Path, Path],
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    """A ready declaration cannot omit the defining producer's external/reference/proof metadata."""
    one, rollout, readiness = observed_transport_reports
    payload = json.loads(readiness.read_text())
    payload.update(
        full_fidelity_claim_admissible=True,
        blocked_reasons=[],
        claim_status="full-fidelity differentiable transport claim admitted",
        external_reference_admitted=True,
        external_reference_artifact_sha256="a" * 64,
        controller_formal_artifact_sha256="b" * 64,
    )
    payload[field] = value
    changed = tmp_path / "authored-missing-prerequisite.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    result = validate_differentiable_transport_latency(one, rollout, readiness_report=changed)
    assert result["status"] == "fail" and result["full_fidelity_ready"] is False
    assert result["readiness_entry"]["status"] == "fail"
    assert any(field in e["field"] for e in result["errors"])
