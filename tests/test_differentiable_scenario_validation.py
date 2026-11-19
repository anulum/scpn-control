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
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from validation import benchmark_differentiable_scenario as producer
from validation import validate_differentiable_scenario as validator
from validation.validate_differentiable_scenario import validate_differentiable_scenario_report


def _campaign_metadata() -> dict[str, object]:
    """Return campaign metadata the validator admits."""
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
    """Return a passing gradient audit within its tolerance."""
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
    """Return readiness evidence that keeps the claim blocked on physics traceability."""
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
    """Return a timing context with one warm-up and two timed runs."""
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
    """Return a complete bounded report the validator admits."""
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
    """A complete bounded report passes with its claim still blocked."""
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(_report()), encoding="utf-8")

    result = validate_differentiable_scenario_report(path)

    assert result["status"] == "pass"
    assert result["claim_admissible"] is False
    assert result["blocked_reasons"] == ["physics_traceability"]


def test_differentiable_scenario_validator_accepts_blocked_jax_report(tmp_path: Path) -> None:
    """A blocked report that names the missing JAX backend passes."""
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
    """Duplicate JSON keys are refused by name."""
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version": 1, "schema_version": 2}', encoding="utf-8")

    result = validate_differentiable_scenario_report(path)

    assert result["status"] == "fail"
    assert result["errors"][0]["error"] == "duplicate JSON key: schema_version"


def test_differentiable_scenario_validator_rejects_promoted_claim(tmp_path: Path) -> None:
    """A report that opens the claim fails on admissibility and blocked reasons."""
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
    """An audit flag that disagrees with its errors and tolerance fails."""
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
    """The committed readiness report is admitted with its claim blocked."""
    root = Path(__file__).resolve().parents[1]
    result = validate_differentiable_scenario_report(
        root / "validation" / "reports" / "differentiable_scenario_readiness.json"
    )

    assert result["status"] == "pass"
    assert result["claim_admissible"] is False


def _validate(tmp_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Write ``payload`` as JSON and return the validator's result for it."""
    path = tmp_path / "report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return validate_differentiable_scenario_report(path)


def test_report_with_the_producers_own_command_is_admitted(tmp_path: Path) -> None:
    """The command line the producer records for its outputs passes, so fresh producer reports are admissible."""
    payload: dict[str, Any] = _report()
    payload["benchmark_context"]["command"] = producer._producer_command(
        tmp_path / "out dir" / "readiness.json", tmp_path / "out dir" / "readiness.md"
    )
    assert _validate(tmp_path, payload)["status"] == "pass"


@pytest.mark.parametrize(
    "command",
    [
        "python validation/other.py",
        f"python {validator.PRODUCER_SCRIPT} --json-out a.json",
        f"python {validator.PRODUCER_SCRIPT} --json-out a.json --md-out b.md --extra",
        f"python {validator.PRODUCER_SCRIPT} --md-out b.md --json-out a.json",
        f"python {validator.PRODUCER_SCRIPT} --json-out '' --md-out b.md",
        f"python {validator.PRODUCER_SCRIPT} --json-out 'a.json",
        None,
    ],
)
def test_other_commands_are_refused(command: str | None, tmp_path: Path) -> None:
    """Another script, a partial or reordered option list, an empty path, bad quoting or no string fails."""
    payload: dict[str, Any] = _report()
    payload["benchmark_context"]["command"] = command
    result = _validate(tmp_path, payload)
    assert result["status"] == "fail"
    assert [error["field"] for error in result["errors"]] == ["benchmark_context.command"]


def _set(section: str | None, field: str, value: object) -> Callable[[dict[str, Any]], None]:
    """Return a mutation that sets ``field`` (in ``section`` when given) to ``value``."""

    def mutate(payload: dict[str, Any]) -> None:
        """Apply the change to ``payload`` in place."""
        (payload if section is None else payload[section])[field] = value

    return mutate


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (_set(None, "status", "unknown"), "status"),
        (_set(None, "claim_status", "open"), "claim_status"),
        (_set(None, "campaign_metadata", []), "campaign_metadata"),
        (_set("campaign_metadata", "schema_version", 2), "campaign_metadata.schema_version"),
        (_set("campaign_metadata", "n_rho", 2), "campaign_metadata.n_rho"),
        (_set("campaign_metadata", "n_steps", 0), "campaign_metadata.n_steps"),
        (_set("campaign_metadata", "flux_grid_shape", [13]), "campaign_metadata.flux_grid_shape"),
        (_set("campaign_metadata", "dt", 0.0), "campaign_metadata.dt"),
        (_set("campaign_metadata", "gradient_tolerance", -1.0), "campaign_metadata.gradient_tolerance"),
        (_set("campaign_metadata", "jax_enable_x64", "yes"), "campaign_metadata.jax_enable_x64"),
        (_set("campaign_metadata", "equilibrium_params", [1.0]), "campaign_metadata.equilibrium_params"),
        (_set("campaign_metadata", "inputs_sha256", "g" * 64), "campaign_metadata.inputs_sha256"),
        (_set(None, "gradient_audit", "audit"), "gradient_audit"),
        (_set("gradient_audit", "loss", -1.0), "gradient_audit.loss"),
        (_set("gradient_audit", "epsilon", 0.0), "gradient_audit.epsilon"),
        (_set("gradient_audit", "tolerance", "tight"), "gradient_audit.tolerance"),
        (_set("gradient_audit", "checked_param_indices", [0]), "gradient_audit.checked_param_indices"),
        (_set("gradient_audit", "checked_source_indices", []), "gradient_audit.checked_source_indices"),
        (
            _set("gradient_audit", "checked_source_indices", [[0, 0, 1], [0, 0, 1]]),
            "gradient_audit.checked_source_indices",
        ),
        (_set("gradient_audit", "param_max_abs_error", -1.0), "gradient_audit.param_max_abs_error"),
        (_set("gradient_audit", "passed", "yes"), "gradient_audit.passed"),
        (_set(None, "readiness", None), "readiness"),
        (_set("readiness", "campaign_sha256", "x"), "readiness.campaign_sha256"),
        (_set("readiness", "gradient_tolerance", 0.0), "readiness.gradient_tolerance"),
        (_set("readiness", "audit_passed", False), "readiness.audit_passed"),
        (_set("readiness", "latency_p95_ms", -1.0), "readiness.latency_p95_ms"),
        (_set("readiness", "traceability_passed", "no"), "readiness.traceability_passed"),
        (_set("readiness", "claim_status", "open"), "readiness.claim_status"),
        (_set(None, "benchmark_context", 1), "benchmark_context"),
        (_set("benchmark_context", "isolation", "isolated"), "benchmark_context.isolation"),
        (_set("benchmark_context", "warmup_runs", 2), "benchmark_context.warmup_runs"),
        (_set("benchmark_context", "timed_runs", 0), "benchmark_context.timed_runs"),
        (_set("benchmark_context", "durations_ms", []), "benchmark_context.durations_ms"),
        (_set("benchmark_context", "durations_ms", [20.0]), "benchmark_context.durations_ms"),
    ],
)
def test_each_check_refuses_its_field(mutate: Callable[[dict[str, Any]], None], field: str, tmp_path: Path) -> None:
    """One bad value fails the report and names its own field among the errors."""
    payload: dict[str, Any] = _report()
    mutate(payload)
    result = _validate(tmp_path, payload)
    assert result["status"] == "fail"
    assert field in {error["field"] for error in result["errors"]}


@pytest.mark.parametrize(
    ("changes", "field"),
    [({"reason": "backend missing"}, "reason"), ({"claim_status": "gradient claim"}, "claim_status")],
)
def test_blocked_report_must_name_jax_and_claim_nothing(changes: dict[str, str], field: str, tmp_path: Path) -> None:
    """A blocked report fails unless it names the missing JAX backend and keeps its no-claim status."""
    payload = {
        "schema_version": validator.SCHEMA_VERSION,
        "status": "blocked",
        "reason": "JAX is required for coupled differentiable scenario gradient evidence",
        "claim_status": "no coupled-scenario gradient claim; JAX backend unavailable",
        **changes,
    }
    result = _validate(tmp_path, payload)
    assert result["status"] == "fail"
    assert [error["field"] for error in result["errors"]] == [field]


def test_unreadable_or_non_object_json_fails_on_the_json_field(tmp_path: Path) -> None:
    """A missing file and a JSON array both fail with a single json finding."""
    missing = validate_differentiable_scenario_report(tmp_path / "absent.json")
    array = tmp_path / "array.json"
    array.write_text("[1, 2]", encoding="utf-8")
    for result in (missing, validate_differentiable_scenario_report(array)):
        assert result["status"] == "fail"
        assert [error["field"] for error in result["errors"]] == ["json"]


def test_summary_fields_are_none_without_a_readiness_object(tmp_path: Path) -> None:
    """Without a readiness object the summary reports neither admissibility nor blocked reasons."""
    payload: dict[str, Any] = _report()
    payload["readiness"] = "pending"
    result = _validate(tmp_path, payload)
    assert result["claim_admissible"] is None
    assert result["blocked_reasons"] is None


def test_command_line_prints_json_or_findings(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The command prints the result as JSON on request, or a status line with one line per finding."""
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_report()), encoding="utf-8")
    assert validator.main([str(good), "--json-out"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    payload: dict[str, Any] = _report()
    payload["benchmark_context"]["warmup_runs"] = 3
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    assert validator.main([str(bad)]) == 1
    out = capsys.readouterr().out
    assert out.startswith("differentiable scenario validation: fail")
    assert "- benchmark_context.warmup_runs: expected one warmup run" in out


def test_quoted_output_paths_round_trip_through_the_command_check(tmp_path: Path) -> None:
    """Paths with spaces stay single tokens in the recorded command and are accepted."""
    command = shlex.join(["python", validator.PRODUCER_SCRIPT, "--json-out", "a b.json", "--md-out", "c d.md"])
    payload: dict[str, Any] = _report()
    payload["benchmark_context"]["command"] = command
    assert _validate(tmp_path, payload)["status"] == "pass"


def test_an_integer_too_large_for_a_float_is_a_finding_not_a_crash(tmp_path: Path) -> None:
    """A 400-digit integer in a numeric field fails that field instead of raising OverflowError."""
    payload: dict[str, Any] = _report()
    payload["campaign_metadata"]["dt"] = 10**400
    result = _validate(tmp_path, payload)
    assert result["status"] == "fail"
    assert "campaign_metadata.dt" in {error["field"] for error in result["errors"]}


def test_command_line_text_mode_without_findings_prints_only_the_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A passing report in text mode prints one status line and no findings."""
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_report()), encoding="utf-8")
    assert validator.main([str(good)]) == 0
    assert capsys.readouterr().out == "differentiable scenario validation: pass\n"


def test_script_entry_answers_help(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Run as a script, the validator hands its command line to main, whose parser prints help."""
    import runpy
    import sys

    monkeypatch.setattr(sys, "argv", [str(validator.__file__), "--help"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(validator.__file__), run_name="__main__")
    assert stopped.value.code == 0
    assert "usage:" in capsys.readouterr().out
