# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable Transport Latency Validation Tests

"""Behavioral tests for differentiable transport gradient-latency evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import validation.benchmark_differentiable_transport_latency as latency_benchmark
from scpn_control.benchmark_records import CAMPAIGN_ENV
from scpn_control.core.differentiable_transport import has_jax
from validation.benchmark_differentiable_transport_latency import _controller_formal_digest
from validation.validate_differentiable_transport_latency import validate_differentiable_transport_latency


def _runtime_metadata() -> dict[str, object]:
    """Return the runtime metadata block shared by the latency report fixtures."""
    return {
        "schema_version": 1,
        "measured_at_unix_s": 1_717_171_717.0,
        "python_version": "3.12.0",
        "platform": "Linux-test",
        "machine": "x86_64",
        "processor": "",
        "jax_version": "0.6.2",
        "jaxlib_version": "0.6.2",
        "jax_default_backend": "cpu",
        "jax_devices": ["TFRT_CPU_0"],
        "jax_enable_x64": True,
    }


def _one_step_report() -> dict[str, object]:
    """Return an admissible one-step gradient-latency report payload."""
    return {
        "schema_version": 1,
        "backend": "jax",
        "dtype": "float64",
        "n_rho": 21,
        "channel_count": 4,
        "warmup_runs": 1,
        "timed_runs": 5,
        "p50_ms": 10.0,
        "p95_ms": 12.0,
        "max_ms": 13.0,
        "runtime_metadata": _runtime_metadata(),
        "claim_status": "local audited gradient-admission latency only; not a real-time control-loop guarantee",
        "audit": {
            "loss": 0.1,
            "epsilon": 5.0e-5,
            "tolerance": 2.0e-3,
            "checked_indices": [[0, 5], [1, 10], [2, 7], [3, 12]],
            "chi_max_abs_error": 1.0e-6,
            "source_max_abs_error": 1.0e-6,
            "passed": True,
        },
    }


def _rollout_report() -> dict[str, object]:
    """Return an admissible rollout source-gradient latency report payload."""
    payload = _one_step_report()
    payload["claim_status"] = (
        "local audited rollout source-gradient latency only; not a real-time control-loop guarantee"
    )
    payload["n_steps"] = 4
    audit = payload["audit"]
    assert isinstance(audit, dict)
    audit.pop("chi_max_abs_error")
    audit["checked_indices"] = [[0, 0, 5], [1, 1, 8], [2, 2, 10], [3, 3, 12]]
    return payload


def _readiness_report() -> dict[str, object]:
    """Return a blocked full-fidelity readiness payload consistent with the latency fixtures."""
    return {
        "schema_version": 1,
        "backend": "jax",
        "campaign_sha256": "a" * 64,
        "gradient_latency_report_sha256": "b" * 64,
        "gradient_audit_sha256": "c" * 64,
        "rollout_latency_report_sha256": "d" * 64,
        "rollout_audit_sha256": "e" * 64,
        "external_reference_artifact_sha256": None,
        "external_reference_admitted": False,
        "controller_formal_artifact_sha256": "f" * 64,
        "n_rho": 21,
        "rollout_steps": 4,
        "channel_order": ["electron_temperature", "ion_temperature", "electron_density", "impurity_density"],
        "equilibrium_coupled": True,
        "full_fidelity_claim_admissible": False,
        "blocked_reasons": ["external_reference_artifact_sha256"],
        "claim_status": "bounded differentiable transport readiness only; full-fidelity claim remains blocked",
    }


def test_controller_formal_digest_requires_current_passing_schema(tmp_path: Path) -> None:
    """The formal digest is taken only from a present, passing report of the current schema."""
    report_path = tmp_path / "formal.json"
    digest = "a" * 64
    payload = {
        "schema_version": "scpn-control.z3-formal-report.v2",
        "status": "pass",
        "holds": True,
        "payload_sha256": digest,
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    assert _controller_formal_digest(report_path) == digest
    payload["schema_version"] = "scpn-control.z3-formal-report.v1"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    assert _controller_formal_digest(report_path) is None
    assert _controller_formal_digest(tmp_path / "missing.json") is None


def test_differentiable_transport_latency_admits_complete_reports(tmp_path: Path) -> None:
    """Complete one-step and rollout reports are both admitted."""
    one_step = tmp_path / "one_step.json"
    rollout = tmp_path / "rollout.json"
    one_step.write_text(json.dumps(_one_step_report()), encoding="utf-8")
    rollout.write_text(json.dumps(_rollout_report()), encoding="utf-8")

    report = validate_differentiable_transport_latency(one_step, rollout, require_admitted=True)

    assert report["status"] == "pass"
    assert report["admitted_reports"] == 2
    assert report["blocked_reports"] == 0
    assert report["errors"] == []


def test_differentiable_transport_latency_validates_full_fidelity_readiness(tmp_path: Path) -> None:
    """A blocked readiness report is validated and keeps the full-fidelity claim closed."""
    one_step = tmp_path / "one_step.json"
    rollout = tmp_path / "rollout.json"
    readiness = tmp_path / "readiness.json"
    one_step.write_text(json.dumps(_one_step_report()), encoding="utf-8")
    rollout.write_text(json.dumps(_rollout_report()), encoding="utf-8")
    readiness.write_text(json.dumps(_readiness_report()), encoding="utf-8")

    report = validate_differentiable_transport_latency(
        one_step,
        rollout,
        readiness_report=readiness,
        require_admitted=True,
    )

    assert report["status"] == "pass"
    assert report["full_fidelity_ready"] is False
    assert report["readiness_entry"]["status"] == "blocked"
    assert report["readiness_entry"]["blocked_reasons"] == ["external_reference_artifact_sha256"]


def test_differentiable_transport_latency_rejects_malformed_formal_artifact_digest(tmp_path: Path) -> None:
    """A readiness report with a malformed formal-artifact digest fails on that field."""
    one_step = tmp_path / "one_step.json"
    rollout = tmp_path / "rollout.json"
    readiness = tmp_path / "readiness.json"
    readiness_payload = _readiness_report()
    readiness_payload["controller_formal_artifact_sha256"] = "not-a-digest"
    one_step.write_text(json.dumps(_one_step_report()), encoding="utf-8")
    rollout.write_text(json.dumps(_rollout_report()), encoding="utf-8")
    readiness.write_text(json.dumps(readiness_payload), encoding="utf-8")

    report = validate_differentiable_transport_latency(one_step, rollout, readiness_report=readiness)

    assert report["status"] == "fail"
    assert any(error["field"] == "controller_formal_artifact_sha256" for error in report["errors"])


def test_differentiable_transport_latency_rejects_inconsistent_readiness(tmp_path: Path) -> None:
    """Readiness that claims admissibility while listing blocked reasons fails."""
    one_step = tmp_path / "one_step.json"
    rollout = tmp_path / "rollout.json"
    readiness = tmp_path / "readiness.json"
    readiness_payload = _readiness_report()
    readiness_payload["full_fidelity_claim_admissible"] = True
    one_step.write_text(json.dumps(_one_step_report()), encoding="utf-8")
    rollout.write_text(json.dumps(_rollout_report()), encoding="utf-8")
    readiness.write_text(json.dumps(readiness_payload), encoding="utf-8")

    report = validate_differentiable_transport_latency(one_step, rollout, readiness_report=readiness)

    assert report["status"] == "fail"
    assert any(error["field"] == "readiness.blocked_reasons" for error in report["errors"])


def test_differentiable_transport_latency_rejects_unordered_latency(tmp_path: Path) -> None:
    """A p95 latency below the median fails the latency ordering check."""
    one_step_payload = _one_step_report()
    one_step_payload["p95_ms"] = 9.0
    one_step = tmp_path / "one_step.json"
    rollout = tmp_path / "rollout.json"
    one_step.write_text(json.dumps(one_step_payload), encoding="utf-8")
    rollout.write_text(json.dumps(_rollout_report()), encoding="utf-8")

    report = validate_differentiable_transport_latency(one_step, rollout)

    assert report["status"] == "fail"
    assert any(error["field"] == "latency" for error in report["errors"])


def test_differentiable_transport_latency_rejects_failed_audit(tmp_path: Path) -> None:
    """A report whose gradient audit failed is refused."""
    one_step_payload = _one_step_report()
    audit = one_step_payload["audit"]
    assert isinstance(audit, dict)
    audit["passed"] = False
    one_step = tmp_path / "one_step.json"
    rollout = tmp_path / "rollout.json"
    one_step.write_text(json.dumps(one_step_payload), encoding="utf-8")
    rollout.write_text(json.dumps(_rollout_report()), encoding="utf-8")

    report = validate_differentiable_transport_latency(one_step, rollout)

    assert report["status"] == "fail"
    assert any(error["field"] == "audit.passed" for error in report["errors"])


def test_differentiable_transport_latency_rejects_missing_runtime_metadata(tmp_path: Path) -> None:
    """A report without runtime metadata is refused."""
    one_step_payload = _one_step_report()
    one_step_payload.pop("runtime_metadata")
    one_step = tmp_path / "one_step.json"
    rollout = tmp_path / "rollout.json"
    one_step.write_text(json.dumps(one_step_payload), encoding="utf-8")
    rollout.write_text(json.dumps(_rollout_report()), encoding="utf-8")

    report = validate_differentiable_transport_latency(one_step, rollout)

    assert report["status"] == "fail"
    assert any(error["field"] == "runtime_metadata" for error in report["errors"])


def test_differentiable_transport_latency_rejects_duplicate_keys(tmp_path: Path) -> None:
    """Duplicate JSON keys are refused by name."""
    one_step = tmp_path / "one_step.json"
    rollout = tmp_path / "rollout.json"
    one_step.write_text('{"schema_version": 1, "schema_version": 2}', encoding="utf-8")
    rollout.write_text(json.dumps(_rollout_report()), encoding="utf-8")

    report = validate_differentiable_transport_latency(one_step, rollout)

    assert report["status"] == "fail"
    assert any(error["error"] == "duplicate JSON key: schema_version" for error in report["errors"])


def test_repository_differentiable_transport_latency_evidence_is_admitted() -> None:
    """The committed latency, rollout and readiness reports are admitted, with full fidelity still blocked."""
    root = Path(__file__).resolve().parents[1]
    report = validate_differentiable_transport_latency(
        root / "validation" / "reports" / "differentiable_transport_latency.json",
        root / "validation" / "reports" / "differentiable_transport_rollout_latency.json",
        readiness_report=root / "validation" / "reports" / "differentiable_transport_full_fidelity_readiness.json",
        require_admitted=True,
    )

    assert report["status"] == "pass"
    assert report["admitted_reports"] == 2
    assert report["full_fidelity_ready"] is False


LATENCY_REPORTS = (
    "JSON_REPORT",
    "MD_REPORT",
    "ROLLOUT_JSON_REPORT",
    "ROLLOUT_MD_REPORT",
    "READINESS_JSON_REPORT",
    "READINESS_MD_REPORT",
)


def _redirect_latency_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Move the producer's six report paths and their directory into ``tmp_path``."""
    out = tmp_path / "reports"
    monkeypatch.setattr(latency_benchmark, "REPORT_DIR", out)
    paths: dict[str, Path] = {}
    for name in LATENCY_REPORTS:
        paths[name] = out / Path(getattr(latency_benchmark, name)).name
        monkeypatch.setattr(latency_benchmark, name, paths[name])
    return paths


def test_latency_main_without_jax_writes_three_blocked_pairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without JAX every report pair records the block and makes no latency claim."""
    monkeypatch.delenv(CAMPAIGN_ENV, raising=False)
    paths = _redirect_latency_reports(tmp_path, monkeypatch)
    monkeypatch.setattr(latency_benchmark, "has_jax", lambda: False)
    latency_benchmark.main()
    for name in ("JSON_REPORT", "ROLLOUT_JSON_REPORT", "READINESS_JSON_REPORT"):
        assert json.loads(paths[name].read_text(encoding="utf-8"))["status"] == "blocked"
    for name in ("MD_REPORT", "ROLLOUT_MD_REPORT", "READINESS_MD_REPORT"):
        assert "Status: `blocked`." in paths[name].read_text(encoding="utf-8")


@pytest.mark.skipif(not has_jax(), reason="JAX not installed")
def test_latency_main_with_jax_writes_reports_the_validator_admits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real JAX run writes one-step, rollout and readiness reports that the repository validator admits."""
    monkeypatch.delenv(CAMPAIGN_ENV, raising=False)
    paths = _redirect_latency_reports(tmp_path, monkeypatch)
    latency_benchmark.main()
    report = validate_differentiable_transport_latency(
        paths["JSON_REPORT"], paths["ROLLOUT_JSON_REPORT"], require_admitted=True
    )
    assert report["status"] == "pass", report["errors"]
    assert report["admitted_reports"] == 2
    rollout = json.loads(paths["ROLLOUT_JSON_REPORT"].read_text(encoding="utf-8"))
    assert rollout["n_steps"] == 4 and rollout["n_rho"] == 21
    readiness = json.loads(paths["READINESS_JSON_REPORT"].read_text(encoding="utf-8"))
    assert readiness["rollout_steps"] == 4
    assert "- P95 latency [ms]:" in paths["MD_REPORT"].read_text(encoding="utf-8")
    assert "- Rollout steps: `4`" in paths["ROLLOUT_MD_REPORT"].read_text(encoding="utf-8")
    assert "- Full-fidelity admissible:" in paths["READINESS_MD_REPORT"].read_text(encoding="utf-8")


def test_latency_script_refuses_the_committed_reports_without_a_campaign(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run as a script, the producer stops at the recorded-campaign guard."""
    import runpy

    monkeypatch.delenv(CAMPAIGN_ENV, raising=False)
    with pytest.raises(RuntimeError, match="requires tools/run_recorded_benchmark.py"):
        runpy.run_path(str(latency_benchmark.__file__), run_name="__main__")


def test_latency_import_path_changes_only_for_a_script_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A package import leaves the import path alone; a script run from a stripped path puts src first."""
    import importlib
    import runpy

    root = Path(str(latency_benchmark.__file__)).resolve().parents[1]
    stripped = [entry for entry in sys.path if Path(entry or ".").resolve() not in {root, root / "src"}]
    monkeypatch.setattr(sys, "path", list(stripped))
    importlib.reload(latency_benchmark)
    assert sys.path == stripped
    monkeypatch.delenv(CAMPAIGN_ENV, raising=False)
    with pytest.raises(RuntimeError, match="requires tools/run_recorded_benchmark.py"):
        runpy.run_path(str(latency_benchmark.__file__), run_name="__main__")
    assert sys.path[0] == str(root / "src")
