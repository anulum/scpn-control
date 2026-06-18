# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable scenario evidence benchmark
"""Publish bounded evidence for the coupled differentiable scenario facade."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "differentiable_scenario_readiness.json"
MD_REPORT = REPORT_DIR / "differentiable_scenario_readiness.md"
SCHEMA_VERSION = "scpn-control.differentiable-scenario-readiness.v1"


def _profiles(rho: np.ndarray) -> np.ndarray:
    te = 8.0 * np.exp(-((rho - 0.35) ** 2) / 0.03) + 0.2
    ti = 6.0 * np.exp(-((rho - 0.42) ** 2) / 0.04) + 0.2
    ne = 4.0 + 0.8 * (1.0 - rho**2)
    nz = 0.03 + 0.02 * np.exp(-((rho - 0.65) ** 2) / 0.02)
    return np.stack([te, ti, ne, nz])


def _scenario_fixture() -> tuple[np.ndarray, ...]:
    from scpn_control.core.differentiable_transport import differentiable_transport_rollout

    rho = np.linspace(0.05, 1.0, 16)
    profiles = _profiles(rho)
    chi = np.stack(
        [
            0.20 + 0.02 * rho,
            0.16 + 0.02 * rho,
            0.04 + 0.005 * rho,
            0.012 + 0.001 * rho,
        ]
    )
    edge = np.array([0.2, 0.2, 4.0, 0.03])
    dt = np.array([8.0e-4])
    sources = np.zeros((3, 4, rho.size))
    sources[:, 0, 4:8] = 0.01
    sources[:, 2, 3:7] = 0.004
    target = np.asarray(
        differentiable_transport_rollout(profiles, chi, sources, rho, float(dt[0]), edge, use_jax=False)
    )
    target[:, 0, 5:9] += 0.05
    target[:, 2, 4:8] += 0.02
    r_grid = np.linspace(1.2, 2.2, rho.size)
    z_grid = np.linspace(-0.8, 0.8, 13)
    params = np.array([1.3, 0.7])
    return params, profiles, chi, sources, target, rho, r_grid, z_grid, dt, edge


def _loadavg() -> tuple[float, float, float] | None:
    try:
        return tuple(float(value) for value in os.getloadavg())
    except OSError:
        return None


def _blocked_payload(reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "reason": reason,
        "claim_status": "no coupled-scenario gradient claim; JAX backend unavailable",
    }


def _write_markdown(payload: dict[str, Any]) -> None:
    status = str(payload["status"])
    readiness = payload.get("readiness", {})
    raw_blocked_reasons = readiness.get("blocked_reasons", []) if isinstance(readiness, dict) else []
    blocked_reasons = list(raw_blocked_reasons) if isinstance(raw_blocked_reasons, list | tuple) else []
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "<!-- Commercial license available -->",
        "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
        "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
        "<!-- ORCID: 0009-0009-3560-0851 -->",
        "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
        "<!-- SCPN Control — Differentiable scenario readiness report -->",
        "",
        "# Differentiable Scenario Readiness",
        "",
        f"- Status: `{status}`",
        f"- Schema: `{payload['schema_version']}`",
        f"- Claim status: `{payload['claim_status']}`",
    ]
    if status == "blocked":
        lines.extend(["", f"Reason: {payload['reason']}"])
    else:
        campaign = payload["campaign_metadata"]
        audit = payload["gradient_audit"]
        benchmark = payload["benchmark_context"]
        lines.extend(
            [
                f"- Campaign digest: `{readiness['campaign_sha256']}`",
                f"- Audit digest: `{readiness['gradient_audit_sha256']}`",
                f"- Backend: `{campaign['backend']}`",
                f"- Radial points: `{campaign['n_rho']}`",
                f"- Rollout steps: `{campaign['n_steps']}`",
                f"- Flux grid: `{list(campaign['flux_grid_shape'])}`",
                f"- Gradient tolerance: `{campaign['gradient_tolerance']}`",
                f"- Audit passed: `{audit['passed']}`",
                f"- p95 audit latency: `{readiness['latency_p95_ms']}` ms",
                f"- Claim admissible: `{readiness['claim_admissible']}`",
                f"- Blocked reasons: `{blocked_reasons}`",
                f"- Timing context: `{benchmark['isolation']}`",
                "",
                "The timing value is local non-isolated admission evidence only. It is not",
                "a hardware benchmark, real-time guarantee, or facility-control claim.",
            ]
        )
    MD_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Generate differentiable-scenario readiness evidence."""

    from scpn_control.core.differentiable_scenario import (
        assert_differentiable_scenario_gradient_consistent,
        differentiable_scenario_readiness_evidence,
        has_jax,
        scenario_campaign_metadata,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not has_jax():
        payload = _blocked_payload("JAX is required for coupled differentiable scenario gradient evidence")
        JSON_REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_markdown(payload)
        return

    params, profiles, chi, sources, target, rho, r_grid, z_grid, dt, edge = _scenario_fixture()
    tolerance = 5.0e-4
    assert_differentiable_scenario_gradient_consistent(
        params,
        profiles,
        chi,
        sources,
        target,
        rho,
        r_grid,
        z_grid,
        float(dt[0]),
        edge,
        tolerance=tolerance,
    )
    load_before = _loadavg()
    durations_ms: list[float] = []
    audit = None
    for _ in range(5):
        started = time.perf_counter()
        audit = assert_differentiable_scenario_gradient_consistent(
            params,
            profiles,
            chi,
            sources,
            target,
            rho,
            r_grid,
            z_grid,
            float(dt[0]),
            edge,
            tolerance=tolerance,
        )
        durations_ms.append((time.perf_counter() - started) * 1000.0)
    assert audit is not None
    load_after = _loadavg()
    metadata = scenario_campaign_metadata(
        params,
        profiles,
        chi,
        sources,
        rho,
        r_grid,
        z_grid,
        float(dt[0]),
        edge,
        backend="jax",
        gradient_tolerance=tolerance,
    )
    p95_ms = float(np.percentile(np.asarray(durations_ms, dtype=float), 95.0))
    readiness = differentiable_scenario_readiness_evidence(
        metadata,
        audit,
        latency_p95_ms=p95_ms,
        traceability_passed=False,
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "claim_status": "bounded coupled differentiable scenario evidence only; full-fidelity claim remains blocked",
        "campaign_metadata": asdict(metadata),
        "gradient_audit": asdict(audit),
        "readiness": asdict(readiness),
        "benchmark_context": {
            "command": "python validation/benchmark_differentiable_scenario.py",
            "isolation": "local_non_isolated_admission_smoke",
            "warmup_runs": 1,
            "timed_runs": len(durations_ms),
            "durations_ms": durations_ms,
            "loadavg_before": load_before,
            "loadavg_after": load_after,
        },
    }
    JSON_REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(payload)


if __name__ == "__main__":
    main()
