#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disruption-sequence bounded validation
"""Validate disruption-sequence phase ordering against repository-owned contracts.

The disruption-sequence model is a bounded control-facing approximation. This
validator checks exact algebraic identities and deterministic branch contracts
inside ``src/scpn_control/core/disruption_sequence.py``:

1. Thermal-quench and current-quench durations are finite and positive.
2. Total duration equals thermal-quench plus current-quench duration.
3. Wall heat load equals thermal-quench deposition plus runaway termination load.
4. Vessel force equals the vertical halo-force convention.
5. The current-quench trace starts at the configured plasma current and decays
   monotonically.
6. The post-TQ temperature remains finite, positive, and below the pre-TQ
   temperature.
7. The SPI branch changes the density-dependent current-quench phase while
   leaving the pre-mitigation thermal-quench state unchanged.

Measured labelled disruption windows, phase labels, wall-contact geometry,
impurity-radiation metadata, and mitigation campaign artefacts remain required
before facility mitigation claims can be promoted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scpn_control.core.disruption_sequence import DisruptionConfig, DisruptionSequence

DISRUPTION_SEQUENCE_SCHEMA_VERSION = "scpn-control.disruption-sequence-validation.v1"


@dataclass(frozen=True)
class DisruptionSequenceValidationResult:
    """Outcome of the bounded disruption-sequence validation."""

    config: DisruptionConfig
    spi_density_target_20: float
    unmitigated_tq_duration_ms: float
    unmitigated_cq_duration_ms: float
    mitigated_cq_duration_ms: float
    unmitigated_post_tq_temperature_eV: float
    mitigated_post_tq_temperature_eV: float
    total_duration_rel_error: float
    wall_heat_load_rel_error: float
    vessel_force_rel_error: float
    sideways_force_rel_error: float
    current_initial_rel_error: float
    current_monotonic: bool
    current_finite: bool
    halo_limit_product: float
    mitigation_metadata: dict[str, float | str]
    exact_tol: float
    production_claim_allowed: bool
    phase_order_passed: bool
    temperature_passed: bool
    current_trace_passed: bool
    halo_force_passed: bool
    mitigation_branch_passed: bool
    passed: bool


def default_config() -> DisruptionConfig:
    """ITER-like disruption-sequence configuration used for local validation."""
    return DisruptionConfig(
        R0=6.2,
        a=2.0,
        B0=5.3,
        kappa=1.7,
        Ip_MA=15.0,
        W_th_MJ=350.0,
        Te_pre_keV=20.0,
        ne_pre_20=1.0,
        dBr_over_B_trigger=3e-3,
    )


def validate_disruption_sequence(
    *,
    config: DisruptionConfig | None = None,
    spi_density_target_20: float = 50.0,
    exact_tol: float = 1e-12,
) -> DisruptionSequenceValidationResult:
    """Validate deterministic disruption-sequence contracts without facility claims."""
    cfg = config or default_config()
    _positive_float("spi_density_target_20", spi_density_target_20)
    if spi_density_target_20 <= cfg.ne_pre_20:
        raise ValueError("spi_density_target_20 must exceed the pre-disruption density")

    sequence = DisruptionSequence(cfg)
    unmitigated = sequence.run()
    mitigated = sequence.with_mitigation(spi_density_target_20)

    expected_total_duration = unmitigated.tq_result.tau_tq_ms + unmitigated.cq_result.cq_duration_ms
    total_duration_rel_error = _relative_error(unmitigated.total_duration_ms, expected_total_duration)

    expected_wall_heat = unmitigated.tq_result.heat_load_MJ_m2 + unmitigated.re_result.wall_heat_load_MJ_m2
    wall_heat_load_rel_error = _relative_error(unmitigated.wall_heat_load_MJ_m2, expected_wall_heat)

    expected_vessel_force = unmitigated.halo_result.F_z_MN
    vessel_force_rel_error = _relative_error(unmitigated.vessel_force_MN, expected_vessel_force)

    expected_sideways = 0.5 * unmitigated.halo_result.F_z_MN
    sideways_force_rel_error = _relative_error(unmitigated.halo_result.F_sideways_MN, expected_sideways)

    current_trace = np.asarray(unmitigated.cq_result.Ip_trace, dtype=float)
    current_finite = bool(np.all(np.isfinite(current_trace)))
    current_monotonic = bool(np.all(np.diff(current_trace) <= 1e-12))
    current_initial_rel_error = _relative_error(float(current_trace[0]), cfg.Ip_MA)

    pre_tq_temperature_eV = cfg.Te_pre_keV * 1000.0
    temperature_passed = (
        math.isfinite(unmitigated.tq_result.post_tq_Te_eV)
        and 0.0 < unmitigated.tq_result.post_tq_Te_eV < pre_tq_temperature_eV
        and mitigated.tq_result.post_tq_Te_eV == unmitigated.tq_result.post_tq_Te_eV
    )
    halo_limit_product = unmitigated.halo_result.f_halo * unmitigated.halo_result.tpf

    phase_order_passed = (
        math.isfinite(unmitigated.tq_result.tau_tq_ms)
        and unmitigated.tq_result.tau_tq_ms > 0.0
        and math.isfinite(unmitigated.cq_result.cq_duration_ms)
        and unmitigated.cq_result.cq_duration_ms > 0.0
        and total_duration_rel_error <= exact_tol
        and wall_heat_load_rel_error <= exact_tol
    )
    current_trace_passed = current_finite and current_monotonic and current_initial_rel_error <= exact_tol
    halo_force_passed = (
        vessel_force_rel_error <= exact_tol
        and sideways_force_rel_error <= exact_tol
        and unmitigated.halo_result.within_iter_limits == (halo_limit_product < 0.75)
    )
    mitigation_branch_passed = (
        mitigated.cq_result.cq_duration_ms != unmitigated.cq_result.cq_duration_ms
        and mitigated.tq_result.tau_tq_ms == unmitigated.tq_result.tau_tq_ms
        and mitigated.tq_result.post_tq_Te_eV == unmitigated.tq_result.post_tq_Te_eV
    )
    passed = (
        phase_order_passed
        and temperature_passed
        and current_trace_passed
        and halo_force_passed
        and mitigation_branch_passed
    )

    return DisruptionSequenceValidationResult(
        config=cfg,
        spi_density_target_20=float(spi_density_target_20),
        unmitigated_tq_duration_ms=float(unmitigated.tq_result.tau_tq_ms),
        unmitigated_cq_duration_ms=float(unmitigated.cq_result.cq_duration_ms),
        mitigated_cq_duration_ms=float(mitigated.cq_result.cq_duration_ms),
        unmitigated_post_tq_temperature_eV=float(unmitigated.tq_result.post_tq_Te_eV),
        mitigated_post_tq_temperature_eV=float(mitigated.tq_result.post_tq_Te_eV),
        total_duration_rel_error=total_duration_rel_error,
        wall_heat_load_rel_error=wall_heat_load_rel_error,
        vessel_force_rel_error=vessel_force_rel_error,
        sideways_force_rel_error=sideways_force_rel_error,
        current_initial_rel_error=current_initial_rel_error,
        current_monotonic=current_monotonic,
        current_finite=current_finite,
        halo_limit_product=float(halo_limit_product),
        mitigation_metadata={
            "mode": "bounded_spi_density_branch",
            "spi_density_target_20": float(spi_density_target_20),
            "unmitigated_density_20": float(cfg.ne_pre_20),
            "unmitigated_zeff": 1.5,
            "mitigated_zeff": 1.0,
        },
        exact_tol=float(exact_tol),
        production_claim_allowed=False,
        phase_order_passed=phase_order_passed,
        temperature_passed=temperature_passed,
        current_trace_passed=current_trace_passed,
        halo_force_passed=halo_force_passed,
        mitigation_branch_passed=mitigation_branch_passed,
        passed=passed,
    )


def build_evidence(result: DisruptionSequenceValidationResult, *, target_id: str) -> dict[str, Any]:
    """Build a canonical sealed evidence payload for a validation result."""
    if not isinstance(target_id, str) or not target_id.strip():
        raise ValueError("target_id must be a non-empty string")
    payload: dict[str, Any] = {
        "schema_version": DISRUPTION_SEQUENCE_SCHEMA_VERSION,
        "target_id": target_id.strip(),
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "config": asdict(result.config),
        "phase_order": {
            "unmitigated_tq_duration_ms": result.unmitigated_tq_duration_ms,
            "unmitigated_cq_duration_ms": result.unmitigated_cq_duration_ms,
            "total_duration_rel_error": result.total_duration_rel_error,
            "wall_heat_load_rel_error": result.wall_heat_load_rel_error,
            "current_trace_monotonic": result.current_monotonic,
            "current_trace_finite": result.current_finite,
            "current_initial_rel_error": result.current_initial_rel_error,
        },
        "temperature": {
            "unmitigated_post_tq_temperature_eV": result.unmitigated_post_tq_temperature_eV,
            "mitigated_post_tq_temperature_eV": result.mitigated_post_tq_temperature_eV,
            "passed": result.temperature_passed,
        },
        "halo": {
            "vessel_force_rel_error": result.vessel_force_rel_error,
            "sideways_force_rel_error": result.sideways_force_rel_error,
            "halo_limit_product": result.halo_limit_product,
            "passed": result.halo_force_passed,
        },
        "mitigation": {
            "mitigated_cq_duration_ms": result.mitigated_cq_duration_ms,
            "metadata": result.mitigation_metadata,
            "passed": result.mitigation_branch_passed,
        },
        "exact_tol": result.exact_tol,
        "production_claim_allowed": result.production_claim_allowed,
        "phase_order_passed": result.phase_order_passed,
        "temperature_passed": result.temperature_passed,
        "current_trace_passed": result.current_trace_passed,
        "halo_force_passed": result.halo_force_passed,
        "mitigation_branch_passed": result.mitigation_branch_passed,
        "passed": result.passed,
        "claim_boundary": (
            "Bounded local disruption-sequence validation only; measured labelled disruption windows "
            "and mitigation campaign artefacts remain required for facility claims."
        ),
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Validate a sealed disruption-sequence evidence payload."""
    if payload.get("schema_version") != DISRUPTION_SEQUENCE_SCHEMA_VERSION:
        raise ValueError("unsupported disruption-sequence evidence schema_version")
    digest = payload.get("payload_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError("payload_sha256 must be a SHA-256 hex digest")
    if _payload_sha256(payload) != digest:
        raise ValueError("payload_sha256 does not match canonical payload")
    if payload.get("production_claim_allowed") is not False:
        raise ValueError("production_claim_allowed must remain false for bounded sequence evidence")
    if payload.get("passed") is not True:
        raise ValueError("evidence payload must be passing")
    return True


def write_markdown_report(payload: Mapping[str, Any], output_path: str | Path) -> None:
    """Write a compact human-readable validation report."""
    path = Path(output_path)
    phase = payload["phase_order"]
    halo = payload["halo"]
    mitigation = payload["mitigation"]
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "<!-- Commercial license available -->",
        "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
        "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
        "<!-- ORCID: 0009-0009-3560-0851 -->",
        "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
        "<!-- SCPN Control — Disruption-sequence validation report -->",
        "",
        "# Disruption Sequence Bounded Validation",
        "",
        f"- Status: {'pass' if payload['passed'] else 'fail'}",
        "- Production claim allowed: false",
        f"- Payload SHA-256: `{payload['payload_sha256']}`",
        f"- Generated UTC: {payload['generated_utc']}",
        "",
        "## Checked Contracts",
        "",
        f"- Total-duration relative error: {phase['total_duration_rel_error']:.3e}",
        f"- Wall-heat-load relative error: {phase['wall_heat_load_rel_error']:.3e}",
        f"- Initial-current relative error: {phase['current_initial_rel_error']:.3e}",
        f"- Current trace finite and monotone: {phase['current_trace_finite']} / {phase['current_trace_monotonic']}",
        f"- Vessel-force relative error: {halo['vessel_force_rel_error']:.3e}",
        f"- Sideways-force relative error: {halo['sideways_force_rel_error']:.3e}",
        f"- SPI branch mitigated CQ duration: {mitigation['mitigated_cq_duration_ms']:.6g} ms",
        "",
        "## Claim Boundary",
        "",
        str(payload["claim_boundary"]),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    canonical_payload = {key: value for key, value in payload.items() if key != "payload_sha256"}
    encoded = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relative_error(measured: float, expected: float) -> float:
    return float(abs(float(measured) - float(expected)) / max(abs(float(expected)), 1e-300))


def _positive_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for disruption-sequence bounded validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print the sealed evidence payload as JSON")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for the sealed JSON evidence payload",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Optional path for a human-readable Markdown report",
    )
    args = parser.parse_args(argv)

    result = validate_disruption_sequence()
    payload = build_evidence(result, target_id="disruption-sequence-bounded-contract")
    validate_evidence_payload(payload)

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown_report(payload, args.output_md)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        status = "pass" if payload["passed"] else "fail"
        print(f"Status: {status}")
        print("production claim allowed: false")
        print(f"payload_sha256: {payload['payload_sha256']}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
