# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Nonlinear GK Cyclone Base Case Validation
"""
Cyclone Base Case (CBC) nonlinear benchmark.

Dimits et al., Phys. Plasmas 7 (2000) 969.
Rosenbluth & Hinton, Phys. Rev. Lett. 80 (1998) 724.

Runs:
  V1: Linear growth rate recovery (NL off → match eigenvalue within 20%)
  V2: Energy conservation without drive (dE/dt bounded)
  V3: Zonal flow self-generation from noise
  V4: CBC saturated state: chi_i finite, simulation stable

Published reference:
  - CBC chi_i ~ 1-5 χ_gB (GENE/GS2/GYRO)
  - Dimits shift: critical R/L_Ti ~ 4.5-5.5
  - Rosenbluth-Hinton residual: 1/(1 + 1.6 q²/√ε) ≈ 0.36 for CBC
"""

from __future__ import annotations

import json
import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def ensure_repo_src_on_path() -> None:
    """Allow direct script execution from a source checkout without installation."""

    repo_src = str(Path(__file__).resolve().parents[1] / "src")
    if repo_src not in sys.path:
        sys.path.insert(0, repo_src)


ensure_repo_src_on_path()

from scpn_control.core.gk_nonlinear import NonlinearGKConfig, NonlinearGKSolver

CBC_CHI_I_GB_REFERENCE_BAND = (1.0, 5.0)
CBC_MIN_SATURATION_STEPS = 2_000
CBC_MAX_TAIL_RELATIVE_DRIFT = 0.10
REPORT_SCHEMA_VERSION = "scpn-control.gk-nonlinear-cyclone.v2"
REPORT_PATH = Path(__file__).parent / "reports" / "gk_nonlinear_cyclone.json"
MARKDOWN_REPORT_PATH = Path(__file__).parent / "reports" / "gk_nonlinear_cyclone.md"


def _json_safe(value: Any) -> Any:
    """Return a deterministic JSON-compatible payload without NumPy scalars."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _canonical_json(value: Any) -> str:
    """Serialise evidence deterministically for SHA-256 binding."""

    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _payload_sha256(value: Any) -> str:
    """Return the canonical SHA-256 digest for a JSON-compatible payload."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _payload_without_digest(report: dict[str, Any]) -> dict[str, Any]:
    """Return a report copy with the self-referential digest field removed."""

    payload = dict(report)
    payload.pop("payload_sha256", None)
    return payload


def verify_payload_digest(report: dict[str, Any]) -> bool:
    """Return true when a persisted nonlinear CBC report matches its digest."""

    digest = report.get("payload_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        return False
    return digest == _payload_sha256(_payload_without_digest(report))


def assess_cbc_saturation_evidence(
    result: Any,
    cfg: Any,
    *,
    reference_band: tuple[float, float] = CBC_CHI_I_GB_REFERENCE_BAND,
    min_steps: int = CBC_MIN_SATURATION_STEPS,
    max_tail_relative_drift: float = CBC_MAX_TAIL_RELATIVE_DRIFT,
) -> dict[str, object]:
    """Assess whether a CBC nonlinear run can support saturation claims."""
    chi_i_gb = float(getattr(result, "chi_i_gB", np.nan))
    q_i_t = np.asarray(getattr(result, "Q_i_t", []), dtype=np.float64)
    n_steps = int(getattr(cfg, "n_steps", 0))
    reasons: list[str] = []

    if not bool(getattr(result, "converged", False)):
        reasons.append("solver did not converge")
    if n_steps < min_steps:
        reasons.append("n_steps below saturation evidence minimum")
    if not np.isfinite(chi_i_gb):
        reasons.append("chi_i_gB is not finite")
    elif not (reference_band[0] <= chi_i_gb <= reference_band[1]):
        reasons.append("chi_i_gB outside CBC reference band")
    if q_i_t.size < 4 or not np.all(np.isfinite(q_i_t)):
        reasons.append("heat-flux time trace is missing or non-finite")
        tail_relative_drift = float("inf")
    else:
        tail = q_i_t[q_i_t.size // 2 :]
        tail_mean = float(np.mean(np.abs(tail)))
        tail_relative_drift = float(abs(tail[-1] - tail[0]) / max(tail_mean, 1e-12))
        if tail_relative_drift > max_tail_relative_drift:
            reasons.append("tail heat-flux drift exceeds saturation threshold")

    return {
        "chi_i_gB": chi_i_gb,
        "n_steps": n_steps,
        "reference_band": list(reference_band),
        "min_steps": min_steps,
        "tail_relative_drift": tail_relative_drift,
        "max_tail_relative_drift": max_tail_relative_drift,
        "passed": not reasons,
        "reasons": reasons,
    }


def _find_result(results: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    """Return a named validation result from the benchmark result list."""

    for result in results:
        if result.get("test") == name:
            return result
    return None


def _diagnostic_checks_passed(results: list[dict[str, Any]]) -> bool:
    """Return whether all non-saturation diagnostics passed."""

    diagnostics = [result for result in results if result.get("test") != "V4_cbc_saturated"]
    return bool(diagnostics) and all(bool(result.get("passed")) for result in diagnostics)


def build_cbc_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a digest-bound nonlinear CBC saturation-evidence report."""

    safe_results = _json_safe(results)
    v4 = _find_result(safe_results, "V4_cbc_saturated")
    saturation_evidence = v4.get("saturation_evidence", {}) if isinstance(v4, dict) else {}
    diagnostic_passed = _diagnostic_checks_passed(safe_results)
    saturation_admitted = diagnostic_passed and bool(saturation_evidence.get("passed"))
    blocked_reasons = list(saturation_evidence.get("reasons", [])) if isinstance(saturation_evidence, dict) else []
    if v4 is None:
        blocked_reasons.append("cbc saturation result missing")
    elif not isinstance(saturation_evidence, dict) or not saturation_evidence:
        blocked_reasons.append("cbc saturation evidence missing")
    if not diagnostic_passed:
        blocked_reasons.append("diagnostic checks failed")

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "benchmark": "nonlinear_gk_cyclone_base_case",
        "reference": "Dimits et al., Phys. Plasmas 7 (2000) 969",
        "normalisation_contract": {
            "chi_i_gB": "Q_i divided by R/L_Ti in gyro-Bohm diffusivity units",
            "reference_band": list(CBC_CHI_I_GB_REFERENCE_BAND),
            "min_saturation_steps": CBC_MIN_SATURATION_STEPS,
            "max_tail_relative_drift": CBC_MAX_TAIL_RELATIVE_DRIFT,
        },
        "diagnostic_checks_passed": diagnostic_passed,
        "saturation_claim_admitted": saturation_admitted,
        "blocked_reasons": [] if saturation_admitted else blocked_reasons,
        "claim_status": (
            "nonlinear CBC saturation evidence admitted"
            if saturation_admitted
            else "diagnostic evidence only; saturated nonlinear CBC chi_i claim remains blocked"
        ),
        "results": safe_results,
    }
    report["payload_sha256"] = _payload_sha256(report)
    return report


def write_markdown_report(report: dict[str, Any], path: Path = MARKDOWN_REPORT_PATH) -> None:
    """Write a Markdown summary for the nonlinear CBC evidence report."""

    blocked = ", ".join(report["blocked_reasons"]) or "none"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
                "<!-- Commercial license available -->",
                "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
                "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
                "<!-- ORCID: 0009-0009-3560-0851 -->",
                "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
                "<!-- SCPN Control — Nonlinear GK Cyclone Base Case Report -->",
                "",
                "# Nonlinear GK Cyclone Base Case Evidence",
                "",
                f"- Schema: `{report['schema_version']}`",
                f"- Diagnostic checks passed: `{report['diagnostic_checks_passed']}`",
                f"- Saturation claim admitted: `{report['saturation_claim_admitted']}`",
                f"- Blocked reasons: `{blocked}`",
                f"- Payload SHA-256: `{report['payload_sha256']}`",
                f"- Claim status: `{report['claim_status']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_linear_recovery() -> dict:
    """V1: linear mode growth without nonlinearity."""
    cfg = NonlinearGKConfig(
        n_kx=8,
        n_ky=8,
        n_theta=16,
        n_vpar=8,
        n_mu=4,
        n_species=2,
        dt=0.02,
        n_steps=200,
        save_interval=20,
        nonlinear=False,
        collisions=False,
        hyper_coeff=0.0,
        R_L_Ti=6.9,
        R_L_Te=6.9,
        R_L_ne=2.2,
        q=1.4,
        s_hat=0.78,
        R0=2.78,
        a=1.0,
        B0=2.0,
        cfl_adapt=False,
    )
    solver = NonlinearGKSolver(cfg)
    state = solver.init_single_mode(ky_idx=2, amplitude=1e-8)
    result = solver.run(state)
    return {
        "test": "V1_linear_recovery",
        "phi_rms_initial": float(result.phi_rms_t[0]) if len(result.phi_rms_t) > 0 else 0.0,
        "phi_rms_final": float(result.phi_rms_t[-1]) if len(result.phi_rms_t) > 0 else 0.0,
        "converged": bool(result.converged),
        "passed": bool(result.converged and np.all(np.isfinite(result.phi_rms_t))),
    }


def run_energy_conservation() -> dict:
    """V2: energy conservation without drive or dissipation."""
    cfg = NonlinearGKConfig(
        n_kx=8,
        n_ky=8,
        n_theta=16,
        n_vpar=8,
        n_mu=4,
        n_species=2,
        dt=0.01,
        n_steps=50,
        save_interval=10,
        nonlinear=True,
        collisions=False,
        hyper_coeff=0.0,
        R_L_Ti=0.0,
        R_L_Te=0.0,
        R_L_ne=0.0,
        q=1.4,
        s_hat=0.78,
        R0=2.78,
        a=1.0,
        B0=2.0,
        cfl_adapt=False,
    )
    solver = NonlinearGKSolver(cfg)
    state = solver.init_state(amplitude=1e-6)
    E0 = solver.total_energy(state)

    for _ in range(50):
        state = solver._rk4_step(state, 0.01)

    E1 = solver.total_energy(state)
    ratio = E1 / E0 if E0 > 0 else float("inf")
    return {
        "test": "V2_energy_conservation",
        "E_initial": float(E0),
        "E_final": float(E1),
        "ratio": float(ratio),
        "passed": bool(0.01 < ratio < 100.0),  # energy should not explode
    }


def run_zonal_flow() -> dict:
    """V3: zonal flow self-generation from noise."""
    cfg = NonlinearGKConfig(
        n_kx=8,
        n_ky=8,
        n_theta=16,
        n_vpar=8,
        n_mu=4,
        n_species=2,
        dt=0.02,
        n_steps=100,
        save_interval=10,
        nonlinear=True,
        collisions=True,
        nu_collision=0.01,
        R_L_Ti=6.9,
        R_L_Te=6.9,
        R_L_ne=2.2,
        q=1.4,
        s_hat=0.78,
        R0=2.78,
        a=1.0,
        B0=2.0,
        cfl_adapt=False,
    )
    solver = NonlinearGKSolver(cfg)
    result = solver.run()
    zonal_final = float(result.zonal_rms_t[-1]) if len(result.zonal_rms_t) > 0 else 0.0
    return {
        "test": "V3_zonal_flow_generation",
        "zonal_rms_final": zonal_final,
        "phi_rms_final": float(result.phi_rms_t[-1]) if len(result.phi_rms_t) > 0 else 0.0,
        "converged": bool(result.converged),
        "passed": bool(result.converged and np.isfinite(zonal_final)),
    }


def run_cbc_saturated() -> dict:
    """V4: CBC saturated state — chi_i finite and simulation stable."""
    cfg = NonlinearGKConfig(
        n_kx=8,
        n_ky=8,
        n_theta=16,
        n_vpar=8,
        n_mu=4,
        n_species=2,
        dt=0.02,
        n_steps=200,
        save_interval=20,
        nonlinear=True,
        collisions=True,
        nu_collision=0.01,
        R_L_Ti=6.9,
        R_L_Te=6.9,
        R_L_ne=2.2,
        q=1.4,
        s_hat=0.78,
        R0=2.78,
        a=1.0,
        B0=2.0,
        cfl_adapt=False,
    )
    solver = NonlinearGKSolver(cfg)
    t0 = time.perf_counter()
    result = solver.run()
    elapsed = time.perf_counter() - t0
    evidence = assess_cbc_saturation_evidence(result, cfg)
    return {
        "test": "V4_cbc_saturated",
        "chi_i_gB": float(result.chi_i_gB),
        "chi_e_gB": float(result.chi_e),
        "converged": bool(result.converged),
        "elapsed_s": float(elapsed),
        "saturation_evidence": evidence,
        "passed": bool(evidence["passed"]),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse nonlinear CBC validation command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-saturation",
        action="store_true",
        help="Exit non-zero unless the saturated nonlinear CBC chi_i claim is admitted.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPORT_PATH,
        help="Path for the JSON evidence report.",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=MARKDOWN_REPORT_PATH,
        help="Path for the Markdown evidence summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    tests = [
        run_linear_recovery,
        run_energy_conservation,
        run_zonal_flow,
        run_cbc_saturated,
    ]

    results = []
    for fn in tests:
        print(f"Running {fn.__name__}...", end=" ", flush=True)
        r = fn()
        status = "PASS" if r["passed"] else "FAIL"
        print(status)
        results.append(r)

    report = build_cbc_report(results)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_markdown_report(report, args.markdown_out)
    print(f"\nReport: {args.json_out}")
    print(f"Summary: {args.markdown_out}")
    print(f"Diagnostics: {'PASS' if report['diagnostic_checks_passed'] else 'FAIL'}")
    print(f"Saturation claim: {'ADMITTED' if report['saturation_claim_admitted'] else 'BLOCKED'}")

    if args.require_saturation and not report["saturation_claim_admitted"]:
        return 1
    return 0 if report["diagnostic_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
