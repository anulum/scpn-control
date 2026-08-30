#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — normalized DGKF H-infinity validation.

"""Validate the normalized DGKF controller against independent identities.

The validation exercises the production flight-simulator factory and checks
the normalization identities, both Riccati residuals, the complete central
controller formula, the strict spectral condition, augmented closed-loop
stability, and a dense independent frequency-response sweep. The sweep is a
finite numerical corroboration, not an exact H-infinity norm oracle; theorem
admission rests on the normalized assumptions and strict DGKF existence tests.
No facility, reactor, saturation, sampled-data stability, structured
uncertainty, or classical gain-margin claim is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json

# The validator invokes one fixed local Git metadata command with no caller input.
import subprocess  # nosec B404
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from scpn_control.control.h_infinity_controller import get_flight_sim_controller

H_INFINITY_SCHEMA_VERSION = "scpn-control.h-infinity-validation.v1"
ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE_PATHS = (
    "src/scpn_control/control/h_infinity_controller.py",
    "validation/validate_h_infinity_control.py",
)


@dataclass(frozen=True)
class HInfinityValidationResult:
    """Deterministic metrics for one normalized DGKF synthesis."""

    gamma: float
    normalization_max_residual: float
    riccati_x_relative_residual: float
    riccati_y_relative_residual: float
    controller_formula_relative_error: float
    spectral_feasibility_margin: float
    dominant_closed_loop_real_part: float
    frequency_sweep_peak: float
    frequency_sweep_peak_over_gamma: float
    frequency_samples: int
    passed: bool


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_commit() -> str:
    # The static argv contains no caller-controlled values and never uses a shell.
    completed = subprocess.run(  # nosec B603, B607
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _formula_relative_error(controller: Any) -> float:
    gamma_squared = controller.gamma**2
    expected_f = -controller.B2.T @ controller.X
    expected_l = -controller.Y @ controller.C2.T
    expected_z = np.linalg.solve(
        np.eye(controller.n) - controller.Y @ controller.X / gamma_squared,
        np.eye(controller.n),
    )
    expected_ak = (
        controller.A
        + controller.B1 @ controller.B1.T @ controller.X / gamma_squared
        + controller.B2 @ expected_f
        + expected_z @ expected_l @ controller.C2
    )
    expected_bk = -expected_z @ expected_l
    differences = np.concatenate(
        (
            (controller.F - expected_f).ravel(),
            (controller.L - expected_l).ravel(),
            (controller.Z - expected_z).ravel(),
            (controller.Ak - expected_ak).ravel(),
            (controller.Bk - expected_bk).ravel(),
            (controller.Ck - expected_f).ravel(),
        )
    )
    references = np.concatenate(
        (
            expected_f.ravel(),
            expected_l.ravel(),
            expected_z.ravel(),
            expected_ak.ravel(),
            expected_bk.ravel(),
            expected_f.ravel(),
        )
    )
    reference_norm = float(np.linalg.norm(references))
    return float(np.linalg.norm(differences) / max(1.0, reference_norm))


def _frequency_peak(controller: Any, frequencies: npt.NDArray[np.float64]) -> float:
    state, disturbance, performance, feedthrough = controller.closed_loop_realization()
    identity = np.eye(state.shape[0])
    peak = 0.0
    for frequency in frequencies:
        transfer = performance @ np.linalg.solve(1j * frequency * identity - state, disturbance) + feedthrough
        peak = max(peak, float(np.linalg.svd(transfer, compute_uv=False)[0]))
    return peak


def validate_h_infinity_control() -> HInfinityValidationResult:
    """Run the bounded normalized-DGKF validation."""
    controller = get_flight_sim_controller()
    normalization_max = max(controller.normalization_residual_norms())
    residual_x, residual_y = controller.riccati_residual_norms()
    x_scale = 1.0 + np.linalg.norm(controller.C1.T @ controller.C1, ord="fro")
    y_scale = 1.0 + np.linalg.norm(controller.B1 @ controller.B1.T, ord="fro")
    relative_x = float(residual_x / x_scale)
    relative_y = float(residual_y / y_scale)
    formula_error = _formula_relative_error(controller)
    dominant_pole = float(np.max(np.real(controller.closed_loop_eigenvalues)))
    frequencies = np.concatenate(([0.0], np.logspace(-4, 6, 20_001)))
    frequency_peak = _frequency_peak(controller, frequencies)
    peak_ratio = frequency_peak / controller.gamma
    passed = bool(
        normalization_max <= 1.0e-12
        and relative_x <= 1.0e-8
        and relative_y <= 1.0e-8
        and formula_error <= 1.0e-12
        and controller.robust_feasibility_margin() > 0.0
        and dominant_pole < 0.0
        and frequency_peak < controller.gamma
    )
    return HInfinityValidationResult(
        gamma=float(controller.gamma),
        normalization_max_residual=float(normalization_max),
        riccati_x_relative_residual=relative_x,
        riccati_y_relative_residual=relative_y,
        controller_formula_relative_error=formula_error,
        spectral_feasibility_margin=controller.robust_feasibility_margin(),
        dominant_closed_loop_real_part=dominant_pole,
        frequency_sweep_peak=frequency_peak,
        frequency_sweep_peak_over_gamma=float(peak_ratio),
        frequency_samples=int(frequencies.size),
        passed=passed,
    )


def build_evidence(
    result: HInfinityValidationResult,
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a digest-sealed bounded-model evidence payload."""
    timestamp = generated_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    payload: dict[str, Any] = {
        "schema_version": H_INFINITY_SCHEMA_VERSION,
        "generated_at": timestamp,
        "source_commit": _source_commit(),
        "runtime_source_sha256": {path: _sha256(ROOT / path) for path in RUNTIME_SOURCE_PATHS},
        "precision": "float64",
        "reference": {
            "title": "State-space solutions to standard H2 and H-infinity control problems",
            "authors": "Doyle, Glover, Khargonekar, Francis",
            "doi": "10.1109/9.29425",
            "result": "Theorem 3 normalized central controller",
        },
        "claim_boundary": {
            "model": "normalized continuous-time standard plant; D11=D22=0",
            "scientific_admission": True,
            "public_claim_allowed": False,
            "production_admission": False,
            "excluded": [
                "facility or reactor validation",
                "saturated H-infinity guarantee",
                "arbitrary sampled-data stability",
                "structured uncertainty or D-K synthesis",
                "classical gain margin",
            ],
            "frequency_sweep_classification": "finite numerical corroboration, not exact norm proof",
        },
        "result": asdict(result),
    }
    payload["payload_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Validate schema, seal, required source digests, and passing status."""
    if payload.get("schema_version") != H_INFINITY_SCHEMA_VERSION:
        raise ValueError("unsupported H-infinity evidence schema_version")
    provided_digest = payload.get("payload_sha256")
    if not isinstance(provided_digest, str) or len(provided_digest) != 64:
        raise ValueError("payload_sha256 must be a SHA-256 hex digest")
    unsigned = dict(payload)
    unsigned.pop("payload_sha256", None)
    expected_digest = hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    if provided_digest != expected_digest:
        raise ValueError("payload_sha256 does not match payload bytes")
    source_digests = payload.get("runtime_source_sha256")
    if not isinstance(source_digests, dict) or set(source_digests) != set(RUNTIME_SOURCE_PATHS):
        raise ValueError("runtime_source_sha256 does not cover the exact owner set")
    for source_path, digest in source_digests.items():
        if digest != _sha256(ROOT / source_path):
            raise ValueError(f"runtime source digest mismatch: {source_path}")
    result = payload.get("result")
    if not isinstance(result, dict) or result.get("passed") is not True:
        raise ValueError("H-infinity validation result is not passing")
    return True


def _markdown(payload: Mapping[str, Any]) -> str:
    result = payload["result"]
    boundary = payload["claim_boundary"]
    return f"""# Normalized DGKF H-infinity validation

- Generated: `{payload["generated_at"]}`
- Source commit: `{payload["source_commit"]}`
- Schema: `{payload["schema_version"]}`
- Payload seal: `{payload["payload_sha256"]}`
- Overall: `{"PASS" if result["passed"] else "FAIL"}`

| Check | Result |
|---|---:|
| Admitted gamma | {result["gamma"]:.12g} |
| Maximum normalization residual | {result["normalization_max_residual"]:.3e} |
| X Riccati relative residual | {result["riccati_x_relative_residual"]:.3e} |
| Y Riccati relative residual | {result["riccati_y_relative_residual"]:.3e} |
| Central-controller formula relative error | {result["controller_formula_relative_error"]:.3e} |
| Spectral feasibility margin | {result["spectral_feasibility_margin"]:.12g} |
| Dominant augmented closed-loop pole real part | {result["dominant_closed_loop_real_part"]:.12g} |
| Frequency-sweep peak | {result["frequency_sweep_peak"]:.12g} |
| Frequency-sweep peak / gamma | {result["frequency_sweep_peak_over_gamma"]:.12g} |
| Frequency samples | {result["frequency_samples"]} |

## Claim boundary

This admits only `{boundary["model"]}`. The frequency sweep is
`{boundary["frequency_sweep_classification"]}`. Production admission is
`{str(boundary["production_admission"]).lower()}`. Excluded claims:

""" + "".join(f"- {item}\n" for item in boundary["excluded"])


def write_reports(payload: Mapping[str, Any], json_path: Path, markdown_path: Path) -> None:
    """Write canonical JSON and Markdown evidence reports."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(payload), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run validation, validate its seal, and optionally write reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "validation" / "reports" / "h_infinity_control.json",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=ROOT / "validation" / "reports" / "h_infinity_control.md",
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    result = validate_h_infinity_control()
    payload = build_evidence(result)
    validate_evidence_payload(payload)
    if not args.no_write:
        write_reports(payload, args.json_out, args.markdown_out)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
