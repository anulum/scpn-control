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
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from scpn_control.control.h_infinity_controller import get_flight_sim_controller
from validation import h_infinity_evidence as _evidence
from validation.report_output_paths import checked_report_destination

H_INFINITY_SCHEMA_VERSION = _evidence.SCHEMA_VERSION
ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE_PATHS = _evidence.RUNTIME_SOURCE_PATHS


@dataclass(frozen=True)
class HInfinityValidationResult:
    """Frozen metrics for one normalized DGKF synthesis and finite sweep.

    Relative residuals, formula error and peak/gamma ratio are dimensionless.
    The feasibility margin is gamma squared minus rho(XY), in squared normalized
    gain units. The dominant pole real part uses s^-1; gamma and sweep peak use
    normalized gain units. ``frequency_samples`` counts the fixed sweep and
    ``passed`` declares its bounded checks. Construction is unchecked; report
    serialisation verifies domains and verdict consistency before returning.
    """

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
    """Observe local file bytes by SHA-256; filesystem errors propagate."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_commit() -> str:
    """Observe repository HEAD through the fixed native Git command."""
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
    """Retain the historical compact UTF-8 serializer used by existing callers."""
    return _evidence.canonical_payload_bytes(payload)


def _formula_relative_error(controller: Any) -> float:
    """Compare all central-controller matrices with the normalized DGKF formula."""
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
    """Return the largest sampled singular gain at the supplied frequencies."""
    state, disturbance, performance, feedthrough = controller.closed_loop_realization()
    identity = np.eye(state.shape[0])
    peak = 0.0
    for frequency in frequencies:
        transfer = performance @ np.linalg.solve(1j * frequency * identity - state, disturbance) + feedthrough
        peak = max(peak, float(np.linalg.svd(transfer, compute_uv=False)[0]))
    return peak


def validate_h_infinity_control() -> HInfinityValidationResult:
    """Run the actual flight-simulator factory and fixed 20002-frequency sweep.

    Return immutable float64 metrics with the original residual, feasibility,
    stability and sampled-gain thresholds. Frequencies include zero and 20001
    logarithmic points from 1e-4 to 1e6 rad/s. This call writes no files. Native
    synthesis/linear-algebra failures propagate, and finite sampling supplies
    corroboration without an exact norm or facility-control admission.
    """
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
    """Serialise consistent metrics with actual HEAD/source-byte observations.

    ``generated_at=None`` observes the UTC clock; a supplied timestamp must be
    aware UTC. Returned v1 data is detached and sealed with historical compact
    UTF-8 JSON. All three declared runtime owners are hashed, including the
    decoder. Coherent failing results declare local scientific admission False;
    public and production admission always remain False. ValueError rejects
    malformed domains/verdicts; file/Git errors propagate. No files are written,
    clean-tree guarantee, producer authentication or independent run is supplied.
    """
    timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z") if generated_at is None else generated_at
    payload: dict[str, Any] = {
        "schema_version": H_INFINITY_SCHEMA_VERSION,
        "generated_at": timestamp,
        "source_commit": _source_commit(),
        "runtime_source_sha256": {path: _sha256(ROOT / path) for path in RUNTIME_SOURCE_PATHS},
        "precision": "float64",
        "reference": dict(_evidence.REFERENCE),
        "claim_boundary": {
            "model": _evidence.MODEL,
            "scientific_admission": result.passed,
            "public_claim_allowed": False,
            "production_admission": False,
            "excluded": list(_evidence.EXCLUDED),
            "frequency_sweep_classification": _evidence.SWEEP_CLASSIFICATION,
        },
        "result": asdict(result),
    }
    payload["payload_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    _evidence.inspect_evidence_payload(payload, expected_sources=payload["runtime_source_sha256"])
    return payload


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Admit only a consistent passing bounded report bound to current source bytes.

    Return True after exact v1 structure, finite metrics, original thresholds,
    fixed sweep, ratio and restricted claim checks. Invalid or failing reports
    raise ValueError. Three local source files are observed afresh; OSError
    propagates. The capture-time commit label is format-checked, not required to
    equal a later HEAD with unchanged source bytes. Sequential reads provide no coherent
    snapshot or producer/facility/run authentication. Old source-bound reports
    remain historical and can fail this current-source check.
    """
    passed = _evidence.inspect_evidence_payload(
        payload,
        expected_sources={path: _sha256(ROOT / path) for path in RUNTIME_SOURCE_PATHS},
    )
    if not passed:
        raise ValueError("H-infinity validation result is not passing")
    return True


def _markdown(payload: Mapping[str, Any]) -> str:
    """Render checked bounded-model values and their explicit claim exclusions."""
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
    """Check a coherent report and directly replace distinct JSON/Markdown outputs.

    Relative paths use cwd, parents are created and existing files overwritten.
    Outputs must not alias each other or the three observed runtime sources;
    path/report refusals raise ValueError before writes. OSError/RuntimeError
    from filesystem inspection or writes propagate.
    A failure after the JSON write can leave a partial pair. Both coherent pass
    and fail reports are supported, without rollback or facility admission.
    """
    protected = [ROOT / name for name in RUNTIME_SOURCE_PATHS]
    checked_report_destination(json_path, inputs=[markdown_path, *protected])
    checked_report_destination(markdown_path, inputs=[json_path, *protected])
    _evidence.inspect_evidence_payload(
        payload,
        expected_sources={path: _sha256(ROOT / path) for path in RUNTIME_SOURCE_PATHS},
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(payload), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixed real validator or read-only check of an existing sealed report.

    ``--check-report PATH`` reads unique-key UTF-8 JSON, checks current sources
    and passing bounded metrics, and prints the accepted payload. It never
    writes outputs. Without it, run the fixed 20002-sample producer and write the
    requested JSON/Markdown pair unless ``--no-write`` is supplied. Existing
    default paths remain under validation/reports; relative paths use cwd.
    Return 0 for pass, 1 for coherent failure or authored report/IO/Git refusal.
    Argparse raises SystemExit(0) for help or (2) for malformed arguments. No old
    report is resealed by check mode and no facility-control claim is admitted.
    """
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
    parser.add_argument("--check-report", type=Path, help="check an existing sealed report without writing outputs")
    args = parser.parse_args(argv)
    try:
        if args.check_report is not None:
            payload = _evidence.read_report(args.check_report)
            validate_evidence_payload(payload)
            passed = True
        else:
            result = validate_h_infinity_control()
            payload = build_evidence(result)
            passed = result.passed
            if not args.no_write:
                write_reports(payload, args.json_out, args.markdown_out)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"H-infinity validation refused: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
