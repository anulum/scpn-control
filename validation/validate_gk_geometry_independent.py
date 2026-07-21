#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — independent Miller geometry cross-validation
"""Cross-validate production Miller geometry against an independent reference.

The production analytic metric in :mod:`scpn_control.core.gk_geometry` is compared
against the structurally independent finite-difference reference in
:mod:`gk_geometry_independent_reference` across circular, shaped, high-shear, and
finite shaping-shear (``s_kappa``/``s_delta``) local equilibria. Agreement within a
physical tolerance is the evidence that closes the Miller geometry validation gap
for tracker #47; the reference differentiates the flux-surface definition directly
and therefore cannot mask a shared analytic-derivative error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from numpy.typing import NDArray

from scpn_control.core.gk_geometry import miller_geometry  # noqa: E402
from validation.gk_geometry_independent_reference import independent_miller_metric  # noqa: E402

FloatArray = NDArray[np.float64]

_REPORT_SCHEMA = "scpn-control.gk-geometry-independent-crosscheck.v1"
_REL_TOLERANCE = 1.0e-6
_ABS_TOLERANCE = 1.0e-8
_N_THETA = 128
_N_PERIOD = 1
_COMPARE_FIELDS = ("R", "Z", "jacobian", "g_rr", "g_rt", "g_tt", "B_toroidal", "b_dot_grad_theta")
_UNITS = {
    "R": "m",
    "Z": "m",
    "jacobian": "m2",
    "g_rr": "dimensionless",
    "g_rt": "m-1",
    "g_tt": "m-2",
    "B_toroidal": "T",
    "b_dot_grad_theta": "m-1",
}

# Cases span circular, fixed-shaping, high magnetic shear, and both signs of
# finite shaping-shear so the s_kappa/s_delta radial-derivative terms are exercised.
_CASES: tuple[dict[str, Any], ...] = (
    {
        "case": "circular_cyclone_limit",
        "parameters": {
            "R0": 2.78,
            "a": 1.0,
            "rho": 0.5,
            "kappa": 1.0,
            "delta": 0.0,
            "s_kappa": 0.0,
            "s_delta": 0.0,
            "q": 1.4,
            "dR_dr": 0.0,
            "B0": 2.0,
        },
    },
    {
        "case": "shaped_positive_triangularity",
        "parameters": {
            "R0": 2.78,
            "a": 1.0,
            "rho": 0.5,
            "kappa": 1.7,
            "delta": 0.3,
            "s_kappa": 0.0,
            "s_delta": 0.0,
            "q": 1.9,
            "dR_dr": -0.08,
            "B0": 2.0,
        },
    },
    {
        "case": "high_shear_local_equilibrium",
        "parameters": {
            "R0": 2.78,
            "a": 1.0,
            "rho": 0.6,
            "kappa": 1.85,
            "delta": 0.25,
            "s_kappa": 0.0,
            "s_delta": 0.0,
            "q": 3.2,
            "dR_dr": -0.12,
            "B0": 2.0,
        },
    },
    {
        "case": "positive_shaping_shear",
        "parameters": {
            "R0": 2.78,
            "a": 1.0,
            "rho": 0.5,
            "kappa": 1.7,
            "delta": 0.3,
            "s_kappa": 0.4,
            "s_delta": 0.3,
            "q": 1.9,
            "dR_dr": -0.08,
            "B0": 2.0,
        },
    },
    {
        "case": "negative_shaping_shear",
        "parameters": {
            "R0": 2.78,
            "a": 1.0,
            "rho": 0.5,
            "kappa": 1.6,
            "delta": -0.2,
            "s_kappa": -0.35,
            "s_delta": -0.25,
            "q": 2.1,
            "dR_dr": -0.05,
            "B0": 2.0,
        },
    },
)


def _production_fields(parameters: dict[str, Any]) -> dict[str, Any]:
    geometry = miller_geometry(**parameters, n_theta=_N_THETA, n_period=_N_PERIOD)
    B0 = float(parameters["B0"])
    R0 = float(parameters["R0"])
    return {
        "theta": np.asarray(geometry.theta, dtype=np.float64),
        "R": np.asarray(geometry.R, dtype=np.float64),
        "Z": np.asarray(geometry.Z, dtype=np.float64),
        "jacobian": np.asarray(geometry.jacobian, dtype=np.float64),
        "g_rr": np.asarray(geometry.g_rr, dtype=np.float64),
        "g_rt": np.asarray(geometry.g_rt, dtype=np.float64),
        "g_tt": np.asarray(geometry.g_tt, dtype=np.float64),
        "B_toroidal": B0 * R0 / np.asarray(geometry.R, dtype=np.float64),
        "b_dot_grad_theta": np.asarray(geometry.b_dot_grad_theta, dtype=np.float64),
    }


def _reference_fields(parameters: dict[str, Any], theta: FloatArray) -> dict[str, FloatArray]:
    ref = independent_miller_metric(theta=theta, **parameters)
    return {
        "R": ref.R,
        "Z": ref.Z,
        "jacobian": ref.jacobian,
        "g_rr": ref.g_rr,
        "g_rt": ref.g_rt,
        "g_tt": ref.g_tt,
        "B_toroidal": ref.B_toroidal,
        "b_dot_grad_theta": ref.b_dot_grad_theta,
    }


def _field_agreement(production: FloatArray, reference: FloatArray) -> tuple[float, float, bool]:
    diff = np.abs(production - reference)
    max_abs = float(np.max(diff))
    denom = np.abs(reference)
    mask = denom > _ABS_TOLERANCE
    max_rel = float(np.max(diff[mask] / denom[mask])) if bool(np.any(mask)) else 0.0
    agrees = bool(np.all(np.isclose(production, reference, rtol=_REL_TOLERANCE, atol=_ABS_TOLERANCE)))
    return max_abs, max_rel, agrees


def validate_gk_geometry_independent() -> dict[str, Any]:
    """Compare production Miller geometry against the independent reference."""
    report = _new_report()
    entries: list[dict[str, Any]] = report["entries"]
    findings: list[str] = report["findings"]

    for case in _CASES:
        name = str(case["case"])
        parameters = dict(case["parameters"])
        production = _production_fields(parameters)
        reference = _reference_fields(parameters, production["theta"])
        field_results: dict[str, Any] = {}
        case_ok = True
        for field in _COMPARE_FIELDS:
            max_abs, max_rel, agrees = _field_agreement(production[field], reference[field])
            field_results[field] = {"max_abs_error": max_abs, "max_rel_error": max_rel, "agrees": agrees}
            if not agrees:
                case_ok = False
                findings.append(
                    f"{name}.{field}: production diverges from independent reference "
                    f"(max_abs={max_abs:.3e}, max_rel={max_rel:.3e})"
                )
        shaped = abs(float(parameters["s_kappa"])) + abs(float(parameters["s_delta"])) > 0.0
        entry = {
            "case": name,
            "samples": int(production["theta"].size),
            "shaping_shear_active": shaped,
            "fields": field_results,
            "status": "pass" if case_ok else "fail",
            "case_sha256": _json_sha256({"case": name, "parameters": parameters}),
        }
        entries.append(entry)
        if not case_ok:
            report["status"] = "fail"

    report["cases"] = len(entries)
    return _finalise_report(report)


def _new_report() -> dict[str, Any]:
    return {
        "schema_version": _REPORT_SCHEMA,
        "status": "pass",
        "reference_implementation": "validation/gk_geometry_independent_reference.py",
        "method": "fourth-order central finite differences of the Miller flux-surface definition",
        "model_reference": "Miller et al., Phys. Plasmas 5 (1998) 973",
        "tolerances": {"absolute": _ABS_TOLERANCE, "relative": _REL_TOLERANCE},
        "grid": {"n_theta": _N_THETA, "n_period": _N_PERIOD},
        "units": dict(_UNITS),
        "public_claims": {
            "bounded_local_miller_geometry_independently_verified": False,
            "full_equilibrium_reconstruction": False,
            "full_equilibrium_blocked_reason": (
                "Local Miller metric only; full equilibrium reconstruction requires external equilibrium-code evidence."
            ),
        },
        "cases": 0,
        "entries": [],
        "findings": [],
        "payload_sha256": None,
    }


def _json_sha256(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _finalise_report(report: dict[str, Any]) -> dict[str, Any]:
    report["public_claims"]["bounded_local_miller_geometry_independently_verified"] = report["status"] == "pass"
    payload = dict(report)
    payload["payload_sha256"] = None
    report["payload_sha256"] = _json_sha256(payload)
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the cross-check and optionally persist the evidence report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", help="Write JSON report to this path")
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report to stdout")
    args = parser.parse_args(argv)

    report = validate_gk_geometry_independent()
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"GK geometry independent cross-check: {report['status']} cases={report['cases']}")
        for finding in report["findings"]:
            print(f"FINDING {finding}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
