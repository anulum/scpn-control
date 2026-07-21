#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — independent ballooning-boundary cross-validation
"""Cross-validate the analytic s-alpha ballooning boundary against a numerical solve.

The production ``ballooning_stability`` in :mod:`scpn_control.core.stability_mhd`
uses the analytic first-stability approximation ``alpha_crit = s(1 - s/2)`` (low
shear) / ``0.6 s`` (high shear). This validator compares that approximation, at a
range of magnetic shear values, against the structurally independent numerical
solve of the s-alpha ballooning ODE in :mod:`ballooning_independent_reference`.

The evidence is a BOUNDED agreement (the analytic fit is an approximation, not an
identity): in the first-stability regime ``s in [0.5, 2.0]`` the analytic
``alpha_crit`` is required to track the numerically resolved marginal boundary to
within a documented tolerance. The low-shear second-stability regime (``s`` <~ 0.4)
is excluded because the single first-boundary picture no longer holds there; that
exclusion is recorded, not hidden. The public ideal-MHD stability claim stays
blocked — this is an internal-consistency bound, not external-code validation.
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

from scpn_control.core.stability_mhd import QProfile, ballooning_stability  # noqa: E402
from validation.ballooning_independent_reference import ballooning_alpha_crit  # noqa: E402

_REPORT_SCHEMA = "scpn-control.stability-ballooning-independent-crosscheck.v1"
# The analytic fit approximates the numerical boundary; the prototype spread over
# s in [0.5, 2.0] is <~5%, so a 10% band bounds it with margin.
_REL_TOLERANCE = 0.10
# First-stability regime where the single-boundary picture is well defined.
_SHEAR_VALUES = (0.5, 0.7, 1.0, 1.5, 2.0)


def _module_alpha_crit(shear_values: tuple[float, ...]) -> list[float]:
    """Read the production analytic alpha_crit at each shear via ballooning_stability."""
    n = len(shear_values)
    rho = np.linspace(0.0, 1.0, n).astype(np.float64)
    q = (1.0 + rho).astype(np.float64)  # strictly positive, q_min=1 at axis, q_edge=2
    shear = np.asarray(shear_values, dtype=np.float64)
    alpha = np.zeros(n, dtype=np.float64)
    qp = QProfile(
        rho=rho,
        q=q,
        shear=shear,
        alpha_mhd=alpha,
        q_min=float(np.min(q)),
        q_min_rho=float(rho[int(np.argmin(q))]),
        q_edge=float(q[-1]),
    )
    result = ballooning_stability(qp)
    return [float(v) for v in result.alpha_crit]


def validate_stability_mhd_ballooning() -> dict[str, Any]:
    """Compare the analytic ballooning boundary against the numerical reference."""
    report = _new_report()
    entries: list[dict[str, Any]] = report["entries"]
    findings: list[str] = report["findings"]

    module_alpha = _module_alpha_crit(_SHEAR_VALUES)
    max_rel = 0.0
    for s, a_mod in zip(_SHEAR_VALUES, module_alpha, strict=True):
        a_num = ballooning_alpha_crit(float(s))
        rel = abs(a_mod - a_num) / a_num if a_num > 0.0 else float("inf")
        max_rel = max(max_rel, rel)
        agrees = rel <= _REL_TOLERANCE
        if not agrees:
            findings.append(
                f"s={s}: analytic alpha_crit={a_mod:.4f} diverges from numerical "
                f"ballooning boundary {a_num:.4f} (rel={rel:.3f} > {_REL_TOLERANCE})"
            )
            report["status"] = "fail"
        entries.append(
            {
                "shear": float(s),
                "alpha_crit_analytic": a_mod,
                "alpha_crit_numeric": a_num,
                "relative_error": rel,
                "agrees": agrees,
                "case_sha256": _json_sha256({"shear": float(s)}),
            }
        )

    report["cases"] = len(entries)
    report["max_relative_error"] = max_rel
    return _finalise_report(report)


def _new_report() -> dict[str, Any]:
    return {
        "schema_version": _REPORT_SCHEMA,
        "status": "pass",
        "reference_implementation": "validation/ballooning_independent_reference.py",
        "method": (
            "numerical solve of the s-alpha ballooning ODE (Connor-Hastie-Taylor 1978) for the "
            "marginal first-stability alpha_crit, compared to the production analytic approximation"
        ),
        "model_reference": "Connor, Hastie & Taylor, Phys. Rev. Lett. 40 (1978) 396",
        "tolerance": {"relative": _REL_TOLERANCE},
        "shear_regime": {
            "values": list(_SHEAR_VALUES),
            "note": "first-stability regime; low-shear second stability excluded",
        },
        "units": {
            "shear": "dimensionless",
            "alpha_crit_analytic": "dimensionless",
            "alpha_crit_numeric": "dimensionless",
            "relative_error": "dimensionless",
        },
        "public_claims": {
            "bounded_ballooning_boundary_independently_verified": False,
            "ideal_mhd_stability_public_claim": False,
            "ideal_mhd_stability_blocked_reason": (
                "Bounded internal-consistency check of the analytic s-alpha approximation against a "
                "numerical ballooning solve; external ideal-MHD stability-code validation is still outstanding."
            ),
        },
        "cases": 0,
        "max_relative_error": 0.0,
        "entries": [],
        "findings": [],
        "payload_sha256": None,
    }


def _json_sha256(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _finalise_report(report: dict[str, Any]) -> dict[str, Any]:
    report["public_claims"]["bounded_ballooning_boundary_independently_verified"] = report["status"] == "pass"
    payload = dict(report)
    payload["payload_sha256"] = None
    report["payload_sha256"] = _json_sha256(payload)
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the ballooning cross-check and optionally persist the evidence report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", help="Write JSON report to this path")
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report to stdout")
    args = parser.parse_args(argv)

    report = validate_stability_mhd_ballooning()
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"Ballooning boundary cross-check: {report['status']} cases={report['cases']} "
            f"max_rel_error={report['max_relative_error']:.3f}"
        )
        for finding in report["findings"]:
            print(f"FINDING {finding}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
