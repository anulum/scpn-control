#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Ci Rmse Gate.

"""Check complete bounded regression reports without granting physics validation.

Read both JSON carriers from ``artifacts`` relative to the working directory,
or from ``--artifact-dir``. Require populated, available confinement, magnetic
axis and beta lanes plus a sampled synthetic disruption false-positive rate.
Missing or invalid evidence fails; no metric defaults to zero. Thresholds are
repository regression limits, not facility or publication acceptance criteria.
Byte/content checks here do not attest how a supplied report was generated.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

THRESHOLDS: dict[str, float] = {
    "confinement_itpa_tau_rmse_s": 0.20,
    "sparc_axis_rmse_m": 2.50,
    "beta_iter_sparc_beta_n_rmse": 0.10,
    "disruption_fpr": 0.15,
}


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON keys that could select ambiguous regression inputs."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_json_number(token: str) -> float:
    """Refuse nonstandard constants and numeric tokens outside finite float range."""
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    return value


def _read_report(path: Path) -> dict[str, object]:
    """Read one unambiguous UTF-8 JSON object, propagating file/format failures."""
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_float=_finite_json_number,
        parse_constant=_finite_json_number,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: report must be an object")
    return payload


def _metric(lane: object, key: str, count_key: str) -> float:
    """Require an available populated lane and a nonnegative real metric."""
    if not isinstance(lane, dict) or lane.get("skipped", False) is not False or "error" in lane:
        raise ValueError("lane must be an available object without errors")
    count = lane.get(count_key)
    if type(count) is not int or count <= 0:
        raise ValueError(f"{count_key} must be a positive integer")
    value = lane.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a finite nonnegative real number")
    try:
        metric = float(value)
    except OverflowError as exc:
        raise ValueError(f"{key} must fit finite float range") from exc
    if not math.isfinite(metric) or metric < 0:
        raise ValueError(f"{key} must be finite and nonnegative")
    return metric


def main(argv: list[str] | None = None) -> int:
    """Read all required carriers and compare four metrics to fixed regression limits.

    Parameters
    ----------
    argv
        CLI arguments, excluding the executable. ``None`` reads process
        arguments. ``--artifact-dir`` defaults to working-directory ``artifacts``.
        Dashboard metrics are tau RMSE in seconds, magnetic axis RMSE in metres
        and dimensionless normalised beta RMSE. Disruption FPR is a fraction.

    Returns
    -------
    int
        Zero only when every required lane is available, populated, finite and
        within its inclusive limit. One for invalid/missing carriers or any
        regression. Argparse exits with two for unsupported arguments. Every
        independently invalid metric is reported; source files are unchanged.

    Notes
    -----
    The dashboard must use ``scpn-control.rmse-dashboard.v1``; unversioned
    legacy carriers are refused because they could substitute references for
    unavailable predictions. The reference carrier must use its bounded v1 schema.
    ``disruption_synthetic.n_safe`` must be positive and FPR must lie in [0, 1].
    Passing these numerical comparisons grants no provenance, held-out model,
    measured-shot, facility, actuation or publication authority.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)
    artifact_dir = args.artifact_dir.resolve()
    try:
        data = _read_report(artifact_dir / "rmse_dashboard_ci.json")
        reference = _read_report(artifact_dir / "reference_evidence_validation.json")
        if data.get("schema") != "scpn-control.rmse-dashboard.v1":
            raise ValueError("unsupported RMSE dashboard schema; regenerate the dashboard")
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"ERROR: invalid RMSE regression carrier: {exc}")
        return 1

    failures: list[str] = []
    for lane_name, metric_name, threshold_key, unit in (
        ("confinement_itpa", "tau_rmse_s", "confinement_itpa_tau_rmse_s", "s"),
        ("sparc_axis", "axis_rmse_m", "sparc_axis_rmse_m", "m"),
        ("beta_iter_sparc", "beta_n_rmse", "beta_iter_sparc_beta_n_rmse", "dimensionless"),
    ):
        try:
            value = _metric(data.get(lane_name), metric_name, "count")
        except ValueError as exc:
            failures.append(f"{lane_name}: {exc}")
            continue
        threshold = THRESHOLDS[threshold_key]
        if value > threshold:
            failures.append(f"{lane_name}: {metric_name} {value:.4f} {unit} > {threshold:.4f} {unit}")
        else:
            print(f"PASS  {lane_name}: {metric_name} {value:.4f} {unit} <= {threshold:.4f} {unit}")

    try:
        if reference.get("schema") != "scpn-control.reference-evidence-validation.v1":
            raise ValueError("unsupported bounded reference-evidence schema")
        lanes = reference.get("lanes")
        if not isinstance(lanes, dict):
            raise ValueError("reference lanes must be an object")
        fpr = _metric(lanes.get("disruption_synthetic"), "false_positive_rate", "n_safe")
        if fpr > 1:
            raise ValueError("false_positive_rate must lie in [0, 1]")
        if fpr > THRESHOLDS["disruption_fpr"]:
            failures.append(f"disruption FPR: {fpr:.4f} > {THRESHOLDS['disruption_fpr']:.4f}")
        else:
            print(f"PASS  disruption FPR: {fpr:.4f} <= {THRESHOLDS['disruption_fpr']:.4f}")
    except ValueError as exc:
        failures.append(f"disruption FPR: {exc}")

    if failures:
        print("\nFAILED RMSE regression gate:")
        for failure in failures:
            print(f"  FAIL  {failure}")
        return 1
    print("\nAll bounded RMSE regression comparisons passed; no facility validation admitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
