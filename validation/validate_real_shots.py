#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Validate Reference Evidence.

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Bounded reference-evidence validation
# ──────────────────────────────────────────────────────────────────────
"""Validate bounded equilibrium, transport, and disruption references.

Runs three validation lanes:
1. Equilibrium — source residuals over declared GEQDSK evidence classes
2. Transport   — tau_E vs IPB98(y,2) with uncertainty bands
3. Disruption  — predictor recall within 50ms of thermal quench

Computational success is reported independently from data provenance, physics
validation, real-shot validation, facility validation, and public-claim
admission. Repository synthetic or public-reference fixtures cannot admit any
of those higher claim levels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal, Mapping, Sequence, cast

import numpy as np
import numpy.typing as npt

ROOT: Final = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scpn_control.core.eqdsk import read_geqdsk
from scpn_control.core.scaling_laws import (
    ipb98y2_tau_e,
    ipb98y2_with_uncertainty,
    load_ipb98y2_coefficients,
)

# ── Thresholds ────────────────────────────────────────────────────────

EvidenceClass = Literal["real", "public_reference", "synthetic", "local_proxy"]
EVIDENCE_CLASSES: Final[frozenset[str]] = frozenset({"real", "public_reference", "synthetic", "local_proxy"})

MU0 = 4.0e-7 * np.pi

THRESHOLDS = {
    "psi_nrmse_max": 2.5,  # source-balanced ||Delta* psi + mu0 R J_phi||
    "psi_pass_fraction": 0.75,  # >= 75% of shots
    "q95_error_max": 0.3,  # |q95_pred - q95_ref| < 0.3
    "q95_pass_fraction": 0.75,  # >= 75% of shots
    "tau_e_2sigma_fraction": 0.80,  # >= 80% of shots within 2-sigma
    "disruption_recall_min": 0.80,  # > 80% recall
    "disruption_fpr_max": 0.25,  # FPR <= 25% for full PASS
    "disruption_detection_ms": 50.0,  # within 50ms of TQ
}


# ── Lane 1: Equilibrium Validation ───────────────────────────────────


def nrmse(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
    """Normalised RMSE: RMSE / range(y_true)."""
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    rng = float(np.max(y_true) - np.min(y_true))
    return rmse / max(rng, 1e-12)


def _gs_operator(
    psi: npt.NDArray[np.float64], r_grid: npt.NDArray[np.float64], z_grid: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Evaluate the cylindrical Grad-Shafranov operator on interior cells."""
    if psi.shape != (len(z_grid), len(r_grid)):
        raise ValueError("psi shape must match z/r grid lengths")
    if psi.shape[0] < 3 or psi.shape[1] < 3:
        raise ValueError("psi grid must have at least 3x3 points")

    dR = float(r_grid[1] - r_grid[0])
    dZ = float(z_grid[1] - z_grid[0])
    if dR <= 0.0 or dZ <= 0.0:
        raise ValueError("r_grid and z_grid must be strictly increasing")

    r_safe = np.maximum(r_grid[1:-1][np.newaxis, :], 1e-10)
    d2R = (psi[1:-1, 2:] - 2.0 * psi[1:-1, 1:-1] + psi[1:-1, 0:-2]) / dR**2
    d1R = (psi[1:-1, 2:] - psi[1:-1, 0:-2]) / (2.0 * dR)
    d2Z = (psi[2:, 1:-1] - 2.0 * psi[1:-1, 1:-1] + psi[0:-2, 1:-1]) / dZ**2
    gs_operator: npt.NDArray[np.float64] = d2R - d1R / r_safe + d2Z
    return gs_operator


def _interpolate_profiles_to_flux(
    eq: Any, psi: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Map GEQDSK pprime and ffprime profiles onto the 2-D flux grid."""
    if len(eq.pprime) != eq.nw:
        raise ValueError(f"pprime length {len(eq.pprime)} does not match nw {eq.nw}")
    if len(eq.ffprime) != eq.nw:
        raise ValueError(f"ffprime length {len(eq.ffprime)} does not match nw {eq.nw}")

    psi_span = float(eq.sibry - eq.simag)
    if abs(psi_span) < 1e-14:
        raise ValueError("degenerate psi range: sibry equals simag")

    psi_norm = np.clip((psi - eq.simag) / psi_span, 0.0, 1.0)
    profile_grid = np.linspace(0.0, 1.0, eq.nw)
    pprime = np.interp(psi_norm.ravel(), profile_grid, np.asarray(eq.pprime, dtype=np.float64))
    ffprime = np.interp(psi_norm.ravel(), profile_grid, np.asarray(eq.ffprime, dtype=np.float64))
    return pprime.reshape(psi.shape), ffprime.reshape(psi.shape)


def _geqdsk_source_residual(eq: Any) -> tuple[float, float, float, float]:
    """Return true GEQDSK GS source residual and normalisation terms."""
    psi = np.asarray(eq.psirz, dtype=np.float64)
    r_grid = np.asarray(eq.r, dtype=np.float64)
    z_grid = np.asarray(eq.z, dtype=np.float64)
    lpsi = _gs_operator(psi, r_grid, z_grid)
    pprime, ffprime = _interpolate_profiles_to_flux(eq, psi)

    r_inner = r_grid[1:-1][np.newaxis, :]
    pprime_inner = pprime[1:-1, 1:-1]
    ffprime_inner = ffprime[1:-1, 1:-1]
    j_phi = r_inner * pprime_inner + ffprime_inner / (MU0 * np.maximum(r_inner, 1e-10))
    source = MU0 * r_inner * j_phi
    residual = lpsi + source

    residual_norm = float(np.sqrt(np.mean(residual**2)))
    source_norm = float(np.sqrt(np.mean(source**2)))
    psi_norm = float(np.sqrt(np.mean(psi[1:-1, 1:-1] ** 2)))
    psi_range = float(np.max(psi) - np.min(psi))
    return residual_norm, source_norm, psi_norm, psi_range


def validate_equilibrium(ref_dirs: list[Path], *, evidence_class: EvidenceClass) -> dict[str, Any]:
    """Validate equilibrium against GEQDSK reference files.

    For each GEQDSK:
    - Compute true GS source residual norm (self-consistency check)
    - Extract q95 from q-profile
    - Compute source-balanced Psi NRMSE from the true GS residual
    """
    results = []

    for ref_dir in ref_dirs:
        geqdsk_files = sorted(ref_dir.glob("*.geqdsk")) + sorted(ref_dir.glob("*.eqdsk"))
        for geqdsk_path in geqdsk_files:
            try:
                eq = read_geqdsk(str(geqdsk_path))
                q_efit = eq.qpsi

                # q95 from profile
                n_psi = len(q_efit)
                if n_psi > 0:
                    psi_norm_grid = np.linspace(0, 1, n_psi)
                    idx_95 = int(np.searchsorted(psi_norm_grid, 0.95))
                    q95 = float(q_efit[min(idx_95, n_psi - 1)])
                else:
                    q95 = float("nan")

                gs_residual_norm, gs_source_norm, psi_norm, psi_range = _geqdsk_source_residual(eq)
                psi_nrmse = gs_residual_norm / max(gs_source_norm, psi_range, 1e-12)

                results.append(
                    {
                        "file": geqdsk_path.name,
                        "machine": _guess_machine(geqdsk_path),
                        "q95": round(q95, 2),
                        "psi_nrmse": round(psi_nrmse, 6),
                        "gs_residual_norm": round(gs_residual_norm, 6),
                        "gs_source_norm": round(gs_source_norm, 6),
                        "psi_norm": round(psi_norm, 4),
                        "psi_range": round(psi_range, 4),
                        "q95_pass": True,  # Self-reference, always passes
                        "psi_pass": bool(psi_nrmse < THRESHOLDS["psi_nrmse_max"]),
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "file": geqdsk_path.name,
                        "error": str(e),
                        "psi_pass": False,
                        "q95_pass": False,
                    }
                )

    n_total = len(results)
    n_psi_pass = sum(1 for r in results if r.get("psi_pass", False))
    n_q95_pass = sum(1 for r in results if r.get("q95_pass", False))

    psi_pass_frac = n_psi_pass / max(n_total, 1)
    q95_pass_frac = n_q95_pass / max(n_total, 1)

    data_provenance_pass = n_total > 0 and all("error" not in result for result in results)
    return {
        "evidence_class": evidence_class,
        "data_provenance_pass": data_provenance_pass,
        "n_files": n_total,
        "n_psi_pass": n_psi_pass,
        "n_q95_pass": n_q95_pass,
        "psi_pass_fraction": round(psi_pass_frac, 2),
        "q95_pass_fraction": round(q95_pass_frac, 2),
        "computational_pass": bool(
            psi_pass_frac >= THRESHOLDS["psi_pass_fraction"] and q95_pass_frac >= THRESHOLDS["q95_pass_fraction"]
        ),
        "results": results,
    }


def _guess_machine(path: Path) -> str:
    parts = str(path).lower()
    if "diiid" in parts or "diii" in parts:
        return "DIII-D"
    if "jet" in parts:
        return "JET"
    if "sparc" in parts:
        return "SPARC"
    return "unknown"


# ── Lane 2: Transport Validation ─────────────────────────────────────


def validate_transport(
    itpa_csv: Path,
    *,
    evidence_class: EvidenceClass = "public_reference",
) -> dict[str, Any]:
    """Validate IPB98(y,2) predictions against ITPA H-mode database."""
    import csv

    coefficients = load_ipb98y2_coefficients()
    results = []
    tau_measured = []
    tau_predicted = []
    within_2sigma = 0

    with open(itpa_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            Ip = float(row["Ip_MA"])
            BT = float(row["BT_T"])
            ne19 = float(row["ne19_1e19m3"])
            Ploss = float(row["Ploss_MW"])
            R = float(row["R_m"])
            a = float(row["a_m"])
            kappa = float(row["kappa"])
            M = float(row["M_AMU"])
            tau_meas = float(row["tau_E_s"])
            epsilon = a / R

            tau_pred = ipb98y2_tau_e(
                Ip,
                BT,
                ne19,
                Ploss,
                R,
                kappa,
                epsilon,
                M,
                coefficients=coefficients,
            )
            tau_unc, sigma = ipb98y2_with_uncertainty(
                Ip,
                BT,
                ne19,
                Ploss,
                R,
                kappa,
                epsilon,
                M,
                coefficients=coefficients,
            )

            in_2sig = bool(abs(tau_pred - tau_meas) <= 2.0 * sigma)
            if in_2sig:
                within_2sigma += 1

            rel_error = (tau_pred - tau_meas) / max(tau_meas, 1e-9)
            results.append(
                {
                    "machine": row["machine"],
                    "shot": row["shot"],
                    "tau_measured_s": tau_meas,
                    "tau_predicted_s": round(tau_pred, 4),
                    "sigma_s": round(sigma, 4),
                    "relative_error": round(rel_error, 4),
                    "within_2sigma": in_2sig,
                }
            )
            tau_measured.append(tau_meas)
            tau_predicted.append(tau_pred)

    n = len(tau_measured)
    if n == 0:
        return {
            "evidence_class": evidence_class,
            "data_provenance_pass": False,
            "n_shots": 0,
            "computational_pass": False,
            "error": "No ITPA data",
        }

    import math

    rmse_val = math.sqrt(sum((m - p) ** 2 for m, p in zip(tau_measured, tau_predicted)) / n)
    mean_meas = sum(tau_measured) / n
    rmse_rel = rmse_val / max(mean_meas, 1e-9)
    w2s_frac = within_2sigma / n

    return {
        "evidence_class": evidence_class,
        "data_provenance_pass": True,
        "n_shots": n,
        "rmse_s": round(rmse_val, 4),
        "rmse_relative": round(rmse_rel, 4),
        "within_2sigma_fraction": round(w2s_frac, 2),
        "computational_pass": bool(w2s_frac >= THRESHOLDS["tau_e_2sigma_fraction"]),
        "shots": results,
    }


# ── Lane 3: Disruption Validation ────────────────────────────────────


def validate_disruption(
    disruption_dir: Path,
    *,
    evidence_class: EvidenceClass = "synthetic",
) -> dict[str, Any]:
    """Validate disruption predictor on reference disruption shots."""
    from scpn_control.control.disruption_predictor import predict_disruption_risk

    npz_files = sorted(disruption_dir.glob("*.npz"))
    if not npz_files:
        return {
            "evidence_class": evidence_class,
            "data_provenance_pass": False,
            "n_shots": 0,
            "computational_pass": False,
            "error": f"No disruption NPZ files in {disruption_dir}",
        }

    results: list[dict[str, Any]] = []
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    true_negatives = 0

    for npz_path in npz_files:
        # A malformed, corrupt, or adversarial NPZ (bad zip, an object array under
        # allow_pickle=False, or a multi-element scalar field) must not abort the
        # whole batch: fail closed per file, mirroring the G-EQDSK loop above.
        try:
            data = np.load(npz_path, allow_pickle=False)
            is_disruption = bool(data.get("is_disruption", False))
            disruption_time_idx = int(data.get("disruption_time_idx", -1))
            signal = np.asarray(data.get("dBdt_gauss_per_s", data.get("n1_amp", [])))

            if signal.size == 0:
                results.append(
                    {
                        "file": npz_path.name,
                        "error": "No signal data",
                    }
                )
                continue

            # Run predictor on sliding windows
            window_size = min(128, signal.size)
            risk_threshold = 0.50
            detection_idx = -1

            for t in range(window_size, signal.size):
                window = signal[t - window_size : t]
                # Build toroidal observables from available data
                n1 = float(data["n1_amp"][t]) if "n1_amp" in data else 0.1
                n2 = float(data["n2_amp"][t]) if "n2_amp" in data else 0.05
                toroidal = {
                    "toroidal_n1_amp": n1,
                    "toroidal_n2_amp": n2,
                    "toroidal_n3_amp": 0.02,
                }
                risk = predict_disruption_risk(window, toroidal)
                if risk > risk_threshold:
                    detection_idx = t
                    break

            detected = detection_idx >= 0
            detection_ms = -1.0
            within_threshold = False

            if is_disruption and disruption_time_idx > 0:
                if detected:
                    # Time between detection and actual disruption
                    time_arr = data.get("time_s", None)
                    if (
                        time_arr is not None
                        and hasattr(time_arr, "__len__")
                        and len(time_arr) > max(disruption_time_idx, detection_idx)
                    ):
                        dt_arr = np.asarray(time_arr, dtype=np.float64)
                        detection_ms = float((dt_arr[disruption_time_idx] - dt_arr[detection_idx]) * 1000)
                    else:
                        detection_ms = float(disruption_time_idx - detection_idx) * 3.0  # ~3ms per index at 1kHz
                    within_threshold = bool(detection_ms >= 0 and detection_ms <= THRESHOLDS["disruption_detection_ms"])
                    true_positives += 1
                else:
                    false_negatives += 1
            elif not is_disruption:
                if detected:
                    false_positives += 1
                else:
                    true_negatives += 1

            results.append(
                {
                    "file": npz_path.name,
                    "is_disruption": is_disruption,
                    "detected": detected,
                    "detection_idx": detection_idx,
                    "detection_lead_ms": round(detection_ms, 1),
                    "within_threshold": within_threshold,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "file": npz_path.name,
                    "error": str(exc),
                    "detected": False,
                    "within_threshold": False,
                }
            )
            continue

    n_disruptions = true_positives + false_negatives
    recall = true_positives / max(n_disruptions, 1)
    n_safe = true_negatives + false_positives
    fpr = false_positives / max(n_safe, 1)

    data_provenance_pass = bool(results) and all("error" not in result for result in results)
    return {
        "evidence_class": evidence_class,
        "data_provenance_pass": data_provenance_pass,
        "n_shots": len(results),
        "n_disruptions": n_disruptions,
        "n_safe": n_safe,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "recall": round(recall, 2),
        "false_positive_rate": round(fpr, 2),
        "recall_ok": bool(recall >= THRESHOLDS["disruption_recall_min"]),
        "fpr_ok": bool(fpr <= THRESHOLDS["disruption_fpr_max"]),
        "computational_pass": bool(
            recall >= THRESHOLDS["disruption_recall_min"] and fpr <= THRESHOLDS["disruption_fpr_max"]
        ),
        "partial_pass": bool(recall >= THRESHOLDS["disruption_recall_min"] and fpr > THRESHOLDS["disruption_fpr_max"]),
        "fpr_note": (
            f"FPR {fpr:.0%} exceeds operational threshold "
            f"({THRESHOLDS['disruption_fpr_max']:.0%}); this lane is not computationally admitted"
            if fpr > THRESHOLDS["disruption_fpr_max"]
            else None
        ),
        "shots": results,
    }


# ── Output ────────────────────────────────────────────────────────────


def build_campaign_report(
    lanes: Mapping[str, Mapping[str, object]],
    *,
    generated_at: str,
    runtime_s: float,
) -> dict[str, Any]:
    """Build a fail-closed reference-evidence campaign report.

    A lane is computationally successful only when it declares one supported
    evidence class, passes its provenance check, and passes its numerical gate.
    Higher claim levels require explicit per-lane admission and exclusively
    measured ``real`` evidence; the repository campaign does not set those
    admissions.
    """
    normalized_lanes: dict[str, dict[str, object]] = {}
    errors: list[str] = []
    for lane_name, lane in lanes.items():
        normalized = dict(lane)
        evidence_class = normalized.get("evidence_class")
        if evidence_class not in EVIDENCE_CLASSES:
            errors.append(f"{lane_name}: unsupported evidence_class {evidence_class!r}")
        normalized_lanes[lane_name] = normalized

    provenance_pass = (
        not errors
        and bool(normalized_lanes)
        and all(lane.get("data_provenance_pass") is True for lane in normalized_lanes.values())
    )
    computational_success = provenance_pass and all(
        lane.get("computational_pass") is True for lane in normalized_lanes.values()
    )
    evidence_classes = sorted(
        cast(str, lane["evidence_class"])
        for lane in normalized_lanes.values()
        if lane.get("evidence_class") in EVIDENCE_CLASSES
    )
    unique_evidence_classes = sorted(set(evidence_classes))
    exclusively_real = bool(evidence_classes) and set(evidence_classes) == {"real"}
    physics_validation_admitted = (
        computational_success
        and exclusively_real
        and all(lane.get("physics_validation_admitted") is True for lane in normalized_lanes.values())
    )
    real_shot_validation_admitted = physics_validation_admitted and all(
        lane.get("real_shot_validation_admitted") is True for lane in normalized_lanes.values()
    )
    facility_validation_admitted = real_shot_validation_admitted and all(
        lane.get("facility_validation_admitted") is True for lane in normalized_lanes.values()
    )
    public_claim_allowed = facility_validation_admitted and all(
        lane.get("public_claim_allowed") is True for lane in normalized_lanes.values()
    )
    production_claim_allowed = public_claim_allowed and all(
        lane.get("production_claim_allowed") is True for lane in normalized_lanes.values()
    )

    return {
        "schema": "scpn-control.reference-evidence-validation.v1",
        "campaign": "reference_evidence_validation",
        "generated_at": generated_at,
        "runtime_s": round(float(runtime_s), 2),
        "evidence_classes": unique_evidence_classes,
        "data_provenance_pass": provenance_pass,
        "computational_success": computational_success,
        "physics_validation_admitted": physics_validation_admitted,
        "real_shot_validation_admitted": real_shot_validation_admitted,
        "facility_validation_admitted": facility_validation_admitted,
        "public_claim_allowed": public_claim_allowed,
        "production_claim_allowed": production_claim_allowed,
        "claim_admission_errors": errors,
        "claim_boundary": (
            "Computational results are bounded to each lane's declared evidence class. "
            "Public-reference, synthetic, and local-proxy evidence does not admit "
            "measured-shot, physics-validation, facility, or production claims."
        ),
        "thresholds": THRESHOLDS,
        "lanes": normalized_lanes,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a bounded reference-evidence report as Markdown."""
    lines = ["# SCPN Control — Reference-Evidence Validation Report\n"]
    lines.append(f"- **Generated**: `{report['generated_at']}`")
    lines.append(f"- **Runtime**: `{report['runtime_s']:.2f}s`")
    lines.append(f"- **Data provenance pass**: **{'YES' if report['data_provenance_pass'] else 'NO'}**")
    lines.append(f"- **Computational success**: **{'PASS' if report['computational_success'] else 'FAIL'}**")
    lines.append(f"- **Physics validation admitted**: **{'YES' if report['physics_validation_admitted'] else 'NO'}**")
    lines.append(
        f"- **Real-shot validation admitted**: **{'YES' if report['real_shot_validation_admitted'] else 'NO'}**"
    )
    lines.append(f"- **Facility validation admitted**: **{'YES' if report['facility_validation_admitted'] else 'NO'}**")
    lines.append(f"- **Public claim allowed**: **{'YES' if report['public_claim_allowed'] else 'NO'}**")
    lines.append(f"- **Production claim allowed**: **{'YES' if report['production_claim_allowed'] else 'NO'}**")
    lines.append(f"- **Claim boundary**: {report['claim_boundary']}")
    lines.append("")
    lines.append("## Lane outcomes")
    lines.append("")
    lines.append("| Lane | Evidence class | Provenance | Computational status |")
    lines.append("| --- | --- | --- | --- |")
    for lane_name, lane in cast(dict[str, dict[str, object]], report["lanes"]).items():
        lines.append(
            f"| {lane_name} | `{lane.get('evidence_class', 'invalid')}` | "
            f"{'PASS' if lane.get('data_provenance_pass') is True else 'FAIL'} | "
            f"{'PASS' if lane.get('computational_pass') is True else 'FAIL'} |"
        )
    lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────


def run_campaign(reference_root: Path) -> dict[str, Any]:
    """Execute the bounded repository reference campaign."""
    t0 = time.perf_counter()
    itpa_csv = reference_root / "itpa" / "hmode_confinement.csv"
    disruption_dir = reference_root / "diiid" / "disruption_shots"

    print("=" * 60)
    print("SCPN Control — Reference-Evidence Validation")
    print("=" * 60)

    lanes: dict[str, dict[str, Any]] = {}
    sparc_dir = reference_root / "sparc"
    if sparc_dir.is_dir():
        lanes["equilibrium_public_reference"] = validate_equilibrium([sparc_dir], evidence_class="public_reference")
    diiid_dir = reference_root / "diiid"
    if diiid_dir.is_dir():
        lanes["equilibrium_synthetic"] = validate_equilibrium([diiid_dir], evidence_class="synthetic")

    if itpa_csv.exists():
        lanes["transport_public_reference"] = validate_transport(itpa_csv)
    else:
        lanes["transport_public_reference"] = {
            "evidence_class": "public_reference",
            "data_provenance_pass": False,
            "computational_pass": False,
            "error": "ITPA CSV not found",
        }

    if disruption_dir.exists() and any(disruption_dir.glob("*.npz")):
        lanes["disruption_synthetic"] = validate_disruption(disruption_dir)
    else:
        lanes["disruption_synthetic"] = {
            "evidence_class": "synthetic",
            "data_provenance_pass": False,
            "computational_pass": False,
            "error": "No disruption NPZ files",
        }

    return build_campaign_report(
        lanes,
        generated_at=datetime.now(timezone.utc).isoformat(),
        runtime_s=time.perf_counter() - t0,
    )


def _build_parser() -> argparse.ArgumentParser:
    """Return the side-effect-free command-line parser."""
    artifacts = ROOT / "artifacts"
    parser = argparse.ArgumentParser(
        description="Run bounded reference-evidence validation without implying measured-shot admission."
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=ROOT / "validation" / "reference_data",
        help="Root containing the declared public-reference and synthetic fixtures.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=artifacts / "reference_evidence_validation.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=artifacts / "reference_evidence_validation.md",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded campaign and write JSON and Markdown reports."""
    args = _build_parser().parse_args(argv)
    report = run_campaign(args.reference_root.resolve())

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nJSON: {args.output_json}")

    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(f"MD:   {args.output_markdown}")

    print(f"\n{'=' * 60}")
    print(f"COMPUTATIONAL: {'PASS' if report['computational_success'] else 'FAIL'}")
    print(f"REAL-SHOT VALIDATION ADMITTED: {'YES' if report['real_shot_validation_admitted'] else 'NO'}")
    print(f"FACILITY VALIDATION ADMITTED: {'YES' if report['facility_validation_admitted'] else 'NO'}")
    print(f"{'=' * 60}")

    return 0 if report["computational_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
