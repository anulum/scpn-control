#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural-equilibrium campaign publisher
"""Publish compact MAST EFM neural-equilibrium campaign evidence into the repo."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CAMPAIGN_SCHEMA = "scpn-control.mast-efm-neural-equilibrium-campaign.v1"


@dataclass(frozen=True)
class CampaignInput:
    """Input paths for one MAST EFM campaign publication."""

    candidate_report: Path
    evaluation_reports: tuple[Path, ...]
    sas_root: Path


def sha256_json(payload: dict[str, Any]) -> str:
    """Return the SHA-256 digest of a canonical JSON payload."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from *path*."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def safe_repo_reference(path_text: str, sas_root: Path) -> str:
    """Return a stable SAS-relative reference for an internal data path."""

    path = Path(path_text)
    try:
        return path.resolve().relative_to(sas_root.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def finite_metric(report: dict[str, Any], key: str) -> float | None:
    """Read one optional finite metric from an evaluator report."""

    value = report.get("metrics", {}).get(key)
    if value is None:
        return None
    result = float(value)
    if not result == result or result in {float("inf"), float("-inf")}:
        raise ValueError(f"metric {key} is not finite")
    return result


def mean(values: list[float]) -> float | None:
    """Return the arithmetic mean for a non-empty list, otherwise ``None``."""

    if not values:
        return None
    return float(sum(values) / len(values))


def maximum(values: list[float]) -> float | None:
    """Return the maximum for a non-empty list, otherwise ``None``."""

    if not values:
        return None
    return float(max(values))


def _shot_id_from_report(report: dict[str, Any]) -> int:
    reference_path = str(report.get("reference_path", ""))
    for token in Path(reference_path).name.split("_"):
        if token.isdigit():
            return int(token)
    raise ValueError(f"cannot infer shot id from reference_path={reference_path!r}")


def build_campaign_report(inputs: CampaignInput) -> dict[str, Any]:
    """Build a compact repository-publishable campaign evidence report."""

    candidate = load_json(inputs.candidate_report)
    evaluations = [load_json(path) for path in inputs.evaluation_reports]
    if not evaluations:
        raise ValueError("at least one evaluation report is required")
    reference_by_path = {item.get("output_path"): item for item in candidate.get("shots", []) if isinstance(item, dict)}
    shot_reports: list[dict[str, Any]] = []
    psi_rmse: list[float] = []
    axis_rmse: list[float] = []
    boundary_mean: list[float] = []
    boundary_p95: list[float] = []
    total_equilibria = 0
    fallback_features: set[str] = set()
    for evaluation in sorted(evaluations, key=_shot_id_from_report):
        shot_id = _shot_id_from_report(evaluation)
        metrics = evaluation.get("metrics", {})
        for feature, note in evaluation.get("feature_mapping_notes", {}).items():
            if isinstance(note, str) and note.startswith("fallback"):
                fallback_features.add(str(feature))
        reference_path = str(evaluation.get("reference_path", ""))
        candidate_shot = reference_by_path.get(reference_path, {})
        total_equilibria += int(evaluation.get("reference_equilibria_count", 0))
        psi_value = finite_metric(evaluation, "psi_rmse_Wb_per_rad")
        axis_value = finite_metric(evaluation, "magnetic_axis_rmse_m")
        boundary_value = finite_metric(evaluation, "boundary_mean_distance_m")
        p95_value = finite_metric(evaluation, "boundary_p95_distance_m")
        if psi_value is not None:
            psi_rmse.append(psi_value)
        if axis_value is not None:
            axis_rmse.append(axis_value)
        if boundary_value is not None:
            boundary_mean.append(boundary_value)
        if p95_value is not None:
            boundary_p95.append(p95_value)
        shot_reports.append(
            {
                "shot_id": shot_id,
                "reference_equilibria_count": int(evaluation.get("reference_equilibria_count", 0)),
                "grid_shape": evaluation.get("grid_shape"),
                "reference_path": safe_repo_reference(reference_path, inputs.sas_root),
                "prediction_path": safe_repo_reference(str(evaluation.get("prediction_path", "")), inputs.sas_root),
                "reference_sha256": evaluation.get("reference_artifact_sha256") or candidate_shot.get("sha256"),
                "prediction_sha256": evaluation.get("prediction_artifact_sha256"),
                "weights_sha256": evaluation.get("weights_sha256"),
                "metrics": {
                    "psi_rmse_Wb_per_rad": psi_value,
                    "magnetic_axis_rmse_m": axis_value,
                    "boundary_mean_distance_m": boundary_value,
                    "boundary_p95_distance_m": p95_value,
                    "pressure_rmse_Pa": metrics.get("pressure_rmse_Pa"),
                    "q_profile_rmse": metrics.get("q_profile_rmse"),
                },
                "admission_ready": bool(evaluation.get("admission_ready", False)),
                "strict_artefact_emitted": bool(evaluation.get("strict_artifact_emitted", False)),
            }
        )
    weights = {str(item.get("weights_sha256")) for item in shot_reports if item.get("weights_sha256")}
    report: dict[str, Any] = {
        "schema_version": CAMPAIGN_SCHEMA,
        "status": "blocked",
        "source": "documented_public_reference",
        "sas_root": inputs.sas_root.as_posix(),
        "candidate_report": safe_repo_reference(str(inputs.candidate_report), inputs.sas_root),
        "candidate_payload_sha256": candidate.get("payload_sha256"),
        "reference_dataset_id": candidate.get("reference_dataset_id"),
        "converted_reference_equilibria_count": int(candidate.get("reference_equilibria_count", 0)),
        "evaluated_reference_equilibria_count": total_equilibria,
        "shot_count": len(shot_reports),
        "shot_ids": [item["shot_id"] for item in shot_reports],
        "grid_shapes": sorted({tuple(item["grid_shape"] or []) for item in shot_reports}),
        "weights_sha256": sorted(weights),
        "fallback_features": sorted(fallback_features),
        "aggregate_metrics": {
            "psi_rmse_Wb_per_rad_mean": mean(psi_rmse),
            "psi_rmse_Wb_per_rad_max": maximum(psi_rmse),
            "magnetic_axis_rmse_m_mean": mean(axis_rmse),
            "magnetic_axis_rmse_m_max": maximum(axis_rmse),
            "boundary_mean_distance_m_mean": mean(boundary_mean),
            "boundary_mean_distance_m_max": maximum(boundary_mean),
            "boundary_p95_distance_m_mean": mean(boundary_p95),
            "boundary_p95_distance_m_max": maximum(boundary_p95),
            "pressure_rmse_Pa_mean": None,
            "q_profile_rmse_mean": None,
        },
        "shots": shot_reports,
        "admission_ready": False,
        "strict_artefact_emitted": False,
        "blocked_reason": (
            "Repository-published campaign evidence covers public MAST EFM flux and derived geometry evaluation only. "
            "Predictive EFIT/P-EFIT admission remains blocked until exact pressure, q-profile, LCFS, magnetic-axis, "
            "and matched public-reference or P-EFIT artefacts pass declared tolerances."
        ),
        "next_processing_steps": [
            "assemble a full-output supervised dataset from converted MAST EFM bundles",
            "train or fine-tune a model that predicts flux, pressure, q-profile, LCFS, and magnetic-axis outputs",
            "replace fallback Ip_MA, Bt_T, and ffprime_scale features with acquired or documented public inputs",
            "emit strict scpn-control.neural-equilibrium-reference.v1 artefacts only after all required outputs pass tolerances",
        ],
    }
    report["payload_sha256"] = sha256_json({**report, "payload_sha256": None})
    return report


def write_campaign_report(report: dict[str, Any], json_out: Path, markdown_out: Path) -> None:
    """Write campaign JSON and Markdown reports."""

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics = report["aggregate_metrics"]
    lines = [
        "# MAST EFM Neural-Equilibrium Campaign Evidence",
        "",
        f"Schema: `{report['schema_version']}`",
        f"Status: `{report['status']}`",
        f"Reference dataset: `{report['reference_dataset_id']}`",
        f"Shots: {', '.join(str(item) for item in report['shot_ids'])}",
        f"Evaluated equilibria: {report['evaluated_reference_equilibria_count']}",
        f"Candidate payload SHA-256: `{report['candidate_payload_sha256']}`",
        f"Campaign payload SHA-256: `{report['payload_sha256']}`",
        "",
        "## Aggregate metrics",
        "",
        f"- Flux RMSE mean: `{metrics['psi_rmse_Wb_per_rad_mean']}` Wb/rad",
        f"- Flux RMSE max: `{metrics['psi_rmse_Wb_per_rad_max']}` Wb/rad",
        f"- Magnetic-axis RMSE mean: `{metrics['magnetic_axis_rmse_m_mean']}` m",
        f"- LCFS mean-distance mean: `{metrics['boundary_mean_distance_m_mean']}` m",
        f"- LCFS p95-distance mean: `{metrics['boundary_p95_distance_m_mean']}` m",
        f"- Pressure RMSE: `{metrics['pressure_rmse_Pa_mean']}`",
        f"- q-profile RMSE: `{metrics['q_profile_rmse_mean']}`",
        "",
        "## Shot metrics",
        "",
        "| Shot | Slices | Flux RMSE | Axis RMSE | LCFS mean distance | LCFS p95 distance |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for shot in report["shots"]:
        shot_metrics = shot["metrics"]
        lines.append(
            "| {shot_id} | {count} | {psi} | {axis} | {boundary} | {p95} |".format(
                shot_id=shot["shot_id"],
                count=shot["reference_equilibria_count"],
                psi=shot_metrics["psi_rmse_Wb_per_rad"],
                axis=shot_metrics["magnetic_axis_rmse_m"],
                boundary=shot_metrics["boundary_mean_distance_m"],
                p95=shot_metrics["boundary_p95_distance_m"],
            )
        )
    lines.extend(
        [
            "",
            "## Admission boundary",
            "",
            report["blocked_reason"],
            "",
            "Fallback features still present: " + ", ".join(f"`{item}`" for item in report["fallback_features"]),
            "",
            "## Next processing steps",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["next_processing_steps"])
    lines.append("")
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", required=True, type=Path)
    parser.add_argument("--evaluation-report", required=True, action="append", type=Path)
    parser.add_argument("--sas-root", default=Path("/mnt/data_sas/DATASETS/SCPN-CONTROL"), type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--report-out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Publish one campaign report from candidate and per-shot evidence."""

    args = parse_args()
    report = build_campaign_report(
        CampaignInput(
            candidate_report=args.candidate_report,
            evaluation_reports=tuple(args.evaluation_report),
            sas_root=args.sas_root,
        )
    )
    write_campaign_report(report, args.json_out, args.report_out)


if __name__ == "__main__":
    main()
