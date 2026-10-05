# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Evaluate mast efm neural equilibrium.

"""Evaluate current neural-equilibrium predictions against converted MAST EFM bundles.

This script emits fail-closed prediction evidence only. The strict predictive
EFIT/P-EFIT admission artefact remains blocked until the model path supplies the
full public-reference contract: flux, pressure, q-profile, boundary, axis, exact
reference lineage, tolerances, and independently reviewable payload hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scpn_control._npz import save_npz_arrays
from scpn_control.core.neural_equilibrium import NeuralEquilibriumAccelerator
from validation.mast_efm_evaluation_features import FeatureProjection as FeatureProjection
from validation.mast_efm_evaluation_features import build_feature_projection as build_feature_projection
from validation.mast_efm_evaluation_geometry import evaluate_flux_geometry as evaluate_flux_geometry
from validation.neural_equilibrium_dataset_contracts import ensure_distinct_outputs
from validation.neural_equilibrium_dataset_tensors import load_verified_npz

EVALUATION_SCHEMA = "scpn-control.mast-efm-neural-equilibrium-evaluation.v1"


def sha256_file(path: str | Path) -> str:
    """Stream selected local file bytes into a SHA-256 hex digest; filesystem errors propagate.

    This observes one pathname and does not authenticate its physical provenance
    or freeze concurrent updates. Reference decoding separately verifies bytes
    against the selected digest before using the captured archive.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: dict[str, Any]) -> str:
    """Hash sorted compact ASCII JSON, refusing nonfinite numbers or unsupported objects.

    All supplied keys participate, including an already-present digest field;
    the caller owns excluding that field when constructing a report's digest.
    No physical/source admission is implied by this reproducibility checksum.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def masked_rmse(
    predicted: NDArray[np.floating[Any]] | NDArray[np.integer[Any]],
    observed: NDArray[np.floating[Any]] | NDArray[np.integer[Any]],
    mask: NDArray[np.bool_] | NDArray[np.integer[Any]],
) -> float:
    """Compute same-unit RMSE on jointly finite observations selected by a mask.

    All three shapes must match and at least one jointly finite masked point
    must exist. Missing/nonfinite points are excluded, so callers must report
    observation completeness separately. Scale finite residuals before the
    square reduction to avoid overflow for representable large RMSE values.
    An unrepresentable residual raises ValueError instead of emitting infinity.

    >>> masked_rmse(np.array([1e200]), np.zeros(1), np.ones(1, dtype=bool))
    1e+200
    """
    predicted_arr = np.asarray(predicted, dtype=np.float64)
    observed_arr = np.asarray(observed, dtype=np.float64)
    mask_arr = np.asarray(mask, dtype=bool)
    if predicted_arr.shape != observed_arr.shape:
        raise ValueError("predicted and observed arrays must have identical shapes")
    if mask_arr.shape != observed_arr.shape:
        raise ValueError("mask and observed arrays must have identical shapes")
    valid = mask_arr & np.isfinite(predicted_arr) & np.isfinite(observed_arr)
    if not np.any(valid):
        raise ValueError("at least one finite masked point is required")
    with np.errstate(over="ignore", invalid="ignore"):
        residual = predicted_arr[valid] - observed_arr[valid]
    if not np.all(np.isfinite(residual)):
        raise ValueError("masked residuals must be representable in float64")
    scale = float(np.max(np.abs(residual)))
    if scale == 0.0:
        return 0.0
    return scale * float(np.sqrt(np.mean((residual / scale) ** 2)))


def load_reference_bundle(path: str | Path) -> dict[str, NDArray[Any]]:
    """Hash one selected NPZ, then verify and decode the same captured bytes without pickle.

    Return independent in-memory arrays. Initial hashing IO failures propagate;
    capture/ZIP/array failures and a changed selected digest raise ValueError.
    Multi-file consistency and physical source authentication remain caller
    responsibilities.
    """
    bundle_path = Path(path)
    return load_verified_npz(bundle_path, sha256_file(bundle_path))


def evaluate_reference_bundle(
    reference_path: str | Path,
    weights_path: str | Path,
    prediction_path: str | Path,
    *,
    ffprime_reference: float | None = None,
) -> dict[str, Any]:
    """Write diagnostic inference from one NPZ reference and existing accelerator weights.

    Source-derived features use the producer's definitions, with an optional
    explicit training-campaign FF-prime reference. Grid dimensions must match.
    A single-row accelerator prediction is restored to the batch dimension.
    Output aliases of either input refuse before inference or writes. Reference
    decoding is bound to captured bytes. Weights are captured once, hashed and
    loaded from a temporary snapshot in the selected prediction directory.
    Reports always retain false predictive admission and incomplete pressure/q
    metrics. Existing non-alias prediction outputs can be replaced. No training
    is performed; source authentication and independent validation are separate.
    """
    ensure_distinct_outputs([Path(prediction_path)], protected=[Path(reference_path), Path(weights_path)])
    reference_sha256 = sha256_file(reference_path)
    reference = load_verified_npz(Path(reference_path), reference_sha256)
    projection = build_feature_projection(reference, ffprime_reference=ffprime_reference)
    psi_reference = np.asarray(reference["psirz_Wb_per_rad"], dtype=np.float64)
    psi_mask = np.asarray(reference["psirz_valid_mask"], dtype=bool)

    accelerator = NeuralEquilibriumAccelerator()
    weights_bytes = Path(weights_path).read_bytes()
    weights_sha256 = hashlib.sha256(weights_bytes).hexdigest()
    prediction_file = Path(prediction_path)
    prediction_file.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=prediction_file.parent, prefix=".mast-eval-") as snapshot_dir:
        snapshot_path = Path(snapshot_dir) / "weights.npz"
        snapshot_path.write_bytes(weights_bytes)
        accelerator.load_weights(snapshot_path)
    prediction = np.asarray(accelerator.predict(projection.features), dtype=np.float64)
    if psi_reference.shape[0] == 1 and prediction.ndim == 2:
        prediction = prediction[np.newaxis, :, :]
    if prediction.shape != psi_reference.shape:
        raise ValueError(
            "model prediction grid does not match reference grid: "
            f"prediction={prediction.shape}, reference={psi_reference.shape}"
        )
    geometry_metrics, geometry_arrays = evaluate_flux_geometry(prediction, reference)

    flux_rmse = masked_rmse(prediction, psi_reference, psi_mask)
    prediction_arrays: dict[str, ArrayLike] = {
        "psi_prediction_Wb_per_rad": prediction,
        "psi_reference_Wb_per_rad": psi_reference,
        "psi_valid_mask": psi_mask,
        "feature_projection": projection.features,
        "feature_names": np.asarray(projection.feature_names),
        "reference_path": np.asarray(str(Path(reference_path))),
        "weights_sha256": np.asarray(weights_sha256),
        "coordinate_grid_provenance": np.asarray(geometry_metrics["coordinate_grid_provenance"]),
        **geometry_arrays,
    }
    save_npz_arrays(prediction_file, prediction_arrays)

    report: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "schema_version": EVALUATION_SCHEMA,
        "status": "pass",
        "reference_path": str(Path(reference_path)),
        "weights_path": str(Path(weights_path)),
        "prediction_path": str(prediction_file),
        "reference_artifact_sha256": reference_sha256,
        "weights_sha256": weights_sha256,
        "prediction_artifact_sha256": sha256_file(prediction_file),
        "reference_equilibria_count": int(psi_reference.shape[0]),
        "grid_shape": [int(psi_reference.shape[1]), int(psi_reference.shape[2])],
        "feature_names": list(projection.feature_names),
        "feature_mapping_notes": projection.mapping_notes,
        "fallback_features": [name for name, note in projection.mapping_notes.items() if note.startswith("fallback:")],
        "ffprime_campaign_reference": ffprime_reference,
        "metric_units": {
            "psi_rmse_Wb_per_rad": "Wb/rad",
            "psi_rmse_Wb": "legacy alias of psi_rmse_Wb_per_rad; no 2*pi conversion",
            "boundary_rmse_m": "legacy alias of directed boundary_mean_distance_m; not root mean square",
        },
        "metrics": {
            "psi_rmse_Wb_per_rad": flux_rmse,
            "psi_rmse_Wb": flux_rmse,
            "pressure_rmse_Pa": None,
            "q_profile_rmse": None,
            "boundary_mean_distance_m": geometry_metrics["boundary_mean_distance_m"],
            "boundary_p95_distance_m": geometry_metrics["boundary_p95_distance_m"],
            "boundary_rmse_m": geometry_metrics["boundary_mean_distance_m"],
            "magnetic_axis_rmse_m": geometry_metrics["magnetic_axis_rmse_m"],
        },
        "geometry_evidence": {
            "coordinate_grid_provenance": geometry_metrics["coordinate_grid_provenance"],
            "derived_lcfs_success_count": geometry_metrics["derived_lcfs_success_count"],
            "reference_lcfs_contract": "nearest-distance residual from predicted psi_boundary contour to converted reference LCFS points",
            "admission_use": "diagnostic evidence only; strict predictive admission remains blocked",
        },
        "admission_ready": False,
        "strict_artifact_emitted": False,
        "blocked_reason": (
            "Current model path predicts poloidal flux only and the converted public EFM bundle "
            "does not supply complete exact inputs for strict predictive EFIT/P-EFIT admission."
        ),
        "required_follow_up": [
            "Add exact-model pressure, q-profile, LCFS, and magnetic-axis prediction surfaces.",
            "Replace synthetic-domain fallback features with acquired diagnostic inputs or documented public-reference artefacts.",
            "Define admission tolerances against matched public P-EFIT or documented public reference artefacts.",
        ],
    }
    report["payload_sha256"] = sha256_json(report)
    return report


def write_report(report: dict[str, Any], json_out: str | Path | None, markdown_out: str | Path | None) -> None:
    """Write optional diagnostic JSON/Markdown without aliasing inputs or prediction.

    Selected output paths must differ by resolved path and existing inode.
    Report fields must follow this evaluator's schema; publication and scientific
    admission are not performed. Multi-file writes are not transactional.
    """
    outputs = [Path(path) for path in (json_out, markdown_out) if path is not None]
    protected = [Path(report[key]) for key in ("reference_path", "weights_path", "prediction_path")]
    ensure_distinct_outputs(outputs, protected=protected)
    if json_out is not None:
        json_path = Path(json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown_out is not None:
        markdown_path = Path(markdown_out)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = report["metrics"]
        lines = [
            "# MAST EFM Neural-Equilibrium Evaluation",
            "",
            f"Schema: `{report['schema']}`",
            f"Reference equilibria: {report['reference_equilibria_count']}",
            f"Grid shape: {report['grid_shape'][0]} x {report['grid_shape'][1]}",
            f"Flux RMSE: {metrics['psi_rmse_Wb_per_rad']:.12g} Wb/rad",
            f"Magnetic-axis RMSE: {metrics['magnetic_axis_rmse_m']} m",
            f"Boundary mean distance: {metrics['boundary_mean_distance_m']} m",
            f"Coordinate grid: {report['geometry_evidence']['coordinate_grid_provenance']}",
            f"Admission ready: {report['admission_ready']}",
            f"Strict artefact emitted: {report['strict_artifact_emitted']}",
            "",
            "## Blocked reason",
            "",
            report["blocked_reason"],
            "",
            "## Required follow-up",
            "",
        ]
        lines.extend(f"- {item}" for item in report["required_follow_up"])
        lines.append("")
        markdown_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse selected local inputs/outputs and optional campaign FF-prime reference; help exits 0, usage errors 2."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", required=True, type=Path)
    parser.add_argument("--weights-path", required=True, type=Path)
    parser.add_argument("--prediction-path", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--report-out", required=True, type=Path)
    parser.add_argument("--ffprime-reference", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Return 0 for diagnostic evidence or 1 with an authored refusal for selected IO/model/value failures.

    Validate every output alias before inference. A successful process does not
    grant predictive admission. Parser help and usage retain argparse's exits.
    """
    args = parse_args(argv)
    try:
        ensure_distinct_outputs(
            [args.prediction_path, args.json_out, args.report_out],
            protected=[args.reference_path, args.weights_path],
        )
        report = evaluate_reference_bundle(
            args.reference_path, args.weights_path, args.prediction_path, ffprime_reference=args.ffprime_reference
        )
        write_report(report, args.json_out, args.report_out)
    except (OSError, ValueError, KeyError, RuntimeError, IndexError, TypeError) as exc:
        print(f"MAST EFM evaluation FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
