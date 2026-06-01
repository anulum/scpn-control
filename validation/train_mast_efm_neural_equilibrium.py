#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural-equilibrium trainer
"""Prepare or execute deterministic full-output MAST EFM baseline training."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_DATASET_MODULE = importlib.import_module("validation.build_mast_efm_neural_equilibrium_dataset")
_CAMPAIGN_MODULE = importlib.import_module("validation.plan_neural_equilibrium_training_campaign")
DATASET_SCHEMA: str = _DATASET_MODULE.DATASET_SCHEMA
FEATURE_NAMES: tuple[str, ...] = _DATASET_MODULE.FEATURE_NAMES
TARGET_KEYS: tuple[str, ...] = _DATASET_MODULE.TARGET_KEYS
CAMPAIGN_PLAN_SCHEMA: str = _CAMPAIGN_MODULE.REPORT_SCHEMA

TRAINING_SCHEMA = "scpn-control.mast-efm-neural-equilibrium-training.v1"
DEFAULT_DATASET_REPORT = ROOT / "validation" / "reports" / "mast_efm_neural_equilibrium_dataset.json"
DEFAULT_CAMPAIGN_PLAN = ROOT / "validation" / "reports" / "neural_equilibrium_training_campaign_plan.json"
DEFAULT_DATASET_PATH = Path(
    "/mnt/data_sas/DATASETS/SCPN-CONTROL/processed/neural_equilibrium/mast_efm_supervised_dataset.npz"
)
DEFAULT_WEIGHTS_OUT = Path(
    "/mnt/data_sas/DATASETS/SCPN-CONTROL/models/neural_equilibrium/mast_efm_full_output_baseline_weights.npz"
)
DEFAULT_JSON_OUT = ROOT / "validation" / "reports" / "mast_efm_neural_equilibrium_training_launch.json"
DEFAULT_MD_OUT = ROOT / "validation" / "reports" / "mast_efm_neural_equilibrium_training_launch.md"
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class TrainingInputs:
    """Input paths and controls for MAST EFM training."""

    dataset_report: Path
    campaign_plan: Path
    dataset_path: Path
    weights_out: Path
    execute: bool = False
    ridge_alpha: float = 1.0e-6
    max_flux_components: int = 32


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _validate_reports(dataset_report: dict[str, Any], campaign_plan: dict[str, Any]) -> None:
    if dataset_report.get("schema_version") != DATASET_SCHEMA:
        raise ValueError("dataset report has unsupported schema_version")
    if dataset_report.get("status") != "blocked":
        raise ValueError("dataset report must preserve blocked predictive-admission state")
    if campaign_plan.get("schema_version") != CAMPAIGN_PLAN_SCHEMA:
        raise ValueError("campaign plan has unsupported schema_version")
    if campaign_plan.get("status") != "prepared":
        raise ValueError("campaign plan must be prepared before training")


def _load_dataset(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _require_keys(data: dict[str, NDArray[Any]], keys: tuple[str, ...] | list[str]) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"dataset is missing required keys: {', '.join(missing)}")


def _dataset_metadata(data: dict[str, NDArray[Any]], dataset_report: dict[str, Any]) -> dict[str, Any]:
    _require_keys(data, ["features", "feature_names", "split", "shot_id", "time_s", "lcfs_point_count", *TARGET_KEYS])
    features = np.asarray(data["features"], dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES) or not np.all(np.isfinite(features)):
        raise ValueError(f"features must be finite with shape (n, {len(FEATURE_NAMES)})")
    feature_names = tuple(str(item) for item in data["feature_names"].tolist())
    if feature_names != FEATURE_NAMES:
        raise ValueError("dataset feature_names do not match the declared training contract")
    labels = np.asarray(data["split"]).astype(str)
    if labels.shape != (features.shape[0],):
        raise ValueError("split labels must have one value per feature row")
    split_counts = {name: int(np.count_nonzero(labels == name)) for name in SPLITS}
    if split_counts != dict(dataset_report["split_counts"]):
        raise ValueError("dataset split_counts do not match the dataset report")
    psirz = np.asarray(data["psirz_Wb_per_rad"], dtype=np.float64)
    if psirz.ndim != 3 or psirz.shape[0] != features.shape[0] or not np.all(np.isfinite(psirz)):
        raise ValueError("psirz_Wb_per_rad must be finite with shape (n, z, r)")
    mask_targets = {
        "psirz_valid_mask": "psirz_Wb_per_rad",
        "pprime_valid_mask": "pprime_Pa_per_Wb_rad",
        "q_profile_valid_mask": "q_profile",
        "lcfs_valid_mask": "lcfs_r_m",
    }
    for key, target_key in mask_targets.items():
        mask = np.asarray(data[key], dtype=bool)
        if mask.shape != np.asarray(data[target_key], dtype=np.float64).shape:
            raise ValueError(f"{key} shape does not match its target array")
    axis = np.column_stack(
        [
            np.asarray(data["magnetic_axis_r_m"], dtype=np.float64),
            np.asarray(data["magnetic_axis_z_m"], dtype=np.float64),
        ]
    )
    if axis.shape != (features.shape[0], 2) or not np.all(np.isfinite(axis)):
        raise ValueError("magnetic-axis targets must be finite per-equilibrium values")
    lcfs_count = np.asarray(data["lcfs_point_count"], dtype=np.int64)
    if lcfs_count.shape != (features.shape[0],) or np.any(lcfs_count < 1):
        raise ValueError("lcfs_point_count must be positive per equilibrium")
    return {
        "equilibria_count": int(features.shape[0]),
        "feature_count": int(features.shape[1]),
        "grid_shape": [int(psirz.shape[1]), int(psirz.shape[2])],
        "split_counts": split_counts,
        "target_keys": list(TARGET_KEYS),
        "max_lcfs_points": int(np.max(lcfs_count)),
    }


def _standardise_train(
    features: NDArray[np.float64], train_mask: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    mean = features[train_mask].mean(axis=0)
    std = features[train_mask].std(axis=0)
    std[std < 1.0e-12] = 1.0
    return (features - mean) / std, mean, std


def _ridge_fit(x: NDArray[np.float64], y: NDArray[np.float64], ridge_alpha: float) -> NDArray[np.float64]:
    x_aug = np.column_stack([x, np.ones(x.shape[0])])
    gram = x_aug.T @ x_aug + ridge_alpha * np.eye(x_aug.shape[1])
    gram[-1, -1] -= ridge_alpha
    return np.linalg.solve(gram, x_aug.T @ y)


def _ridge_predict(x: NDArray[np.float64], coeff: NDArray[np.float64]) -> NDArray[np.float64]:
    x_aug = np.column_stack([x, np.ones(x.shape[0])])
    return np.asarray(x_aug @ coeff, dtype=np.float64)


def _pca_fit(
    y_train: NDArray[np.float64], n_components: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
    mean = y_train.mean(axis=0)
    centred = y_train - mean
    _, singular_values, vt = np.linalg.svd(centred, full_matrices=False)
    usable = min(max(1, int(n_components)), vt.shape[0])
    components = vt[:usable]
    coeffs = centred @ components.T
    total = float(np.sum(singular_values**2))
    explained = float(np.sum(singular_values[:usable] ** 2) / max(total, 1.0e-30))
    return mean, components, coeffs, explained


def _masked_rmse(
    predicted: NDArray[np.float64], observed: NDArray[np.float64], mask: NDArray[np.bool_]
) -> float | None:
    valid = mask & np.isfinite(predicted) & np.isfinite(observed)
    if not np.any(valid):
        return None
    residual = predicted[valid] - observed[valid]
    return float(np.sqrt(np.mean(residual**2)))


def _fill_masked_columns(
    values: NDArray[np.float64], mask: NDArray[np.bool_], train_mask: NDArray[np.bool_]
) -> NDArray[np.float64]:
    filled = np.asarray(values, dtype=np.float64).copy()
    valid_train = mask[train_mask] & np.isfinite(filled[train_mask])
    defaults = np.zeros(filled.shape[1], dtype=np.float64)
    for col in range(filled.shape[1]):
        column_valid = valid_train[:, col]
        if np.any(column_valid):
            defaults[col] = float(np.mean(filled[train_mask][column_valid, col]))
    invalid = ~mask | ~np.isfinite(filled)
    rows, cols = np.nonzero(invalid)
    filled[rows, cols] = defaults[cols]
    return filled


def _split_metrics(
    data: dict[str, NDArray[Any]],
    labels: NDArray[np.str_],
    predictions: dict[str, NDArray[np.float64]],
) -> dict[str, dict[str, float | None]]:
    metrics: dict[str, dict[str, float | None]] = {}
    psirz = np.asarray(data["psirz_Wb_per_rad"], dtype=np.float64)
    pprime = np.asarray(data["pprime_Pa_per_Wb_rad"], dtype=np.float64)
    q_profile = np.asarray(data["q_profile"], dtype=np.float64)
    lcfs_r = np.asarray(data["lcfs_r_m"], dtype=np.float64)
    lcfs_z = np.asarray(data["lcfs_z_m"], dtype=np.float64)
    axis = np.column_stack(
        [
            np.asarray(data["magnetic_axis_r_m"], dtype=np.float64),
            np.asarray(data["magnetic_axis_z_m"], dtype=np.float64),
        ]
    )
    for split in SPLITS:
        split_mask = labels == split
        axis_error = np.linalg.norm(predictions["axis"][split_mask] - axis[split_mask], axis=1)
        metrics[split] = {
            "psi_rmse_Wb_per_rad": _masked_rmse(
                predictions["psirz"][split_mask],
                psirz[split_mask],
                np.asarray(data["psirz_valid_mask"], dtype=bool)[split_mask],
            ),
            "pprime_rmse_Pa_per_Wb_rad": _masked_rmse(
                predictions["pprime"][split_mask],
                pprime[split_mask],
                np.asarray(data["pprime_valid_mask"], dtype=bool)[split_mask],
            ),
            "q_profile_rmse": _masked_rmse(
                predictions["q_profile"][split_mask],
                q_profile[split_mask],
                np.asarray(data["q_profile_valid_mask"], dtype=bool)[split_mask],
            ),
            "lcfs_r_rmse_m": _masked_rmse(
                predictions["lcfs_r"][split_mask],
                lcfs_r[split_mask],
                np.asarray(data["lcfs_valid_mask"], dtype=bool)[split_mask],
            ),
            "lcfs_z_rmse_m": _masked_rmse(
                predictions["lcfs_z"][split_mask],
                lcfs_z[split_mask],
                np.asarray(data["lcfs_valid_mask"], dtype=bool)[split_mask],
            ),
            "magnetic_axis_rmse_m": float(np.sqrt(np.mean(axis_error**2))) if axis_error.size else None,
        }
    return metrics


def _execute_training(
    data: dict[str, NDArray[Any]],
    inputs: TrainingInputs,
) -> tuple[dict[str, Any], str]:
    start = time.perf_counter()
    features = np.asarray(data["features"], dtype=np.float64)
    labels = np.asarray(data["split"]).astype(str)
    train_mask = labels == "train"
    if int(np.count_nonzero(train_mask)) < 2:
        raise ValueError("at least two training equilibria are required")
    x, x_mean, x_std = _standardise_train(features, train_mask)
    psirz = np.asarray(data["psirz_Wb_per_rad"], dtype=np.float64)
    y_flux = psirz.reshape(psirz.shape[0], -1)
    flux_mean, flux_components, flux_train_coeffs, flux_explained = _pca_fit(
        y_flux[train_mask],
        inputs.max_flux_components,
    )
    flux_regression = _ridge_fit(x[train_mask], flux_train_coeffs, inputs.ridge_alpha)
    flux_coeffs = _ridge_predict(x, flux_regression)
    flux_pred = (flux_coeffs @ flux_components + flux_mean).reshape(psirz.shape)

    pprime = np.asarray(data["pprime_Pa_per_Wb_rad"], dtype=np.float64)
    q_profile = np.asarray(data["q_profile"], dtype=np.float64)
    lcfs_r = np.asarray(data["lcfs_r_m"], dtype=np.float64)
    lcfs_z = np.asarray(data["lcfs_z_m"], dtype=np.float64)
    pprime_filled = _fill_masked_columns(pprime, np.asarray(data["pprime_valid_mask"], dtype=bool), train_mask)
    q_filled = _fill_masked_columns(q_profile, np.asarray(data["q_profile_valid_mask"], dtype=bool), train_mask)
    lcfs_mask = np.asarray(data["lcfs_valid_mask"], dtype=bool)
    lcfs_r_filled = _fill_masked_columns(lcfs_r, lcfs_mask, train_mask)
    lcfs_z_filled = _fill_masked_columns(lcfs_z, lcfs_mask, train_mask)
    axis = np.column_stack(
        [
            np.asarray(data["magnetic_axis_r_m"], dtype=np.float64),
            np.asarray(data["magnetic_axis_z_m"], dtype=np.float64),
        ]
    )
    regressions = {
        "pprime": _ridge_fit(x[train_mask], pprime_filled[train_mask], inputs.ridge_alpha),
        "q_profile": _ridge_fit(x[train_mask], q_filled[train_mask], inputs.ridge_alpha),
        "lcfs_r": _ridge_fit(x[train_mask], lcfs_r_filled[train_mask], inputs.ridge_alpha),
        "lcfs_z": _ridge_fit(x[train_mask], lcfs_z_filled[train_mask], inputs.ridge_alpha),
        "axis": _ridge_fit(x[train_mask], axis[train_mask], inputs.ridge_alpha),
    }
    predictions = {
        "psirz": flux_pred,
        "pprime": _ridge_predict(x, regressions["pprime"]),
        "q_profile": _ridge_predict(x, regressions["q_profile"]),
        "lcfs_r": _ridge_predict(x, regressions["lcfs_r"]),
        "lcfs_z": _ridge_predict(x, regressions["lcfs_z"]),
        "axis": _ridge_predict(x, regressions["axis"]),
    }
    metrics = _split_metrics(data, labels, predictions)
    inputs.weights_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        inputs.weights_out,
        schema_version=np.asarray([TRAINING_SCHEMA]),
        feature_names=np.asarray(FEATURE_NAMES),
        x_mean=x_mean,
        x_std=x_std,
        ridge_alpha=np.asarray([inputs.ridge_alpha]),
        flux_mean=flux_mean,
        flux_components=flux_components,
        flux_regression=flux_regression,
        flux_explained_variance=np.asarray([flux_explained]),
        pprime_regression=regressions["pprime"],
        q_profile_regression=regressions["q_profile"],
        lcfs_r_regression=regressions["lcfs_r"],
        lcfs_z_regression=regressions["lcfs_z"],
        axis_regression=regressions["axis"],
        lcfs_point_count=np.asarray(data["lcfs_point_count"], dtype=np.int64),
        grid_shape=np.asarray(psirz.shape[1:], dtype=np.int64),
    )
    return (
        {
            "execution_mode": "execute",
            "weights_path": str(inputs.weights_out),
            "weights_sha256": _sha256_file(inputs.weights_out),
            "flux_components": int(flux_components.shape[0]),
            "flux_explained_variance": flux_explained,
            "ridge_alpha": float(inputs.ridge_alpha),
            "train_time_s": time.perf_counter() - start,
            "holdout_metrics": metrics,
        },
        _sha256_file(inputs.weights_out),
    )


def build_training_report(inputs: TrainingInputs) -> dict[str, Any]:
    """Build a dry-run launch report or execute deterministic local training."""

    dataset_report = _load_json_object(inputs.dataset_report)
    campaign_plan = _load_json_object(inputs.campaign_plan)
    _validate_reports(dataset_report, campaign_plan)
    dataset_exists = inputs.dataset_path.is_file()
    if inputs.execute and not dataset_exists:
        raise FileNotFoundError(f"dataset payload is required for --execute: {inputs.dataset_path}")

    dataset_sha256: str | None = None
    dataset_metadata: dict[str, Any] | None = None
    execution_payload: dict[str, Any]
    if dataset_exists:
        dataset_sha256 = _sha256_file(inputs.dataset_path)
        if dataset_sha256 != dataset_report["dataset_sha256"]:
            raise ValueError("dataset payload SHA-256 does not match the dataset report")
        data = _load_dataset(inputs.dataset_path)
        dataset_metadata = _dataset_metadata(data, dataset_report)
        if inputs.execute:
            execution_payload, _ = _execute_training(data, inputs)
        else:
            execution_payload = {
                "execution_mode": "dry_run",
                "weights_path": str(inputs.weights_out),
                "weights_sha256": None,
                "holdout_metrics": None,
            }
    else:
        execution_payload = {
            "execution_mode": "dry_run",
            "weights_path": str(inputs.weights_out),
            "weights_sha256": None,
            "holdout_metrics": None,
        }

    report: dict[str, Any] = {
        "schema_version": TRAINING_SCHEMA,
        "status": "executed" if inputs.execute else "prepared",
        "admission_ready": False,
        "strict_artefact_emitted": False,
        "claim_boundary": (
            "This report prepares or executes a deterministic repository baseline. "
            "It is not predictive EFIT/P-EFIT admission evidence."
        ),
        "dataset_report": str(inputs.dataset_report),
        "campaign_plan": str(inputs.campaign_plan),
        "dataset_path": str(inputs.dataset_path),
        "dataset_exists_on_this_host": dataset_exists,
        "dataset_sha256": dataset_sha256 or dataset_report["dataset_sha256"],
        "dataset_metadata": dataset_metadata,
        "required_targets": list(TARGET_KEYS),
        "fallback_features": dataset_report["fallback_features"],
        "blocked_before_admission": [
            "replace fallback Ip_MA, Bt_T, and ffprime_scale with acquired or documented public inputs",
            "run --execute on admitted storage and publish holdout metrics for train, validation, and test splits",
            "validate the exact trained weight checksum through the strict neural-equilibrium reference gate",
        ],
        "run_command": (
            "python validation/train_mast_efm_neural_equilibrium.py --execute "
            f"--dataset-path {inputs.dataset_path} --weights-out {inputs.weights_out}"
        ),
        **execution_payload,
    }
    report["payload_sha256"] = _sha256_json({**report, "payload_sha256": None})
    return report


def write_report(report: dict[str, Any], json_out: Path, markdown_out: Path) -> None:
    """Write JSON and Markdown launch reports."""

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# MAST EFM Neural-Equilibrium Training Launch",
        "",
        f"Schema: `{report['schema_version']}`",
        f"Status: `{report['status']}`",
        f"Execution mode: `{report['execution_mode']}`",
        f"Dataset path: `{report['dataset_path']}`",
        f"Dataset SHA-256: `{report['dataset_sha256']}`",
        f"Dataset exists on this host: `{report['dataset_exists_on_this_host']}`",
        f"Weights path: `{report['weights_path']}`",
        "",
        "## Claim boundary",
        "",
        report["claim_boundary"],
        "",
        "## Run command",
        "",
        "```bash",
        report["run_command"],
        "```",
        "",
        "## Required targets",
        "",
    ]
    lines.extend(f"- `{key}`" for key in report["required_targets"])
    lines.extend(["", "## Admission blockers", ""])
    lines.extend(f"- {item}" for item in report["blocked_before_admission"])
    if report["holdout_metrics"] is not None:
        lines.extend(["", "## Holdout metrics", ""])
        lines.append("```json")
        lines.append(json.dumps(report["holdout_metrics"], indent=2, sort_keys=True))
        lines.append("```")
    lines.append("")
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-report", default=DEFAULT_DATASET_REPORT, type=Path)
    parser.add_argument("--campaign-plan", default=DEFAULT_CAMPAIGN_PLAN, type=Path)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, type=Path)
    parser.add_argument("--weights-out", default=DEFAULT_WEIGHTS_OUT, type=Path)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT, type=Path)
    parser.add_argument("--report-out", default=DEFAULT_MD_OUT, type=Path)
    parser.add_argument("--ridge-alpha", default=1.0e-6, type=float)
    parser.add_argument("--max-flux-components", default=32, type=int)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Prepare or execute the training lane."""

    args = parse_args()
    report = build_training_report(
        TrainingInputs(
            dataset_report=args.dataset_report,
            campaign_plan=args.campaign_plan,
            dataset_path=args.dataset_path,
            weights_out=args.weights_out,
            execute=args.execute,
            ridge_alpha=args.ridge_alpha,
            max_flux_components=args.max_flux_components,
        )
    )
    write_report(report, args.json_out, args.report_out)


if __name__ == "__main__":
    main()
