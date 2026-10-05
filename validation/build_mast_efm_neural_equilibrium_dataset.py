#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural-equilibrium dataset builder

"""Build blocked supervised MAST EFM format artifacts from validated selected references.

The canonical metadata remains a blocked declaration and does not make the
external tensor corpus locally available.
>>> canonical = ROOT / "validation/reports/mast_efm_neural_equilibrium_dataset.json"
>>> report = validate_dataset_report(load_json(canonical))
>>> report["status"], report["equilibria_count"]
('blocked', 527)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.neural_equilibrium_dataset_contracts import DATASET_SCHEMA as DATASET_SCHEMA
from validation.neural_equilibrium_dataset_contracts import DEFAULT_TEST_SHOTS as DEFAULT_TEST_SHOTS
from validation.neural_equilibrium_dataset_contracts import DEFAULT_TRAIN_SHOTS as DEFAULT_TRAIN_SHOTS
from validation.neural_equilibrium_dataset_contracts import DEFAULT_VALIDATION_SHOTS as DEFAULT_VALIDATION_SHOTS
from validation.neural_equilibrium_dataset_contracts import FALLBACK_FEATURES as FALLBACK_FEATURES
from validation.neural_equilibrium_dataset_contracts import FEATURE_NAMES as FEATURE_NAMES
from validation.neural_equilibrium_dataset_contracts import FEATURE_SOURCE_POLICY as FEATURE_SOURCE_POLICY
from validation.neural_equilibrium_dataset_contracts import RAGGED_LCFS_KEYS as RAGGED_LCFS_KEYS
from validation.neural_equilibrium_dataset_contracts import TARGET_KEYS as TARGET_KEYS
from validation.neural_equilibrium_dataset_contracts import DatasetInput as DatasetInput
from validation.neural_equilibrium_dataset_contracts import load_json as load_json
from validation.neural_equilibrium_dataset_contracts import read_candidate
from validation.neural_equilibrium_dataset_contracts import safe_storage_reference as safe_storage_reference
from validation.neural_equilibrium_dataset_contracts import sha256_file as sha256_file
from validation.neural_equilibrium_dataset_contracts import sha256_json as sha256_json
from validation.neural_equilibrium_dataset_features import _campaign_ffprime_reference, _public_feature_availability
from validation.neural_equilibrium_dataset_features import build_feature_matrix as build_feature_matrix
from validation.neural_equilibrium_dataset_features import validate_feature_matrix as validate_feature_matrix
from validation.neural_equilibrium_dataset_reporting import validate_dataset_report as validate_dataset_report
from validation.neural_equilibrium_dataset_reporting import write_report as write_report
from validation.neural_equilibrium_dataset_tensors import assemble_targets, load_reference, persist_dataset
from validation.neural_equilibrium_training_rendering import ensure_distinct_outputs


def build_dataset(inputs: DatasetInput) -> dict[str, Any]:
    """Validate candidate/selected bytes/tensors/split custody, then persist blocked supervised NPZ and metadata.

    Each selected converter declaration and SHA must match. Descending grids
    are normalised with flux/masks; valid LCFS coordinates are compacted in
    original order and padded NaN/False with true counts. No interpolation or
    authentic physical/admission evidence is invented. Reads/writes are
    sequential; supported domain/IO failures refuse without implying rollback.
    """
    candidate, selections = read_candidate(inputs)
    ensure_distinct_outputs(
        [inputs.output_npz],
        protected=[inputs.candidate_report, *(path for _, path in selections)],
    )
    rows = [load_reference(shot, path) for shot, path in selections]
    labels_by_shot = {
        shot: name
        for name, shots in (
            ("train", inputs.train_shots),
            ("validation", inputs.validation_shots),
            ("test", inputs.test_shots),
        )
        for shot in shots
    }
    ffprime_reference = _campaign_ffprime_reference(rows)
    fallback_features, feature_source_policy = _public_feature_availability(rows, ffprime_reference)
    target_payload = assemble_targets(rows)
    feature_payload = np.concatenate(
        [build_feature_matrix(row, ffprime_reference=ffprime_reference) for row in rows], axis=0
    )
    labels = np.concatenate(
        [np.full(shot["selected_time_count"], labels_by_shot[shot["shot_id"]], dtype="U10") for shot, _ in selections]
    )
    shot_ids = np.concatenate([row["shot_id"] for row in rows]).astype(np.int64)
    times = np.concatenate([row["time_s"] for row in rows]).astype(np.float64)
    counts = target_payload["lcfs_point_count"]
    r_grid = np.asarray(rows[0]["r_grid_m"], dtype=np.float64)
    z_grid = np.asarray(rows[0]["z_grid_m"], dtype=np.float64)
    payload = {
        **target_payload,
        "features": feature_payload,
        "feature_names": np.asarray(FEATURE_NAMES),
        "split": labels,
        "shot_id": shot_ids,
        "time_s": times,
        "r_grid_m": r_grid,
        "z_grid_m": z_grid,
    }
    persist_dataset(inputs.output_npz, payload)
    shot_reports = [
        {
            "shot_id": shot["shot_id"],
            "split": labels_by_shot[shot["shot_id"]],
            "equilibria_count": shot["selected_time_count"],
            "reference_path": safe_storage_reference(str(path), inputs.storage_root),
            "reference_sha256": shot["sha256"],
            "grid_shape": shot["grid_shape"],
            "time_start_s": float(row["time_s"][0]),
            "time_end_s": float(row["time_s"][-1]),
        }
        for (shot, path), row in zip(selections, rows, strict=True)
    ]
    split_counts = {name: int(np.count_nonzero(labels == name)) for name in ("train", "validation", "test")}
    report: dict[str, Any] = {
        "schema_version": DATASET_SCHEMA,
        "status": "blocked",
        "source": "documented_public_reference",
        "candidate_report": safe_storage_reference(str(inputs.candidate_report), inputs.storage_root),
        "candidate_payload_sha256": candidate.get("payload_sha256"),
        "reference_dataset_id": candidate.get("reference_dataset_id"),
        "dataset_path": safe_storage_reference(str(inputs.output_npz), inputs.storage_root),
        "dataset_sha256": sha256_file(inputs.output_npz),
        "feature_names": list(FEATURE_NAMES),
        "fallback_features": list(fallback_features),
        "feature_source_policy": feature_source_policy,
        "generated_at_utc": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
        "target_keys": list(TARGET_KEYS),
        "ragged_target_policy": {
            "keys": list(RAGGED_LCFS_KEYS),
            "padding": "NaN for coordinates and False for validity mask",
            "point_count_key": "lcfs_point_count",
            "max_lcfs_points": int(counts.max()),
        },
        "shot_count": len(shot_reports),
        "equilibria_count": int(feature_payload.shape[0]),
        "split_counts": split_counts,
        "split_policy": {
            "train_shots": list(inputs.train_shots),
            "validation_shots": list(inputs.validation_shots),
            "test_shots": list(inputs.test_shots),
            "policy": "shot-held-out deterministic split; no random time-slice leakage across holdout shots",
        },
        "grid_shape": [
            int(target_payload["psirz_Wb_per_rad"].shape[1]),
            int(target_payload["psirz_Wb_per_rad"].shape[2]),
        ],
        "r_grid_m": {"count": int(r_grid.size), "min": float(r_grid[0]), "max": float(r_grid[-1])},
        "z_grid_m": {"count": int(z_grid.size), "min": float(z_grid[0]), "max": float(z_grid[-1])},
        "shots": shot_reports,
        "reference_paths": [shot["reference_path"] for shot in shot_reports],
        "admission_ready": False,
        "strict_artefact_emitted": False,
        "blocked_reason": (
            "This is a supervised public-MAST-EFM dataset for training and holdout evaluation. "
            + (
                "Predictive EFIT/P-EFIT admission remains blocked because no trained full-output pressure/q-profile/LCFS "
                "predictive artefact has passed tolerances."
                if not fallback_features
                else "Predictive EFIT/P-EFIT admission remains blocked because Ip_MA, Bt_T, and ffprime_scale are fallback "
                "features and no trained full-output pressure/q-profile/LCFS predictive artefact has passed tolerances."
            )
        ),
        "next_processing_steps": [
            "train a full-output model on the train split and evaluate only once on validation/test shot splits",
            "keep public-source feature policy fixed while training and holdout evaluation are performed",
            "emit compact holdout metrics and keep large weights/predictions on storage-host storage by SHA-256",
            "run validate_neural_equilibrium_reference.py only after full predictive artefacts and tolerances exist",
        ],
    }
    report["payload_sha256"] = sha256_json({**report, "payload_sha256": None})
    validate_dataset_report(report)
    return report


def _parse_shots(value: str) -> tuple[int, ...]:
    """Parse comma-separated positive distinct decimal shots; malformed controls retain argparse usage exit2."""
    try:
        items = value.split(",")
        if any(not item.strip() or not item.strip().isdecimal() for item in items):
            raise ValueError("shot IDs must be positive decimal integers")
        result = tuple(int(item.strip()) for item in items)
        if any(item <= 0 for item in result) or len(set(result)) != len(result):
            raise ValueError("shot IDs must be positive and distinct")
        return result
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit argv/process controls; help/usage retain argparse0/2 and no scientific execution authority."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", required=True, type=Path)
    parser.add_argument("--storage-root", default=Path("/data/SCPN-CONTROL"), type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--report-out", required=True, type=Path)
    parser.add_argument("--train-shots", type=_parse_shots, default=DEFAULT_TRAIN_SHOTS)
    parser.add_argument("--validation-shots", type=_parse_shots, default=DEFAULT_VALIDATION_SHOTS)
    parser.add_argument("--test-shots", type=_parse_shots, default=DEFAULT_TEST_SHOTS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build/write actual selected reference artifacts; success0/domain-IO refusal1 with authored FAIL text.

    All three outputs are distinct and protected against candidate/reference
    inputs before building. Sequential failures may leave earlier outputs; no
    transaction success, remote storage mount or physical admission is implied.
    """
    args = parse_args(argv)
    try:
        inputs = DatasetInput(
            args.candidate_report,
            args.storage_root,
            args.output_npz,
            args.train_shots,
            args.validation_shots,
            args.test_shots,
        )
        _, selections = read_candidate(inputs)
        ensure_distinct_outputs(
            [args.output_npz, args.json_out, args.report_out],
            protected=[args.candidate_report, *(path for _, path in selections)],
        )
        report = build_dataset(inputs)
        write_report(report, args.json_out, args.report_out)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        RecursionError,
        RuntimeError,
        FloatingPointError,
    ) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
