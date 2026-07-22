#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST programme-label and Ip-proxy sensitivity audit
"""Audit Ip-proxy stability without promoting weak labels to ground truth.

The audit verifies a disruption dataset report and every referenced shot NPZ,
replays the versioned Ip detector over a declared parameter grid, and reports
shot-level transitions relative to a declared reference point.  An optional
programme-label manifest may carry an explicitly declared *weak expectation*;
agreement with it is descriptive only and can never establish independent
validation.  If that manifest is absent, disagreement is ``not_computable``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import numpy as np

from validation.mast_disruption_shot_label import ProgrammeClass, derive_ip_quench_proxy
from validation.mast_source_object_manifest import canonical_json_sha256, file_sha256

AUDIT_SCHEMA = "scpn-control.mast-label-proxy-sensitivity-audit.v1.0.0"
PROGRAMME_LABEL_MANIFEST_SCHEMA = "scpn-control.mast-programme-weak-labels.v1.0.0"
DEFAULT_PARAMETER_GRID: tuple[tuple[float, float], ...] = tuple(
    (drop_fraction, quench_window_ms) for drop_fraction in (0.7, 0.8, 0.9) for quench_window_ms in (2.5, 5.0, 10.0)
)
DEFAULT_REFERENCE_POINT = (0.8, 5.0)
_OUTCOMES = ("ambiguous", "disruption", "non_disruption")
_PROGRAMME_CLASSES = frozenset({"spontaneous", "forced_vde", "control", "other", "unknown"})
WeakExpectation = Literal["disruption", "non_disruption", "no_expectation"]
_WEAK_EXPECTATIONS = frozenset({"disruption", "non_disruption", "no_expectation"})
_DATASET_SCHEMAS = frozenset(
    {
        "scpn-control.mast-disruption-supervised-dataset.v1",
        "scpn-control.mast-disruption-supervised-dataset.v2.0.0",
    }
)
_SHA256_LENGTH = 64


class LabelProxyAuditError(ValueError):
    """Raised when audit input is ambiguous, unsafe, or fails integrity checks."""


@dataclass(frozen=True, order=True)
class DetectorPoint:
    """One bounded Ip-quench detector parameter point."""

    drop_fraction: float
    quench_window_ms: float

    def __post_init__(self) -> None:
        """Reject parameter points outside the production detector contract."""
        if not math.isfinite(self.drop_fraction) or not 0.0 < self.drop_fraction < 1.0:
            raise LabelProxyAuditError("drop_fraction must be finite and within (0, 1)")
        if not math.isfinite(self.quench_window_ms) or self.quench_window_ms <= 0.0:
            raise LabelProxyAuditError("quench_window_ms must be finite and positive")

    @property
    def key(self) -> str:
        """Return a deterministic human-readable parameter key."""
        return f"drop={self.drop_fraction:.12g};window_ms={self.quench_window_ms:.12g}"

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-ready representation."""
        return {
            "drop_fraction": self.drop_fraction,
            "quench_window_ms": self.quench_window_ms,
        }


@dataclass(frozen=True)
class WeakProgrammeLabel:
    """One programme metadata class and explicitly declared weak expectation."""

    shot_id: int
    programme_class: ProgrammeClass
    proxy_expectation: WeakExpectation


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise LabelProxyAuditError(f"duplicate JSON key {key!r}")
        payload[key] = value
    return payload


def _load_json(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError, LabelProxyAuditError) as exc:
        raise LabelProxyAuditError(f"cannot read verified JSON {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise LabelProxyAuditError(f"JSON root must be an object: {path}")
    return payload


def _required_string(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise LabelProxyAuditError(f"{field} must be a non-empty string")
    return value


def _required_int(payload: Mapping[str, object], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise LabelProxyAuditError(f"{field} must be an integer")
    return value


def _required_float(payload: Mapping[str, object], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabelProxyAuditError(f"{field} must be numeric")
    return float(value)


def _validate_sha256(value: str, *, field: str) -> str:
    if len(value) != _SHA256_LENGTH or any(char not in "0123456789abcdef" for char in value):
        raise LabelProxyAuditError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _verify_self_digest(payload: Mapping[str, object], *, field: str = "payload_sha256") -> str:
    digest = _validate_sha256(_required_string(payload, field), field=field)
    digest_input = dict(payload)
    digest_input[field] = None
    if canonical_json_sha256(digest_input) != digest:
        raise LabelProxyAuditError(f"{field} does not match the JSON payload")
    return digest


def _normalise_grid(points: Sequence[tuple[float, float]]) -> tuple[DetectorPoint, ...]:
    if not points:
        raise LabelProxyAuditError("parameter grid must not be empty")
    normalised = tuple(sorted(DetectorPoint(*point) for point in points))
    if len(set(normalised)) != len(normalised):
        raise LabelProxyAuditError("parameter grid contains duplicate points")
    return normalised


def _safe_artifact_path(dataset_dir: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or len(pure.parts) != 1 or pure.parts[0] in {"", ".", ".."}:
        raise LabelProxyAuditError(f"unsafe shot artifact path {relative!r}")
    try:
        root = dataset_dir.resolve(strict=True)
        candidate = (root / pure.as_posix()).resolve(strict=True)
    except OSError as exc:
        raise LabelProxyAuditError(f"cannot resolve shot artifact {relative!r}: {exc}") from exc
    if candidate.parent != root or not candidate.is_file():
        raise LabelProxyAuditError(f"shot artifact escapes dataset directory: {relative!r}")
    return candidate


def _programme_labels(path: Path | None) -> tuple[dict[int, WeakProgrammeLabel], dict[str, object]]:
    if path is None:
        return {}, {
            "status": "not_computable",
            "reason": "programme_label_manifest_not_supplied",
            "manifest_file_sha256": None,
            "manifest_payload_sha256": None,
        }
    payload = _load_json(path)
    expected = {"schema_version", "machine", "authority", "source_uri", "labels", "payload_sha256"}
    if set(payload) != expected:
        raise LabelProxyAuditError("programme-label manifest fields do not match the schema")
    if _required_string(payload, "schema_version") != PROGRAMME_LABEL_MANIFEST_SCHEMA:
        raise LabelProxyAuditError("unsupported programme-label manifest schema")
    if _required_string(payload, "machine") != "MAST":
        raise LabelProxyAuditError("programme-label manifest machine must be MAST")
    if _required_string(payload, "authority") != "programme_metadata":
        raise LabelProxyAuditError("programme-label manifest authority must be programme_metadata")
    _required_string(payload, "source_uri")
    manifest_digest = _verify_self_digest(payload)
    raw_labels = payload.get("labels")
    if not isinstance(raw_labels, list):
        raise LabelProxyAuditError("programme-label labels must be an array")
    labels: dict[int, WeakProgrammeLabel] = {}
    for raw in raw_labels:
        if not isinstance(raw, Mapping) or set(raw) != {"shot_id", "programme_class", "proxy_expectation"}:
            raise LabelProxyAuditError("programme-label record fields do not match the schema")
        shot_id = _required_int(raw, "shot_id")
        programme_class = _required_string(raw, "programme_class")
        expectation = _required_string(raw, "proxy_expectation")
        if shot_id <= 0:
            raise LabelProxyAuditError("programme-label shot_id must be positive")
        if programme_class not in _PROGRAMME_CLASSES:
            raise LabelProxyAuditError(f"unsupported programme_class {programme_class!r}")
        if expectation not in _WEAK_EXPECTATIONS:
            raise LabelProxyAuditError(f"unsupported proxy_expectation {expectation!r}")
        if shot_id in labels:
            raise LabelProxyAuditError(f"duplicate programme label for shot {shot_id}")
        labels[shot_id] = WeakProgrammeLabel(
            shot_id=shot_id,
            programme_class=cast(ProgrammeClass, programme_class),
            proxy_expectation=cast(WeakExpectation, expectation),
        )
    return labels, {
        "status": "available",
        "reason": None,
        "manifest_file_sha256": file_sha256(path),
        "manifest_payload_sha256": manifest_digest,
    }


def _dataset_records(
    report_path: Path,
    dataset_dir: Path,
    reference: DetectorPoint,
) -> tuple[str, str, str, str, str, list[tuple[int, Path]]]:
    payload = _load_json(report_path)
    report_digest = _verify_self_digest(payload)
    schema_version = _required_string(payload, "schema_version")
    if schema_version not in _DATASET_SCHEMAS:
        raise LabelProxyAuditError(f"unsupported dataset report schema {schema_version!r}")
    if payload.get("synthetic") is not False:
        raise LabelProxyAuditError("dataset report must declare synthetic:false")
    if payload.get("status") != "blocked":
        raise LabelProxyAuditError("dataset report must remain status:blocked")
    label_algorithm = payload.get("label_algorithm")
    if not isinstance(label_algorithm, Mapping):
        raise LabelProxyAuditError("dataset report label_algorithm must be an object")
    source_reference = DetectorPoint(
        _required_float(label_algorithm, "drop_fraction"),
        _required_float(label_algorithm, "quench_window_ms"),
    )
    if source_reference != reference:
        raise LabelProxyAuditError("reference point does not match the source dataset label algorithm")
    dataset_id = _required_string(payload, "dataset_id")
    n_shots = _required_int(payload, "n_shots")
    raw_records = payload.get("shots")
    if not isinstance(raw_records, list) or len(raw_records) != n_shots:
        raise LabelProxyAuditError("dataset report shot count does not match shots")
    records: list[tuple[int, Path]] = []
    seen_shots: set[int] = set()
    seen_paths: set[str] = set()
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise LabelProxyAuditError("dataset shot record must be an object")
        shot_id = _required_int(raw, "shot_id")
        relative = _required_string(raw, "npz")
        expected_sha = _validate_sha256(_required_string(raw, "checksum_sha256"), field="checksum_sha256")
        if shot_id <= 0 or shot_id in seen_shots:
            raise LabelProxyAuditError(f"invalid or duplicate dataset shot_id {shot_id}")
        if relative in seen_paths:
            raise LabelProxyAuditError(f"duplicate dataset artifact path {relative!r}")
        artifact = _safe_artifact_path(dataset_dir, relative)
        if file_sha256(artifact) != expected_sha:
            raise LabelProxyAuditError(f"shot {shot_id} artifact checksum mismatch")
        seen_shots.add(shot_id)
        seen_paths.add(relative)
        records.append((shot_id, artifact))
    records.sort()
    return (
        dataset_id,
        schema_version,
        file_sha256(report_path),
        report_digest,
        canonical_json_sha256(label_algorithm),
        records,
    )


def _load_proxy_inputs(
    shot_id: int, artifact: Path
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    try:
        with np.load(artifact, allow_pickle=False) as archive:
            if "Ip_MA" not in archive.files or "time_s" not in archive.files:
                raise LabelProxyAuditError(f"shot {shot_id} artifact lacks Ip_MA or time_s")
            ip_ma = np.asarray(archive["Ip_MA"], dtype=np.float64)
            time_s = np.asarray(archive["time_s"], dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise LabelProxyAuditError(f"cannot load shot {shot_id} artifact: {exc}") from exc
    return ip_ma, time_s


def _programme_comparison(
    labels: Mapping[int, WeakProgrammeLabel],
    label_source: dict[str, object],
    reference_outcomes: Mapping[int, str],
) -> dict[str, object]:
    dataset_ids = set(reference_outcomes)
    manifest_ids = set(labels)
    comparison_ids = sorted(
        shot_id for shot_id in dataset_ids & manifest_ids if labels[shot_id].proxy_expectation != "no_expectation"
    )
    comparison: dict[str, object] = {
        **label_source,
        "authority": "programme_metadata" if labels else None,
        "independent_of_ip_features": False,
        "programme_label_count": len(labels),
        "matched_dataset_count": len(dataset_ids & manifest_ids),
        "comparison_count": len(comparison_ids),
        "unmatched_programme_shot_ids": sorted(manifest_ids - dataset_ids),
        "dataset_shots_without_programme_label": sorted(dataset_ids - manifest_ids),
        "disagreement_count": None,
        "disagreement_rate": None,
        "disagreement_shot_ids": [],
        "independent_validation_claim": False,
    }
    if not comparison_ids:
        if labels:
            comparison["status"] = "not_computable"
            comparison["reason"] = "no_matched_programme_labels_with_declared_proxy_expectation"
        return comparison
    disagreements = [
        shot_id for shot_id in comparison_ids if reference_outcomes[shot_id] != labels[shot_id].proxy_expectation
    ]
    comparison.update(
        {
            "status": "descriptive_only",
            "reason": "programme_metadata_is_a_weak_non_independent_comparator",
            "disagreement_count": len(disagreements),
            "disagreement_rate": len(disagreements) / len(comparison_ids),
            "disagreement_shot_ids": disagreements,
        }
    )
    return comparison


def audit_label_proxy_sensitivity(
    *,
    dataset_report_path: Path,
    dataset_dir: Path,
    generated_at: str,
    parameter_grid: Sequence[tuple[float, float]] = DEFAULT_PARAMETER_GRID,
    reference_point: tuple[float, float] = DEFAULT_REFERENCE_POINT,
    programme_labels_path: Path | None = None,
) -> dict[str, object]:
    """Verify source artefacts and build a raw-data-free sensitivity report."""
    if not generated_at:
        raise LabelProxyAuditError("generated_at must be non-empty")
    points = _normalise_grid(parameter_grid)
    reference = DetectorPoint(*reference_point)
    if reference not in points:
        raise LabelProxyAuditError("reference point must be present in the parameter grid")
    dataset_id, dataset_schema, report_file_sha, report_payload_sha, label_algorithm_sha, records = _dataset_records(
        dataset_report_path,
        dataset_dir,
        reference,
    )
    if not records:
        raise LabelProxyAuditError("dataset report contains no shots")
    labels, label_source = _programme_labels(programme_labels_path)

    results: dict[int, dict[DetectorPoint, tuple[str, str]]] = {}
    for shot_id, artifact in records:
        ip_ma, time_s = _load_proxy_inputs(shot_id, artifact)
        point_results: dict[DetectorPoint, tuple[str, str]] = {}
        for point in points:
            proxy = derive_ip_quench_proxy(
                ip_ma,
                time_s,
                shot_id=shot_id,
                drop_fraction=point.drop_fraction,
                quench_window_ms=point.quench_window_ms,
            )
            point_results[point] = (proxy.record.outcome, proxy.record.algorithm_digest)
        results[shot_id] = point_results

    reference_outcomes = {shot_id: point_results[reference][0] for shot_id, point_results in results.items()}
    sensitivity_points: list[dict[str, object]] = []
    for point in points:
        outcomes = {shot_id: point_results[point][0] for shot_id, point_results in results.items()}
        disagreements = sorted(
            shot_id for shot_id, outcome in outcomes.items() if outcome != reference_outcomes[shot_id]
        )
        counts = {outcome: sum(value == outcome for value in outcomes.values()) for outcome in _OUTCOMES}
        sensitivity_points.append(
            {
                **point.to_dict(),
                "parameter_key": point.key,
                "algorithm_digest": next(iter(results.values()))[point][1],
                "outcome_counts": counts,
                "disagreement_from_reference_count": len(disagreements),
                "disagreement_from_reference_rate": len(disagreements) / len(records),
                "disagreement_from_reference_shot_ids": disagreements,
            }
        )

    shot_stability: list[dict[str, object]] = []
    for shot_id, point_results in sorted(results.items()):
        outcomes_by_parameter = {point.key: point_results[point][0] for point in points}
        unique_outcomes = sorted(set(outcomes_by_parameter.values()))
        shot_stability.append(
            {
                "shot_id": shot_id,
                "reference_outcome": reference_outcomes[shot_id],
                "stable_across_grid": len(unique_outcomes) == 1,
                "unique_outcomes": unique_outcomes,
                "outcomes_by_parameter": outcomes_by_parameter,
            }
        )
    unstable_ids = [record["shot_id"] for record in shot_stability if not record["stable_across_grid"]]

    report: dict[str, object] = {
        "schema_version": AUDIT_SCHEMA,
        "status": "blocked",
        "dataset_id": dataset_id,
        "source_dataset_schema": dataset_schema,
        "source_dataset_report_file_sha256": report_file_sha,
        "source_dataset_report_payload_sha256": report_payload_sha,
        "source_label_algorithm_sha256": label_algorithm_sha,
        "verified_shot_artifact_count": len(records),
        "reference_point": reference.to_dict(),
        "parameter_grid": [point.to_dict() for point in points],
        "sensitivity_points": sensitivity_points,
        "stable_shot_count": len(records) - len(unstable_ids),
        "unstable_shot_count": len(unstable_ids),
        "unstable_shot_ids": unstable_ids,
        "shot_stability": shot_stability,
        "programme_comparison": _programme_comparison(labels, label_source, reference_outcomes),
        "claim_boundary": {
            "scientific_validation": False,
            "independent_label_validation": False,
            "cohort_admission": False,
            "training_admission": False,
            "facility_prediction": False,
            "control_admission": False,
        },
        "generated_at": generated_at,
        "payload_sha256": None,
    }
    report["payload_sha256"] = canonical_json_sha256(report)
    return report


def _parse_point(value: str) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("point must be DROP_FRACTION,QUENCH_WINDOW_MS")
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("point values must be numeric") from exc


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--programme-labels", type=Path)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--point", type=_parse_point, action="append")
    parser.add_argument("--reference-point", type=_parse_point, default=DEFAULT_REFERENCE_POINT)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the verified audit and write its deterministic JSON report."""
    args = _parse_args(argv)
    report = audit_label_proxy_sensitivity(
        dataset_report_path=args.dataset_report,
        dataset_dir=args.dataset_dir,
        generated_at=args.generated_at,
        parameter_grid=args.point or DEFAULT_PARAMETER_GRID,
        reference_point=args.reference_point,
        programme_labels_path=args.programme_labels,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "proxy sensitivity: "
        f"{report['unstable_shot_count']}/{report['verified_shot_artifact_count']} unstable "
        f"(programme={cast(Mapping[str, object], report['programme_comparison'])['status']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
