# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural-equilibrium dataset builder

"""Validate compact producer declarations and persist rendered report pairs."""

from __future__ import annotations

import json
import math
from pathlib import Path, PureWindowsPath
from typing import Any

from validation.neural_equilibrium_campaign_inputs import validate_campaign_dataset_metadata
from validation.neural_equilibrium_dataset_contracts import (
    FALLBACK_FEATURES,
    FEATURE_NAMES,
    FEATURE_SOURCE_POLICY,
    RAGGED_LCFS_KEYS,
    TARGET_KEYS,
    digest,
    ensure_distinct_outputs,
    integer,
    text,
)


def _relative_reference(value: Any, field: str) -> str:
    """Require a portable relative storage declaration without absolute/drive/traversal/control spellings."""
    selected = text(value, field)
    if (
        selected.startswith("/")
        or PureWindowsPath(selected).drive
        or "\\" in selected
        or any(part in ("", ".", "..") for part in selected.split("/"))
    ):
        raise ValueError(f"{field} must use a safe relative storage path")
    return selected


def _finite_number(value: Any, field: str) -> float:
    """Require genuine finite representable numeric metadata; booleans and overflow refuse."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def validate_dataset_report(report: dict[str, Any]) -> dict[str, Any]:
    """Validate finite producer declarations and their self-digest without claiming physical source authenticity.

    The supplied mapping is returned unchanged. This checks supported blocked
    metadata, feature/target/ragged/split/shot bindings and renderable fields;
    direct report validation does not inspect the storage-host tensor corpus.

    >>> canonical = Path(__file__).resolve().parent / "reports/mast_efm_neural_equilibrium_dataset.json"
    >>> validated = validate_dataset_report(json.loads(canonical.read_text(encoding="utf-8")))
    >>> validated["status"], validated["admission_ready"]
    ('blocked', False)
    """
    validate_campaign_dataset_metadata(report)
    digest(report.get("payload_sha256"), "payload_sha256")
    if report.get("admission_ready") is not False or report.get("strict_artefact_emitted") is not False:
        raise ValueError("dataset report must preserve literal blocked admission flags")
    if report.get("source") != "documented_public_reference":
        raise ValueError("dataset report must declare documented_public_reference source")
    if report.get("feature_names") != list(FEATURE_NAMES) or report.get("target_keys") != list(TARGET_KEYS):
        raise ValueError("dataset report must preserve all feature/target names and order")
    if not set(report["fallback_features"]).issubset(FALLBACK_FEATURES):
        raise ValueError("dataset report fallback_features must use supported source features")
    digest(report.get("candidate_payload_sha256"), "candidate_payload_sha256")
    for field in ("candidate_report", "dataset_path"):
        _relative_reference(report.get(field), field)
    for field, count in (("z_grid_m", report["grid_shape"][0]), ("r_grid_m", report["grid_shape"][1])):
        grid = report.get(field)
        if not isinstance(grid, dict) or integer(grid.get("count"), field + ".count") != count:
            raise ValueError("dataset report grid descriptor count must match grid_shape")
        lower = _finite_number(grid.get("min"), field + ".min")
        upper = _finite_number(grid.get("max"), field + ".max")
        if lower > upper or (count > 1 and lower == upper):
            raise ValueError("dataset report grid descriptor must declare increasing bounds")
    ragged = report["ragged_target_policy"]
    if ragged["keys"] != list(RAGGED_LCFS_KEYS) or ragged["point_count_key"] != "lcfs_point_count":
        raise ValueError("dataset report must preserve LCFS ragged keys/count policy")
    if "NaN" not in ragged["padding"] or "False" not in ragged["padding"]:
        raise ValueError("dataset report must declare NaN/False LCFS padding")
    references = report.get("reference_paths")
    shots = report.get("shots")
    if (
        not isinstance(references, list)
        or not isinstance(shots, list)
        or len(shots) != integer(report.get("shot_count"), "shot_count")
    ):
        raise ValueError("dataset report must declare shot/reference arrays and matching shot_count")
    if (
        len(references) != len(shots)
        or any(not isinstance(v, str) for v in references)
        or len(set(references)) != len(references)
    ):
        raise ValueError("dataset report must declare distinct per-shot reference paths")
    split_policy = report.get("split_policy")
    if not isinstance(split_policy, dict):
        raise ValueError("dataset report split_policy must be an object")
    assignments: dict[int, str] = {}
    for label in ("train", "validation", "test"):
        selected = split_policy.get(label + "_shots")
        if not isinstance(selected, list) or not selected:
            raise ValueError("dataset report must declare nonempty split shot lists")
        for shot_id in selected:
            integer(shot_id, "split shot_id")
            if shot_id in assignments:
                raise ValueError("dataset report split shots must be distinct")
            assignments[shot_id] = label
    text(split_policy.get("policy"), "split_policy.policy")
    totals = {"train": 0, "validation": 0, "test": 0}
    seen: set[int] = set()
    for shot, reference in zip(shots, references, strict=True):
        if not isinstance(shot, dict):
            raise ValueError("dataset report shot must be an object")
        shot_id = integer(shot.get("shot_id"), "shot_id")
        if shot_id in seen or assignments.get(shot_id) != shot.get("split"):
            raise ValueError("dataset report shot IDs must match their distinct split assignments")
        seen.add(shot_id)
        if shot.get("reference_path") != reference or shot.get("grid_shape") != report["grid_shape"]:
            raise ValueError("dataset report shot reference/grid bindings must match")
        _relative_reference(reference, "reference_path")
        start = _finite_number(shot.get("time_start_s"), "time_start_s")
        end = _finite_number(shot.get("time_end_s"), "time_end_s")
        count = integer(shot.get("equilibria_count"), "shot.equilibria_count")
        if start < 0 or end < start or (count > 1 and end == start) or (count == 1 and end != start):
            raise ValueError("dataset report time bounds must match nonnegative ordered shot observations")
        digest(shot.get("reference_sha256"), "reference_sha256")
        totals[shot["split"]] += integer(shot.get("equilibria_count"), "shot.equilibria_count")
    if seen != set(assignments) or totals != report["split_counts"]:
        raise ValueError("dataset report shot counts must match declared partitions")
    for field in ("blocked_reason", "generated_at_utc"):
        text(report.get(field), field)
    steps = report.get("next_processing_steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("dataset report must declare next_processing_steps")
    for step in steps:
        text(step, "next_processing_steps")
    policies = report.get("feature_source_policy")
    required_policies = set(FALLBACK_FEATURES) - set(report["fallback_features"])
    if not isinstance(policies, dict) or set(policies) != required_policies:
        raise ValueError("dataset report source policy must match nonfallback features")
    for feature, policy in policies.items():
        if not isinstance(policy, dict):
            raise ValueError("feature source policy must be an object")
        for field in ("source_key", "transform"):
            text(policy.get(field), "feature_source_policy." + field)
        for field, expected in FEATURE_SOURCE_POLICY[feature].items():
            if policy.get(field) != expected:
                raise ValueError("feature source policy must preserve supported source units/transform/clip")
        if feature == "ffprime_scale" and _finite_number(policy.get("campaign_reference"), "campaign_reference") <= 0:
            raise ValueError("feature source policy campaign_reference must be positive")
    return report


def _render_report(report: dict[str, Any]) -> str:
    """Render validated producer metadata; no numerical or scientific evidence is generated."""
    lines = [
        "# MAST EFM Neural-Equilibrium Supervised Dataset",
        "",
        f"Schema: `{report['schema_version']}`",
        f"Status: `{report['status']}`",
        f"Reference dataset: `{report['reference_dataset_id']}`",
        f"Dataset path: `{report['dataset_path']}`",
        f"Dataset SHA-256: `{report['dataset_sha256']}`",
        f"Equilibria: {report['equilibria_count']}",
        f"Grid shape: {report['grid_shape'][0]} x {report['grid_shape'][1]}",
        f"Maximum LCFS points: {report['ragged_target_policy']['max_lcfs_points']}",
        f"Split counts: train={report['split_counts']['train']}, validation={report['split_counts']['validation']}, test={report['split_counts']['test']}",
        "",
        "## Split policy",
        "",
        f"- Train shots: {', '.join(str(item) for item in report['split_policy']['train_shots'])}",
        f"- Validation shots: {', '.join(str(item) for item in report['split_policy']['validation_shots'])}",
        f"- Test shots: {', '.join(str(item) for item in report['split_policy']['test_shots'])}",
        f"- Policy: {report['split_policy']['policy']}",
        "",
        "## Targets",
        "",
    ]
    lines.extend(f"- `{key}`" for key in report["target_keys"])
    lines.extend(
        [
            "",
            "LCFS coordinates are padded with NaN values, LCFS validity masks are padded with False values, "
            f"and `{report['ragged_target_policy']['point_count_key']}` records the real point count per slice.",
            "",
            "## Admission boundary",
            "",
            report["blocked_reason"],
            "",
            "Fallback features: "
            + (
                ", ".join(f"`{item}`" for item in report["fallback_features"])
                if report["fallback_features"]
                else "none"
            ),
            "",
            "## Feature source policy",
            "",
        ]
    )
    for feature, policy in report.get("feature_source_policy", {}).items():
        lines.append(f"- `{feature}` from `{policy['source_key']}` using `{policy['transform']}`")
    lines.extend(
        [
            "",
            "## Next processing steps",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["next_processing_steps"])
    lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], json_out: Path, markdown_out: Path) -> None:
    """Validate and render finite blocked metadata before sequential JSON/Markdown writes.

    Pair path/symlink/hardlink aliases refuse. A second-file IO failure may leave
    JSON; supported schema/render/IO failures become ValueError and imply no
    complete report. This declaration writer does not locate/verify remote NPZ.
    """
    try:
        validate_dataset_report(report)
        markdown = _render_report(report)
        ensure_distinct_outputs([json_out, markdown_out])
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(encoded, encoding="utf-8")
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(markdown, encoding="utf-8")
    except (OSError, ValueError, TypeError, KeyError, IndexError, RecursionError, RuntimeError) as exc:
        raise ValueError(f"cannot write dataset report: {exc}") from exc
