# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Original source audit declarations and persistence.
"""Validate complete original/converted source bindings before rendering or trainer consumption.

A consistency digest is not a source signature. Direct validation checks a
supplied declaration; the producer performs actual source decoding/reference
comparison. The trainer consumes these complete bindings without authenticating
remote measurements. Report pair persistence remains sequential.
"""

from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
from typing import Any

from validation.mast_efm_feature_audit_reporting import validate_audit_dataset_bindings, validate_audit_report
from validation.mast_efm_original_source_policy import (
    AUDIT_SCHEMA,
    FEATURE_SOURCE_POLICY,
    aggregate_feature_status,
    classify_feature_sources,
)
from validation.mast_efm_reference_contracts import json_sha256
from validation.neural_equilibrium_campaign_inputs import canonical_campaign_digest
from validation.neural_equilibrium_dataset_contracts import (
    FALLBACK_FEATURES,
    digest,
    ensure_distinct_outputs,
    integer,
    text,
)


def _relative(value: Any) -> str:
    """Require a portable local relative manifest name without ambiguous path components."""
    name = text(value, "source snapshot path")
    if (
        name.startswith("/")
        or PureWindowsPath(name).drive
        or "\\" in name
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise ValueError("source snapshot path must be a safe relative name")
    return name


def _validate_snapshot(snapshot: Any) -> None:
    """Validate sorted unique per-file digest/size metadata and exact manifest/self bindings."""
    if not isinstance(snapshot, dict):
        raise ValueError("source_snapshot must be an object")
    files = snapshot.get("files")
    if not isinstance(files, list) or len(files) != integer(snapshot.get("file_count"), "source file_count"):
        raise ValueError("source_snapshot files must match nonempty file_count")
    names: list[str] = []
    metadata_digest = None
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("source snapshot file must be an object")
        name = _relative(entry.get("path"))
        names.append(name)
        file_digest = digest(entry.get("sha256"), "source file sha256")
        size = entry.get("size_bytes")
        if type(size) is not int or size < 0:
            raise ValueError("source file size_bytes must be a nonnegative integer")
        if name == ".zmetadata":
            metadata_digest = file_digest
    if names != sorted(set(names)) or metadata_digest is None:
        raise ValueError("source snapshot paths must be sorted/distinct and contain .zmetadata")
    if digest(snapshot.get("metadata_sha256"), "metadata_sha256") != metadata_digest:
        raise ValueError("source snapshot metadata_sha256 does not match captured metadata")
    if digest(snapshot.get("snapshot_sha256"), "snapshot_sha256") != json_sha256(files):
        raise ValueError("source snapshot snapshot_sha256 does not match captured file declarations")


def validate_original_audit_report(audit: dict[str, Any]) -> dict[str, Any]:
    """Validate the complete finite self-bound declaration and recompute readiness from every shot.

    Legacy descriptor-only reports refuse. PASS declarations must carry actual
    source snapshot manifests, complete converted-feature evidence and matching
    per-shot reference IDs/paths/digests/counts. This is not re-reading the corpus.

    >>> directory = Path(__file__).resolve().parent / "reports"
    >>> try:
    ...     validate_original_audit_report(json.loads((directory / "mast_efm_original_feature_source_audit.json").read_text()))
    ... except ValueError as error:
    ...     print("schema_version" in str(error))
    True
    """
    json.dumps(audit, allow_nan=False)
    if audit.get("schema_version") != AUDIT_SCHEMA:
        raise ValueError("original source audit has unsupported schema_version")
    if digest(audit.get("payload_sha256"), "payload_sha256") != canonical_campaign_digest(audit):
        raise ValueError("original source payload_sha256 does not match its contents")
    converted = audit.get("converted_feature_audit")
    if not isinstance(converted, dict):
        raise ValueError("original source audit must contain a complete converted_feature_audit")
    validate_audit_report(converted)
    for field in ("dataset_report", "storage_root", "reference_dataset_id"):
        if text(audit.get(field), field) != converted[field]:
            raise ValueError(f"original source {field} does not match converted evidence")
    if audit.get("fallback_features") != list(FALLBACK_FEATURES):
        raise ValueError("original source audit must preserve supported fallback features")
    shots = audit.get("shots")
    if (
        not isinstance(shots, list)
        or len(shots) != integer(audit.get("shot_count"), "shot_count")
        or len(shots) != converted["reference_count"]
    ):
        raise ValueError("original source shots must match declared nonempty shot_count/converted references")
    candidates = {name for policy in FEATURE_SOURCE_POLICY.values() for name in policy["candidates"]}
    for shot, reference in zip(shots, converted["shots"], strict=True):
        if not isinstance(shot, dict):
            raise ValueError("original source shot must be an object")
        for field in ("shot_id", "reference_path", "reference_sha256", "equilibria_count"):
            if shot.get(field) != reference[field] or (
                field in {"shot_id", "equilibria_count"} and type(shot.get(field)) is not int
            ):
                raise ValueError(f"original source shot {field} does not match converted evidence")
        if shot.get("zarr_path") != f"mast/level1/shot_{shot['shot_id']}/efm.zarr":
            raise ValueError("original source zarr_path must match its selected shot")
        variables = shot.get("source_variables")
        if (
            not isinstance(variables, dict)
            or not set(variables).issubset(candidates)
            or any(not isinstance(v, dict) for v in variables.values())
        ):
            raise ValueError("original source_variables must contain supported metadata objects")
        if shot.get("feature_status") != classify_feature_sources(variables):
            raise ValueError("original feature_status does not match preferred source metadata")
        _validate_snapshot(shot.get("source_snapshot"))
        check = shot.get("conversion_check")
        if not isinstance(check, dict) or check.get("status") not in {"pass", "blocked"}:
            raise ValueError("original source conversion_check must declare pass or blocked")
        errors = check.get("errors")
        if not isinstance(errors, list) or any(not isinstance(e, str) or not e.strip() for e in errors):
            raise ValueError("original source conversion errors must be nonempty strings")
        observed = check.get("observed_time_count")
        if observed is not None:
            integer(observed, "observed_time_count")
        passed = check["status"] == "pass"
        if (
            bool(errors) == passed
            or check.get("reference_arrays_match") is not passed
            or type(check.get("matched_reference_count")) is not int
            or check["matched_reference_count"] != (shot["equilibria_count"] if passed else 0)
            or (passed and (observed is None or observed < shot["equilibria_count"]))
        ):
            raise ValueError("original source conversion readiness must match observed reference evidence")
    aggregate = aggregate_feature_status(shots)
    blocked = [feature for feature, entry in aggregate.items() if entry["status"] != "source_found_requires_rebuild"]
    conversion_blocked = [shot["shot_id"] for shot in shots if shot["conversion_check"]["status"] != "pass"]
    ready = not blocked and not conversion_blocked and converted["status"] == "pass"
    if (
        audit.get("feature_status") != aggregate
        or audit.get("blocked_features") != blocked
        or audit.get("conversion_blocked_shots") != conversion_blocked
    ):
        raise ValueError("original source aggregate metadata/conversion blockers do not match selected shots")
    if audit.get("can_rebuild_dataset_now") is not ready or audit.get("status") != (
        "source_ready" if ready else "blocked"
    ):
        raise ValueError("original source readiness must match all converted and original observations")
    steps = audit.get("next_processing_steps")
    if not isinstance(steps, list) or not steps or any(not isinstance(step, str) or not step.strip() for step in steps):
        raise ValueError("original source next_processing_steps must contain nonempty strings")
    return audit


def validate_original_audit_bindings(
    audit: dict[str, Any], dataset: dict[str, Any], *, dataset_report_sha256: str
) -> dict[str, Any]:
    """Bind the actual trainer's selected full dataset declaration to the complete original audit.

    Reuse the consumed converted-feature contract for exact captured raw/payload/
    NPZ digest, candidate/dataset locator, identity/count and per-shot bindings.
    No alias/synthetic schema projection is used; the nested report is produced
    by the real converted auditor and independently validated.
    """
    validate_original_audit_report(audit)
    validate_audit_dataset_bindings(
        audit["converted_feature_audit"], dataset, dataset_report_sha256=dataset_report_sha256
    )
    return audit


def write_report(audit: dict[str, Any], json_out: Path, markdown_out: Path) -> None:
    """Validate/render before sequential writes, protecting every selected input and captured source file."""
    try:
        validate_original_audit_report(audit)
        encoded = json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n"
        lines = [
            "# MAST EFM Original Feature-Source Audit",
            "",
            f"Schema: `{audit['schema_version']}`",
            f"Status: `{audit['status']}`",
            f"Can rebuild dataset now: `{audit['can_rebuild_dataset_now']}`",
            f"Reference dataset: `{audit['reference_dataset_id']}`",
            f"Shot count: {audit['shot_count']}",
            "",
            "## Feature source status",
            "",
            "| Feature | Status | Selected source | Transform | Resolution |",
            "|---|---|---|---|---|",
        ]
        for feature, entry in audit["feature_status"].items():
            selected = entry["selected_source"] or ", ".join(entry["selected_sources"]) or "none"
            lines.append(
                f"| `{feature}` | `{entry['status']}` | `{selected}` | `{entry['required_transform']}` | {entry['resolution']} |"
            )
        lines += ["", "## Original conversion checks", ""]
        for shot in audit["shots"]:
            lines.append(
                f"- Shot {shot['shot_id']}: {shot['conversion_check']['status']}; snapshot `{shot['source_snapshot']['snapshot_sha256']}`"
            )
            lines.extend(f"  - {error}" for error in shot["conversion_check"]["errors"])
        lines += [
            "",
            "Local conversion equivalence and consistency digests do not authenticate physical acquisition or admit predictive claims.",
            "",
            "## Next processing steps",
            "",
        ]
        lines.extend(f"- {step}" for step in audit["next_processing_steps"])
        markdown = "\n".join([*lines, ""])
        storage = Path(audit["storage_root"])
        converted = audit["converted_feature_audit"]
        protected = [
            Path(audit["dataset_report"]),
            storage / converted["dataset_path"],
            storage / converted["candidate_report"],
            *(storage / shot["reference_path"] for shot in audit["shots"]),
        ]
        for shot in audit["shots"]:
            source = storage / shot["zarr_path"]
            if any(path.resolve().is_relative_to(source.resolve()) for path in (json_out, markdown_out)):
                raise ValueError("original audit outputs must not overwrite an original Zarr source")
            protected.extend(source / entry["path"] for entry in shot["source_snapshot"]["files"])
        ensure_distinct_outputs([json_out, markdown_out], protected=protected)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(encoded, encoding="utf-8")
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(markdown, encoding="utf-8")
    except (OSError, ValueError, TypeError, KeyError, RecursionError, RuntimeError) as exc:
        raise ValueError(f"cannot write original-source audit: {exc}") from exc
