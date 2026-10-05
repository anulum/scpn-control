# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM source-audit declarations and persistence.
"""Validate finite source-audit declarations before rendering and pair persistence.

Self-digests and internal bindings are consistency checks, not source signatures.
The producer verifies local inputs; this writer checks declarations without
reopening the corpus. Pair writes are sequential and do not promise rollback.
"""

from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
from typing import Any

from validation.mast_efm_feature_audit_inputs import AUDIT_SCHEMA, feature_status_for_shots
from validation.neural_equilibrium_campaign_inputs import canonical_campaign_digest
from validation.neural_equilibrium_dataset_contracts import (
    FALLBACK_FEATURES,
    FEATURE_SOURCE_POLICY,
    digest,
    ensure_distinct_outputs,
    integer,
    text,
)
from validation.neural_equilibrium_dataset_reporting import validate_dataset_report


def _strings(value: Any, field: str) -> list[str]:
    """Require a distinct string list, permitting explicit empty inventory members."""
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a string list")
    result = [text(item, field) for item in value]
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must contain distinct members")
    return result


def _relative(value: Any, field: str) -> str:
    """Require portable storage-relative custody declarations without traversal or drive spellings."""
    selected = text(value, field)
    if (
        selected.startswith("/")
        or PureWindowsPath(selected).drive
        or "\\" in selected
        or any(part in ("", ".", "..") for part in selected.split("/"))
    ):
        raise ValueError(f"{field} must use a safe relative storage path")
    return selected


def validate_audit_report(audit: dict[str, Any]) -> dict[str, Any]:
    """Validate finite schema/self-digest, per-shot source coverage and exact aggregate status.

    Return the supplied mapping unchanged. This validates declarations rather
    than reading local source bytes or authenticating original measurements.

    >>> validate_audit_report({})
    Traceback (most recent call last):
        ...
    ValueError: feature audit has unsupported schema_version
    """
    if not isinstance(audit, dict):
        raise ValueError("feature audit must be a JSON object")
    computed = canonical_campaign_digest(audit)
    if audit.get("schema_version") != AUDIT_SCHEMA:
        raise ValueError("feature audit has unsupported schema_version")
    if digest(audit.get("payload_sha256"), "payload_sha256") != computed:
        raise ValueError("feature audit payload_sha256 does not match its contents")
    for field in ("dataset_report", "storage_root", "reference_dataset_id"):
        text(audit.get(field), field)
    for field in ("dataset_report_sha256", "dataset_payload_sha256", "dataset_sha256"):
        digest(audit.get(field), field)
    for field in ("dataset_path", "candidate_report"):
        _relative(audit.get(field), field)
    if audit.get("fallback_features") != list(FALLBACK_FEATURES):
        raise ValueError("feature audit must declare all supported fallback features")
    shots = audit.get("shots")
    if not isinstance(shots, list) or len(shots) != integer(audit.get("reference_count"), "reference_count"):
        raise ValueError("feature audit must declare matching nonempty reference_count/shots")
    ids: set[int] = set()
    paths: set[str] = set()
    for shot in shots:
        if not isinstance(shot, dict):
            raise ValueError("feature audit shot must be an object")
        shot_id = integer(shot.get("shot_id"), "shot_id")
        relative = _relative(shot.get("reference_path"), "reference_path")
        if shot_id in ids or relative in paths:
            raise ValueError("feature audit shot IDs and reference paths must be distinct")
        ids.add(shot_id)
        paths.add(relative)
        digest(shot.get("reference_sha256"), "reference_sha256")
        n = integer(shot.get("equilibria_count"), "equilibria_count")
        keys = _strings(shot.get("keys"), "keys")
        if keys != sorted(keys) or integer(shot.get("key_count"), "key_count") != len(keys):
            raise ValueError("feature audit key_count must match sorted keys")
        shapes = shot.get("shapes")
        if not isinstance(shapes, dict) or set(shapes) != set(keys):
            raise ValueError("feature audit shapes must match inventory keys")
        for shape in shapes.values():
            if not isinstance(shape, list) or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape
            ):
                raise ValueError("feature audit shapes must declare nonnegative integer dimensions")
        sourced = _strings(shot.get("sourced_features"), "sourced_features")
        expected = [feature for feature in FALLBACK_FEATURES if FEATURE_SOURCE_POLICY[feature]["source_key"] in keys]
        if sourced != expected or any(shapes[FEATURE_SOURCE_POLICY[f]["source_key"]] != [n] for f in sourced):
            raise ValueError("feature audit sourced features must match canonical per-row channels")
    status = feature_status_for_shots(shots)
    blocked = [feature for feature, entry in status.items() if entry["status"] == "blocked"]
    if (
        audit.get("feature_status") != status
        or audit.get("blocked_features") != blocked
        or audit.get("status") != ("blocked" if blocked else "pass")
    ):
        raise ValueError("feature audit aggregate status must match per-shot canonical source coverage")
    if audit.get("all_reference_keys") != sorted({key for shot in shots for key in shot["keys"]}):
        raise ValueError("feature audit union inventory must match selected references")
    steps = _strings(audit.get("next_processing_steps"), "next_processing_steps")
    if not steps:
        raise ValueError("feature audit must declare next_processing_steps")
    return audit


def validate_audit_dataset_bindings(
    audit: dict[str, Any], dataset: dict[str, Any], *, dataset_report_sha256: str
) -> dict[str, Any]:
    """Require a complete audit of the exact captured producer declaration and selected references.

    The caller supplies the SHA of the same JSON bytes it decoded. Dataset,
    payload and per-shot byte/count bindings must agree. This validates
    declarations, without reopening source NPZ or authenticating acquisition.

    The preserved scientific audit has a stale self-digest and cannot admit a
    new run merely because its status label says PASS:

    >>> directory = Path(__file__).resolve().parent / "reports"
    >>> audit = json.loads((directory / "mast_efm_feature_provenance_audit.json").read_text())
    >>> dataset = json.loads((directory / "mast_efm_neural_equilibrium_dataset.json").read_text())
    >>> try:
    ...     validate_audit_dataset_bindings(audit, dataset, dataset_report_sha256="0" * 64)
    ... except ValueError as error:
    ...     print("payload_sha256" in str(error))
    True
    """
    validate_audit_report(audit)
    validate_dataset_report(dataset)
    if audit["dataset_report_sha256"] != digest(dataset_report_sha256, "dataset_report_sha256"):
        raise ValueError("feature provenance dataset_report_sha256 does not match selected declaration bytes")
    bindings = {
        "reference_dataset_id": dataset["reference_dataset_id"],
        "dataset_payload_sha256": dataset["payload_sha256"],
        "dataset_sha256": dataset["dataset_sha256"],
        "dataset_path": dataset["dataset_path"],
        "candidate_report": dataset["candidate_report"],
        "reference_count": dataset["shot_count"],
    }
    for field, expected in bindings.items():
        if audit[field] != expected:
            raise ValueError(f"feature provenance {field} does not match the selected dataset report")
    for observed, selected in zip(audit["shots"], dataset["shots"], strict=True):
        for field in ("shot_id", "reference_path", "reference_sha256", "equilibria_count"):
            if observed[field] != selected[field]:
                raise ValueError(f"feature provenance shot {field} does not match the selected dataset report")
    return audit


def _render(audit: dict[str, Any]) -> str:
    """Render validated channel completeness and explicit original-source admission boundary."""
    lines = [
        "# MAST EFM Feature-Provenance Audit",
        "",
        f"Schema: `{audit['schema_version']}`",
        f"Status: `{audit['status']}`",
        f"Reference dataset: `{audit['reference_dataset_id']}`",
        f"Reference bundles: {audit['reference_count']}",
        "",
        "## Fallback feature status",
        "",
        "| Feature | Status | Present keys | Complete references | Resolution |",
        "|---|---|---|---|---|",
    ]
    for feature, entry in audit["feature_status"].items():
        present = ", ".join(f"`{key}`" for key in entry["present_keys"]) or "none"
        lines.append(
            f"| `{feature}` | `{entry['status']}` | {present} | {entry['complete_reference_count']}/{entry['reference_count']} | {entry['resolution']} |"
        )
    lines.extend(
        [
            "",
            "## Available reference keys",
            "",
            ", ".join(f"`{key}`" for key in audit["all_reference_keys"]),
            "",
            "## Source custody and admission",
            "",
            f"Dataset declaration bytes: `{audit['dataset_report_sha256']}`",
            "Every selected reference was SHA-bound before channel inspection. PASS means complete converted channels; it does not authenticate original measurements or grant predictive admission.",
            "",
            "## Next processing steps",
            "",
        ]
    )
    lines.extend(f"- {step}" for step in audit["next_processing_steps"])
    return "\n".join([*lines, ""])


def write_report(audit: dict[str, Any], json_out: Path, markdown_out: Path) -> None:
    """Validate/render before writing distinct outputs protected against declared input aliases.

    Unsupported declarations and ordinary IO failures raise ValueError. Writes
    are sequential; a second-file failure may leave the first JSON file.
    """
    try:
        validate_audit_report(audit)
        encoded = json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n"
        markdown = _render(audit)
        protected = [
            Path(audit["dataset_report"]),
            Path(audit["storage_root"]) / audit["dataset_path"],
            Path(audit["storage_root"]) / audit["candidate_report"],
            *(Path(audit["storage_root"]) / shot["reference_path"] for shot in audit["shots"]),
        ]
        ensure_distinct_outputs([json_out, markdown_out], protected=protected)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(encoded, encoding="utf-8")
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(markdown, encoding="utf-8")
    except (OSError, ValueError, TypeError, KeyError, RecursionError, RuntimeError) as exc:
        raise ValueError(f"cannot write feature-provenance audit: {exc}") from exc
