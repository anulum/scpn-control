#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM feature-provenance audit
"""Audit whether prepared MAST EFM bundles contain non-fallback feature sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.build_mast_efm_neural_equilibrium_dataset import DATASET_SCHEMA, FALLBACK_FEATURES

AUDIT_SCHEMA = "scpn-control.mast-efm-feature-provenance-audit.v1"
DEFAULT_DATASET_REPORT = ROOT / "validation" / "reports" / "mast_efm_neural_equilibrium_dataset.json"
DEFAULT_STORAGE_ROOT = Path("/data/SCPN-CONTROL")
DEFAULT_JSON_OUT = ROOT / "validation" / "reports" / "mast_efm_feature_provenance_audit.json"
DEFAULT_MD_OUT = ROOT / "validation" / "reports" / "mast_efm_feature_provenance_audit.md"

FEATURE_CANDIDATES = {
    "Ip_MA": ("Ip_MA", "plasma_current_MA", "plasma_current_A", "ip", "Ip", "current_A"),
    "Bt_T": ("Bt_T", "bcentr_T", "b_tor_T", "toroidal_field_T", "Bt", "bcentr"),
    "ffprime_scale": ("ffprime_scale", "ffprime_rms_T_rad", "ffprime", "ffprime_Wb_per_rad", "fpol", "fpol_profile"),
}


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


def _safe_path(storage_root: Path, relative_path: str) -> Path:
    path = (storage_root / relative_path).resolve()
    root = storage_root.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"reference path escapes storage root: {relative_path}") from exc
    return path


def _array_summary(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        keys = sorted(payload.files)
        shapes = {key: [int(item) for item in np.asarray(payload[key]).shape] for key in keys}
    return {"keys": keys, "shapes": shapes}


def build_audit(dataset_report_path: Path, storage_root: Path) -> dict[str, Any]:
    """Build the provenance audit from prepared MAST EFM reference bundles."""

    dataset_report = _load_json_object(dataset_report_path)
    if dataset_report.get("schema_version") != DATASET_SCHEMA:
        raise ValueError("dataset report has unsupported schema_version")
    references = dataset_report.get("reference_paths")
    if not isinstance(references, list) or not references:
        raise ValueError("dataset report must declare reference_paths")
    shot_reports: list[dict[str, Any]] = []
    all_keys: set[str] = set()
    for reference in references:
        if not isinstance(reference, str):
            raise ValueError("reference_paths entries must be strings")
        path = _safe_path(storage_root, reference)
        if not path.is_file():
            raise FileNotFoundError(f"reference bundle is missing: {path}")
        summary = _array_summary(path)
        all_keys.update(summary["keys"])
        shot_reports.append(
            {
                "reference_path": reference,
                "key_count": len(summary["keys"]),
                "keys": summary["keys"],
                "shapes": summary["shapes"],
            }
        )
    feature_status: dict[str, dict[str, Any]] = {}
    for feature in FALLBACK_FEATURES:
        candidates = FEATURE_CANDIDATES[feature]
        present = sorted(set(candidates) & all_keys)
        feature_status[feature] = {
            "status": "resolved" if present else "blocked",
            "candidate_keys": list(candidates),
            "present_keys": present,
            "resolution": "direct public bundle key available"
            if present
            else "not present in converted public EFM bundles",
        }
    blocked = [feature for feature, entry in feature_status.items() if entry["status"] != "resolved"]
    next_processing_steps = (
        [
            "keep the converted public feature-source keys fixed while training and holdout evaluation are performed",
            "rebuild the supervised dataset whenever converted reference bundles are regenerated",
        ]
        if not blocked
        else [
            "inspect the original public MAST Level 1 EFM/Zarr metadata for plasma-current and toroidal-field channels",
            "acquire or document public FF-prime/fpol provenance or keep ffprime_scale blocked",
            "rebuild the supervised dataset after any non-fallback feature sources are admitted",
        ]
    )
    audit: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA,
        "status": "blocked" if blocked else "pass",
        "dataset_report": str(dataset_report_path),
        "storage_root": str(storage_root),
        "reference_dataset_id": dataset_report.get("reference_dataset_id"),
        "reference_count": len(shot_reports),
        "fallback_features": list(FALLBACK_FEATURES),
        "feature_status": feature_status,
        "blocked_features": blocked,
        "all_reference_keys": sorted(all_keys),
        "shots": shot_reports,
        "next_processing_steps": next_processing_steps,
    }
    audit["payload_sha256"] = _sha256_json({**audit, "payload_sha256": None})
    return audit


def write_report(audit: dict[str, Any], json_out: Path, markdown_out: Path) -> None:
    """Write JSON and Markdown audit reports."""

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
        "| Feature | Status | Present keys | Resolution |",
        "|---|---|---|---|",
    ]
    for feature, entry in audit["feature_status"].items():
        present = ", ".join(f"`{key}`" for key in entry["present_keys"]) or "none"
        lines.append(f"| `{feature}` | `{entry['status']}` | {present} | {entry['resolution']} |")
    lines.extend(["", "## Available reference keys", ""])
    lines.append(", ".join(f"`{key}`" for key in audit["all_reference_keys"]))
    lines.extend(["", "## Next processing steps", ""])
    lines.extend(f"- {item}" for item in audit["next_processing_steps"])
    lines.append("")
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-report", default=DEFAULT_DATASET_REPORT, type=Path)
    parser.add_argument("--storage-root", default=DEFAULT_STORAGE_ROOT, type=Path)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT, type=Path)
    parser.add_argument("--report-out", default=DEFAULT_MD_OUT, type=Path)
    return parser.parse_args()


def main() -> None:
    """Run the feature-provenance audit."""

    args = parse_args()
    audit = build_audit(args.dataset_report, args.storage_root)
    write_report(audit, args.json_out, args.report_out)


if __name__ == "__main__":
    main()
