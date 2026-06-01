#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM original feature-source audit
"""Audit original public MAST Level 1 EFM Zarr metadata for feature sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.build_mast_efm_neural_equilibrium_dataset import DATASET_SCHEMA, FALLBACK_FEATURES

AUDIT_SCHEMA = "scpn-control.mast-efm-original-feature-source-audit.v1"
DEFAULT_DATASET_REPORT = ROOT / "validation" / "reports" / "mast_efm_neural_equilibrium_dataset.json"
DEFAULT_SAS_ROOT = Path("/mnt/data_sas/DATASETS/SCPN-CONTROL")
DEFAULT_JSON_OUT = ROOT / "validation" / "reports" / "mast_efm_original_feature_source_audit.json"
DEFAULT_MD_OUT = ROOT / "validation" / "reports" / "mast_efm_original_feature_source_audit.md"

SHOT_RE = re.compile(r"mast_efm_shot_(?P<shot_id>\d+)_reference[.]npz$")

FEATURE_SOURCE_POLICY: dict[str, dict[str, Any]] = {
    "Ip_MA": {
        "candidates": ("plasma_current_x", "plasma_current_c", "plasma_current_rz"),
        "preferred": "plasma_current_x",
        "required_units": "A",
        "required_dims": ("time",),
        "required_transform": "A_to_MA",
        "source_kind": "measured_total_plasma_current",
        "resolved_status": "source_found_requires_rebuild",
        "resolution": "measured total plasma current is available in original public EFM metadata",
    },
    "Bt_T": {
        "candidates": ("bphi_rmag", "bphi_rgeom", "bvac_rmag", "bvac_rgeom", "bvac_val"),
        "preferred": "bphi_rmag",
        "required_units": "T",
        "required_dims": ("time",),
        "required_transform": "identity_T",
        "source_kind": "total_toroidal_field_at_magnetic_axis",
        "resolved_status": "source_found_requires_rebuild",
        "resolution": "total toroidal field at the magnetic axis is available in original public EFM metadata",
    },
    "ffprime_scale": {
        "candidates": ("ffprime", "ffprime_coefs", "fpsi_c"),
        "preferred": "ffprime",
        "required_units": "T-rad",
        "required_dims": ("time", "psi_norm"),
        "required_transform": "profile_to_training_scalar",
        "source_kind": "ffprime_profile",
        "resolved_status": "source_found_requires_policy",
        "resolution": "FF-prime profile is available, but the scalar training-feature reduction must be specified before rebuild",
    },
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


def _safe_path(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    resolved_root = root.resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"path escapes SAS root: {relative_path}") from exc
    return path


def _normalise_dims(attrs: dict[str, Any]) -> list[str]:
    dims = attrs.get("_ARRAY_DIMENSIONS", attrs.get("dims", []))
    if not isinstance(dims, list):
        return []
    return [str(item) for item in dims]


def _variable_summary(metadata: dict[str, Any], variable_name: str) -> dict[str, Any]:
    zarray = metadata.get(f"{variable_name}/.zarray", {})
    zattrs = metadata.get(f"{variable_name}/.zattrs", {})
    if not isinstance(zarray, dict) or not isinstance(zattrs, dict):
        return {}
    return {
        "attrs": dict(zattrs),
        "chunks": list(zarray.get("chunks", [])),
        "description": zattrs.get("description"),
        "dims": _normalise_dims(zattrs),
        "dtype": zarray.get("dtype"),
        "mds_name": zattrs.get("mds_name"),
        "quality": zattrs.get("quality"),
        "shape": list(zarray.get("shape", [])),
        "uda_name": zattrs.get("uda_name"),
        "units": zattrs.get("units"),
    }


def load_zarr_candidate_metadata(zarr_path: Path) -> dict[str, dict[str, Any]]:
    """Load candidate variable metadata from consolidated Zarr JSON."""

    metadata_path = zarr_path / ".zmetadata"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"consolidated Zarr metadata is missing: {metadata_path}")
    payload = _load_json_object(metadata_path)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path} does not contain consolidated metadata")
    candidate_names = sorted({name for policy in FEATURE_SOURCE_POLICY.values() for name in policy["candidates"]})
    variables: dict[str, dict[str, Any]] = {}
    for name in candidate_names:
        summary = _variable_summary(metadata, name)
        if summary:
            variables[name] = summary
    return variables


def _select_candidate(feature: str, variables: dict[str, dict[str, Any]]) -> dict[str, Any]:
    policy = FEATURE_SOURCE_POLICY[feature]
    candidates = list(policy["candidates"])
    present = [name for name in candidates if name in variables]
    selected = None
    for name in candidates:
        variable = variables.get(name)
        if not variable:
            continue
        attrs = variable.get("attrs", {})
        if not isinstance(attrs, dict):
            attrs = {}
        units = variable.get("units", attrs.get("units"))
        dims = variable.get("dims", _normalise_dims(attrs))
        if str(units or "") != policy["required_units"]:
            continue
        if tuple(dims) != tuple(policy["required_dims"]):
            continue
        selected = name
        break
    if selected is None:
        return {
            "candidate_sources": candidates,
            "present_sources": present,
            "required_dims": list(policy["required_dims"]),
            "required_transform": policy["required_transform"],
            "required_units": policy["required_units"],
            "resolution": "required source metadata is not present in the original public EFM Zarr store",
            "selected_source": None,
            "source_kind": policy["source_kind"],
            "status": "blocked",
        }
    source = variables[selected]
    attrs = source.get("attrs", {})
    if not isinstance(attrs, dict):
        attrs = {}
    return {
        "candidate_sources": candidates,
        "present_sources": present,
        "required_dims": list(policy["required_dims"]),
        "required_transform": policy["required_transform"],
        "required_units": policy["required_units"],
        "resolution": policy["resolution"],
        "selected_source": selected,
        "selected_source_metadata": {
            "description": source.get("description", attrs.get("description")),
            "dims": source.get("dims", _normalise_dims(attrs)),
            "dtype": source.get("dtype"),
            "mds_name": source.get("mds_name", attrs.get("mds_name")),
            "quality": source.get("quality", attrs.get("quality")),
            "shape": source.get("shape"),
            "uda_name": source.get("uda_name", attrs.get("uda_name")),
            "units": source.get("units", attrs.get("units")),
        },
        "source_kind": policy["source_kind"],
        "status": policy["resolved_status"],
    }


def classify_feature_sources(variables: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Classify original metadata support for fallback training features."""

    return {feature: _select_candidate(feature, variables) for feature in FALLBACK_FEATURES}


def _shot_ids_from_dataset_report(dataset_report: dict[str, Any]) -> list[int]:
    shots = dataset_report.get("shots")
    if isinstance(shots, list) and shots:
        shot_ids = [int(item["shot_id"]) for item in shots if isinstance(item, dict) and "shot_id" in item]
        if shot_ids:
            return sorted(dict.fromkeys(shot_ids))
    references = dataset_report.get("reference_paths")
    if not isinstance(references, list):
        raise ValueError("dataset report must declare shots or reference_paths")
    shot_ids = []
    for reference in references:
        if not isinstance(reference, str):
            continue
        match = SHOT_RE.search(reference)
        if match:
            shot_ids.append(int(match.group("shot_id")))
    if not shot_ids:
        raise ValueError("dataset report does not expose MAST EFM shot identifiers")
    return sorted(dict.fromkeys(shot_ids))


def _aggregate_feature_status(shots: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    aggregate: dict[str, dict[str, Any]] = {}
    for feature in FALLBACK_FEATURES:
        entries = [shot["feature_status"][feature] for shot in shots]
        statuses = [entry["status"] for entry in entries]
        selected_sources = sorted({entry["selected_source"] for entry in entries if entry["selected_source"]})
        present_sources = sorted({source for entry in entries for source in entry["present_sources"]})
        if any(status == "blocked" for status in statuses):
            status = "blocked"
        elif any(status == "source_found_requires_policy" for status in statuses):
            status = "source_found_requires_policy"
        else:
            status = "source_found_requires_rebuild"
        exemplar = entries[0]
        aggregate[feature] = {
            "candidate_sources": exemplar["candidate_sources"],
            "present_sources": present_sources,
            "required_dims": exemplar["required_dims"],
            "required_transform": exemplar["required_transform"],
            "required_units": exemplar["required_units"],
            "resolution": exemplar["resolution"]
            if len({entry["resolution"] for entry in entries}) == 1
            else "source metadata differs across shots",
            "selected_source": selected_sources[0] if len(selected_sources) == 1 else None,
            "selected_sources": selected_sources,
            "source_kind": exemplar["source_kind"],
            "status": status,
        }
    return aggregate


def build_original_feature_source_audit(dataset_report_path: Path, sas_root: Path) -> dict[str, Any]:
    """Build the original public Zarr feature-source audit."""

    dataset_report = _load_json_object(dataset_report_path)
    if dataset_report.get("schema_version") != DATASET_SCHEMA:
        raise ValueError("dataset report has unsupported schema_version")
    shot_ids = _shot_ids_from_dataset_report(dataset_report)
    shot_reports: list[dict[str, Any]] = []
    for shot_id in shot_ids:
        relative_zarr_path = f"mast/level1/shot_{shot_id}/efm.zarr"
        zarr_path = _safe_path(sas_root, relative_zarr_path)
        variables = load_zarr_candidate_metadata(zarr_path)
        shot_reports.append(
            {
                "feature_status": classify_feature_sources(variables),
                "shot_id": shot_id,
                "source_variables": variables,
                "zarr_path": relative_zarr_path,
            }
        )
    feature_status = _aggregate_feature_status(shot_reports)
    blocked_features = [
        feature for feature, entry in feature_status.items() if entry["status"] != "source_found_requires_rebuild"
    ]
    audit: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA,
        "status": "blocked" if blocked_features else "source_ready",
        "can_rebuild_dataset_now": not blocked_features,
        "blocked_features": blocked_features,
        "dataset_report": str(dataset_report_path),
        "fallback_features": list(FALLBACK_FEATURES),
        "feature_status": feature_status,
        "next_processing_steps": _next_processing_steps(blocked_features),
        "reference_dataset_id": dataset_report.get("reference_dataset_id"),
        "sas_root": str(sas_root),
        "shot_count": len(shot_reports),
        "shots": shot_reports,
    }
    audit["payload_sha256"] = _sha256_json({**audit, "payload_sha256": None})
    return audit


def _next_processing_steps(blocked_features: list[str]) -> list[str]:
    steps = [
        "rebuild the supervised dataset only after every fallback feature has an admitted original public source",
        "record the source-variable policy in the dataset report before training",
    ]
    if "ffprime_scale" in blocked_features:
        steps.insert(0, "define ffprime profile reduction before rebuilding the supervised dataset")
    if "Bt_T" in blocked_features:
        steps.insert(0, "confirm total versus vacuum toroidal-field policy before rebuilding Bt_T")
    if "Ip_MA" in blocked_features:
        steps.insert(0, "admit a plasma-current source and unit conversion before rebuilding Ip_MA")
    return steps


def write_report(audit: dict[str, Any], json_out: Path, markdown_out: Path) -> None:
    """Write JSON and Markdown original-source audit reports."""

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "<!-- Commercial license available -->",
        "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
        "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
        "<!-- ORCID: 0009-0009-3560-0851 -->",
        "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
        "<!-- SCPN Control — MAST EFM original feature-source audit report -->",
        "",
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
        selected = entry.get("selected_source") or ", ".join(entry.get("selected_sources", [])) or "none"
        lines.append(
            f"| `{feature}` | `{entry['status']}` | `{selected}` | "
            f"`{entry['required_transform']}` | {entry['resolution']} |"
        )
    lines.extend(["", "## Next processing steps", ""])
    lines.extend(f"- {item}" for item in audit["next_processing_steps"])
    lines.append("")
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-report", default=DEFAULT_DATASET_REPORT, type=Path)
    parser.add_argument("--sas-root", default=DEFAULT_SAS_ROOT, type=Path)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT, type=Path)
    parser.add_argument("--report-out", default=DEFAULT_MD_OUT, type=Path)
    return parser.parse_args()


def main() -> None:
    """Run the original public Zarr feature-source audit."""

    args = parse_args()
    audit = build_original_feature_source_audit(args.dataset_report, args.sas_root)
    write_report(audit, args.json_out, args.report_out)


if __name__ == "__main__":
    main()
