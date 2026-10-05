#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM original feature-source audit
"""Audit actual captured original MAST conversion and selected reference equivalence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from validation.audit_mast_efm_feature_provenance import build_audit
from validation.mast_efm_feature_audit_inputs import read_dataset_declaration
from validation.mast_efm_original_source_inputs import inspect_original_sources
from validation.mast_efm_original_source_inputs import load_zarr_candidate_metadata as load_zarr_candidate_metadata
from validation.mast_efm_original_source_policy import AUDIT_SCHEMA as AUDIT_SCHEMA
from validation.mast_efm_original_source_policy import FEATURE_SOURCE_POLICY as FEATURE_SOURCE_POLICY
from validation.mast_efm_original_source_policy import aggregate_feature_status
from validation.mast_efm_original_source_policy import classify_feature_sources as classify_feature_sources
from validation.mast_efm_original_source_reporting import validate_original_audit_bindings
from validation.mast_efm_original_source_reporting import (
    validate_original_audit_report as validate_original_audit_report,
)
from validation.mast_efm_original_source_reporting import write_report as write_report
from validation.neural_equilibrium_campaign_inputs import canonical_campaign_digest
from validation.neural_equilibrium_dataset_contracts import DATASET_SCHEMA as DATASET_SCHEMA
from validation.neural_equilibrium_dataset_contracts import FALLBACK_FEATURES as FALLBACK_FEATURES

DEFAULT_DATASET_REPORT = ROOT / "validation" / "reports" / "mast_efm_neural_equilibrium_dataset.json"
DEFAULT_STORAGE_ROOT = Path("/data/SCPN-CONTROL")
DEFAULT_JSON_OUT = ROOT / "validation" / "reports" / "mast_efm_original_feature_source_audit.json"
DEFAULT_MD_OUT = ROOT / "validation" / "reports" / "mast_efm_original_feature_source_audit.md"


def _next_processing_steps(blocked: list[str], conversion_blocked: list[int]) -> list[str]:
    """Describe exact outstanding original conversion/source policies without admitting training."""
    steps = ["keep the source-variable policy fixed during training and held-out evaluation"]
    if conversion_blocked:
        steps.insert(0, "repair original conversion/reference mismatches before rebuilding the supervised dataset")
    if not blocked and not conversion_blocked:
        steps.insert(0, "rebuild or verify the supervised dataset from the matched original public source arrays")
    for feature in blocked:
        steps.insert(0, f"admit the preferred converter-supported source and transform for {feature}")
    return steps


def build_original_feature_source_audit(dataset_report_path: Path, storage_root: Path) -> dict[str, Any]:
    """Build v2 original conversion evidence bound to the selected full producer declaration.

    Every store is decoded from its privately captured bytes, compared against
    the SHA-bound converted reference and inventoried with exact file digests.
    Metadata-only v1 readiness is not promoted or resealed. Local engineering
    equivalence does not authenticate physical acquisition or admit predictions.
    """
    dataset, captured_sha = read_dataset_declaration(dataset_report_path)
    converted = build_audit(dataset_report_path, storage_root)
    shots = inspect_original_sources(dataset, storage_root)
    features = aggregate_feature_status(shots)
    blocked = [name for name, entry in features.items() if entry["status"] != "source_found_requires_rebuild"]
    conversion_blocked = [shot["shot_id"] for shot in shots if shot["conversion_check"]["status"] != "pass"]
    ready = not blocked and not conversion_blocked and converted["status"] == "pass"
    audit: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA,
        "status": "source_ready" if ready else "blocked",
        "can_rebuild_dataset_now": ready,
        "blocked_features": blocked,
        "conversion_blocked_shots": conversion_blocked,
        "dataset_report": str(dataset_report_path),
        "storage_root": str(storage_root),
        "fallback_features": list(FALLBACK_FEATURES),
        "feature_status": features,
        "next_processing_steps": _next_processing_steps(blocked, conversion_blocked),
        "reference_dataset_id": dataset["reference_dataset_id"],
        "shot_count": len(shots),
        "shots": shots,
        "converted_feature_audit": converted,
    }
    audit["payload_sha256"] = canonical_campaign_digest(audit)
    validate_original_audit_bindings(audit, dataset, dataset_report_sha256=captured_sha)
    return audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse explicit inspection inputs and output destinations; help/usage preserve argparse exit0/2.

    >>> str(parse_args([]).storage_root)
    '/data/SCPN-CONTROL'
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-report", default=DEFAULT_DATASET_REPORT, type=Path)
    parser.add_argument("--storage-root", default=DEFAULT_STORAGE_ROOT, type=Path)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT, type=Path)
    parser.add_argument("--report-out", default=DEFAULT_MD_OUT, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Produce a complete ready/blocked audit with exit0; ordinary inspection/write refusals exit1."""
    args = parse_args(argv)
    try:
        audit = build_original_feature_source_audit(args.dataset_report, args.storage_root)
        write_report(audit, args.json_out, args.report_out)
    except (OSError, ValueError, TypeError, KeyError, RecursionError, RuntimeError) as exc:
        print(f"Original MAST source audit refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
