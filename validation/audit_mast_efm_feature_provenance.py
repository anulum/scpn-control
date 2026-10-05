#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM feature-provenance audit
"""Audit complete SHA-bound converted MAST feature channels without granting admission.

PASS requires every selected shot to supply the actual producer's canonical
Ip_MA, Bt_T and positive FF-prime RMS vectors. Alternative aliases are inventory
hints, not admitted transformations. The original acquisition process and the
supervised tensor corpus require their separate gates.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.mast_efm_feature_audit_inputs import AUDIT_SCHEMA as AUDIT_SCHEMA
from validation.mast_efm_feature_audit_inputs import FEATURE_CANDIDATES as FEATURE_CANDIDATES
from validation.mast_efm_feature_audit_inputs import (
    feature_status_for_shots,
    inspect_reference_sources,
    read_dataset_declaration,
    reference_paths,
)
from validation.mast_efm_feature_audit_reporting import validate_audit_report as validate_audit_report
from validation.mast_efm_feature_audit_reporting import write_report as write_report
from validation.neural_equilibrium_campaign_inputs import canonical_campaign_digest
from validation.neural_equilibrium_dataset_contracts import DATASET_SCHEMA as DATASET_SCHEMA
from validation.neural_equilibrium_dataset_contracts import FALLBACK_FEATURES as FALLBACK_FEATURES
from validation.neural_equilibrium_training_rendering import ensure_distinct_outputs

DEFAULT_DATASET_REPORT = ROOT / "validation" / "reports" / "mast_efm_neural_equilibrium_dataset.json"
DEFAULT_STORAGE_ROOT = Path("/data/SCPN-CONTROL")
DEFAULT_JSON_OUT = ROOT / "validation" / "reports" / "mast_efm_feature_provenance_audit.json"
DEFAULT_MD_OUT = ROOT / "validation" / "reports" / "mast_efm_feature_provenance_audit.md"


def build_audit(dataset_report_path: Path, storage_root: Path) -> dict[str, Any]:
    """Verify producer declarations and captured reference bytes, then audit all-shot channel completeness.

    Missing supported channels yield a blocked report; malformed present channels
    or stale byte/metadata bindings refuse. This checks converted source vectors,
    not original measurements, target correctness or supervised tensor contents.
    """
    dataset_report, declaration_sha = read_dataset_declaration(dataset_report_path)
    shots = inspect_reference_sources(dataset_report, storage_root)
    status = feature_status_for_shots(shots)
    blocked = [feature for feature, entry in status.items() if entry["status"] == "blocked"]
    if set(blocked) != set(dataset_report["fallback_features"]):
        raise ValueError("dataset fallback_features disagree with actual selected source channels")
    steps = (
        [
            "keep converted channels and their byte bindings fixed during training and holdout evaluation",
            "rebuild the supervised dataset whenever converted reference bundles are regenerated",
        ]
        if not blocked
        else [
            "inspect original metadata for missing canonical per-equilibrium source channels",
            "admit any source aliases through the converter and rebuild the supervised dataset before training",
        ]
    )
    audit: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA,
        "status": "blocked" if blocked else "pass",
        "dataset_report": str(dataset_report_path.resolve()),
        "dataset_report_sha256": declaration_sha,
        "dataset_payload_sha256": dataset_report["payload_sha256"],
        "dataset_sha256": dataset_report["dataset_sha256"],
        "dataset_path": dataset_report["dataset_path"],
        "candidate_report": dataset_report["candidate_report"],
        "storage_root": str(storage_root.resolve()),
        "reference_dataset_id": dataset_report["reference_dataset_id"],
        "reference_count": len(shots),
        "fallback_features": list(FALLBACK_FEATURES),
        "feature_status": status,
        "blocked_features": blocked,
        "all_reference_keys": sorted({key for shot in shots for key in shot["keys"]}),
        "shots": shots,
        "next_processing_steps": steps,
    }
    audit["payload_sha256"] = canonical_campaign_digest(audit)
    return validate_audit_report(audit)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse optional explicit arguments, retaining historical defaults and argparse help/usage exits.

    >>> parse_args(["--storage-root", "/tmp/selected-mast-source"]).storage_root
    PosixPath('/tmp/selected-mast-source')
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-report", default=DEFAULT_DATASET_REPORT, type=Path)
    parser.add_argument("--storage-root", default=DEFAULT_STORAGE_ROOT, type=Path)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT, type=Path)
    parser.add_argument("--report-out", default=DEFAULT_MD_OUT, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run real source inspection with success0, authored domain failure1 and argparse usage2.

    Output aliases are refused before inspection/persistence. A blocked audit is
    a completed inspection, not predictive admission, and therefore exits zero.
    """
    args = parse_args(argv)
    try:
        dataset, _ = read_dataset_declaration(args.dataset_report)
        ensure_distinct_outputs(
            [args.json_out, args.report_out],
            protected=[
                args.dataset_report,
                *reference_paths(dataset, args.storage_root),
                args.storage_root / dataset["dataset_path"],
                args.storage_root / dataset["candidate_report"],
            ],
        )
        audit = build_audit(args.dataset_report, args.storage_root)
        write_report(audit, args.json_out, args.report_out)
    except (OSError, ValueError, TypeError, RecursionError, RuntimeError) as exc:
        print(f"FAIL: feature-provenance audit: {exc}", file=sys.stderr)
        return 1
    print(f"Feature-provenance audit: {audit['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
