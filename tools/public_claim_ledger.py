#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Public claim admission ledger
"""Generate the fail-closed ledger of validation reports eligible for public claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

_IMPORT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_IMPORT_ROOT))

from tools.validation_report_freshness import (
    DEFAULT_LIFECYCLE_REGISTRY,
    ROOT,
    ValidationReportFreshness,
    ValidationReportFreshnessMatrix,
    build_validation_report_freshness_matrix,
    parse_datetime,
)

DEFAULT_REPORTS_ROOT = ROOT / "validation" / "reports"
DEFAULT_OUTPUT = ROOT / "validation" / "public_claim_ledger.json"


def _repo_relative(path: Path) -> str:
    """Return a stable repository-relative path when possible."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _claim_record(report: ValidationReportFreshness) -> dict[str, object]:
    """Return one immutable, admitted public-claim record."""
    lifecycle = report.lifecycle
    if lifecycle.report_commit is None:
        raise ValueError(f"public claim report lacks immutable report commit: {lifecycle.path}")
    return {
        "report_path": lifecycle.path,
        "report_sha256": lifecycle.report_sha256,
        "report_commit": lifecycle.report_commit,
        "evidence_time_utc": report.evidence_time.isoformat().replace("+00:00", "Z"),
        "evidence_class": lifecycle.evidence_class,
        "claim_boundary": {
            "current_evidence": lifecycle.current_evidence,
            "scientific_admission": lifecycle.scientific_admission,
            "production_admission": lifecycle.production_admission,
            "public_claim_allowed": lifecycle.public_claim_allowed,
            "rationale": lifecycle.claim_rationale,
        },
        "source_commit": lifecycle.provenance["source_commit"],
        "dependency_lock_sha256": lifecycle.provenance["dependency_lock_sha256"],
        "refresh_artifact_path": lifecycle.refresh_artifact_path,
        "refresh_artifact_sha256": lifecycle.refresh_artifact_sha256,
    }


def _registry_metadata(registry_path: Path) -> tuple[str, str]:
    """Return the registry digest and its declared source commit."""
    raw = registry_path.read_bytes()
    payload: object = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("lifecycle registry must contain a JSON object")
    source_commit = payload.get("registry_source_commit")
    if not isinstance(source_commit, str) or not source_commit.strip():
        raise ValueError("lifecycle registry requires registry_source_commit")
    return hashlib.sha256(raw).hexdigest(), source_commit


def _ledger_from_matrix(
    matrix: ValidationReportFreshnessMatrix,
    *,
    registry_path: Path,
) -> dict[str, object]:
    """Build a deterministic public-claim ledger from validated lifecycle data."""
    registry_sha256, registry_source_commit = _registry_metadata(registry_path)
    claims = sorted(
        (_claim_record(report) for report in matrix.current_admitted_reports),
        key=lambda claim: cast(str, claim["report_path"]),
    )
    return {
        "schema_version": "scpn-control.public-claim-ledger.v1",
        "generated_from": {
            "lifecycle_registry_path": _repo_relative(registry_path),
            "lifecycle_registry_sha256": registry_sha256,
            "registry_source_commit": registry_source_commit,
        },
        "admission_policy": {
            "freshness_max_age_days": matrix.max_age_days,
            "requires_current_evidence": True,
            "requires_scientific_admission": True,
            "requires_public_claim_permission": True,
        },
        "public_claim_count": len(claims),
        "claims": claims,
    }


def build_public_claim_ledger(
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    *,
    registry_path: Path = DEFAULT_LIFECYCLE_REGISTRY,
    as_of: datetime | None = None,
    max_age_days: int = 21,
) -> dict[str, object]:
    """Validate lifecycle metadata and return the admitted public-claim ledger."""
    matrix = build_validation_report_freshness_matrix(
        reports_root,
        as_of=as_of or datetime.now(tz=timezone.utc),
        max_age_days=max_age_days,
        registry_path=registry_path,
    )
    return _ledger_from_matrix(matrix, registry_path=registry_path)


def main(argv: list[str] | None = None) -> int:
    """Generate or verify the canonical public-claim ledger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    parser.add_argument("--registry", default=str(DEFAULT_LIFECYCLE_REGISTRY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--as-of", help="UTC timestamp for deterministic freshness evaluation")
    parser.add_argument("--max-age-days", type=int, default=21)
    parser.add_argument("--check", action="store_true", help="Fail when the committed ledger differs")
    args = parser.parse_args(argv)

    try:
        payload = build_public_claim_ledger(
            Path(args.reports_root),
            registry_path=Path(args.registry),
            as_of=parse_datetime(args.as_of) if args.as_of else None,
            max_age_days=args.max_age_days,
        )
        rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        output = Path(args.output)
        if args.check:
            if output.read_text(encoding="utf-8") != rendered:
                print(f"Public claim ledger drift: {output}", file=sys.stderr)
                return 1
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered, encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        print(f"Public claim ledger failed: {exc}", file=sys.stderr)
        return 1

    print(f"Public claim ledger: claims={payload['public_claim_count']} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
