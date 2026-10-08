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
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

_IMPORT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_IMPORT_ROOT))

from tools.report_inventory_output import InventoryOutputError, publish_inventory_outputs
from tools.validation_report_freshness import (
    DEFAULT_LIFECYCLE_REGISTRY,
    ROOT,
    LifecycleRegistryError,
    ValidationReportFreshness,
    ValidationReportFreshnessMatrix,
    build_validation_report_freshness_matrix,
    parse_datetime,
)

DEFAULT_REPORTS_ROOT = ROOT / "validation" / "reports"
DEFAULT_OUTPUT = ROOT / "validation" / "public_claim_ledger.json"


def _repo_relative(path: Path) -> str:
    """Return a canonical repository-relative path or the supplied external name.

    Parameters
    ----------
    path : pathlib.Path
        Registry or artifact path selected by the caller.

    Returns
    -------
    str
        Resolved repository-relative POSIX spelling when contained, otherwise the
        original POSIX spelling. No external containment is implied.
    """
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _claim_record(report: ValidationReportFreshness) -> dict[str, object]:
    """Serialise one validated report's declared public claim.

    Parameters
    ----------
    report : ValidationReportFreshness
        Fresh report selected by the validated matrix's declaration filter.

    Returns
    -------
    dict of str to object
        Source/refresh digests, UTC evidence time and retained claim/provenance.

    Raises
    ------
    LifecycleRegistryError
        Selected report lacks its required immutable report commit declaration.

    Notes
    -----
    This serialises declarations; it does not independently attest an experiment.
    """
    lifecycle = report.lifecycle
    if lifecycle.report_commit is None:
        raise LifecycleRegistryError(f"public claim report lacks immutable report commit: {lifecycle.path}")
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
    """Read the registry's digest and its declared source commit.

    Parameters
    ----------
    registry_path : pathlib.Path
        Consumed UTF-8 JSON registry, opened again for the ledger byte binding.

    Returns
    -------
    tuple of (str, str)
        SHA-256 digest and nonblank declared source commit.

    Raises
    ------
    LifecycleRegistryError
        Top-level object or source commit declaration is absent.
    OSError, ValueError
        Registry cannot be read or decoded.

    Notes
    -----
    This is a second read after matrix validation. Callers must coordinate input
    changes; it is not a transaction across concurrent registry modification.
    """
    raw = registry_path.read_bytes()
    payload: object = json.loads(raw)
    if not isinstance(payload, dict):
        raise LifecycleRegistryError("lifecycle registry must contain a JSON object")
    source_commit = payload.get("registry_source_commit")
    if not isinstance(source_commit, str) or not source_commit.strip():
        raise LifecycleRegistryError("lifecycle registry requires registry_source_commit")
    return hashlib.sha256(raw).hexdigest(), source_commit


def _ledger_from_matrix(
    matrix: ValidationReportFreshnessMatrix,
    *,
    registry_path: Path,
) -> dict[str, object]:
    """Select the validated matrix's declared public claims for the ledger schema.

    Parameters
    ----------
    matrix : ValidationReportFreshnessMatrix
        Digest-validated inventory with the caller's advisory freshness window.
    registry_path : pathlib.Path
        Consumed registry, reread for its digest and declared source commit.

    Returns
    -------
    dict of str to object
        Version-one ledger with sorted claims and explicit admission requirements.

    Raises
    ------
    LifecycleRegistryError
        Registry metadata or a selected report's commit declaration is invalid.
    OSError, ValueError
        Registry bytes cannot be read or parsed.

    Notes
    -----
    Selection requires fresh/current/scientific/public declarations, not production
    admission. Declaration consistency does not establish independent physical truth.
    """
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
    """Validate lifecycle inputs and return the declared public-claim ledger.

    Parameters
    ----------
    reports_root : pathlib.Path, optional
        Selected report directory, defaulting to the canonical validation corpus.
    registry_path : pathlib.Path, optional
        Lifecycle registry binding the selected report and refresh bytes.
    as_of : datetime.datetime or None, optional
        Evaluation time; null uses the current UTC clock and naive values denote UTC.
    max_age_days : int, optional
        Exact nonnegative advisory window; the registry retains its audited 21 days.

    Returns
    -------
    dict of str to object
        Version-one ledger, preserving the public API and canonical field meanings.

    Raises
    ------
    LifecycleRegistryError
        Lifecycle, provenance, digest, scalar or selected-claim declarations fail.
    OSError, ValueError
        Inputs cannot be read or decoded.

    Notes
    -----
    No producer is executed and no file is published. Git/host fields remain
    validated declarations. The registry is reread for its digest after validation.
    """
    matrix = build_validation_report_freshness_matrix(
        reports_root,
        as_of=as_of or datetime.now(tz=UTC),
        max_age_days=max_age_days,
        registry_path=registry_path,
    )
    return _ledger_from_matrix(matrix, registry_path=registry_path)


def main(argv: list[str] | None = None) -> int:
    """Generate or check the ledger through the actual public CLI.

    Parameters
    ----------
    argv : list of str or None, optional
        Process options for input roots, output, time/window and read-only check mode.

    Returns
    -------
    int
        Zero on publication or exact check; one on drift or input/output refusal.

    Raises
    ------
    SystemExit
        Argument parsing rejects syntax or displays help.

    Notes
    -----
    Builds one validated matrix and serialises its ledger once. Publication shares
    source protection and handled-failure recovery with the freshness inventory.
    Authored refusal types retain their deliberate messages; other caught exceptions
    use fixed caller-safe text. Check mode reads without replacing any file.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    parser.add_argument("--registry", default=str(DEFAULT_LIFECYCLE_REGISTRY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--as-of", help="UTC timestamp for deterministic freshness evaluation")
    parser.add_argument("--max-age-days", type=int, default=21)
    parser.add_argument("--check", action="store_true", help="Fail when the committed ledger differs")
    args = parser.parse_args(argv)

    try:
        registry_path = Path(args.registry)
        matrix = build_validation_report_freshness_matrix(
            Path(args.reports_root),
            registry_path=registry_path,
            as_of=parse_datetime(args.as_of) if args.as_of is not None else datetime.now(tz=UTC),
            max_age_days=args.max_age_days,
        )
        payload = _ledger_from_matrix(matrix, registry_path=registry_path)
        rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        output = Path(args.output)
        if args.check:
            if output.read_text(encoding="utf-8") != rendered:
                print(f"Public claim ledger drift: {output}", file=sys.stderr)
                return 1
        else:
            publish_inventory_outputs(matrix, ((output, rendered.encode("utf-8")),), registry_path=registry_path)
    except (LifecycleRegistryError, InventoryOutputError) as exc:
        print(f"Public claim ledger failed: {exc}", file=sys.stderr)
        return 1
    except (OSError, TypeError, ValueError):
        print("Public claim ledger inputs or output could not be inspected", file=sys.stderr)
        return 1

    print(f"Public claim ledger: claims={payload['public_claim_count']} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
