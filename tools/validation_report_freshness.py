# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Validation report freshness inventory

"""Validation report freshness inventory."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.report_inventory_output import InventoryOutputError, write_inventory_outputs
from tools.report_inventory_types import (
    ROOT as ROOT,
)
from tools.report_inventory_types import (
    ValidationReportClassification as ValidationReportClassification,
)
from tools.report_inventory_types import (
    ValidationReportFreshness as ValidationReportFreshness,
)
from tools.report_inventory_types import (
    ValidationReportFreshnessMatrix as ValidationReportFreshnessMatrix,
)
from tools.report_inventory_types import (
    ValidationReportRefreshPlan as ValidationReportRefreshPlan,
)
from tools.report_lifecycle_registry import (
    load_validation_report_lifecycle_registry as load_validation_report_lifecycle_registry,
)
from tools.report_lifecycle_types import (
    EvidenceClass as EvidenceClass,
)
from tools.report_lifecycle_types import (
    FreshnessBucket as FreshnessBucket,
)
from tools.report_lifecycle_types import (
    LifecycleRefreshStatus as LifecycleRefreshStatus,
)
from tools.report_lifecycle_types import (
    LifecycleRegistryError as LifecycleRegistryError,
)
from tools.report_lifecycle_types import (
    RefreshPlanStatus as RefreshPlanStatus,
)
from tools.report_lifecycle_types import (
    ValidationReportLifecycle as ValidationReportLifecycle,
)
from tools.report_lifecycle_values import _normalize_datetime, _read_json_object, _validate_max_age_days
from tools.report_lifecycle_values import parse_datetime as parse_datetime

DEFAULT_LIFECYCLE_REGISTRY = ROOT / "validation" / "report_lifecycle_registry.json"


def build_validation_report_freshness_matrix(
    reports_root: Path,
    *,
    as_of: datetime,
    max_age_days: int,
    registry_path: Path = DEFAULT_LIFECYCLE_REGISTRY,
) -> ValidationReportFreshnessMatrix:
    """Build a digest-bound freshness matrix for validation report artifacts.

    Parameters
    ----------
    reports_root : pathlib.Path
        Existing report directory, including a supported external corpus copy.
    as_of : datetime.datetime
        Evaluation time; naive timestamps denote UTC.
    max_age_days : int
        Exact nonnegative advisory window without an upper bound. The registry
        independently retains its audited 21-day policy.
    registry_path : pathlib.Path, optional
        Lifecycle registry binding the selected reports and refreshes.

    Returns
    -------
    ValidationReportFreshnessMatrix
        Sorted inventory retaining source/refresh bindings, original caveats
        and effective ages. Inputs are read without alteration.

    Raises
    ------
    LifecycleRegistryError
        Registry, scalar, corpus, digest or declaration checks fail.
    OSError, ValueError
        Report root, input bytes or timestamps cannot be inspected or parsed.

    Notes
    -----
    This verifies local byte bindings and declaration consistency. It neither
    runs producers nor verifies Git objects, host authenticity or physical truth.
    """
    _validate_max_age_days(max_age_days)
    if not reports_root.exists():
        raise ValueError(f"reports root does not exist: {reports_root}")
    if not reports_root.is_dir():
        raise ValueError(f"reports root is not a directory: {reports_root}")
    normalized_as_of = _normalize_datetime(as_of)
    lifecycle_by_path = load_validation_report_lifecycle_registry(
        registry_path,
        reports_root=reports_root,
        as_of=normalized_as_of,
        max_age_days=max_age_days,
    )
    reports = tuple(
        _report_freshness(
            reports_root / Path(lifecycle.path).relative_to("validation/reports"),
            lifecycle,
            as_of=normalized_as_of,
            max_age_days=max_age_days,
        )
        for lifecycle in sorted(lifecycle_by_path.values(), key=lambda item: item.path)
    )
    return ValidationReportFreshnessMatrix(
        reports_root=reports_root,
        as_of=normalized_as_of,
        max_age_days=max_age_days,
        reports=reports,
    )


def _report_freshness(
    path: Path,
    lifecycle: ValidationReportLifecycle,
    *,
    as_of: datetime,
    max_age_days: int,
) -> ValidationReportFreshness:
    """Combine a validated lifecycle with its effective timestamp and source marker.

    Parameters
    ----------
    path : pathlib.Path
        Report path; indexed owner-local bytes may be absent.
    lifecycle : ValidationReportLifecycle
        Registry record whose byte/admission bindings were checked.
    as_of : datetime.datetime
        Normalised UTC evaluation time.
    max_age_days : int
        Validated advisory age window.

    Returns
    -------
    ValidationReportFreshness
        Effective age, staleness and advisory refresh plan.

    Raises
    ------
    LifecycleRegistryError
        Embedded claim-marker presence disagrees with the registry.
    OSError, ValueError
        Existing report cannot be read or decoded as a JSON object.
    """
    if path.is_file():
        source_claim_boundary_present = _contains_claim_boundary(_read_json_object(path))
        if source_claim_boundary_present != lifecycle.source_claim_boundary_present:
            raise LifecycleRegistryError(f"source claim-boundary drift for {lifecycle.path}")
    effective_time = lifecycle.refresh_evidence_time or lifecycle.evidence_time
    age_days = max((as_of - effective_time).days, 0)
    classification = ValidationReportClassification(
        bucket=lifecycle.bucket,
        rationale=lifecycle.claim_rationale,
    )
    return ValidationReportFreshness(
        path=path,
        evidence_time=effective_time,
        evidence_time_source="lifecycle_refresh"
        if lifecycle.refresh_evidence_time is not None
        else lifecycle.evidence_time_source,
        age_days=age_days,
        stale=age_days > max_age_days,
        claim_boundary_present=True,
        lifecycle=lifecycle,
        classification=classification,
        refresh_plan=_build_refresh_plan(classification, lifecycle),
    )


def _build_refresh_plan(
    classification: ValidationReportClassification,
    lifecycle: ValidationReportLifecycle,
) -> ValidationReportRefreshPlan:
    """Classify preserved command text without executing or reconstructing it.

    Parameters
    ----------
    classification : ValidationReportClassification
        Audited lifecycle bucket and its rationale.
    lifecycle : ValidationReportLifecycle
        Validated record carrying preserved commands.

    Returns
    -------
    ValidationReportRefreshPlan
        Exact-command, manual-reconstruction or nonlocal advisory status.
        Ellipses make commands partial; no commands are also manual work.
    """
    if classification.bucket != "rerunnable_local":
        return ValidationReportRefreshPlan(
            status="not_rerunnable_local",
            commands=(),
            rationale="Report is not classified as rerunnable-local.",
        )

    preserved_commands = lifecycle.refresh_commands
    if preserved_commands and all("..." not in command for command in preserved_commands):
        return ValidationReportRefreshPlan(
            status="ready_exact_command",
            commands=preserved_commands,
            rationale="Exact command text is preserved in the lifecycle registry.",
        )

    if preserved_commands:
        return ValidationReportRefreshPlan(
            status="manual_reconstruction_required",
            commands=preserved_commands,
            rationale="Only abbreviated or partial command text is present; rerun requires manual producer reconstruction.",
        )

    return ValidationReportRefreshPlan(
        status="manual_reconstruction_required",
        commands=(),
        rationale="No exact command metadata is preserved in the lifecycle registry.",
    )


def _contains_claim_boundary(value: object) -> bool:
    """Find an embedded claim/boundary key recursively in decoded JSON.

    Parameters
    ----------
    value : object
        Decoded source payload or one nested object/array/scalar.

    Returns
    -------
    bool
        Whether any object key contains ``claim`` or ``boundary`` ignoring case.
        This is a marker-presence check, not validation of its meaning.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            if "claim" in key.lower() or "boundary" in key.lower():
                return True
            if _contains_claim_boundary(item):
                return True
    elif isinstance(value, list):
        return any(_contains_claim_boundary(item) for item in value)
    return False


def main(argv: list[str] | None = None) -> int:
    """Generate the advisory inventory through supported CLI options.

    Parameters
    ----------
    argv : list of str or None, optional
        Options for report/registry roots, time, window, stdout format, file
        outputs and stale failure. Null uses process arguments.

    Returns
    -------
    int
        Zero on success; one for input/output refusal or requested stale failure.
        Stale failure is evaluated after otherwise successful file publication.

    Raises
    ------
    SystemExit
        Argument parser rejects syntax or displays help.

    Notes
    -----
    Authored refusal types may expose their messages. Other caught input/output
    exceptions use fixed caller-safe text. JSON stdout takes precedence when
    both stdout-format flags are present. Files use paired publication custody.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-root",
        default=str(ROOT / "validation" / "reports"),
        help="Directory containing validation report JSON artifacts",
    )
    parser.add_argument(
        "--registry",
        default=str(DEFAULT_LIFECYCLE_REGISTRY),
        help="Digest-bound lifecycle registry for the report corpus",
    )
    parser.add_argument("--as-of", help="UTC timestamp for deterministic freshness checks; defaults to now")
    parser.add_argument("--max-age-days", type=int, default=21, help="Freshness window in days")
    parser.add_argument("--json-out", action="store_true", help="Emit JSON to stdout")
    parser.add_argument("--markdown-out", action="store_true", help="Emit Markdown to stdout")
    parser.add_argument("--output-json", help="Write JSON to this path")
    parser.add_argument("--output-md", help="Write Markdown to this path")
    parser.add_argument("--fail-on-stale", action="store_true", help="Return non-zero when stale reports exist")
    args = parser.parse_args(argv)

    try:
        as_of = parse_datetime(args.as_of) if args.as_of is not None else datetime.now(tz=UTC)
        matrix = build_validation_report_freshness_matrix(
            Path(args.reports_root),
            as_of=as_of,
            max_age_days=args.max_age_days,
            registry_path=Path(args.registry),
        )
    except LifecycleRegistryError as exc:
        print(f"Validation report freshness failed: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError):
        print("Validation report freshness inputs could not be inspected", file=sys.stderr)
        return 1

    try:
        write_inventory_outputs(
            matrix,
            json_path=Path(args.output_json) if args.output_json else None,
            markdown_path=Path(args.output_md) if args.output_md else None,
            registry_path=Path(args.registry),
        )
    except InventoryOutputError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError):
        print("Validation report freshness outputs could not be published", file=sys.stderr)
        return 1

    if args.json_out:
        print(json.dumps(matrix.to_dict(), indent=2, sort_keys=True))
    elif args.markdown_out:
        print(matrix.to_markdown(), end="")
    else:
        print(
            "Validation report freshness: "
            f"reports={len(matrix.reports)} "
            f"stale={len(matrix.stale_reports)} "
            f"max_age_days={matrix.max_age_days} "
            f"claim_boundary_missing={matrix.claim_boundary_missing} "
            f"source_claim_boundary_missing={matrix.source_claim_boundary_missing}"
        )

    if args.fail_on_stale and matrix.stale_reports:
        print(f"Stale validation reports detected: {len(matrix.stale_reports)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
