# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Evidence gap matrix generator
"""Generate diagnostic work packages from declared physics traceability metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evidence_gap_models import _STATUS_SEVERITY
from tools.evidence_gap_models import ROOT as ROOT
from tools.evidence_gap_models import EvidenceGapMatrix as EvidenceGapMatrix
from tools.evidence_gap_models import EvidenceWorkPackage as EvidenceWorkPackage
from tools.evidence_gap_models import ExternalValidationTracker as ExternalValidationTracker
from tools.evidence_gap_models import TraceabilityEntry as TraceabilityEntry
from tools.evidence_gap_registry import EvidenceGapRegistryError, load_gap_registry
from tools.inventory_file_output import InventoryOutputError, publish_guarded_outputs


def build_evidence_gap_matrix(registry: Path) -> EvidenceGapMatrix:
    """Build deterministic work packages from declared planning metadata.

    Parameters
    ----------
    registry : pathlib.Path
        UTF-8 registry containing entries and external tracker declarations.

    Returns
    -------
    EvidenceGapMatrix
        Complete declared inventory, including unresolved tracker diagnostics.

    Raises
    ------
    EvidenceGapRegistryError
        Consumed metadata is malformed, ambiguous, or outside the vocabulary.
    OSError, UnicodeError, json.JSONDecodeError
        Input cannot be read or decoded.

    Notes
    -----
    No producer, source-marker check, external request or physical admission
    runs here. Full traceability qualification remains a separate validator.
    """
    entries, trackers = load_gap_registry(registry)
    return EvidenceGapMatrix(registry, entries, trackers, tuple(_build_work_packages(entries, trackers)))


def _build_work_packages(
    entries: tuple[TraceabilityEntry, ...],
    trackers: tuple[ExternalValidationTracker, ...],
) -> list[EvidenceWorkPackage]:
    tracker_by_issue = {tracker.issue: tracker for tracker in trackers}
    grouped: dict[int, list[TraceabilityEntry]] = {tracker.issue: [] for tracker in trackers}
    for entry in entries:
        issue = entry.external_validation_tracker_issue
        if issue is not None and issue in tracker_by_issue:
            grouped[issue].append(entry)

    packages = [
        EvidenceWorkPackage(tracker=tracker_by_issue[issue], entries=tuple(sorted(items, key=_entry_sort_key)))
        for issue, items in grouped.items()
        if items
    ]
    return sorted(
        packages,
        key=lambda package: (
            -package.severity,
            -package.blocked_public_claims,
            -package.open_fidelity_gaps,
            package.tracker.issue,
        ),
    )


def _entry_sort_key(entry: TraceabilityEntry) -> tuple[int, str, str]:
    return (-_STATUS_SEVERITY.get(entry.fidelity_status, 0), entry.component, entry.module_path)


def main(argv: list[str] | None = None) -> int:
    """Render the planning matrix and publish requested files with input custody.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments, or process arguments when omitted.

    Returns
    -------
    int
        Zero after complete publication/rendering; one after metadata refusal
        or a caught native input/output failure.

    Raises
    ------
    SystemExit
        Argument parsing requested help or refused invalid options.

    Notes
    -----
    JSON stdout takes precedence over Markdown stdout. Both distinct requested
    files are published. Protected registries and declaration artifacts remain
    read-only; paired publication has handled-failure recovery, not crash
    atomicity. Cooperating callers must coordinate concurrent writers.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default=str(ROOT / "validation/physics_traceability.json"),
        help="Physics traceability registry JSON path",
    )
    parser.add_argument("--json-out", action="store_true", help="Emit the matrix JSON to stdout")
    parser.add_argument("--markdown-out", action="store_true", help="Emit the matrix Markdown to stdout")
    parser.add_argument("--output-json", help="Write matrix JSON to this path")
    parser.add_argument("--output-md", help="Write matrix Markdown to this path")
    args = parser.parse_args(argv)
    try:
        registry = Path(args.registry)
        matrix = build_evidence_gap_matrix(registry)
        output_json = (json.dumps(matrix.to_dict(), indent=2, sort_keys=True) + "\n").encode("utf-8")
        output_md = matrix.to_markdown().encode("utf-8")
        outputs: list[tuple[Path, bytes]] = []
        if args.output_json:
            outputs.append((Path(args.output_json), output_json))
        if args.output_md:
            outputs.append((Path(args.output_md), output_md))
        selected_root = registry.absolute().parent.parent
        publish_guarded_outputs(
            tuple(outputs),
            protected_files=(
                registry,
                ROOT / "validation/physics_traceability.json",
                ROOT / "validation/report_lifecycle_registry.json",
                ROOT / "validation/public_claim_ledger.json",
                selected_root / "validation/report_lifecycle_registry.json",
                selected_root / "validation/public_claim_ledger.json",
            ),
            protected_roots=(
                ROOT / "validation/reports",
                ROOT / "validation/report_refreshes",
                selected_root / "validation/reports",
                selected_root / "validation/report_refreshes",
            ),
        )
        if args.json_out:
            print(output_json.decode("utf-8"), end="")
        elif args.markdown_out:
            print(output_md.decode("utf-8"), end="")
        else:
            print(
                "Evidence gap matrix: "
                f"entries={len(matrix.entries)} "
                f"open_fidelity_gaps={matrix.open_fidelity_gaps} "
                f"public_claim_blocked={matrix.public_claim_blocked} "
                f"work_packages={len(matrix.work_packages)}"
            )
    except (EvidenceGapRegistryError, InventoryOutputError) as error:
        print(f"Evidence gap matrix failed: {error}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError):
        print("Evidence gap matrix failed: inputs or outputs could not be inspected", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
