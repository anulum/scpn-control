# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Validation inventory output custody.
"""Derive freshness input custody for the shared guarded file publisher."""

from __future__ import annotations

import json
from pathlib import Path

from tools.inventory_file_output import InventoryOutputError as InventoryOutputError
from tools.inventory_file_output import publish_guarded_outputs
from tools.report_inventory_types import ROOT, ValidationReportFreshnessMatrix
from tools.report_lifecycle_registry import _repository_root_for_reports


def write_inventory_outputs(
    matrix: ValidationReportFreshnessMatrix,
    *,
    json_path: Path | None,
    markdown_path: Path | None,
    registry_path: Path,
) -> None:
    """Publish requested JSON/Markdown files while preserving input custody.

    Parameters
    ----------
    matrix : ValidationReportFreshnessMatrix
        Already validated inventory. Publication does not rerun producers or
        grant scientific, execution or source authenticity admission.
    json_path, markdown_path : pathlib.Path or None
        Optional caller-relative destinations. Existing regular outputs may be
        replaced; symlinks, input aliases and shared destinations are refused.
    registry_path : pathlib.Path
        Registry consumed by the inventory, protected from replacement.

    Raises
    ------
    InventoryOutputError
        Unsafe destination or incomplete recovery. Retained recovery paths are
        attached to the exception; no exception from the operating system is
        interpolated into its authored message.
    OSError, ValueError, TypeError
        Staging, serialisation or publication fails after successful recovery.

    Notes
    -----
    Report and refresh namespaces remain read-only. All requested files are
    staged before replacement; handled failures restore predecessors or remove
    newly published outputs. Recovery refuses to overwrite changed outputs.
    Atomic replacement is per file, not a power-loss transaction across files.
    Cooperating callers must coordinate concurrent publications themselves.
    """
    outputs: list[tuple[Path, bytes]] = []
    if json_path is not None:
        outputs.append(
            (json_path.absolute(), (json.dumps(matrix.to_dict(), indent=2, sort_keys=True) + "\n").encode("utf-8"))
        )
    if markdown_path is not None:
        outputs.append((markdown_path.absolute(), matrix.to_markdown().encode("utf-8")))
    publish_inventory_outputs(
        matrix,
        tuple(outputs),
        registry_path=registry_path,
        additional_protected_files=(
            ROOT / "validation/public_claim_ledger.json",
            _repository_root_for_reports(matrix.reports_root) / "validation/public_claim_ledger.json",
        ),
    )


def publish_inventory_outputs(
    matrix: ValidationReportFreshnessMatrix,
    outputs: tuple[tuple[Path, bytes], ...],
    *,
    registry_path: Path,
    additional_protected_files: tuple[Path, ...] = (),
) -> None:
    """Publish caller-serialised inventories with the consumed matrix's custody.

    Parameters
    ----------
    matrix : ValidationReportFreshnessMatrix
        Validated source inventory defining report and refresh input bindings.
    outputs : tuple of (pathlib.Path, bytes)
        Complete payloads and distinct caller-relative destinations. The caller
        owns schema and claim selection; publication grants no admission.
    registry_path : pathlib.Path
        Consumed registry, protected along with reports and refresh artifacts.
    additional_protected_files : tuple of pathlib.Path, optional
        Consumer-selected artifacts that this publication must not replace,
        including reserved paths whose files do not yet exist.

    Raises
    ------
    InventoryOutputError
        Unsafe destination, alias or incomplete handled-failure recovery.
    OSError, ValueError, TypeError
        Filesystem inspection, staging or publication fails after recovery.

    Notes
    -----
    Shares input custody and transaction logic across freshness and public-claim
    schema consumers without changing either consumer's serialisation.
    """
    outputs = tuple((path.absolute(), payload) for path, payload in outputs)
    repository_root = _repository_root_for_reports(matrix.reports_root)
    protected_files = (
        registry_path,
        *additional_protected_files,
        *(report.path for report in matrix.reports),
        *(
            repository_root / report.lifecycle.refresh_artifact_path
            for report in matrix.reports
            if report.lifecycle.refresh_artifact_path is not None
        ),
    )
    protected_roots = (matrix.reports_root, repository_root / "validation/report_refreshes")
    publish_guarded_outputs(outputs, protected_files=protected_files, protected_roots=protected_roots)
