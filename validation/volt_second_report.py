# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Guarded volt-second report publication
"""Publish declared analytic reports without overwriting source namespaces."""

from __future__ import annotations

import json
from pathlib import Path

from tools.inventory_file_output import publish_guarded_outputs
from validation.volt_second_evidence import VoltSecondEvidence, validate_evidence_payload


def write_report(evidence: VoltSecondEvidence, json_path: Path) -> None:
    """Publish a checked JSON report and its Markdown summary with recovery.

    Parameters
    ----------
    evidence : VoltSecondEvidence
        Complete consistent v1 declaration; failed outcomes remain writable.
    json_path : pathlib.Path
        Destination whose sibling with .md suffix holds the summary. Outputs
        must be distinct regular files outside source/configuration namespaces.

    Raises
    ------
    ValueError
        Evidence or output identity is invalid.
    OSError
        Publication fails after successful handled-failure recovery.

    Notes
    -----
    Each file is atomically replaced. Handled failures restore unchanged prior
    outputs; this is not a crash transaction or a concurrent-writer sandbox.
    """
    validate_evidence_payload(evidence)
    json_bytes = (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode("utf-8")
    md_path = json_path.with_suffix(".md")
    decomposition = evidence["decomposition"]
    monitor = evidence["monitor"]
    lines = [
        "",
        "# Volt-Second Flux-Budget Validation",
        "",
        f"- Schema: `{evidence['schema_version']}`",
        f"- Generated (UTC): {evidence['generated_utc']}",
        f"- Target: `{evidence['target_id']}`",
        f"- Status: **{'pass' if evidence['passed'] else 'fail'}**",
        "",
        f"## Exact flux relations (relative error, gate < {evidence['exact_tol']:.1e})",
        "",
        "| relation | value |",
        "| --- | --- |",
        f"| inductive flux L_p I_p | {evidence['inductive_rel_error']:.3e} |",
        f"| Ejima startup flux C_E mu0 R0 I_p | {evidence['ejima_rel_error']:.3e} |",
        f"| resistive ramp integral | {evidence['resistive_ramp_rel_error']:.3e} |",
        f"| flat-top budget closure | {evidence['flat_top_closure_rel_error']:.3e} |",
        f"| scenario decomposition (max) | {decomposition['max_rel_error']:.3e} |",
        f"| consumption integrator (max) | {monitor['max_rel_error']:.3e} |",
        f"| flux scaling laws (max) | {evidence['max_scaling_rel_error']:.3e} |",
        "",
        "## Ramp optimiser",
        "",
        f"- linear ramp: {evidence['ramp_optimizer']['is_linear']}; "
        f"endpoint relative error: {evidence['ramp_optimizer']['end_rel_error']:.3e}",
        "",
        "## Budget margin",
        "",
        f"- margin closed-form absolute error: {decomposition['margin_abs_error']:.3e} V s "
        f"(gate < {evidence['margin_abs_tol']:.1e})",
    ]
    root = Path(__file__).resolve().parents[1]
    validation = root / "validation"
    protected = tuple(path for path in (*root.iterdir(), *validation.iterdir()) if path.is_file())
    roots = tuple(
        root / name for name in ("src", "tools", "tests", "docs", "papers", "weights", "scpn-control-rs", ".git")
    )
    roots += tuple(path for path in validation.iterdir() if path.is_dir() and path.name != "reports")
    publish_guarded_outputs(
        ((json_path, json_bytes), (md_path, ("\n".join(lines) + "\n").encode("utf-8"))),
        protected_files=protected,
        protected_roots=roots,
    )
