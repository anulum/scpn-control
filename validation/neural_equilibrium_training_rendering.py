# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural-equilibrium trainer
"""Persist validated launch and template JSON/Markdown with explicit output custody.

Launch writers verify execute-mode local weight bytes. Both writers validate
finite declarations and render before persistence. Pair writes are sequential;
a second-file failure may leave the first file and does not imply success.

Output custody is checked before writes, including canonical and hardlink aliases:

>>> from tempfile import TemporaryDirectory
>>> with TemporaryDirectory() as directory:
...     selected = Path(directory) / "report.json"
...     ensure_distinct_outputs([selected, selected])
Traceback (most recent call last):
...
ValueError: training outputs must be distinct and must not overwrite protected inputs/weights
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from validation.neural_equilibrium_dataset_contracts import ensure_distinct_outputs as ensure_distinct_outputs
from validation.neural_equilibrium_training_evidence import validate_result_templates, validate_training_report


def _render_launch(report: dict[str, Any]) -> str:
    """Render launch sections only; public writer translates missing/invalid render fields."""
    lines = [
        "# MAST EFM Neural-Equilibrium Training Launch",
        "",
        f"Schema: `{report['schema_version']}`",
        f"Status: `{report['status']}`",
        f"Execution mode: `{report['execution_mode']}`",
        f"Dataset path: `{report['dataset_path']}`",
        f"Dataset SHA-256: `{report['dataset_sha256']}`",
        f"Dataset exists on this host: `{report['dataset_exists_on_this_host']}`",
        f"Weights path: `{report['weights_path']}`",
        "",
        "## Execution host policy",
        "",
        report["execution_host_policy"],
        "",
        "## Claim boundary",
        "",
        report["claim_boundary"],
        "",
        "## Pre-run admission",
        "",
        f"Status: `{report['pre_run_admission']['status']}`",
        f"Dataset SHA-256 verified: `{report['pre_run_admission']['dataset_sha256_verified']}`",
        f"Source provenance: `{report['pre_run_admission']['source_provenance']['status']}`",
        f"Compute execution: `{report['pre_run_admission']['compute_execution']['status']}`",
        "",
        "## Run command",
        "",
        "```bash",
        report["run_command"],
        "```",
        "",
        "## Required targets",
        "",
    ]
    lines.extend(f"- `{key}`" for key in report["required_targets"])
    lines.extend(["", "## Admission blockers", ""])
    lines.extend(f"- {item}" for item in report["blocked_before_admission"])
    if report["holdout_metrics"] is not None:
        lines.extend(["", "## Holdout metrics", ""])
        lines.append("```json")
        lines.append(json.dumps(report["holdout_metrics"], indent=2, sort_keys=True))
        lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _render_templates(templates: dict[str, Any]) -> str:
    """Render the declared template schemas; no future output is simulated."""
    lines = [
        "# MAST EFM Neural-Equilibrium Result Templates",
        "",
        f"Schema: `{templates['schema_version']}`",
        f"Expected dataset SHA-256: `{templates['expected_dataset_sha256']}`",
        "",
        templates["claim_boundary"],
        "",
        "## Output policy",
        "",
        templates["expected_weight_path_policy"],
        "",
        "## Template schemas",
        "",
    ]
    for key in ("holdout_metrics", "latency_metrics", "gpu_cost", "admission_certificate"):
        template = templates[key]
        lines.extend(
            [
                f"### `{key}`",
                "",
                f"- Schema: `{template['schema_version']}`",
                f"- Acceptance policy: {template['acceptance_policy']}",
                "",
            ]
        )
    return "\n".join(lines)


def _write_pair(payload: dict[str, Any], markdown: str, json_out: Path, markdown_out: Path) -> None:
    """Persist already rendered finite JSON and Markdown sequentially; failures do not imply rollback."""
    ensure_distinct_outputs([json_out, markdown_out])
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(encoded, encoding="utf-8")
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text(markdown, encoding="utf-8")


def write_report(report: dict[str, Any], json_out: Path, markdown_out: Path) -> None:
    """Validate/render a launch before writing distinct JSON/Markdown output paths.

    Execute-mode launch reports additionally verify their selected weight bytes.
    Supported schema/render/IO failures become ValueError. Writes are sequential,
    so a second-write failure may leave JSON; no complete-report success is implied.
    """
    try:
        validate_training_report(report)
        if report["execution_mode"] == "execute":
            validate_training_report(report, require_executed=True)
        _write_pair(report, _render_launch(report), json_out, markdown_out)
    except (OSError, ValueError, TypeError, KeyError, IndexError, RecursionError, RuntimeError) as exc:
        raise ValueError(f"cannot write training launch report: {exc}") from exc


def write_result_templates(templates: dict[str, Any], json_out: Path, markdown_out: Path) -> None:
    """Validate/render future schemas before sequential JSON/Markdown persistence.

    This does not emit measured holdout/latency/cost/admission results. Supported
    schema/render/IO failures become ValueError; second-write partial output is
    possible and must not be counted as a complete pair.
    """
    try:
        validate_result_templates(templates)
        _write_pair(templates, _render_templates(templates), json_out, markdown_out)
    except (OSError, ValueError, TypeError, KeyError, IndexError, RecursionError, RuntimeError) as exc:
        raise ValueError(f"cannot write training result templates: {exc}") from exc
