# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Static zero-frequency mu declarations and deprecated command compatibility

"""Static zero-frequency mu declarations and deprecated command compatibility.

Registered by the reference facade; preserve actual Click options, defaults and
report/refusal semantics. Admission limits are defined by each source validator.
"""

from __future__ import annotations

import json

import click


def _run_static_mu_analysis_reference(
    artifact_root: str | None,
    require_reference_artifacts: bool,
    output_json: str | None,
    json_out: bool,
    *,
    default_directory: str,
) -> None:
    """Inspect Static mu-analysis declarations and write an input-protected report with authored operational refusal."""
    from validation.validate_static_mu_analysis_reference import (
        ROOT,
        validate_static_mu_analysis_reference,
        write_static_mu_analysis_reference_report,
    )

    path = artifact_root or str(ROOT / "validation" / "reports" / default_directory)
    try:
        report = validate_static_mu_analysis_reference(path, require_reference_artifacts=require_reference_artifacts)
        if output_json is not None:
            write_static_mu_analysis_reference_report(report, output_json, artifact_root=path)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        click.echo("Static mu-analysis reference FAILED: could not inspect artifacts or write report", err=True)
        raise click.exceptions.Exit(1) from None
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(
            f"Static mu-analysis reference: {report['status']} reference_artifacts={report['reference_artifacts']}"
        )
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-static-mu-analysis-reference")
@click.option(
    "--artifact-root",
    help="Directory or JSON artifact containing persisted static mu-analysis reference evidence",
)
@click.option("--require-reference-artifacts", is_flag=True, help="Fail if no reference artifacts are present")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_static_mu_analysis_reference_command(
    artifact_root: str | None,
    require_reference_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Validate persisted static structured-mu reference artifacts."""
    _run_static_mu_analysis_reference(
        artifact_root,
        require_reference_artifacts,
        output_json,
        json_out,
        default_directory="static_mu_analysis_reference",
    )


@click.command("validate-mu-synthesis-reference", hidden=True)
@click.option("--artifact-root", help="Directory or JSON artifact containing persisted legacy reference evidence")
@click.option("--require-reference-artifacts", is_flag=True, help="Fail if no reference artifacts are present")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_mu_synthesis_reference_compatibility_command(
    artifact_root: str | None,
    require_reference_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Forward the deprecated validator command to static mu analysis."""
    click.echo(
        "DEPRECATED: validate-mu-synthesis-reference will be removed in 0.25.0; "
        "use validate-static-mu-analysis-reference.",
        err=True,
    )
    _run_static_mu_analysis_reference(
        artifact_root,
        require_reference_artifacts,
        output_json,
        json_out,
        default_directory="mu_synthesis_reference",
    )
