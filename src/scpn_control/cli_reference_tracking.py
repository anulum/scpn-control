# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — RZIP, free-boundary and digital-twin declaration checks

"""RZIP, free-boundary and digital-twin declaration checks.

Registered by the reference facade; preserve actual Click options, defaults and
report/refusal semantics. Admission limits are defined by each source validator.
"""

from __future__ import annotations

import json

import click


@click.command("validate-rzip-reference")
@click.option("--artifact-root", help="Directory or JSON artifact containing persisted RZIP reference evidence")
@click.option("--require-reference-artifacts", is_flag=True, help="Fail if no reference artifacts are present")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_rzip_reference_command(
    artifact_root: str | None,
    require_reference_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Inspect RZIP declarations and write an input-protected report with authored operational refusal."""
    from validation.validate_rzip_reference import ROOT, validate_rzip_reference, write_rzip_reference_report

    path = artifact_root or str(ROOT / "validation" / "reports" / "rzip_reference")
    try:
        report = validate_rzip_reference(path, require_reference_artifacts=require_reference_artifacts)
        if output_json is not None:
            write_rzip_reference_report(report, output_json, artifact_root=path)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        click.echo("RZIP reference FAILED: could not inspect artifacts or write report", err=True)
        raise click.exceptions.Exit(1) from None
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"RZIP reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-free-boundary-reference")
@click.option(
    "--artifact-root", help="Directory or JSON artifact containing persisted free-boundary reference evidence"
)
@click.option("--require-reference-artifacts", is_flag=True, help="Fail if no reference artifacts are present")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_free_boundary_reference_command(
    artifact_root: str | None,
    require_reference_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Inspect Free-boundary declarations and write an input-protected report with authored operational refusal."""
    from validation.validate_free_boundary_reference import (
        ROOT,
        validate_free_boundary_reference,
        write_free_boundary_reference_report,
    )

    path = artifact_root or str(ROOT / "validation" / "reports" / "free_boundary_reference")
    try:
        report = validate_free_boundary_reference(path, require_reference_artifacts=require_reference_artifacts)
        if output_json is not None:
            write_free_boundary_reference_report(report, output_json, artifact_root=path)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        click.echo("Free-boundary reference FAILED: could not inspect artifacts or write report", err=True)
        raise click.exceptions.Exit(1) from None
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"Free-boundary reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-digital-twin-reference")
@click.option("--artifact-root", help="Directory or JSON artifact containing persisted digital twin reference evidence")
@click.option("--require-reference-artifacts", is_flag=True, help="Fail if no reference artifacts are present")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_digital_twin_reference_command(
    artifact_root: str | None,
    require_reference_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Inspect Digital twin declarations and write an input-protected report with authored operational refusal."""
    from validation.validate_digital_twin_reference import (
        ROOT,
        validate_digital_twin_reference,
        write_digital_twin_reference_report,
    )

    path = artifact_root or str(ROOT / "validation" / "reports" / "digital_twin_reference")
    try:
        report = validate_digital_twin_reference(path, require_reference_artifacts=require_reference_artifacts)
        if output_json is not None:
            write_digital_twin_reference_report(report, output_json, artifact_root=path)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        click.echo("Digital twin reference FAILED: could not inspect artifacts or write report", err=True)
        raise click.exceptions.Exit(1) from None
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"Digital twin reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)
