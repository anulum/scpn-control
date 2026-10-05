# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — ELM, EPED, MARFE, NTM and disruption declaration checks

"""ELM, EPED, MARFE, NTM and disruption declaration checks.

Registered by the reference facade; preserve actual Click options, defaults and
report/refusal semantics. Admission limits are defined by each source validator.
"""

from __future__ import annotations

import json

import click


@click.command("validate-elm-reference")
@click.option("--artifact-root", help="Directory or JSON artifact containing persisted ELM reference evidence")
@click.option("--require-reference-artifacts", is_flag=True, help="Fail if no reference artifacts are present")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_elm_reference_command(
    artifact_root: str | None,
    require_reference_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Inspect ELM declarations and write an input-protected report with authored operational refusal."""
    from validation.validate_elm_reference import (
        ROOT,
        validate_elm_reference,
        write_elm_reference_report,
    )

    path = artifact_root or str(ROOT / "validation" / "reports" / "elm_reference")
    try:
        report = validate_elm_reference(path, require_reference_artifacts=require_reference_artifacts)
        if output_json is not None:
            write_elm_reference_report(report, output_json, artifact_root=path)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        click.echo("ELM reference FAILED: could not inspect artifacts or write report", err=True)
        raise click.exceptions.Exit(1) from None
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"ELM reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-eped-reference")
@click.option("--artifact-root", help="Directory or JSON artifact containing persisted EPED reference evidence")
@click.option("--require-reference-artifacts", is_flag=True, help="Fail if no reference artifacts are present")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_eped_reference_command(
    artifact_root: str | None,
    require_reference_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Inspect EPED declarations and write an input-protected report with authored operational refusal."""
    from validation.validate_eped_reference import ROOT, validate_eped_reference, write_eped_reference_report

    path = artifact_root or str(ROOT / "validation" / "reports" / "eped_reference")
    try:
        report = validate_eped_reference(path, require_reference_artifacts=require_reference_artifacts)
        if output_json is not None:
            write_eped_reference_report(report, output_json, artifact_root=path)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        click.echo("EPED reference FAILED: could not inspect artifacts or write report", err=True)
        raise click.exceptions.Exit(1) from None
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"EPED reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-marfe-reference")
@click.option("--artifact-root", help="Directory or JSON artifact containing persisted MARFE reference evidence")
@click.option("--require-reference-artifacts", is_flag=True, help="Fail if no reference artifacts are present")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_marfe_reference_command(
    artifact_root: str | None,
    require_reference_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Inspect MARFE declarations and write an input-protected report with authored operational refusal."""
    from validation.validate_marfe_reference import ROOT, validate_marfe_reference, write_marfe_reference_report

    path = artifact_root or str(ROOT / "validation" / "reports" / "marfe_reference")
    try:
        report = validate_marfe_reference(path, require_reference_artifacts=require_reference_artifacts)
        if output_json is not None:
            write_marfe_reference_report(report, output_json, artifact_root=path)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        click.echo("MARFE reference FAILED: could not inspect artifacts or write report", err=True)
        raise click.exceptions.Exit(1) from None
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"MARFE reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-ntm-reference")
@click.option("--artifact-root", help="Directory or JSON artifact containing persisted NTM reference evidence")
@click.option("--require-reference-artifacts", is_flag=True, help="Fail if no reference artifacts are present")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_ntm_reference_command(
    artifact_root: str | None,
    require_reference_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Inspect NTM declarations and write an input-protected report with authored operational refusal."""
    from validation.validate_ntm_reference import ROOT, validate_ntm_reference, write_ntm_reference_report

    path = artifact_root or str(ROOT / "validation" / "reports" / "ntm_reference")
    try:
        report = validate_ntm_reference(path, require_reference_artifacts=require_reference_artifacts)
        if output_json is not None:
            write_ntm_reference_report(report, output_json, artifact_root=path)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        click.echo("NTM reference FAILED: could not inspect artifacts or write report", err=True)
        raise click.exceptions.Exit(1) from None
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"NTM reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-disruption-reference")
@click.option("--artifact-root", help="Directory or JSON artifact containing persisted disruption reference evidence")
@click.option("--require-reference-artifacts", is_flag=True, help="Fail if no reference artifacts are present")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_disruption_reference_command(
    artifact_root: str | None,
    require_reference_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Inspect Disruption declarations and write an input-protected report with authored operational refusal."""
    from validation.validate_disruption_reference import (
        ROOT,
        validate_disruption_reference,
        write_disruption_reference_report,
    )

    path = artifact_root or str(ROOT / "validation" / "reports" / "disruption_reference")
    try:
        report = validate_disruption_reference(path, require_reference_artifacts=require_reference_artifacts)
        if output_json is not None:
            write_disruption_reference_report(report, output_json, artifact_root=path)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        click.echo("Disruption reference FAILED: could not inspect artifacts or write report", err=True)
        raise click.exceptions.Exit(1) from None
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"Disruption reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)
