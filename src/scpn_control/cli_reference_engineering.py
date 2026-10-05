# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Current-drive, density, burn and volt-second declaration checks

"""Current-drive, density, burn and volt-second declaration checks.

Registered by the reference facade; preserve actual Click options, defaults and
report/refusal semantics. Admission limits are defined by each source validator.
"""

from __future__ import annotations

import json

import click

from scpn_control.cli_reference_paths import _ReportOutputPath


@click.command(
    "validate-current-drive-reference",
    help="Inspect local reference declarations without fetching or recomputing evidence.\n\n--artifact-root uses caller-relative paths or the validator's default root.\nOptional empty inspection passes; --require-reference-artifacts refuses it.\n--json-out emits the same report as the source API. --output-json creates\nparents and directly replaces its destination before stdout emission.\nInspection failures exit 1 and retain partial entries as diagnostics only;\noutput filesystem failures raise ClickException with fixed text and exit 1.\nClick Path validation rejects an existing directory destination before this\ncallback, retaining its authored usage error and exit 2.\nA pass verifies schema and declared tolerances, not a reference digest,\nprovenance, measured comparison or external/facility claim.",
)
@click.option(
    "--artifact-root", help="Directory or JSON artifact containing persisted current-drive reference evidence"
)
@click.option("--require-reference-artifacts", is_flag=True, help="Fail if no reference artifacts are present")
@click.option("--output-json", type=_ReportOutputPath(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_current_drive_reference_command(
    artifact_root: str | None,
    require_reference_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Inspect local reference declarations without fetching or recomputing evidence.

    --artifact-root uses caller-relative paths or the validator's default root.
    Optional empty inspection passes; --require-reference-artifacts refuses it.
    --json-out emits the same report as the source API. --output-json creates
    parents and replaces unrelated output while protecting selected inputs before stdout emission.
    Inspection failures exit 1 and retain partial entries as diagnostics only;
    output filesystem failures raise ClickException with fixed text and exit 1.
    Click Path validation rejects an existing directory destination before this
    callback, retaining its authored usage error and exit 2.
    A pass verifies schema and declared tolerances, not a reference digest,
    provenance, measured comparison or external/facility claim.
    """
    from validation.validate_current_drive_reference import (
        ROOT,
        validate_current_drive_reference,
        write_current_drive_reference_report,
    )

    path = artifact_root or str(ROOT / "validation" / "reports" / "current_drive_reference")
    report = validate_current_drive_reference(path, require_reference_artifacts=require_reference_artifacts)
    if output_json is not None:
        try:
            write_current_drive_reference_report(report, output_json, artifact_root=path)
        except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
            raise click.ClickException("could not write current-drive reference report") from exc
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"Current-drive reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-density-reference", help="Validate persisted density-control reference artifacts.")
@click.option("--artifact-root", help="Directory or JSON artifact containing persisted density reference evidence")
@click.option("--require-reference-artifacts", is_flag=True, help="Fail if no reference artifacts are present")
@click.option("--output-json", type=_ReportOutputPath(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_density_reference_command(
    artifact_root: str | None,
    require_reference_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Inspect density declarations and protect report output without authenticating reference/model/facility evidence."""
    from validation.validate_density_reference import ROOT, validate_density_reference, write_density_reference_report

    path = artifact_root or str(ROOT / "validation" / "reports" / "density_reference")
    try:
        report = validate_density_reference(path, require_reference_artifacts=require_reference_artifacts)
        if output_json is not None:
            write_density_reference_report(report, output_json, artifact_root=path)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        click.echo("Density reference FAILED: could not inspect artifacts or write report", err=True)
        raise click.exceptions.Exit(1) from None
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"Density reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-volt-second-reference")
@click.option("--artifact-root", help="Directory or JSON artifact containing persisted volt-second reference evidence")
@click.option("--require-reference-artifacts", is_flag=True, help="Fail if no reference artifacts are present")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_volt_second_reference_command(
    artifact_root: str | None,
    require_reference_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Inspect Volt-second declarations and write an input-protected report with authored operational refusal."""
    from validation.validate_volt_second_reference import (
        ROOT,
        validate_volt_second_reference,
        write_volt_second_reference_report,
    )

    path = artifact_root or str(ROOT / "validation" / "reports" / "volt_second_reference")
    try:
        report = validate_volt_second_reference(path, require_reference_artifacts=require_reference_artifacts)
        if output_json is not None:
            write_volt_second_reference_report(report, output_json, artifact_root=path)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        click.echo("Volt-second reference FAILED: could not inspect artifacts or write report", err=True)
        raise click.exceptions.Exit(1) from None
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"Volt-second reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-burn-reference")
@click.option("--artifact-root", help="Directory or JSON artifact containing persisted burn-control reference evidence")
@click.option("--require-reference-artifacts", is_flag=True, help="Fail if no reference artifacts are present")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_burn_reference_command(
    artifact_root: str | None,
    require_reference_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Inspect Burn declarations and write an input-protected report with authored operational refusal."""
    from validation.validate_burn_reference import (
        ROOT,
        validate_burn_reference,
        write_burn_reference_report,
    )

    path = artifact_root or str(ROOT / "validation" / "reports" / "burn_reference")
    try:
        report = validate_burn_reference(path, require_reference_artifacts=require_reference_artifacts)
        if output_json is not None:
            write_burn_reference_report(report, output_json, artifact_root=path)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        click.echo("Burn reference FAILED: could not inspect artifacts or write report", err=True)
        raise click.exceptions.Exit(1) from None
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"Burn reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)
