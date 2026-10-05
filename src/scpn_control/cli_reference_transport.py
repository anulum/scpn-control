# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Blob, neural transport/turbulence and SOC declaration checks

"""Blob, neural transport/turbulence and SOC declaration checks.

Registered by the reference facade; preserve actual Click options, defaults and
report/refusal semantics. Admission limits are defined by each source validator.
"""

from __future__ import annotations

import json

import click


@click.command("validate-blob-transport-reference")
@click.option(
    "--artifact-root", help="Directory or JSON artifact containing persisted blob transport reference evidence"
)
@click.option("--require-reference-artifacts", is_flag=True, help="Fail if no reference artifacts are present")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_blob_transport_reference_command(
    artifact_root: str | None,
    require_reference_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Inspect Blob transport declarations and write an input-protected report with authored operational refusal."""
    from validation.validate_blob_transport_reference import (
        ROOT,
        validate_blob_transport_reference,
        write_blob_transport_reference_report,
    )

    path = artifact_root or str(ROOT / "validation" / "reports" / "blob_transport_reference")
    try:
        report = validate_blob_transport_reference(path, require_reference_artifacts=require_reference_artifacts)
        if output_json is not None:
            write_blob_transport_reference_report(report, output_json, artifact_root=path)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        click.echo("Blob transport reference FAILED: could not inspect artifacts or write report", err=True)
        raise click.exceptions.Exit(1) from None
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"Blob transport reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-neural-transport-reference")
@click.option(
    "--artifact-root", help="Directory or JSON artifact containing persisted neural transport reference evidence"
)
@click.option("--require-reference-artifacts", is_flag=True, help="Fail if no reference artifacts are present")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_neural_transport_reference_command(
    artifact_root: str | None,
    require_reference_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Inspect Neural transport declarations and write an input-protected report with authored operational refusal."""
    from validation.validate_neural_transport_reference import (
        ROOT,
        validate_neural_transport_reference,
        write_neural_transport_reference_report,
    )

    path = artifact_root or str(ROOT / "validation" / "reports" / "neural_transport_reference")
    try:
        report = validate_neural_transport_reference(path, require_reference_artifacts=require_reference_artifacts)
        if output_json is not None:
            write_neural_transport_reference_report(report, output_json, artifact_root=path)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        click.echo("Neural transport reference FAILED: could not inspect artifacts or write report", err=True)
        raise click.exceptions.Exit(1) from None
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(
            f"Neural transport reference: {report['status']} reference_artifacts={report['reference_artifacts']}"
        )
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-neural-turbulence-reference")
@click.option(
    "--artifact-root", help="Directory or JSON artifact containing persisted neural turbulence reference evidence"
)
@click.option("--require-reference-artifacts", is_flag=True, help="Fail if no reference artifacts are present")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_neural_turbulence_reference_command(
    artifact_root: str | None,
    require_reference_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Inspect Neural turbulence declarations and write an input-protected report with authored operational refusal."""
    from validation.validate_neural_turbulence_reference import (
        ROOT,
        validate_neural_turbulence_reference,
        write_neural_turbulence_reference_report,
    )

    path = artifact_root or str(ROOT / "validation" / "reports" / "neural_turbulence_reference")
    try:
        report = validate_neural_turbulence_reference(path, require_reference_artifacts=require_reference_artifacts)
        if output_json is not None:
            write_neural_turbulence_reference_report(report, output_json, artifact_root=path)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        click.echo("Neural turbulence reference FAILED: could not inspect artifacts or write report", err=True)
        raise click.exceptions.Exit(1) from None
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(
            f"Neural turbulence reference: {report['status']} reference_artifacts={report['reference_artifacts']}"
        )
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-soc-reference")
@click.option("--artifact-root", help="Directory or JSON artifact containing persisted SOC reference evidence")
@click.option("--require-reference-artifacts", is_flag=True, help="Fail if no reference artifacts are present")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_soc_reference_command(
    artifact_root: str | None,
    require_reference_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Inspect SOC declarations and write an input-protected report with authored operational refusal."""
    from validation.validate_soc_reference import (
        ROOT,
        validate_soc_reference,
        write_soc_reference_report,
    )

    path = artifact_root or str(ROOT / "validation" / "reports" / "soc_reference")
    try:
        report = validate_soc_reference(path, require_reference_artifacts=require_reference_artifacts)
        if output_json is not None:
            write_soc_reference_report(report, output_json, artifact_root=path)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        click.echo("SOC reference FAILED: could not inspect artifacts or write report", err=True)
        raise click.exceptions.Exit(1) from None
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"SOC reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)
