# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural/equilibrium/trajectory/UQ declarations and IDA same-case admission

"""Neural/equilibrium/trajectory/UQ declarations and IDA same-case admission.

Registered by the reference facade; preserve actual Click options, defaults and
report/refusal semantics. Admission limits are defined by each source validator.
"""

from __future__ import annotations

import json

import click

from scpn_control.cli_reference_paths import _ReportOutputPath as _ReportOutputPath


@click.command("validate-neural-equilibrium-reference")
@click.option(
    "--artifact-root", help="Directory or JSON artifact containing persisted neural equilibrium reference evidence"
)
@click.option("--require-reference-artifacts", is_flag=True, help="Fail if no reference artifacts are present")
@click.option("--output-json", type=_ReportOutputPath(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_neural_equilibrium_reference_command(
    artifact_root: str | None,
    require_reference_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Report neural-reference declaration checks using the shared reader and input-protecting writer.

    Root/output are caller-relative. Required mode rejects zero declarations;
    schema/tolerance findings and authored persistence failures exit one. Cyclic
    output paths receive fixed operational refusal. JSON
    output carries exact input-byte and report checksums, not model or array
    qualification. No P-EFIT, model fitting or download is performed.
    """
    from validation.validate_neural_equilibrium_reference import (
        ROOT,
        NeuralReferenceReportRefusal,
        validate_neural_equilibrium_reference,
        write_neural_equilibrium_reference_report,
    )

    path = artifact_root or str(ROOT / "validation" / "reports" / "neural_equilibrium_reference")
    try:
        report = validate_neural_equilibrium_reference(path, require_reference_artifacts=require_reference_artifacts)
        if output_json is not None:
            write_neural_equilibrium_reference_report(report, output_json, artifact_root=path)
    except NeuralReferenceReportRefusal:
        raise click.ClickException(
            "Neural equilibrium reference FAILED: neural reference report output must not overwrite selected input"
        ) from None
    except (OSError, UnicodeError, ValueError, RuntimeError):
        raise click.ClickException(
            "Neural equilibrium reference FAILED: could not inspect artifacts or write report"
        ) from None
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(
            f"Neural equilibrium reference: {report['status']} reference_artifacts={report['reference_artifacts']}"
        )
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-orbit-reference")
@click.option(
    "--artifact-root", help="Directory or JSON artifact containing persisted orbit-following reference evidence"
)
@click.option("--require-reference-artifacts", is_flag=True, help="Fail if no reference artifacts are present")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_orbit_reference_command(
    artifact_root: str | None,
    require_reference_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Inspect orbit declarations and protect selected input aliases during report persistence.

    Root/output paths are caller-relative; required zero-declaration or schema/
    tolerance findings exit one. Supported write failures use fixed caller-safe
    text. Passing declarations do not authenticate orbit execution or references.
    """
    from validation.validate_orbit_reference import ROOT, validate_orbit_reference, write_orbit_reference_report

    path = artifact_root or str(ROOT / "validation" / "reports" / "orbit_reference")
    try:
        report = validate_orbit_reference(path, require_reference_artifacts=require_reference_artifacts)
        if output_json is not None:
            write_orbit_reference_report(report, output_json, artifact_root=path)
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        raise click.ClickException("Orbit reference FAILED: could not inspect artifacts or write report") from exc
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"Orbit reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-uncertainty-reference")
@click.option("--artifact-root", help="Directory or JSON artifact containing persisted uncertainty reference evidence")
@click.option("--require-reference-artifacts", is_flag=True, help="Fail if no reference artifacts are present")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_uncertainty_reference_command(
    artifact_root: str | None,
    require_reference_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Inspect UQ declarations and persist an input-protected report with authored operational refusal."""
    from validation.validate_uncertainty_reference import (
        ROOT,
        validate_uncertainty_reference,
        write_uncertainty_reference_report,
    )

    path = artifact_root or str(ROOT / "validation" / "reports" / "uncertainty_reference")
    try:
        report = validate_uncertainty_reference(path, require_reference_artifacts=require_reference_artifacts)
        if output_json is not None:
            write_uncertainty_reference_report(report, output_json, artifact_root=path)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        click.echo("Uncertainty reference FAILED: could not inspect artifacts or write report", err=True)
        raise click.exceptions.Exit(1) from None
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"Uncertainty reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-vmec-reference")
@click.option("--artifact-root", help="Directory or JSON artifact containing persisted VMEC reference evidence")
@click.option("--require-reference-artifacts", is_flag=True, help="Fail if no reference artifacts are present")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_vmec_reference_command(
    artifact_root: str | None,
    require_reference_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Inspect VMEC declarations and write an input-protected report with authored operational refusal."""
    from validation.validate_vmec_reference import ROOT, validate_vmec_reference, write_vmec_reference_report

    path = artifact_root or str(ROOT / "validation" / "reports" / "vmec_reference")
    try:
        report = validate_vmec_reference(path, require_reference_artifacts=require_reference_artifacts)
        if output_json is not None:
            write_vmec_reference_report(report, output_json, artifact_root=path)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        click.echo("VMEC reference FAILED: could not inspect artifacts or write report", err=True)
        raise click.exceptions.Exit(1) from None
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"VMEC reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-ida-same-case")
@click.argument(
    "report_path",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
)
@click.option(
    "--fusion-root",
    type=click.Path(exists=True, file_okay=False, path_type=str),
    help="FUSION Git tree used to verify every source digest at the bound commit",
)
@click.option("--json-out", is_flag=True, help="Emit the CONTROL admission record as JSON")
def validate_ida_same_case_command(
    report_path: str,
    fusion_root: str | None,
    json_out: bool,
) -> None:
    """Validate FUSION IDA same-case evidence without granting admission."""
    from scpn_control.core.ida_same_case_evidence import (
        validate_ida_same_case_evidence,
    )

    admission = validate_ida_same_case_evidence(
        report_path,
        fusion_root=fusion_root,
    )
    payload = admission.as_dict()
    if json_out:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        click.echo(
            "IDA same-case evidence: "
            f"{admission.status} artifact_valid={admission.artifact_valid} "
            f"source_verified={admission.source_verified} "
            f"admitted={admission.admitted}"
        )
        for blocker in admission.blockers:
            click.echo(f"BLOCKED {blocker}", err=True)
    raise click.exceptions.Exit(2)
