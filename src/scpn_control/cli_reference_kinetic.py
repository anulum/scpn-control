# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — External GK declarations and bounded geometry/species reference checks

"""External GK declarations and bounded geometry/species reference checks.

Registered by the reference facade; preserve actual Click options, defaults and
report/refusal semantics. Admission limits are defined by each source validator.
"""

from __future__ import annotations

import json

import click

from scpn_control.cli_reference_paths import _ReportOutputPath


def _split_csv_option(value: str | None) -> tuple[str, ...]:
    """Return ordered trimmed nonempty CSV fields, or an empty tuple for an absent option."""
    if value is None:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


@click.command("validate-gk-crosscode", help="Validate real external-code evidence for linear GK agreement.")
@click.option("--evidence-root", help="Directory or JSON report containing external GK evidence")
@click.option("--require-external-runs", is_flag=True, help="Fail if no real external-code evidence is present")
@click.option("--output-json", type=_ReportOutputPath(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_gk_crosscode_command(
    evidence_root: str | None,
    require_external_runs: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Inspect GK declarations with protected persistence; a passing metadata comparison grants no binary/physical admission."""
    from validation.validate_gk_crosscode import ROOT, validate_gk_crosscode_evidence, write_gk_crosscode_report

    evidence_path = evidence_root or str(ROOT / "validation" / "reports" / "gk_crosscode")
    try:
        report = validate_gk_crosscode_evidence(evidence_path, require_external_runs=require_external_runs)
        if output_json is not None:
            write_gk_crosscode_report(report, output_json, evidence_root=evidence_path)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        click.echo("GK cross-code evidence FAILED: could not inspect declarations or write report", err=True)
        raise click.exceptions.Exit(1) from None
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"GK cross-code evidence: {report['status']} external_runs={report['external_runs']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-gk-geometry-reference")
@click.option("--reference-path", help="Immutable Miller geometry reference case JSON")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_gk_geometry_reference_command(
    reference_path: str | None,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Inspect exact-byte local Miller comparisons and protect the single reference during report persistence."""
    from validation.validate_gk_geometry_reference import (
        ROOT,
        validate_gk_geometry_reference,
        write_gk_geometry_reference_report,
    )

    path = reference_path or str(ROOT / "validation" / "reference_data" / "gk_geometry" / "miller_reference_cases.json")
    try:
        report = validate_gk_geometry_reference(path)
        if output_json is not None:
            write_gk_geometry_reference_report(report, output_json, reference_path=path)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        click.echo("GK geometry reference FAILED: could not inspect reference or write report", err=True)
        raise click.exceptions.Exit(1) from None
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"GK geometry reference: {report['status']} cases={report['cases']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-gk-species-reference")
@click.option("--reference-path", help="Immutable GK species and collision reference case JSON")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_gk_species_reference_command(
    reference_path: str | None,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Compare exact-byte bounded species/operator cases and protect the single input during report persistence."""
    from validation.validate_gk_species_reference import (
        ROOT,
        validate_gk_species_reference,
        write_gk_species_reference_report,
    )

    path = reference_path or str(
        ROOT / "validation" / "reference_data" / "gk_species" / "species_collision_reference_cases.json"
    )
    try:
        report = validate_gk_species_reference(path)
        if output_json is not None:
            write_gk_species_reference_report(report, output_json, reference_path=path)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        click.echo("GK species reference FAILED: could not inspect reference or write report", err=True)
        raise click.exceptions.Exit(1) from None
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"GK species reference: {report['status']} cases={report['cases']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-jax-gk-parity")
@click.option("--artifact-root", help="Directory or JSON artifact containing persisted JAX/native GK parity evidence")
@click.option("--require-parity-artifacts", is_flag=True, help="Fail if no persisted parity artifacts are present")
@click.option("--require-cases", help="Comma-separated required parity cases")
@click.option("--require-backends", help="Comma-separated required JAX backends")
@click.option("--output-json", type=_ReportOutputPath(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_jax_gk_parity_command(
    artifact_root: str | None,
    require_parity_artifacts: bool,
    require_cases: str | None,
    require_backends: str | None,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Validate persisted JAX/native GK parity artifacts."""
    from validation.validate_jax_gk_parity import ROOT, validate_jax_gk_parity, write_jax_gk_parity_report

    path = artifact_root or str(ROOT / "validation" / "reports" / "jax_gk_parity")
    report = validate_jax_gk_parity(
        path,
        require_parity_artifacts=require_parity_artifacts,
        require_cases=_split_csv_option(require_cases),
        require_backends=_split_csv_option(require_backends),
    )
    if output_json is not None:
        try:
            write_jax_gk_parity_report(report, output_json, artifact_root=path)
        except (OSError, UnicodeError, ValueError, RuntimeError):
            click.echo("JAX GK parity FAILED: could not write report without overwriting selected input", err=True)
            raise click.exceptions.Exit(1) from None
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"JAX GK parity: {report['status']} parity_artifacts={report['parity_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-gk-ood-calibration", help="Validate persisted GK OOD calibration campaign artifacts.")
@click.option("--artifact-root", help="Directory or JSON artifact containing persisted GK OOD calibration evidence")
@click.option("--require-campaign-artifacts", is_flag=True, help="Fail if no calibration artifacts are present")
@click.option("--output-json", type=_ReportOutputPath(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_gk_ood_calibration_command(
    artifact_root: str | None,
    require_campaign_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Inspect declared OOD campaigns with protected persistence; original acceptance flag grants no installed deployment."""
    from validation.validate_gk_ood_calibration import (
        ROOT,
        validate_gk_ood_calibration,
        write_gk_ood_calibration_report,
    )

    path = artifact_root or str(ROOT / "validation" / "reports" / "gk_ood_calibration")
    try:
        report = validate_gk_ood_calibration(path, require_campaign_artifacts=require_campaign_artifacts)
        if output_json is not None:
            write_gk_ood_calibration_report(report, output_json, artifact_root=path)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        click.echo("GK OOD calibration FAILED: could not inspect campaigns or write report", err=True)
        raise click.exceptions.Exit(1) from None
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"GK OOD calibration: {report['status']} campaign_artifacts={report['campaign_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-gk-interface-artifacts", help="Validate persisted external GK interface parser artifacts.")
@click.option("--artifact-root", help="Directory or JSON artifact containing persisted external GK interface evidence")
@click.option("--require-interface-artifacts", is_flag=True, help="Fail if no external interface artifacts are present")
@click.option("--output-json", type=_ReportOutputPath(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_gk_interface_artifacts_command(
    artifact_root: str | None,
    require_interface_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Inspect declared parser metadata with protected report output; grant no independent source or scientific acceptance."""
    from validation.validate_gk_interface_artifacts import (
        ROOT,
        validate_gk_interface_artifacts,
        write_gk_interface_artifacts_report,
    )

    path = artifact_root or str(ROOT / "validation" / "reports" / "gk_interfaces")
    try:
        report = validate_gk_interface_artifacts(path, require_interface_artifacts=require_interface_artifacts)
        if output_json is not None:
            write_gk_interface_artifacts_report(report, output_json, artifact_root=path)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        click.echo("GK interface artifacts FAILED: could not inspect declarations or write report", err=True)
        raise click.exceptions.Exit(1) from None
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"GK interface artifacts: {report['status']} interface_artifacts={report['interface_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)
