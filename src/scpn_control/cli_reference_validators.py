# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CLI reference-artifact validation commands

"""Click commands that validate persisted reference-data artifacts.

Each command wraps a ``validation.validate_*`` reference check, emits a text or
JSON report, and exits non-zero on failure. They are registered onto the root
``scpn-control`` group via :data:`REFERENCE_VALIDATOR_COMMANDS` in
:mod:`scpn_control.cli`, keeping the CLI entry point a thin dispatcher.
"""

from __future__ import annotations

import json

import click


def _split_csv_option(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


# ── reference-artifact validation commands ────────────────────────────


@click.command("validate-gk-crosscode")
@click.option("--evidence-root", help="Directory or JSON report containing external GK evidence")
@click.option("--require-external-runs", is_flag=True, help="Fail if no real external-code evidence is present")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_gk_crosscode_command(
    evidence_root: str | None,
    require_external_runs: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Validate real external-code evidence for linear GK agreement."""
    from validation.validate_gk_crosscode import ROOT, validate_gk_crosscode_evidence

    evidence_path = evidence_root or str(ROOT / "validation" / "reports" / "gk_crosscode")
    report = validate_gk_crosscode_evidence(evidence_path, require_external_runs=require_external_runs)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    """Validate Miller geometry output against immutable reference cases."""
    from validation.validate_gk_geometry_reference import ROOT, validate_gk_geometry_reference

    path = reference_path or str(ROOT / "validation" / "reference_data" / "gk_geometry" / "miller_reference_cases.json")
    report = validate_gk_geometry_reference(path)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    """Validate GK species normalisation and collision reference cases."""
    from validation.validate_gk_species_reference import ROOT, validate_gk_species_reference

    path = reference_path or str(
        ROOT / "validation" / "reference_data" / "gk_species" / "species_collision_reference_cases.json"
    )
    report = validate_gk_species_reference(path)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
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
    from validation.validate_jax_gk_parity import ROOT, validate_jax_gk_parity

    path = artifact_root or str(ROOT / "validation" / "reports" / "jax_gk_parity")
    report = validate_jax_gk_parity(
        path,
        require_parity_artifacts=require_parity_artifacts,
        require_cases=_split_csv_option(require_cases),
        require_backends=_split_csv_option(require_backends),
    )
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"JAX GK parity: {report['status']} parity_artifacts={report['parity_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-gk-ood-calibration")
@click.option("--artifact-root", help="Directory or JSON artifact containing persisted GK OOD calibration evidence")
@click.option("--require-campaign-artifacts", is_flag=True, help="Fail if no calibration artifacts are present")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_gk_ood_calibration_command(
    artifact_root: str | None,
    require_campaign_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Validate persisted GK OOD calibration campaign artifacts."""
    from validation.validate_gk_ood_calibration import ROOT, validate_gk_ood_calibration

    path = artifact_root or str(ROOT / "validation" / "reports" / "gk_ood_calibration")
    report = validate_gk_ood_calibration(path, require_campaign_artifacts=require_campaign_artifacts)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"GK OOD calibration: {report['status']} campaign_artifacts={report['campaign_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-gk-interface-artifacts")
@click.option("--artifact-root", help="Directory or JSON artifact containing persisted external GK interface evidence")
@click.option("--require-interface-artifacts", is_flag=True, help="Fail if no external interface artifacts are present")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_gk_interface_artifacts_command(
    artifact_root: str | None,
    require_interface_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Validate persisted external GK interface parser artifacts."""
    from validation.validate_gk_interface_artifacts import ROOT, validate_gk_interface_artifacts

    path = artifact_root or str(ROOT / "validation" / "reports" / "gk_interfaces")
    report = validate_gk_interface_artifacts(path, require_interface_artifacts=require_interface_artifacts)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"GK interface artifacts: {report['status']} interface_artifacts={report['interface_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


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
    """Validate persisted blob-transport reference artifacts."""
    from validation.validate_blob_transport_reference import ROOT, validate_blob_transport_reference

    path = artifact_root or str(ROOT / "validation" / "reports" / "blob_transport_reference")
    report = validate_blob_transport_reference(path, require_reference_artifacts=require_reference_artifacts)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"Blob transport reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


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
    """Validate persisted ELM reference artifacts."""
    from validation.validate_elm_reference import ROOT, validate_elm_reference

    path = artifact_root or str(ROOT / "validation" / "reports" / "elm_reference")
    report = validate_elm_reference(path, require_reference_artifacts=require_reference_artifacts)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    """Validate persisted EPED reference artifacts."""
    from validation.validate_eped_reference import ROOT, validate_eped_reference

    path = artifact_root or str(ROOT / "validation" / "reports" / "eped_reference")
    report = validate_eped_reference(path, require_reference_artifacts=require_reference_artifacts)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    """Validate persisted MARFE reference artifacts."""
    from validation.validate_marfe_reference import ROOT, validate_marfe_reference

    path = artifact_root or str(ROOT / "validation" / "reports" / "marfe_reference")
    report = validate_marfe_reference(path, require_reference_artifacts=require_reference_artifacts)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    """Validate persisted NTM reference artifacts."""
    from validation.validate_ntm_reference import ROOT, validate_ntm_reference

    path = artifact_root or str(ROOT / "validation" / "reports" / "ntm_reference")
    report = validate_ntm_reference(path, require_reference_artifacts=require_reference_artifacts)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"NTM reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-neural-equilibrium-reference")
@click.option(
    "--artifact-root", help="Directory or JSON artifact containing persisted neural equilibrium reference evidence"
)
@click.option("--require-reference-artifacts", is_flag=True, help="Fail if no reference artifacts are present")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_neural_equilibrium_reference_command(
    artifact_root: str | None,
    require_reference_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Validate persisted neural-equilibrium reference artifacts."""
    from validation.validate_neural_equilibrium_reference import ROOT, validate_neural_equilibrium_reference

    path = artifact_root or str(ROOT / "validation" / "reports" / "neural_equilibrium_reference")
    report = validate_neural_equilibrium_reference(path, require_reference_artifacts=require_reference_artifacts)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    """Validate persisted neural-transport reference artifacts."""
    from validation.validate_neural_transport_reference import ROOT, validate_neural_transport_reference

    path = artifact_root or str(ROOT / "validation" / "reports" / "neural_transport_reference")
    report = validate_neural_transport_reference(path, require_reference_artifacts=require_reference_artifacts)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    """Validate persisted neural-turbulence reference artifacts."""
    from validation.validate_neural_turbulence_reference import ROOT, validate_neural_turbulence_reference

    path = artifact_root or str(ROOT / "validation" / "reports" / "neural_turbulence_reference")
    report = validate_neural_turbulence_reference(path, require_reference_artifacts=require_reference_artifacts)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    """Validate persisted orbit-following reference artifacts."""
    from validation.validate_orbit_reference import ROOT, validate_orbit_reference

    path = artifact_root or str(ROOT / "validation" / "reports" / "orbit_reference")
    report = validate_orbit_reference(path, require_reference_artifacts=require_reference_artifacts)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    """Validate persisted uncertainty-quantification reference artifacts."""
    from validation.validate_uncertainty_reference import ROOT, validate_uncertainty_reference

    path = artifact_root or str(ROOT / "validation" / "reports" / "uncertainty_reference")
    report = validate_uncertainty_reference(path, require_reference_artifacts=require_reference_artifacts)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    """Validate persisted VMEC-lite reference artifacts."""
    from validation.validate_vmec_reference import ROOT, validate_vmec_reference

    path = artifact_root or str(ROOT / "validation" / "reports" / "vmec_reference")
    report = validate_vmec_reference(path, require_reference_artifacts=require_reference_artifacts)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"VMEC reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


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
    """Validate persisted RZIP vertical-stability reference artifacts."""
    from validation.validate_rzip_reference import ROOT, validate_rzip_reference

    path = artifact_root or str(ROOT / "validation" / "reports" / "rzip_reference")
    report = validate_rzip_reference(path, require_reference_artifacts=require_reference_artifacts)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"RZIP reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-current-drive-reference")
@click.option(
    "--artifact-root", help="Directory or JSON artifact containing persisted current-drive reference evidence"
)
@click.option("--require-reference-artifacts", is_flag=True, help="Fail if no reference artifacts are present")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_current_drive_reference_command(
    artifact_root: str | None,
    require_reference_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Validate persisted current-drive reference artifacts."""
    from validation.validate_current_drive_reference import ROOT, validate_current_drive_reference

    path = artifact_root or str(ROOT / "validation" / "reports" / "current_drive_reference")
    report = validate_current_drive_reference(path, require_reference_artifacts=require_reference_artifacts)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"Current-drive reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-mu-synthesis-reference")
@click.option("--artifact-root", help="Directory or JSON artifact containing persisted mu-synthesis reference evidence")
@click.option("--require-reference-artifacts", is_flag=True, help="Fail if no reference artifacts are present")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_mu_synthesis_reference_command(
    artifact_root: str | None,
    require_reference_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Validate persisted mu-synthesis reference artifacts."""
    from validation.validate_mu_synthesis_reference import ROOT, validate_mu_synthesis_reference

    path = artifact_root or str(ROOT / "validation" / "reports" / "mu_synthesis_reference")
    report = validate_mu_synthesis_reference(path, require_reference_artifacts=require_reference_artifacts)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"Mu-synthesis reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
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
    """Validate persisted volt-second reference artifacts."""
    from validation.validate_volt_second_reference import ROOT, validate_volt_second_reference

    path = artifact_root or str(ROOT / "validation" / "reports" / "volt_second_reference")
    report = validate_volt_second_reference(path, require_reference_artifacts=require_reference_artifacts)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    """Validate persisted DT burn-control reference artifacts."""
    from validation.validate_burn_reference import ROOT, validate_burn_reference

    path = artifact_root or str(ROOT / "validation" / "reports" / "burn_reference")
    report = validate_burn_reference(path, require_reference_artifacts=require_reference_artifacts)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"Burn reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-density-reference")
@click.option("--artifact-root", help="Directory or JSON artifact containing persisted density reference evidence")
@click.option("--require-reference-artifacts", is_flag=True, help="Fail if no reference artifacts are present")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_density_reference_command(
    artifact_root: str | None,
    require_reference_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Validate persisted density-control reference artifacts."""
    from validation.validate_density_reference import ROOT, validate_density_reference

    path = artifact_root or str(ROOT / "validation" / "reports" / "density_reference")
    report = validate_density_reference(path, require_reference_artifacts=require_reference_artifacts)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"Density reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
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
    """Validate persisted free-boundary tracking reference artifacts."""
    from validation.validate_free_boundary_reference import ROOT, validate_free_boundary_reference

    path = artifact_root or str(ROOT / "validation" / "reports" / "free_boundary_reference")
    report = validate_free_boundary_reference(path, require_reference_artifacts=require_reference_artifacts)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"Free-boundary reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
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
    """Validate persisted disruption-mitigation reference artifacts."""
    from validation.validate_disruption_reference import ROOT, validate_disruption_reference

    path = artifact_root or str(ROOT / "validation" / "reports" / "disruption_reference")
    report = validate_disruption_reference(path, require_reference_artifacts=require_reference_artifacts)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"Disruption reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
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
    """Validate persisted tokamak digital-twin reference artifacts."""
    from validation.validate_digital_twin_reference import ROOT, validate_digital_twin_reference

    path = artifact_root or str(ROOT / "validation" / "reports" / "digital_twin_reference")
    report = validate_digital_twin_reference(path, require_reference_artifacts=require_reference_artifacts)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"Digital twin reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
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
    """Validate persisted SOC turbulence-learning reference artifacts."""
    from validation.validate_soc_reference import ROOT, validate_soc_reference

    path = artifact_root or str(ROOT / "validation" / "reports" / "soc_reference")
    report = validate_soc_reference(path, require_reference_artifacts=require_reference_artifacts)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(f"SOC reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
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


REFERENCE_VALIDATOR_COMMANDS: tuple[click.Command, ...] = (
    validate_gk_crosscode_command,
    validate_gk_geometry_reference_command,
    validate_gk_species_reference_command,
    validate_jax_gk_parity_command,
    validate_gk_ood_calibration_command,
    validate_gk_interface_artifacts_command,
    validate_blob_transport_reference_command,
    validate_elm_reference_command,
    validate_eped_reference_command,
    validate_marfe_reference_command,
    validate_ntm_reference_command,
    validate_neural_equilibrium_reference_command,
    validate_neural_transport_reference_command,
    validate_neural_turbulence_reference_command,
    validate_orbit_reference_command,
    validate_uncertainty_reference_command,
    validate_vmec_reference_command,
    validate_rzip_reference_command,
    validate_current_drive_reference_command,
    validate_mu_synthesis_reference_command,
    validate_volt_second_reference_command,
    validate_burn_reference_command,
    validate_density_reference_command,
    validate_free_boundary_reference_command,
    validate_disruption_reference_command,
    validate_digital_twin_reference_command,
    validate_soc_reference_command,
    validate_ida_same_case_command,
)
