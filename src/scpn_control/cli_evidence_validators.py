# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CLI evidence/manifest validation commands

"""Click commands that validate persisted evidence and data-manifest artifacts.

Each command wraps a ``validation.validate_*`` evidence check (or a manifest /
RMSE dashboard entry point), emits a text or JSON report, and exits non-zero on
failure. They are registered onto the root ``scpn-control`` group via
:data:`EVIDENCE_VALIDATOR_COMMANDS` in :mod:`scpn_control.cli`, keeping the CLI
entry point a thin dispatcher over the core operational commands.
"""

from __future__ import annotations

import json
import sys

import click


@click.command("validate")
@click.option("--json-out", is_flag=True, help="Emit JSON")
@click.option("--data-manifest-root", help="Root directory to scan for repository data manifests")
@click.option("--no-data-manifests", is_flag=True, help="Skip repository data-manifest validation")
@click.option("--no-verify-artifacts", is_flag=True, help="Validate manifest metadata without local checksums")
@click.option(
    "--jax-gk-parity-root", help="Directory or JSON artifact containing persisted JAX/native GK parity evidence"
)
@click.option("--no-jax-gk-parity", is_flag=True, help="Skip persisted JAX GK parity evidence validation")
@click.option("--physics-traceability-registry", help="Physics traceability registry JSON path")
@click.option("--no-physics-traceability", is_flag=True, help="Skip physics traceability validation")
@click.option("--multi-shot-campaign-python-report", help="Python/PyO3 multi-shot campaign benchmark JSON report")
@click.option("--multi-shot-campaign-rust-report", help="Rust multi-shot campaign benchmark JSON report")
@click.option(
    "--multi-shot-min-digest-count",
    default=2,
    show_default=True,
    type=int,
    help="Minimum pulsed-MPC admission digests required in each multi-shot campaign report",
)
@click.option("--no-multi-shot-campaign-evidence", is_flag=True, help="Skip multi-shot campaign evidence validation")
@click.option("--runtime-admission-report", help="Runtime-admission benchmark JSON report")
@click.option("--no-runtime-admission-evidence", is_flag=True, help="Skip runtime-admission evidence validation")
@click.option("--native-formal-certificate-report", help="Native formal AOT certificate benchmark JSON report")
@click.option(
    "--native-formal-max-aot-p99-cycle-us",
    default=10.0,
    show_default=True,
    type=float,
    help="Maximum admitted native formal AOT p99 cycle latency in microseconds",
)
@click.option("--no-native-formal-certificate", is_flag=True, help="Skip native formal certificate evidence validation")
def validate(
    json_out: bool,
    data_manifest_root: str | None,
    no_data_manifests: bool,
    no_verify_artifacts: bool,
    jax_gk_parity_root: str | None,
    no_jax_gk_parity: bool,
    physics_traceability_registry: str | None,
    no_physics_traceability: bool,
    multi_shot_campaign_python_report: str | None,
    multi_shot_campaign_rust_report: str | None,
    multi_shot_min_digest_count: int,
    no_multi_shot_campaign_evidence: bool,
    runtime_admission_report: str | None,
    no_runtime_admission_evidence: bool,
    native_formal_certificate_report: str | None,
    native_formal_max_aot_p99_cycle_us: float,
    no_native_formal_certificate: bool,
) -> None:
    """Run import hygiene, provenance, parity, traceability, and formal evidence validation."""
    try:
        from scpn_control.core.integrated_transport_solver import IntegratedTransportSolver  # noqa: F401

        has_transport = True
    except ImportError:
        has_transport = False

    result = {
        "transport_solver_available": has_transport,
        "import_clean": True,
        "status": "pass",
    }
    validation_failed = False

    for mod in ["matplotlib", "torch", "streamlit"]:
        if mod in sys.modules:
            result["import_clean"] = False
            result["status"] = "fail"
            result["contaminated_module"] = mod
            break

    if no_data_manifests:
        result["data_manifests"] = {"status": "skipped"}
    else:
        from validation.validate_data_manifests import ROOT, validate_manifest_directory

        validation_root = data_manifest_root or str(ROOT / "validation" / "reference_data")
        manifest_report = validate_manifest_directory(validation_root, verify_artifacts=not no_verify_artifacts)
        result["data_manifests"] = manifest_report
        if manifest_report["status"] != "pass":
            result["status"] = "fail"
            validation_failed = True

    if no_jax_gk_parity:
        result["jax_gk_parity"] = {"status": "skipped"}
    else:
        from validation.validate_jax_gk_parity import ROOT, validate_jax_gk_parity

        parity_root = jax_gk_parity_root or str(ROOT / "validation" / "reports" / "jax_gk_parity")
        parity_report = validate_jax_gk_parity(
            parity_root,
            require_parity_artifacts=True,
            require_cases=("cyclone_base_case", "tem_kinetic_electron", "stable_mode"),
            require_backends=("cpu", "gpu"),
        )
        result["jax_gk_parity"] = parity_report
        if parity_report["status"] != "pass":
            result["status"] = "fail"
            validation_failed = True

    if no_physics_traceability:
        result["physics_traceability"] = {"status": "skipped"}
    else:
        from validation.validate_physics_traceability import ROOT, validate_physics_traceability

        registry_path = physics_traceability_registry or str(ROOT / "validation" / "physics_traceability.json")
        traceability_report = validate_physics_traceability(registry_path)
        result["physics_traceability"] = traceability_report
        if traceability_report["status"] != "pass":
            result["status"] = "fail"
            validation_failed = True

    if no_multi_shot_campaign_evidence:
        result["multi_shot_campaign"] = {"status": "skipped"}
    else:
        from validation.validate_multi_shot_campaign_evidence import (
            DEFAULT_PYTHON_REPORT,
            DEFAULT_RUST_REPORT,
            validate_multi_shot_campaign_evidence,
        )

        multi_shot_report = validate_multi_shot_campaign_evidence(
            multi_shot_campaign_python_report or DEFAULT_PYTHON_REPORT,
            multi_shot_campaign_rust_report or DEFAULT_RUST_REPORT,
            minimum_digest_count=multi_shot_min_digest_count,
        ).as_dict()
        result["multi_shot_campaign"] = multi_shot_report
        if multi_shot_report["status"] != "pass":
            result["status"] = "fail"
            validation_failed = True

    if no_runtime_admission_evidence:
        result["runtime_admission"] = {"status": "skipped"}
    else:
        from validation.validate_runtime_admission_evidence import (
            DEFAULT_REPORT as DEFAULT_RUNTIME_ADMISSION_REPORT,
        )
        from validation.validate_runtime_admission_evidence import (
            validate_runtime_admission_evidence,
        )

        runtime_report = validate_runtime_admission_evidence(
            runtime_admission_report or DEFAULT_RUNTIME_ADMISSION_REPORT
        ).as_dict()
        result["runtime_admission"] = runtime_report
        if runtime_report["status"] != "pass":
            result["status"] = "fail"
            validation_failed = True

    if no_native_formal_certificate:
        result["native_formal_certificate"] = {"status": "skipped"}
    else:
        from validation.validate_native_formal_certificate_evidence import (
            DEFAULT_REPORT,
            validate_native_formal_certificate_evidence,
        )

        certificate_report_path = native_formal_certificate_report or str(DEFAULT_REPORT)
        certificate_report = validate_native_formal_certificate_evidence(
            certificate_report_path,
            max_aot_p99_cycle_us=native_formal_max_aot_p99_cycle_us,
        ).as_dict()
        result["native_formal_certificate"] = certificate_report
        if certificate_report["status"] != "pass":
            result["status"] = "fail"
            validation_failed = True

    if json_out:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"Transport solver: {'OK' if has_transport else 'MISSING'}")
        click.echo(f"Import clean: {'OK' if result['import_clean'] else 'FAIL'}")
        data_manifests = result["data_manifests"]
        if isinstance(data_manifests, dict) and data_manifests.get("status") == "skipped":
            click.echo("Data manifests: SKIPPED")
        elif isinstance(data_manifests, dict):
            click.echo(
                "Data manifests: "
                f"{data_manifests['status']} "
                f"total={data_manifests['total']} "
                f"real={data_manifests['real']} "
                f"synthetic={data_manifests['synthetic']}"
            )
            for error in data_manifests["errors"]:
                click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
        jax_gk_parity = result["jax_gk_parity"]
        if isinstance(jax_gk_parity, dict) and jax_gk_parity.get("status") == "skipped":
            click.echo("JAX GK parity: SKIPPED")
        elif isinstance(jax_gk_parity, dict):
            click.echo(f"JAX GK parity: {jax_gk_parity['status']} parity_artifacts={jax_gk_parity['parity_artifacts']}")
            for error in jax_gk_parity["errors"]:
                click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
        physics_traceability = result["physics_traceability"]
        if isinstance(physics_traceability, dict) and physics_traceability.get("status") == "skipped":
            click.echo("Physics traceability: SKIPPED")
        elif isinstance(physics_traceability, dict):
            click.echo(
                "Physics traceability: "
                f"{physics_traceability['status']} "
                f"total={physics_traceability['total']} "
                f"open_fidelity_gaps={physics_traceability['open_fidelity_gaps']} "
                f"public_claim_blocked={physics_traceability['public_claim_blocked']}"
            )
            for error in physics_traceability["errors"]:
                index = error.get("index", "-")
                click.echo(f"ERROR {error['path']}[{index}].{error['field']}: {error['error']}", err=True)
        multi_shot_campaign = result["multi_shot_campaign"]
        if isinstance(multi_shot_campaign, dict) and multi_shot_campaign.get("status") == "skipped":
            click.echo("Multi-shot campaign evidence: SKIPPED")
        elif isinstance(multi_shot_campaign, dict):
            surfaces = multi_shot_campaign.get("admitted_surfaces", ())
            surface_count = len(surfaces) if isinstance(surfaces, list) else 0
            click.echo(
                "Multi-shot campaign evidence: "
                f"{multi_shot_campaign['status']} "
                f"surfaces={surface_count} "
                f"pyo3={multi_shot_campaign.get('pyo3_status') or '-'}"
            )
            for error in multi_shot_campaign["errors"]:
                click.echo(f"ERROR multi_shot_campaign: {error}", err=True)
        runtime_admission = result["runtime_admission"]
        if isinstance(runtime_admission, dict) and runtime_admission.get("status") == "skipped":
            click.echo("Runtime admission evidence: SKIPPED")
        elif isinstance(runtime_admission, dict):
            click.echo(
                "Runtime admission evidence: "
                f"{runtime_admission['status']} "
                f"admission={runtime_admission.get('admission_status') or '-'} "
                f"class={runtime_admission.get('benchmark_evidence_class') or '-'}"
            )
            for error in runtime_admission["errors"]:
                click.echo(f"ERROR runtime_admission: {error}", err=True)
        native_formal_certificate = result["native_formal_certificate"]
        if isinstance(native_formal_certificate, dict) and native_formal_certificate.get("status") == "skipped":
            click.echo("Native formal certificate: SKIPPED")
        elif isinstance(native_formal_certificate, dict):
            digest = native_formal_certificate.get("certificate_assumption_sha256") or "-"
            admitted_cases = native_formal_certificate.get("admitted_cases", ())
            admitted_count = len(admitted_cases) if isinstance(admitted_cases, list) else 0
            click.echo(
                "Native formal certificate: "
                f"{native_formal_certificate['status']} "
                f"admitted_cases={admitted_count} "
                f"digest={digest}"
            )
            for error in native_formal_certificate["errors"]:
                click.echo(f"ERROR native_formal_certificate: {error}", err=True)
        click.echo(f"Status: {result['status']}")
    if validation_failed:
        raise click.exceptions.Exit(1)


@click.command("validate-release-evidence")
@click.argument("report", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_release_evidence_command(report: str, json_out: bool) -> None:
    """Validate top-level release evidence report admission."""
    from validation.validate_release_evidence import (
        RELEASE_EVIDENCE_SCHEMA_VERSION,
        validate_release_evidence,
    )

    result = validate_release_evidence(report)
    payload = {
        "schema_version": RELEASE_EVIDENCE_SCHEMA_VERSION,
        "status": result.status,
        "errors": list(result.errors),
        "report_sha256": result.report_sha256,
        "admitted_gates": list(result.admitted_gates),
    }
    if json_out:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        click.echo(f"Release evidence: {result.status}")
        if result.report_sha256 is not None:
            click.echo(f"Report SHA-256: {result.report_sha256}")
        for error in result.errors:
            click.echo(f"ERROR {error}", err=True)
    if result.status != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-manifest")
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option("--json-out", is_flag=True, help="Emit JSON")
@click.option("--verify-artifact", is_flag=True, help="Verify local artefact checksum")
def validate_manifest(manifest: str, json_out: bool, verify_artifact: bool) -> None:
    """Validate real-shot or synthetic-shot data manifest provenance."""
    from scpn_control.core.real_data_manifest import RealDataManifestError, load_real_data_manifest

    try:
        validated = load_real_data_manifest(manifest, verify_artifact=verify_artifact)
    except RealDataManifestError as exc:
        result: dict[str, object] = {
            "status": "fail",
            "error": str(exc),
        }
        if json_out:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"Status: {result['status']}")
            click.echo(f"Error: {result['error']}")
        raise click.exceptions.Exit(1) from exc

    result = {
        "dataset_id": validated.dataset_id,
        "kind": validated.kind,
        "machine": validated.machine,
        "shot": validated.shot,
        "signals": len(validated.signals),
        "source_kind": validated.source.kind,
        "status": "pass",
    }
    if verify_artifact:
        result["artifact_verified"] = True
    if json_out:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"Dataset: {result['dataset_id']}")
        click.echo(f"Kind: {result['kind']}")
        click.echo(f"Machine: {result['machine']}")
        click.echo(f"Shot: {result['shot']}")
        click.echo(f"Signals: {result['signals']}")
        click.echo(f"Source: {result['source_kind']}")
        click.echo(f"Status: {result['status']}")


@click.command("validate-data-manifests")
@click.option("--root", help="Root directory to scan for repository data manifests")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--no-verify-artifacts", is_flag=True, help="Validate metadata without local checksum verification")
@click.option(
    "--require-real-acquisition",
    is_flag=True,
    help="Fail when an acquisition spec has no corresponding acquired MDSplus manifest",
)
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_data_manifests_command(
    root: str | None,
    output_json: str | None,
    no_verify_artifacts: bool,
    require_real_acquisition: bool,
    json_out: bool,
) -> None:
    """Validate repository data manifests, local artefacts, and acquisition specs."""
    from validation.validate_data_manifests import ROOT, validate_manifest_directory

    validation_root = root or str(ROOT / "validation" / "reference_data")
    report = validate_manifest_directory(
        validation_root,
        verify_artifacts=not no_verify_artifacts,
        require_real_acquisition=require_real_acquisition,
    )
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(
            "Data manifests: "
            f"{report['status']} "
            f"total={report['total']} "
            f"real={report['real']} "
            f"synthetic={report['synthetic']} "
            f"acquisition_specs={report['acquisition_specs']['total']}"
        )
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-physics-traceability")
@click.option("--registry", help="Physics traceability registry JSON path")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_physics_traceability_command(
    registry: str | None,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Validate physics fidelity traceability and bounded-claim contracts."""
    from validation.validate_physics_traceability import ROOT, validate_physics_traceability

    registry_path = registry or str(ROOT / "validation" / "physics_traceability.json")
    report = validate_physics_traceability(registry_path)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(
            "Physics traceability: "
            f"{report['status']} "
            f"total={report['total']} "
            f"open_fidelity_gaps={report['open_fidelity_gaps']} "
            f"public_claim_blocked={report['public_claim_blocked']} "
            f"external_validation_trackers={report['external_validation_tracker_count']}"
        )
        for error in report["errors"]:
            index = error.get("index", "-")
            click.echo(f"ERROR {error['path']}[{index}].{error['field']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@click.command("validate-rmse")
@click.option("--json-out", is_flag=True, help="Print JSON report to stdout")
@click.option("--output-json", default="artifacts/rmse_report.json", help="JSON path")
@click.option("--output-md", default="artifacts/rmse_report.md", help="Markdown path")
def validate_rmse(json_out: bool, output_json: str, output_md: str) -> None:
    """Run full RMSE validation dashboard against reference data."""
    from validation.rmse_dashboard import main as rmse_main

    sys.argv = ["rmse_dashboard", "--output-json", output_json, "--output-md", output_md]
    exit_code = rmse_main()
    if json_out:
        from pathlib import Path as _P

        p = _P(output_json)
        if p.exists():
            click.echo(p.read_text(encoding="utf-8"))
    sys.exit(exit_code or 0)


EVIDENCE_VALIDATOR_COMMANDS: tuple[click.Command, ...] = (
    validate,
    validate_release_evidence_command,
    validate_manifest,
    validate_data_manifests_command,
    validate_physics_traceability_command,
    validate_rmse,
)
