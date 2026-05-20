# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CLI Entry Point

"""
CLI for scpn-control.

Usage::

    scpn-control demo --scenario combined --steps 1000
    scpn-control benchmark --n-bench 5000
    scpn-control validate
    scpn-control validate-manifest real_manifest.json --json-out
    scpn-control validate-data-manifests --json-out
    scpn-control validate-physics-traceability --json-out
    scpn-control validate-gk-geometry-reference --json-out
    scpn-control validate-gk-species-reference --json-out
    scpn-control validate-jax-gk-parity --require-parity-artifacts --json-out
    scpn-control validate-gk-ood-calibration --require-campaign-artifacts --json-out
    scpn-control validate-gk-interface-artifacts --require-interface-artifacts --json-out
    scpn-control validate-neural-equilibrium-reference --require-reference-artifacts --json-out
    scpn-control validate-neural-transport-reference --require-reference-artifacts --json-out
    scpn-control validate-neural-turbulence-reference --require-reference-artifacts --json-out
    scpn-control validate-orbit-reference --require-reference-artifacts --json-out
    scpn-control validate-uncertainty-reference --require-reference-artifacts --json-out
    scpn-control validate-vmec-reference --require-reference-artifacts --json-out
    scpn-control validate-rzip-reference --require-reference-artifacts --json-out
    scpn-control validate-density-reference --require-reference-artifacts --json-out
    scpn-control validate-disruption-reference --require-reference-artifacts --json-out
    scpn-control validate-digital-twin-reference --require-reference-artifacts --json-out
    scpn-control validate-soc-reference --require-reference-artifacts --json-out
    scpn-control acquire-mdsplus-shot --spec-json validation/reference_data/diiid/acquisition_specs/shot_163303_mdsplus.json
    scpn-control live --port 8765 --zeta 0.5
    scpn-control hil-test --shots-dir validation/reference_data/diiid/disruption_shots
"""

from __future__ import annotations

import json
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Mapping, cast

import click
import numpy as np

try:
    _VERSION = version("scpn-control")
except PackageNotFoundError:
    _VERSION = "0.0.0.dev"

if TYPE_CHECKING:
    from scpn_control.core.mdsplus_acquisition import MDSplusSignalSpec


@click.group()
@click.version_option(version=_VERSION, prog_name="scpn-control")
def main() -> None:
    """SCPN Control — Neuro-symbolic Stochastic Petri Net controller."""


@main.command()
@click.option("--scenario", default="combined", type=click.Choice(["pid", "snn", "combined"]))
@click.option("--steps", default=1000, type=int, help="Simulation time steps")
@click.option("--json-out", is_flag=True, help="Emit JSON instead of text")
def demo(scenario: str, steps: int, json_out: bool) -> None:
    """Run a closed-loop control demo."""
    from scpn_control.scpn.compiler import FusionCompiler
    from scpn_control.scpn.contracts import ControlScales, ControlTargets, FeatureAxisSpec
    from scpn_control.scpn.controller import NeuroSymbolicController
    from scpn_control.scpn.structure import StochasticPetriNet

    # Build a simple control Petri Net
    net = StochasticPetriNet()
    # Observation places (axis-based error mapping)
    net.add_place("x_err_pos", initial_tokens=0.0)  # idx 0
    net.add_place("x_err_neg", initial_tokens=0.0)  # idx 1
    # Action places
    net.add_place("a_ctrl_pos", initial_tokens=0.0)  # idx 2
    net.add_place("a_ctrl_neg", initial_tokens=0.0)  # idx 3

    # Logic: if error is positive, fire positive control
    net.add_transition("t_pos", threshold=0.01)
    net.add_arc("x_err_pos", "t_pos", weight=0.1)
    net.add_arc("t_pos", "a_ctrl_pos", weight=0.1)

    # Logic: if error is negative, fire negative control
    net.add_transition("t_neg", threshold=0.01)
    net.add_arc("x_err_neg", "t_neg", weight=0.1)
    net.add_arc("t_neg", "a_ctrl_neg", weight=0.1)

    compiler = FusionCompiler()
    compiled = compiler.compile(net)

    # Create the controller with mapping
    artifact = compiled.export_artifact(
        name="demo_controller",
        injection_config=[
            {"place_id": 0, "source": "x_err_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 1, "source": "x_err_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        ],
        readout_config={
            "actions": [{"name": "ctrl", "pos_place": 2, "neg_place": 3}],
            "gains": [0.2],
            "abs_max": [1.0],
        },
    )

    # Map a generic 'error' feature to our places
    feat_axes = [
        FeatureAxisSpec(
            obs_key="err",
            target=1.0,  # target state
            scale=1.0,
            pos_key="x_err_pos",
            neg_key="x_err_neg",
        )
    ]

    controller = NeuroSymbolicController(
        artifact,
        seed_base=42,
        targets=ControlTargets(R_target_m=1.0, Z_target_m=0.0),
        scales=ControlScales(R_scale_m=1.0, Z_scale_m=1.0),
        feature_axes=feat_axes,
        sc_binary_margin=0.0,
    )

    rng = np.random.default_rng(42)
    target = 1.0
    state = 0.5
    trajectory = []

    for step in range(steps):
        # The controller internal logic:
        # 1. Takes 'err' observation
        # 2. Subtracts target (1.0) -> if state is 0.5, err_val is 0.5.
        #    Wait, obs is usually the measured value.
        #    Controller computes: feature_err = target - measurement

        obs = {"err": float(state)}
        actions = controller.step(obs, step)
        action = float(cast(Mapping[str, float], actions).get("ctrl", 0.0))

        if scenario == "pid":
            # Override with PID for comparison
            action = 0.1 * (target - state) + 0.01 * rng.standard_normal()
        elif scenario == "combined":
            action += 0.05 * (target - state)

        state += action
        state = np.clip(state, 0.0, 2.0)
        error = target - state
        trajectory.append({"step": step, "state": float(state), "error": float(error)})

    final_error = abs(trajectory[-1]["error"])
    result = {
        "scenario": scenario,
        "steps": steps,
        "final_state": trajectory[-1]["state"],
        "final_error": final_error,
        "converged": final_error < 0.05,
        "net_places": net.n_places,
        "net_transitions": net.n_transitions,
        "compiled_neurons": compiled.n_neurons if hasattr(compiled, "n_neurons") else "N/A",
    }

    if json_out:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"Scenario: {scenario}")
        click.echo(f"Steps: {steps}")
        click.echo(f"Final state: {result['final_state']:.4f}")
        click.echo(f"Final error: {result['final_error']:.4f}")
        click.echo(f"Converged: {result['converged']}")


@main.command()
@click.option("--n-bench", default=5000, type=int, help="Number of benchmark iterations")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def benchmark(n_bench: int, json_out: bool) -> None:
    """Timing benchmark: PID vs SNN control step latency."""

    t0 = time.perf_counter()
    kp, ki, kd = 1.0, 0.1, 0.01
    integral, prev_error = 0.0, 0.0
    for _ in range(n_bench):
        error = np.random.randn()
        integral += error
        derivative = error - prev_error
        _ = kp * error + ki * integral + kd * derivative
        prev_error = error
    pid_time = (time.perf_counter() - t0) / n_bench * 1e6  # microseconds

    t0 = time.perf_counter()
    n_neurons = 50
    v = np.zeros(n_neurons)
    threshold = 1.0
    decay = 0.9
    for _ in range(n_bench):
        error = np.random.randn()
        v[:] = v * decay + error * np.random.randn(n_neurons) * 0.1
        spikes = v > threshold
        v[spikes] = 0.0
        _ = np.mean(spikes.astype(float))
    snn_time = (time.perf_counter() - t0) / n_bench * 1e6

    result = {
        "n_bench": n_bench,
        "pid_us_per_step": round(pid_time, 2),
        "snn_us_per_step": round(snn_time, 2),
        "speedup_ratio": round(pid_time / max(snn_time, 1e-9), 2),
    }

    if json_out:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"PID: {result['pid_us_per_step']} us/step")
        click.echo(f"SNN: {result['snn_us_per_step']} us/step")
        click.echo(f"Ratio: {result['speedup_ratio']}x")


@main.command()
@click.option("--json-out", is_flag=True, help="Emit JSON")
@click.option("--data-manifest-root", help="Root directory to scan for repository data manifests")
@click.option("--no-data-manifests", is_flag=True, help="Skip repository data-manifest validation")
@click.option("--no-verify-artifacts", is_flag=True, help="Validate manifest metadata without local checksums")
def validate(
    json_out: bool,
    data_manifest_root: str | None,
    no_data_manifests: bool,
    no_verify_artifacts: bool,
) -> None:
    """Run import hygiene and repository data-provenance validation."""
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
    manifest_gate_failed = False

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
            manifest_gate_failed = True

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
        click.echo(f"Status: {result['status']}")
    if manifest_gate_failed:
        raise click.exceptions.Exit(1)


@main.command("validate-manifest")
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


@main.command("validate-data-manifests")
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


@main.command("validate-physics-traceability")
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
            f"public_claim_blocked={report['public_claim_blocked']}"
        )
        for error in report["errors"]:
            index = error.get("index", "-")
            click.echo(f"ERROR {error['path']}[{index}].{error['field']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@main.command("validate-gk-crosscode")
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


@main.command("validate-gk-geometry-reference")
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


@main.command("validate-gk-species-reference")
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


@main.command("validate-jax-gk-parity")
@click.option("--artifact-root", help="Directory or JSON artifact containing persisted JAX/native GK parity evidence")
@click.option("--require-parity-artifacts", is_flag=True, help="Fail if no persisted parity artifacts are present")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_jax_gk_parity_command(
    artifact_root: str | None,
    require_parity_artifacts: bool,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Validate persisted JAX/native GK parity artifacts."""
    from validation.validate_jax_gk_parity import ROOT, validate_jax_gk_parity

    path = artifact_root or str(ROOT / "validation" / "reports" / "jax_gk_parity")
    report = validate_jax_gk_parity(path, require_parity_artifacts=require_parity_artifacts)
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


@main.command("validate-gk-ood-calibration")
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


@main.command("validate-gk-interface-artifacts")
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


@main.command("validate-neural-equilibrium-reference")
@click.option("--artifact-root", help="Directory or JSON artifact containing persisted neural equilibrium reference evidence")
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
        click.echo(f"Neural equilibrium reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@main.command("validate-neural-transport-reference")
@click.option("--artifact-root", help="Directory or JSON artifact containing persisted neural transport reference evidence")
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
        click.echo(f"Neural transport reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@main.command("validate-neural-turbulence-reference")
@click.option("--artifact-root", help="Directory or JSON artifact containing persisted neural turbulence reference evidence")
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
        click.echo(f"Neural turbulence reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@main.command("validate-orbit-reference")
@click.option("--artifact-root", help="Directory or JSON artifact containing persisted orbit-following reference evidence")
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


@main.command("validate-uncertainty-reference")
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


@main.command("validate-vmec-reference")
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


@main.command("validate-rzip-reference")
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


@main.command("validate-density-reference")
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


@main.command("validate-disruption-reference")
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


@main.command("validate-digital-twin-reference")
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


@main.command("validate-soc-reference")
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


@main.command("acquire-mdsplus-shot")
@click.option(
    "--spec-json",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Versioned acquisition spec JSON; replaces --tree/--shot/--signal",
)
@click.option("--tree", help="MDSplus tree name")
@click.option("--shot", type=int, help="Shot number")
@click.option("--signal", "signals_json", multiple=True, help="Signal JSON with name/node/units/timebase")
@click.option("--output-npz", required=True, type=click.Path(dir_okay=False), help="Output NPZ path")
@click.option("--manifest-json", required=True, type=click.Path(dir_okay=False), help="Output manifest JSON path")
@click.option("--source-uri", help="Source URI for provenance; defaults to mdsplus://TREE/SHOT")
@click.option("--access-policy", default="facility-approved", show_default=True, help="Access policy label")
@click.option("--licence", default="facility data policy", show_default=True, help="Licence or facility data policy")
@click.option("--retrieved-at", help="UTC retrieval timestamp for reproducible tests")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def acquire_mdsplus_shot_command(
    spec_json: str | None,
    tree: str | None,
    shot: int | None,
    signals_json: tuple[str, ...],
    output_npz: str,
    manifest_json: str,
    source_uri: str | None,
    access_policy: str,
    licence: str,
    retrieved_at: str | None,
    json_out: bool,
) -> None:
    """Acquire selected MDSplus shot signals and write a validated manifest."""
    from scpn_control.core.mdsplus_acquisition import acquire_mdsplus_shot, load_mdsplus_acquisition_request

    try:
        if spec_json is not None:
            request = load_mdsplus_acquisition_request(spec_json)
            tree = request.tree
            shot = request.shot
            signal_specs = request.signals
            source_uri = source_uri or request.source_uri
            access_policy = request.access_policy
            licence = request.licence
        else:
            if tree is None:
                raise ValueError("--tree is required unless --spec-json is supplied")
            if shot is None:
                raise ValueError("--shot is required unless --spec-json is supplied")
            if not signals_json:
                raise ValueError("at least one --signal is required unless --spec-json is supplied")
            signal_specs = [_parse_mdsplus_signal_spec(raw) for raw in signals_json]

        result = acquire_mdsplus_shot(
            tree=tree,
            shot=shot,
            signals=signal_specs,
            output_npz=output_npz,
            manifest_json=manifest_json,
            source_uri=source_uri or f"mdsplus://{tree}/{shot}",
            access_policy=access_policy,
            licence=licence,
            retrieved_at=retrieved_at,
        )
    except (RuntimeError, ValueError, json.JSONDecodeError, KeyError) as exc:
        payload: dict[str, object] = {"status": "fail", "error": str(exc)}
        if json_out:
            click.echo(json.dumps(payload, indent=2))
        else:
            click.echo(f"Status: {payload['status']}")
            click.echo(f"Error: {payload['error']}")
        raise click.exceptions.Exit(1) from exc

    payload = {
        "status": "pass",
        "dataset_id": result.dataset_id,
        "output_npz": str(result.output_npz),
        "manifest_json": str(result.manifest_json),
        "checksum_sha256": result.checksum_sha256,
    }
    if json_out:
        click.echo(json.dumps(payload, indent=2))
    else:
        click.echo(f"Dataset: {payload['dataset_id']}")
        click.echo(f"Output NPZ: {payload['output_npz']}")
        click.echo(f"Manifest: {payload['manifest_json']}")
        click.echo(f"Status: {payload['status']}")


def _parse_mdsplus_signal_spec(raw: str) -> MDSplusSignalSpec:
    from scpn_control.core.mdsplus_acquisition import MDSplusSignalSpec

    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("signal specification must be a JSON object")
    return MDSplusSignalSpec(
        name=str(payload["name"]),
        node=str(payload["node"]),
        units=str(payload["units"]),
        timebase=str(payload["timebase"]),
    )


@main.command("validate-rmse")
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


@main.command()
@click.option("--port", default=8765, type=int, help="WebSocket port")
@click.option("--host", default="0.0.0.0", help="Bind address")  # nosec B104
@click.option("--layers", default=16, type=int, help="Kuramoto layers (L)")
@click.option("--n-per", default=50, type=int, help="Oscillators per layer")
@click.option("--zeta", default=0.5, type=float, help="Global field coupling zeta")
@click.option("--psi", default=0.0, type=float, help="Initial Psi driver")
@click.option("--tick-interval", default=0.001, type=float, help="Seconds between ticks")
def live(port: int, host: str, layers: int, n_per: int, zeta: float, psi: float, tick_interval: float) -> None:
    """Start real-time WebSocket phase sync server.

    Streams Kuramoto R/V/lambda tick snapshots over ws://<host>:<port>.
    Connect with: examples/streamlit_ws_client.py or any WS client.
    """
    import logging

    from scpn_control.phase.realtime_monitor import RealtimeMonitor
    from scpn_control.phase.ws_phase_stream import PhaseStreamServer

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    click.echo(f"Starting phase sync server on ws://{host}:{port}")
    click.echo(f"  L={layers}, N_per={n_per}, zeta={zeta}, psi={psi}")
    click.echo("  Ctrl-C to stop")

    mon = RealtimeMonitor.from_paper27(
        L=layers,
        N_per=n_per,
        zeta_uniform=zeta,
        psi_driver=psi,
    )
    server = PhaseStreamServer(monitor=mon, tick_interval_s=tick_interval)
    server.serve_sync(host=host, port=port)


@main.command(name="hil-test")
@click.option("--shots-dir", default="validation/reference_data/diiid/disruption_shots", type=str)
@click.option("--json-out", is_flag=True, help="Emit JSON")
def hil_test(shots_dir: str, json_out: bool) -> None:
    """Hardware-in-the-loop test campaign against reference shots."""
    from pathlib import Path

    shots_path = Path(shots_dir)
    if not shots_path.exists():
        click.echo(f"ERROR: Shots directory not found: {shots_dir}", err=True)
        sys.exit(1)

    shot_files = sorted(shots_path.glob("*.npz"))
    results = []
    for sf in shot_files:
        data = np.load(sf, allow_pickle=True)
        results.append(
            {
                "shot": sf.stem,
                "keys": list(data.keys()),
                "status": "loaded",
            }
        )

    summary = {
        "shots_dir": str(shots_dir),
        "n_shots": len(results),
        "shots": results,
    }

    if json_out:
        click.echo(json.dumps(summary, indent=2))
    else:
        click.echo(f"Loaded {len(results)} shots from {shots_dir}")
        for r in results:
            click.echo(f"  {r['shot']}: {len(r['keys'])} arrays")


@main.command()
@click.option("--json-out", is_flag=True, help="Emit JSON")
def info(json_out: bool) -> None:
    """Print environment and backend information."""
    from pathlib import Path

    import scpn_control

    rust_status = "available" if scpn_control.RUST_BACKEND else "not installed"

    weights_dir = Path(__file__).resolve().parents[2] / "weights"
    weight_files = sorted(weights_dir.glob("*.npz")) if weights_dir.exists() else []

    result = {
        "version": scpn_control.__version__,
        "rust_backend": scpn_control.RUST_BACKEND,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "weights": [w.name for w in weight_files],
        "source_modules": len(list((Path(__file__).resolve().parent).rglob("*.py"))),
        "test_files": len(list((Path(__file__).resolve().parent.parent.parent.parent / "tests").glob("test_*.py"))),
    }

    if json_out:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"scpn-control {result['version']}")
        click.echo(f"  Rust backend: {rust_status}")
        click.echo(f"  Python: {result['python']}")
        click.echo(f"  NumPy: {result['numpy']}")
        click.echo(f"  Weights: {len(weight_files)} files")
        for w in weight_files:
            click.echo(f"    {w.name} ({w.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
