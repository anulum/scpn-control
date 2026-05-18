# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CLI Tests

"""Tests for the scpn-control Click CLI entry point."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from scpn_control.cli import main


@pytest.fixture
def runner():
    return CliRunner()


def test_version(runner):
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output


def test_demo_json_out(runner):
    result = runner.invoke(main, ["demo", "--steps", "10", "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    for key in ("scenario", "steps", "final_state", "final_error", "converged"):
        assert key in data
    assert data["steps"] == 10
    assert isinstance(data["final_state"], float)
    assert isinstance(data["converged"], bool)


def test_demo_pid_text(runner):
    result = runner.invoke(main, ["demo", "--scenario", "pid", "--steps", "5"])
    assert result.exit_code == 0
    assert "Scenario: pid" in result.output
    assert "Steps: 5" in result.output


def test_benchmark_json_out(runner):
    result = runner.invoke(main, ["benchmark", "--n-bench", "100", "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    for key in ("n_bench", "pid_us_per_step", "snn_us_per_step", "speedup_ratio"):
        assert key in data
    assert data["n_bench"] == 100
    assert data["pid_us_per_step"] > 0


def test_validate_json_out(runner):
    result = runner.invoke(main, ["validate", "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "status" in data
    assert data["status"] in ("pass", "fail")
    assert data["data_manifests"]["status"] == "pass"
    assert data["data_manifests"]["total"] >= 3


def test_validate_reports_manifest_gate_failures(runner, tmp_path):
    result = runner.invoke(main, ["validate", "--data-manifest-root", str(tmp_path), "--json-out"])

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["data_manifests"]["status"] == "fail"
    assert data["data_manifests"]["errors"][0]["error"] == "no data manifests found"


def test_validate_command_is_import_clean_in_fresh_process():
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-m", "scpn_control.cli", "validate", "--json-out"],
        check=True,
        cwd=repo_root,
        env={"PYTHONPATH": str(repo_root / "src")},
        capture_output=True,
        text=True,
        timeout=30,
    )

    data = json.loads(completed.stdout)
    assert data["import_clean"] is True
    assert "contaminated_module" not in data


def test_validate_manifest_json_out(runner, tmp_path):
    manifest_path = tmp_path / "real_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_id": "diii-d-163303-control-replay",
                "machine": "DIII-D",
                "shot": "163303",
                "synthetic": False,
                "source": {
                    "kind": "mdsplus",
                    "uri": "mdsplus://DIII-D/163303",
                    "access": "facility-approved",
                },
                "retrieved_at": "2026-05-18T01:20:00Z",
                "checksum_sha256": "b" * 64,
                "licence": "facility data policy",
                "signals": [
                    {
                        "name": "plasma_current",
                        "path": "\\\\IP",
                        "units": "A",
                        "timebase": "s",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(main, ["validate-manifest", str(manifest_path), "--json-out"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data == {
        "dataset_id": "diii-d-163303-control-replay",
        "kind": "real",
        "machine": "DIII-D",
        "shot": "163303",
        "signals": 1,
        "source_kind": "mdsplus",
        "status": "pass",
    }


def test_validate_manifest_rejects_mock_as_real(runner, tmp_path):
    manifest_path = tmp_path / "invalid_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_id": "bad-real-claim",
                "machine": "DIII-D",
                "shot": "999999",
                "synthetic": False,
                "source": {
                    "kind": "mock",
                    "uri": "tests/mock_diiid.py",
                    "access": "repository fixture",
                },
                "retrieved_at": "2026-05-18T01:20:00Z",
                "checksum_sha256": "c" * 64,
                "licence": "repository fixture",
                "signals": [
                    {
                        "name": "normalised_beta",
                        "path": "beta_N",
                        "units": "1",
                        "timebase": "s",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(main, ["validate-manifest", str(manifest_path), "--json-out"])

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert "synthetic or mock source" in data["error"]


def test_validate_manifest_verifies_repository_artifact(runner):
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "validation"
        / "reference_data"
        / "diiid"
        / "manifests"
        / "diiid_hmode_1p5MA.geqdsk.manifest.json"
    )

    result = runner.invoke(main, ["validate-manifest", str(manifest_path), "--verify-artifact", "--json-out"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["artifact_verified"] is True


def test_validate_data_manifests_json_out(runner):
    root = Path(__file__).resolve().parents[1] / "validation" / "reference_data"

    result = runner.invoke(main, ["validate-data-manifests", "--root", str(root), "--json-out"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["status"] == "pass"
    assert data["total"] >= 3
    assert data["artifact_coverage"]["covered"] == 21
    assert data["acquisition_specs"]["total"] >= 1


def test_validate_data_manifests_reports_failures(runner, tmp_path):
    result = runner.invoke(main, ["validate-data-manifests", "--root", str(tmp_path), "--json-out"])

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["errors"][0]["error"] == "no data manifests found"


def test_validate_data_manifests_can_require_real_acquisition(runner):
    root = Path(__file__).resolve().parents[1] / "validation" / "reference_data"

    result = runner.invoke(
        main,
        [
            "validate-data-manifests",
            "--root",
            str(root),
            "--require-real-acquisition",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["acquisition_specs"]["pending"] >= 1
    assert any(error["error"] == "missing acquired MDSplus manifest" for error in data["errors"])


def test_validate_physics_traceability_json_out(runner):
    registry = Path(__file__).resolve().parents[1] / "validation" / "physics_traceability.json"

    result = runner.invoke(main, ["validate-physics-traceability", "--registry", str(registry), "--json-out"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["status"] == "pass"
    assert data["open_fidelity_gaps"] >= 5
    assert data["public_claim_blocked"] >= 5


def test_validate_physics_traceability_reports_failures(runner, tmp_path):
    registry = tmp_path / "physics_traceability.json"
    registry.write_text(json.dumps({"schema_version": "1.0", "entries": []}), encoding="utf-8")

    result = runner.invoke(main, ["validate-physics-traceability", "--registry", str(registry), "--json-out"])

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    fields = {error["field"] for error in data["errors"]}
    assert "entries" in fields
    assert "spdx_license_id" in fields


def test_validate_gk_crosscode_requires_external_runs(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "validate-gk-crosscode",
            "--evidence-root",
            str(tmp_path),
            "--require-external-runs",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["errors"][0]["error"] == "no real external GK evidence reports found"


def test_validate_gk_geometry_reference_json_out(runner):
    reference_path = (
        Path(__file__).resolve().parents[1]
        / "validation"
        / "reference_data"
        / "gk_geometry"
        / "miller_reference_cases.json"
    )

    result = runner.invoke(
        main, ["validate-gk-geometry-reference", "--reference-path", str(reference_path), "--json-out"]
    )

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["status"] == "pass"
    assert data["cases"] == 3


def test_validate_gk_species_reference_json_out(runner):
    reference_path = (
        Path(__file__).resolve().parents[1]
        / "validation"
        / "reference_data"
        / "gk_species"
        / "species_collision_reference_cases.json"
    )

    result = runner.invoke(
        main, ["validate-gk-species-reference", "--reference-path", str(reference_path), "--json-out"]
    )

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["status"] == "pass"
    assert data["cases"] == 4


def test_validate_jax_gk_parity_requires_artifacts(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "validate-jax-gk-parity",
            "--artifact-root",
            str(tmp_path),
            "--require-parity-artifacts",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["errors"][0]["error"] == "no JAX GK parity artifacts found"


def test_validate_gk_ood_calibration_requires_artifacts(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "validate-gk-ood-calibration",
            "--artifact-root",
            str(tmp_path),
            "--require-campaign-artifacts",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["errors"][0]["error"] == "no GK OOD calibration artifacts found"


def test_validate_gk_interface_artifacts_requires_artifacts(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "validate-gk-interface-artifacts",
            "--artifact-root",
            str(tmp_path),
            "--require-interface-artifacts",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["errors"][0]["error"] == "no external GK interface artifacts found"


def test_hil_test_nonexistent_dir(runner):
    result = runner.invoke(main, ["hil-test", "--shots-dir", "nonexistent_dir_12345", "--json-out"])
    assert result.exit_code != 0


def test_live_help(runner):
    result = runner.invoke(main, ["live", "--help"])
    assert result.exit_code == 0
    assert "--port" in result.output
    assert "--zeta" in result.output


def test_benchmark_text_output(runner):
    result = runner.invoke(main, ["benchmark", "--n-bench", "50"])
    assert result.exit_code == 0
    assert "PID:" in result.output
    assert "SNN:" in result.output
    assert "Ratio:" in result.output


def test_validate_text_output(runner):
    result = runner.invoke(main, ["validate"])
    assert result.exit_code == 0
    assert "Transport solver:" in result.output
    assert "Import clean:" in result.output
    assert "Status:" in result.output


def test_info_json_out(runner):
    result = runner.invoke(main, ["info", "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "version" in data
    assert "python" in data
    assert "numpy" in data
    assert "rust_backend" in data


def test_info_text_output(runner):
    result = runner.invoke(main, ["info"])
    assert result.exit_code == 0
    assert "scpn-control" in result.output
    assert "Rust backend:" in result.output
    assert "Python:" in result.output
    assert "NumPy:" in result.output


def test_hil_test_with_mock_shots(runner, tmp_path):
    """Cover hil-test loading NPZ files (lines 222-243)."""
    rng = np.random.default_rng(0)
    for name in ("shot_001", "shot_002"):
        np.savez(
            tmp_path / f"{name}.npz",
            psi=rng.standard_normal((10, 10)),
            ip=np.array([15e6]),
        )
    result = runner.invoke(main, ["hil-test", "--shots-dir", str(tmp_path), "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["n_shots"] == 2
    assert len(data["shots"]) == 2
    assert data["shots"][0]["status"] == "loaded"
    assert "psi" in data["shots"][0]["keys"]


def test_hil_test_text_output(runner, tmp_path):
    np.savez(tmp_path / "s42.npz", plasma=np.zeros(5))
    result = runner.invoke(main, ["hil-test", "--shots-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "1 shots" in result.output
    assert "s42" in result.output


def test_hil_test_empty_dir(runner, tmp_path):
    result = runner.invoke(main, ["hil-test", "--shots-dir", str(tmp_path), "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["n_shots"] == 0


def test_info_json_includes_weights_list(runner):
    result = runner.invoke(main, ["info", "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data["weights"], list)


def test_demo_combined_json(runner):
    result = runner.invoke(main, ["demo", "--scenario", "combined", "--steps", "5", "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["scenario"] == "combined"


def test_demo_snn_text(runner):
    result = runner.invoke(main, ["demo", "--scenario", "snn", "--steps", "3"])
    assert result.exit_code == 0
    assert "Scenario: snn" in result.output


def test_validate_rmse_command(runner, tmp_path, monkeypatch):
    """Cover cli.py lines 238-248: validate-rmse imports rmse_dashboard."""
    called = {}

    def fake_rmse_main():
        import json as _json
        from pathlib import Path as _P

        out = _P(tmp_path) / "rmse_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_json.dumps({"status": "pass"}))
        called["invoked"] = True
        return 0

    monkeypatch.setitem(
        __import__("sys").modules,
        "validation.rmse_dashboard",
        type("M", (), {"main": staticmethod(fake_rmse_main)})(),
    )

    result = runner.invoke(
        main,
        [
            "validate-rmse",
            "--json-out",
            "--output-json",
            str(tmp_path / "rmse_report.json"),
            "--output-md",
            str(tmp_path / "rmse_report.md"),
        ],
    )
    assert called.get("invoked")
    assert "pass" in result.output
