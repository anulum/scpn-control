# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Density reference validation tests

"""Exercise declaration-schema behavior, not real density or facility validation.

The inherited fixture supplies illustrative self-declared metadata/errors only;
its source labels and DOI do not establish an authenticated reference corpus.
All checks execute the actual public reader or registered/direct command.
"""

from __future__ import annotations

import doctest
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from click.testing import CliRunner

import validation.validate_density_reference as density_module
from scpn_control.cli import main as control_cli
from validation.validate_density_reference import main, validate_density_reference


def _valid_density_reference_artifact() -> dict[str, object]:
    """Construct the inherited schema-only declaration fixture; its values are not model-run/reference evidence."""
    return {
        "schema_version": "1.0",
        "source": "documented_public_reference",
        "reference_doi": "10.1088/0029-5515/39/12/301",
        "model_id": "density-control-particle-source",
        "model_version": "0.19.0",
        "reference_dataset_id": "density-fuelling-reference-2026-05-20",
        "reference_artifact_sha256": "5" * 64,
        "reference_case_count": 8,
        "executed_at": "2026-05-20T05:00:00Z",
        "radial_grid": {"n_rho": 64, "major_radius_m": 6.2, "minor_radius_m": 2.0},
        "actuator_metadata": {
            "gas_puff_rate_particles_s": 3.0e21,
            "pellet_radius_mm": 2.0,
            "pellet_speed_m_s": 500.0,
            "nbi_energy_keV": 80.0,
            "nbi_power_MW": 5.0,
            "cryopump_speed_m3_s": 8.0,
            "recycling_coefficient": 0.97,
        },
        "units": {
            "density": "m^-3",
            "particle_rate": "s^-1",
            "radius": "m",
            "diffusivity": "m^2/s",
            "pinch_velocity": "m/s",
            "time": "s",
            "greenwald_fraction": "1",
        },
        "metrics": {
            "pellet_deposition_rmse": 0.025,
            "recycling_source_relative_error": 0.04,
            "greenwald_fraction_abs_error": 0.015,
            "density_profile_relative_error": 0.05,
        },
        "tolerances": {
            "pellet_deposition_rmse": 0.05,
            "recycling_source_relative_error": 0.08,
            "greenwald_fraction_abs_error": 0.03,
            "density_profile_relative_error": 0.1,
        },
    }


def test_strict_density_gate_requires_reference_artifacts(tmp_path: Path) -> None:
    """Strict inspection refuses an empty directory instead of claiming reference presence."""
    report = validate_density_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["reference_artifacts"] == 0
    assert report["errors"][0]["error"] == "no density reference artifacts found"


def test_density_gate_accepts_documented_public_reference(tmp_path: Path) -> None:
    """An illustrative DOI declaration passes shape checks without authenticating a reference or its digest."""
    artifact = tmp_path / "density_public_reference.json"
    artifact.write_text(json.dumps(_valid_density_reference_artifact()), encoding="utf-8")

    report = validate_density_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["reference_artifacts"] == 1
    assert report["entries"][0]["source"] == "documented_public_reference"
    assert report["entries"][0]["reference_case_count"] == 8


def test_density_gate_accepts_measured_fuelling_campaign(tmp_path: Path) -> None:
    """Declared shot/diagnostic strings satisfy the schema without proving a measured campaign exists."""
    payload = _valid_density_reference_artifact()
    payload["source"] = "measured_fuelling_campaign"
    payload.pop("reference_doi")
    payload["shot_id"] = "DIII-D:163303"
    payload["diagnostic_uri"] = "mdsplus://DIII-D/163303/electron_density"
    artifact = tmp_path / "measured_density_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_density_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["entries"][0]["source"] == "measured_fuelling_campaign"


def test_density_gate_accepts_external_integrated_modelling(tmp_path: Path) -> None:
    """An allowed code label/scoped URI passes lexical checks without executing the code or reading its artifact."""
    payload = _valid_density_reference_artifact()
    payload["source"] = "external_integrated_modelling"
    payload.pop("reference_doi")
    payload["external_code"] = "ASTRA"
    payload["reference_artifact_uri"] = "file:///validation/reports/density/astra_fuelling_profile.nc"
    artifact = tmp_path / "external_density_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_density_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["entries"][0]["source"] == "external_integrated_modelling"


def test_density_gate_rejects_unscoped_external_artifact_uri(tmp_path: Path) -> None:
    """A local system path cannot satisfy the existing admitted reference URI policy."""
    payload = _valid_density_reference_artifact()
    payload["source"] = "external_integrated_modelling"
    payload.pop("reference_doi")
    payload["external_code"] = "ASTRA"
    payload["reference_artifact_uri"] = "file:///etc/passwd"
    artifact = tmp_path / "bad_external_density_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_density_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "reference_artifact_uri"
    assert "validation/reports" in report["errors"][0]["error"]


def test_density_gate_rejects_synthetic_source(tmp_path: Path) -> None:
    """The reference schema refuses the synthetic source label even when other metadata is well formed."""
    payload = _valid_density_reference_artifact()
    payload["source"] = "synthetic"
    artifact = tmp_path / "synthetic_density_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_density_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "source"


def test_density_gate_rejects_metric_outside_tolerance(tmp_path: Path) -> None:
    """A declared pellet error above its declared tolerance makes the entire declaration fail."""
    payload = _valid_density_reference_artifact()
    metrics = cast(dict[str, object], payload["metrics"])
    metrics["pellet_deposition_rmse"] = 0.2
    artifact = tmp_path / "bad_density_metric.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_density_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "pellet_deposition_rmse"


def test_density_gate_rejects_missing_actuator_metadata(tmp_path: Path) -> None:
    """Required pellet-speed metadata cannot be omitted from an accepted declaration."""
    payload = _valid_density_reference_artifact()
    actuator_metadata = cast(dict[str, object], payload["actuator_metadata"])
    actuator_metadata.pop("pellet_speed_m_s")
    artifact = tmp_path / "bad_density_metadata.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_density_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "actuator_metadata"


@pytest.mark.parametrize(
    "changes,field",
    [
        ({"schema_version": "wrong"}, "schema_version"),
        ({"model_id": " "}, "model_id"),
        ({"reference_artifact_sha256": "f" * 64 + "\n"}, "reference_artifact_sha256"),
        ({"source": []}, "source"),
        ({"source": {}}, "source"),
        ({"source": "documented_public_reference", "reference_doi": None}, "reference"),
        ({"source": "measured_fuelling_campaign"}, "shot_id"),
        ({"source": "external_integrated_modelling", "external_code": {}}, "external_code"),
        ({"source": "external_integrated_modelling", "external_code": "unknown"}, "external_code"),
        ({"radial_grid": []}, "radial_grid"),
        ({"radial_grid": {"n_rho": True}}, "radial_grid"),
        ({"radial_grid": {"n_rho": 1}}, "radial_grid"),
        ({"actuator_metadata": None}, "actuator_metadata"),
        ({"units": []}, "units"),
        ({"reference_case_count": False}, "reference_case_count"),
        ({"reference_case_count": 0}, "reference_case_count"),
        ({"reference_case_count": 1.5}, "reference_case_count"),
        ({"metrics": []}, "metrics"),
        ({"tolerances": []}, "tolerances"),
    ],
)
def test_density_declared_schema_errors_are_structured(tmp_path: Path, changes: dict[str, Any], field: str) -> None:
    """Invalid metadata, including unhashable source/code containers, produces findings rather than tracebacks."""
    payload = _valid_density_reference_artifact()
    payload.update(changes)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()
    report = validate_density_reference(path, require_reference_artifacts=True)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    assert field in {finding["field"] for finding in report["errors"]}
    assert path.read_bytes() == before


@pytest.mark.parametrize("value", [True, "0.01", -1.0, 10**1000])
@pytest.mark.parametrize("block", ["metrics", "tolerances"])
def test_density_invalid_declared_numbers_refuse_without_overflow(tmp_path: Path, value: Any, block: str) -> None:
    """Actual decoded field values outside the float contract cannot crash or satisfy a declared comparison."""
    payload = _valid_density_reference_artifact()
    cast(dict[str, object], payload[block])["pellet_deposition_rmse"] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_density_reference(path, require_reference_artifacts=True)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    assert "pellet_deposition_rmse" in {finding["field"] for finding in report["errors"]}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_density_nonfinite_unused_metadata_is_invalid_json(tmp_path: Path, value: float) -> None:
    """The actual parser refuses nonstandard/nonfinite tokens even outside required metadata fields."""
    payload = _valid_density_reference_artifact()
    payload["unused"] = {"number": value}
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_density_reference(path, require_reference_artifacts=True)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    assert report["errors"][0]["field"] == "json"
    assert "numbers must be finite" in report["errors"][0]["error"]


@pytest.mark.parametrize("raw", [b"{", b"\xff", b'{"source": 1, "source": 2}', b'{"unused": 1e9999}'])
def test_density_read_parse_failures_return_findings(tmp_path: Path, raw: bytes) -> None:
    """Actual malformed, undecodable, ambiguous and overflowing files are refused without synthetic parsing mocks."""
    path = tmp_path / "invalid.json"
    path.write_bytes(raw)
    report = validate_density_reference(path, require_reference_artifacts=True)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    assert report["errors"][0]["field"] == "json"


def test_density_non_object_and_unreadable_candidates_fail(tmp_path: Path) -> None:
    """A nonobject JSON root and a real matching directory cannot count as reference declarations."""
    (tmp_path / "list.json").write_text("[]", encoding="utf-8")
    (tmp_path / "directory.json").mkdir()
    report = validate_density_reference(tmp_path, require_reference_artifacts=True)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    assert {error["field"] for error in report["errors"]} == {"root", "json"}


def test_density_optional_empty_and_required_missing_roots(tmp_path: Path) -> None:
    """Optional absence is vacuous metadata success; strict absence cannot become reference presence."""
    missing = tmp_path / "absent"
    optional = validate_density_reference(missing)
    required = validate_density_reference(missing, require_reference_artifacts=True)
    assert optional["status"] == "pass" and optional["reference_artifacts"] == 0
    assert required["status"] == "fail" and required["reference_artifacts"] == 0
    (tmp_path / "ignored.txt").write_text("not an inspected JSON file", encoding="utf-8")
    assert validate_density_reference(tmp_path)["reference_artifacts"] == 0


def test_density_inspection_is_sorted_fresh_and_accepts_single_any_suffix(tmp_path: Path) -> None:
    """The public reader inspects current bytes in sorted scope and returns independent report containers."""
    for name in ["z.json", "a.json"]:
        (tmp_path / name).write_text(json.dumps(_valid_density_reference_artifact()), encoding="utf-8")
    first = validate_density_reference(tmp_path)
    assert [Path(entry["path"]).name for entry in first["entries"]] == ["a.json", "z.json"]
    first["entries"].clear()
    assert validate_density_reference(tmp_path)["reference_artifacts"] == 2
    single = tmp_path / "declaration.data"
    single.write_text(json.dumps(_valid_density_reference_artifact()), encoding="utf-8")
    assert validate_density_reference(single)["reference_artifacts"] == 1


@pytest.mark.parametrize("recycling", [-0.1, 1.1])
def test_density_declared_recycling_fraction_bounds(tmp_path: Path, recycling: float) -> None:
    """The literal actuator contract rejects negative and greater-than-one recycling fractions."""
    payload = _valid_density_reference_artifact()
    cast(dict[str, object], payload["actuator_metadata"])["recycling_coefficient"] = recycling
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_density_reference(path)
    assert report["status"] == "fail"
    assert "actuator_metadata" in {finding["field"] for finding in report["errors"]}


def test_density_public_main_writes_actual_report_and_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The real main entry creates report parents, writes its declared result and emits readable findings."""
    source = tmp_path / "invalid.json"
    source.write_text('{"source": []}', encoding="utf-8")
    output = tmp_path / "reports" / "result.json"
    assert main(["--artifact-root", str(source), "--output-json", str(output), "--json-out"]) == 1
    displayed = capsys.readouterr()
    assert json.loads(displayed.out) == json.loads(output.read_bytes())
    assert main(["--artifact-root", str(source)]) == 1
    displayed = capsys.readouterr()
    assert "Density reference: fail" in displayed.out and "ERROR" in displayed.err
    assert main(["--artifact-root", str(tmp_path / "absent")]) == 0
    assert "Density reference: pass" in capsys.readouterr().out


@pytest.mark.parametrize("without_site", [False, True])
def test_density_actual_standalone_cli_refuses_malformed_json(tmp_path: Path, without_site: bool) -> None:
    """Exercise the documented file entry, including stdlib-only mode, without repo import-path assistance."""
    source = tmp_path / "invalid.json"
    source.write_text('{"source": []}', encoding="utf-8")
    argv = [sys.executable] + (["-S"] if without_site else [])
    argv += [str(Path(density_module.__file__)), "--artifact-root", str(source), "--json-out"]
    result = subprocess.run(argv, cwd=tmp_path, env={**os.environ, "PYTHONPATH": ""}, text=True, capture_output=True)
    assert result.returncode == 1 and result.stderr == ""
    report = json.loads(result.stdout)
    assert report["status"] == "fail" and "source" in {finding["field"] for finding in report["errors"]}


def test_density_registered_command_executes_same_public_reader(tmp_path: Path) -> None:
    """Invoke the actual root Click registration and keep malformed metadata as exit-one findings."""
    source = tmp_path / "invalid.json"
    source.write_text('{"source": {}}', encoding="utf-8")
    result = CliRunner().invoke(
        control_cli, ["validate-density-reference", "--artifact-root", str(source), "--json-out"]
    )
    assert result.exit_code == 1
    report = json.loads(result.output)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    assert "source" in {finding["field"] for finding in report["errors"]}


def test_density_native_example_executes_actual_owner_bytes() -> None:
    """Execute the native public example on the actual Python file, rather than a fabricated reference."""
    result = doctest.testmod(density_module)
    assert result.failed == 0 and result.attempted == 2
