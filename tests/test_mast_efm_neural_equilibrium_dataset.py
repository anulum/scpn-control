# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural-equilibrium dataset tests

"""Exercise actual public producer/NPZ/report paths with retained engineering format fixtures, never physical MAST evidence."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from zipfile import ZipFile

import numpy as np
import pytest

from validation.build_mast_efm_neural_equilibrium_dataset import (
    DATASET_SCHEMA,
    DatasetInput,
    build_dataset,
    build_feature_matrix,
    main,
    parse_args,
    sha256_file,
    sha256_json,
    validate_dataset_report,
    validate_feature_matrix,
    write_report,
)


def _write_reference(path: Path, shot_id: int, *, n: int = 2, lcfs_points: int = 4) -> None:
    """Retain original converter-shaped engineering arrays; this declares no authentic public MAST source."""
    path.parent.mkdir(parents=True, exist_ok=True)
    psirz = np.full((n, 3, 4), float(shot_id), dtype=np.float64)
    lcfs_r = np.linspace(0.5, 0.9, lcfs_points)
    lcfs_z = np.linspace(-0.2, 0.2, lcfs_points)
    np.savez_compressed(
        path,
        time_s=np.arange(n, dtype=np.float64),
        r_grid_m=np.linspace(0.4, 1.0, 4),
        z_grid_m=np.linspace(-0.2, 0.2, 3),
        psirz_Wb_per_rad=psirz,
        psirz_valid_mask=np.ones_like(psirz, dtype=bool),
        psi_axis_Wb_per_rad=np.full(n, 0.1),
        psi_boundary_Wb_per_rad=np.full(n, 1.1),
        Ip_MA=np.linspace(0.8, 0.9, n),
        Bt_T=np.linspace(0.5, 0.6, n),
        ffprime_rms_T_rad=np.linspace(2.0, 4.0, n),
        pprime_Pa_per_Wb_rad=np.ones((n, 5)),
        pprime_valid_mask=np.ones((n, 5), dtype=bool),
        q_profile=np.full((n, 5), 2.5),
        q_profile_valid_mask=np.ones((n, 5), dtype=bool),
        lcfs_r_m=np.tile(lcfs_r, (n, 1)),
        lcfs_z_m=np.tile(lcfs_z, (n, 1)),
        lcfs_valid_mask=np.ones((n, lcfs_points), dtype=bool),
        magnetic_axis_r_m=np.full(n, 0.7),
        magnetic_axis_z_m=np.zeros(n),
        shot_id=np.full(n, shot_id),
    )


def test_build_dataset_writes_supervised_npz_and_compact_report(tmp_path: Path) -> None:
    """Run actual selected-byte producer and inspect original numerical fixture values, ragged shape and split custody."""
    storage_root = tmp_path / "storage"
    ref_a = storage_root / "converted/neural_equilibrium_reference/mast_efm_shot_1_reference.npz"
    ref_b = storage_root / "converted/neural_equilibrium_reference/mast_efm_shot_2_reference.npz"
    ref_c = storage_root / "converted/neural_equilibrium_reference/mast_efm_shot_3_reference.npz"
    _write_reference(ref_a, 1, lcfs_points=3)
    _write_reference(ref_b, 2, lcfs_points=4)
    _write_reference(ref_c, 3, lcfs_points=5)
    candidate = storage_root / "converted/neural_equilibrium_reference/candidate.json"
    candidate.write_text(json.dumps(_candidate_payload([(1, ref_a), (2, ref_b), (3, ref_c)])), encoding="utf-8")
    output_npz = storage_root / "processed/neural_equilibrium/mast_efm_supervised_dataset.npz"

    report = build_dataset(
        DatasetInput(
            candidate_report=candidate,
            storage_root=storage_root,
            output_npz=output_npz,
            train_shots=(1,),
            validation_shots=(2,),
            test_shots=(3,),
        )
    )

    assert report["schema_version"] == DATASET_SCHEMA
    assert report["status"] == "blocked"
    assert report["equilibria_count"] == 6
    assert report["split_counts"] == {"train": 2, "validation": 2, "test": 2}
    assert report["fallback_features"] == []
    assert report["feature_source_policy"]["Ip_MA"]["source_key"] == "Ip_MA"
    assert report["feature_source_policy"]["Bt_T"]["source_key"] == "Bt_T"
    assert report["feature_source_policy"]["ffprime_scale"]["source_key"] == "ffprime_rms_T_rad"
    assert report["ragged_target_policy"]["max_lcfs_points"] == 5
    assert report["dataset_path"] == "processed/neural_equilibrium/mast_efm_supervised_dataset.npz"
    assert len(report["dataset_sha256"]) == 64
    assert output_npz.exists()
    with np.load(output_npz, allow_pickle=False) as payload:
        assert payload["features"].shape == (6, 12)
        assert np.allclose(payload["features"][:2, 0], [0.8, 0.9])
        assert np.allclose(payload["features"][:2, 1], [0.5, 0.6])
        assert np.all(payload["features"][:, 5] > 0.0)
        assert payload["psirz_Wb_per_rad"].shape == (6, 3, 4)
        assert payload["lcfs_r_m"].shape == (6, 5)
        assert payload["lcfs_point_count"].tolist() == [3, 3, 4, 4, 5, 5]
        assert np.isnan(payload["lcfs_r_m"][0, 3])
        assert not bool(payload["lcfs_valid_mask"][0, 3])
        assert payload["split"].tolist() == ["train", "train", "validation", "validation", "test", "test"]


def test_write_report_records_split_and_admission_boundary(tmp_path: Path) -> None:
    """Persist a complete actual producer declaration and inspect renderable split/source/admission sections."""
    report = build_dataset(_format_inputs(tmp_path))
    report["blocked_reason"] = "predictive claims remain blocked"
    _reseal(report)

    write_report(report, tmp_path / "dataset.json", tmp_path / "dataset.md")

    markdown = (tmp_path / "dataset.md").read_text(encoding="utf-8")
    assert "MAST EFM Neural-Equilibrium Supervised Dataset" in markdown
    assert "train=2, validation=2, test=2" in markdown
    assert "Maximum LCFS points: 5" in markdown
    assert "predictive claims remain blocked" in markdown
    assert "Fallback features: none" in markdown
    assert "`ffprime_scale` from `ffprime_rms_T_rad`" in markdown


def _reseal(payload: dict[str, Any]) -> None:
    """Independently bind the documented null-field digest for engineering schema declarations."""
    payload["payload_sha256"] = hashlib.sha256(
        json.dumps({**payload, "payload_sha256": None}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _candidate_payload(references: list[tuple[int, Path]]) -> dict[str, Any]:
    """Declare actual converter schema/count/grid/hash records around original format arrays, without physical authenticity."""
    from validation.convert_mast_efm_neural_equilibrium_reference import CANDIDATE_SCHEMA

    shots = []
    total = 0
    for shot_id, path in references:
        with np.load(path, allow_pickle=False) as data:
            count = data["time_s"].shape[0]
            total += count
            shots.append(
                {
                    "shot_id": shot_id,
                    "output_path": path.as_posix(),
                    "source_path": "schema_contract_fixture",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "selected_time_count": count,
                    "grid_shape": list(data["psirz_Wb_per_rad"].shape[1:]),
                    "lcfs_points": data["lcfs_r_m"].shape[1],
                    "status": "reference_candidate",
                }
            )
    payload: dict[str, Any] = {
        "schema_version": CANDIDATE_SCHEMA,
        "status": "pass",
        "source": "documented_public_reference",
        "fixture_kind": "engineering_schema_contract",
        "reference_dataset_id": "mast-efm-test",
        "reference_equilibria_count": total,
        "target_schema_status": "reference_only_no_prediction_metrics",
        "admission_ready": False,
        "errors": [],
        "shots": shots,
    }
    _reseal(payload)
    return payload


def _format_inputs(tmp_path: Path) -> DatasetInput:
    """Select actual original format references and a fully bound converter declaration in isolated storage."""
    storage = tmp_path / "storage"
    references = [(shot, storage / f"converted/shot_{shot}.npz") for shot in (1, 2, 3)]
    for shot, path in references:
        _write_reference(path, shot, lcfs_points=shot + 2)
    candidate = storage / "converted/candidate.json"
    candidate.write_text(json.dumps(_candidate_payload(references)))
    return DatasetInput(candidate, storage, storage / "processed/dataset.npz", (1,), (2,), (3,))


def _rebind_candidate(inputs: DatasetInput) -> None:
    """Update selected engineering input byte/count declarations without changing any canonical source evidence."""
    payload = json.loads(inputs.candidate_report.read_text())
    for shot in payload["shots"]:
        path = Path(shot["output_path"])
        shot["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _reseal(payload)
    inputs.candidate_report.write_text(json.dumps(payload))


def test_actual_normalised_ragged_output_reaches_trainer(tmp_path: Path) -> None:
    """Real builder/reader/planner/trainer chain preserves descending-grid coordinates and compact valid LCFS observations."""
    from validation.plan_neural_equilibrium_training_campaign import CampaignInputs, build_plan
    from validation.train_mast_efm_neural_equilibrium import TrainingInputs, build_training_report

    inputs = _format_inputs(tmp_path)
    candidate = json.loads(inputs.candidate_report.read_text())
    for shot in candidate["shots"]:
        path = Path(shot["output_path"])
        with np.load(path, allow_pickle=False) as f:
            data = {key: f[key] for key in f.files}
        data["r_grid_m"] = data["r_grid_m"][::-1].copy()
        data["z_grid_m"] = data["z_grid_m"][::-1].copy()
        data["psirz_Wb_per_rad"] = data["psirz_Wb_per_rad"][:, ::-1, ::-1].copy()
        data["psirz_valid_mask"] = data["psirz_valid_mask"][:, ::-1, ::-1].copy()
        data["lcfs_valid_mask"][0, 1] = False
        data["lcfs_r_m"][0, 1] = np.nan
        data["lcfs_z_m"][0, 1] = np.nan
        np.savez_compressed(path, **data)
    _rebind_candidate(inputs)
    report = build_dataset(inputs)
    assert validate_dataset_report(report) is report
    with np.load(inputs.output_npz, allow_pickle=False) as data:
        assert np.all(np.diff(data["r_grid_m"]) > 0) and np.all(np.diff(data["z_grid_m"]) > 0)
        assert data["lcfs_point_count"].tolist() == [2, 3, 3, 4, 4, 5]
        np.testing.assert_allclose(data["lcfs_r_m"][0, :2], [0.5, 0.9])
        assert not data["lcfs_valid_mask"][0, 2:].any()
    report_path = tmp_path / "dataset.json"
    write_report(report, report_path, tmp_path / "dataset.md")
    root = Path(__file__).resolve().parents[1]
    plan = build_plan(CampaignInputs(report_path, inputs.storage_root, root / "validation/reference_data/qlknn"))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    weights = tmp_path / "uncreated-weights.npz"
    launch = build_training_report(TrainingInputs(report_path, plan_path, inputs.output_npz, weights))
    assert launch["dataset_metadata"]["equilibria_count"] == 6 and launch["admission_ready"] is False
    assert not weights.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_report", "bad"),
        ("storage_root", Path("bad" + chr(0))),
        ("output_npz", Path("bad.bin")),
        ("train_shots", []),
        ("validation_shots", ()),
        ("test_shots", (True,)),
        ("train_shots", (1, 1)),
        ("train_shots", (2,)),
        ("train_shots", (0,)),
        ("test_shots", ("3",)),
    ],
)
def test_public_dataset_controls_refuse_invalid_partitions(tmp_path: Path, field: str, value: Any) -> None:
    """Actual frozen controls refuse coerced paths/splits before input reads."""
    with pytest.raises(ValueError):
        replace(_format_inputs(tmp_path), **{field: value})


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("schema", "supported schema"),
        ("status", "pass conversion"),
        ("admission", "blocked predictive"),
        ("target_status", "reference-only"),
        ("source", "public reference"),
        ("errors", "conversion errors"),
        ("identity", "trimmed"),
        ("digest", "lowercase SHA"),
        ("stale", "does not match"),
        ("shots", "nonempty shots"),
        ("shot_type", "shot must"),
        ("id", "positive integer"),
        ("grid", "positive dimensions"),
        ("grid_count", "positive integer"),
        ("width", "positive integer"),
        ("shot_status", "reference_candidate"),
        ("sha", "lowercase SHA"),
        ("partition", "configured shot"),
        ("count", "match shot counts"),
        ("escape", "within storage_root"),
        ("relative_traversal", "safe local"),
        ("duplicate", "distinct configured"),
    ],
)
def test_actual_public_candidate_contract_refusals(tmp_path: Path, kind: str, message: str) -> None:
    """Resealed actual converter declarations cannot bypass candidate/policy/split/path contracts in the public producer."""
    inputs = _format_inputs(tmp_path)
    report = json.loads(inputs.candidate_report.read_text())
    if kind == "schema":
        report["schema_version"] = "old"
    elif kind == "status":
        report["status"] = "fail"
    elif kind == "admission":
        report["admission_ready"] = True
    elif kind == "target_status":
        report["target_schema_status"] = "prediction"
    elif kind == "source":
        report["source"] = "unknown"
    elif kind == "errors":
        report["errors"] = ["unconverted"]
    elif kind == "identity":
        report["reference_dataset_id"] = " bad "
    elif kind == "digest":
        report["payload_sha256"] = "g" * 64
    elif kind == "stale":
        report["reference_dataset_id"] = "changed"
    elif kind == "shots":
        report["shots"] = []
    elif kind == "shot_type":
        report["shots"][0] = []
    elif kind == "id":
        report["shots"][0]["shot_id"] = True
    elif kind == "grid":
        report["shots"][0]["grid_shape"] = [3]
    elif kind == "grid_count":
        report["shots"][0]["grid_shape"] = [3, False]
    elif kind == "width":
        report["shots"][0]["lcfs_points"] = 0
    elif kind == "shot_status":
        report["shots"][0]["status"] = "executed"
    elif kind == "sha":
        report["shots"][0]["sha256"] = "Z" * 64
    elif kind == "partition":
        report["shots"][0]["shot_id"] = 4
    elif kind == "count":
        report["reference_equilibria_count"] = 7
    elif kind == "escape":
        report["shots"][0]["output_path"] = (tmp_path / "outside.npz").as_posix()
    elif kind == "relative_traversal":
        report["shots"][0]["output_path"] = "../outside.npz"
    else:
        report["shots"][0]["shot_id"] = 2
    if kind not in {"digest", "stale"}:
        _reseal(report)
    inputs.candidate_report.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=message):
        build_dataset(inputs)
    assert not inputs.output_npz.exists()


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "bad_json",
        "root",
        "duplicate_key",
        "nonfinite",
        "overflow",
        "invalid_archive",
        "npy",
        "object",
        "sha",
        "shot",
        "time",
        "axis",
        "flux",
        "mask",
        "nonfinite_valid",
        "profile",
        "lcfs",
        "empty_lcfs",
        "grid",
        "grid_disagreement",
        "profile_disagreement",
        "output_alias",
    ],
)
def test_actual_public_reference_and_io_refusals(tmp_path: Path, kind: str) -> None:
    """Actual files/decoded source arrays fail through full public build, with no substituted kernels or private production calls."""
    inputs = _format_inputs(tmp_path)
    report = json.loads(inputs.candidate_report.read_text())
    selected = Path(report["shots"][0]["output_path"])
    if kind == "missing":
        inputs.candidate_report.unlink()
    elif kind == "bad_json":
        inputs.candidate_report.write_text("{")
    elif kind == "root":
        inputs.candidate_report.write_text("[]")
    elif kind == "duplicate_key":
        inputs.candidate_report.write_text('{"a":1,"a":2}')
    elif kind == "nonfinite":
        inputs.candidate_report.write_text('{"number":NaN}')
    elif kind == "overflow":
        inputs.candidate_report.write_text('{"number":1e400}')
    elif kind == "invalid_archive":
        selected.write_bytes(b"invalid actual NPZ")
        _rebind_candidate(inputs)
    elif kind == "npy":
        with selected.open("wb") as f:
            np.save(f, np.arange(2), allow_pickle=False)
        _rebind_candidate(inputs)
    elif kind == "sha":
        selected.write_bytes(b"changed selected source")
    elif kind == "output_alias":
        inputs = replace(inputs, output_npz=selected)
    else:
        with np.load(selected, allow_pickle=False) as f:
            data = {key: f[key] for key in f.files}
        if kind == "object":
            data["q_profile"] = data["q_profile"].astype(object)
        elif kind == "shot":
            data["shot_id"][:] = 2
        elif kind == "time":
            data["time_s"][1] = 0
        elif kind == "axis":
            data["magnetic_axis_r_m"][0] = np.inf
        elif kind == "flux":
            data["psirz_Wb_per_rad"] = data["psirz_Wb_per_rad"][:, 0, :]
        elif kind == "mask":
            data["q_profile_valid_mask"] = data["q_profile_valid_mask"].astype(str)
        elif kind == "nonfinite_valid":
            data["q_profile"][0, 0] = np.nan
        elif kind == "profile":
            data["q_profile"] = data["q_profile"][:, 0]
        elif kind == "lcfs":
            data["lcfs_z_m"] = data["lcfs_z_m"][:, :1]
        elif kind == "empty_lcfs":
            data["lcfs_valid_mask"][0, :] = False
        elif kind == "grid":
            data["r_grid_m"][0] = data["r_grid_m"][1]
        elif kind == "grid_disagreement":
            data["r_grid_m"] += 0.1
        else:
            data["q_profile"] = data["q_profile"][:, :1]
            data["q_profile_valid_mask"] = data["q_profile_valid_mask"][:, :1]
        np.savez_compressed(selected, **data)
        _rebind_candidate(inputs)
    with pytest.raises(ValueError):
        build_dataset(inputs)
    if kind != "output_alias":
        assert not inputs.output_npz.exists()


@pytest.fixture
def produced_report(tmp_path: Path) -> dict[str, Any]:
    """Produce a complete real format report through the public builder for declaration/writer regressions."""
    return build_dataset(_format_inputs(tmp_path))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "executed"),
        ("admission_ready", True),
        ("strict_artefact_emitted", True),
        ("source", "unknown"),
        ("feature_names", []),
        ("target_keys", []),
        ("fallback_features", ["unknown"]),
        ("candidate_payload_sha256", None),
        (
            "ragged_target_policy",
            {"keys": ["bad"], "point_count_key": "bad", "padding": "NaN False", "max_lcfs_points": 5},
        ),
        (
            "ragged_target_policy",
            {
                "keys": ["lcfs_r_m", "lcfs_z_m", "lcfs_valid_mask"],
                "point_count_key": "lcfs_point_count",
                "padding": "bad",
                "max_lcfs_points": 5,
            },
        ),
        ("reference_paths", None),
        ("reference_paths", []),
        ("shots", [[]]),
        ("split_policy", []),
        ("next_processing_steps", []),
        ("feature_source_policy", []),
    ],
)
def test_public_dataset_report_refuses_invalid_declarations(
    produced_report: dict[str, Any], field: str, value: Any
) -> None:
    """Resealing actual producer output cannot promote admission or bypass supported target/split/source declarations."""
    report = json.loads(json.dumps(produced_report))
    report[field] = value
    _reseal(report)
    with pytest.raises(ValueError):
        validate_dataset_report(report)


@pytest.mark.parametrize("kind", ["stale", "nonfinite", "nonjson", "root", "same", "symlink", "hardlink", "second_io"])
def test_actual_dataset_writer_refusals(produced_report: dict[str, Any], tmp_path: Path, kind: str) -> None:
    """Actual public writer validates before persistence and exposes alias/second-file failure honestly."""
    report = json.loads(json.dumps(produced_report))
    json_out = tmp_path / "out.json"
    md_out = tmp_path / "out.md"
    if kind == "stale":
        report["reference_dataset_id"] = "changed"
    elif kind == "nonfinite":
        report["extra"] = float("nan")
    elif kind == "nonjson":
        report["extra"] = {1, 2}
    elif kind == "root":
        report = cast(Any, [])
    elif kind == "same":
        md_out = json_out
    elif kind == "symlink":
        md_out.symlink_to(json_out)
    elif kind == "hardlink":
        json_out.write_text("custody original")
        os.link(json_out, md_out)
    else:
        md_out.mkdir()
    before = json_out.read_bytes() if json_out.exists() else None
    with pytest.raises(ValueError):
        write_report(report, json_out, md_out)
    if kind == "second_io":
        assert json.loads(json_out.read_text()) == produced_report
    elif before is None:
        assert not json_out.exists()
    else:
        assert json_out.read_bytes() == before


def _cli_args(inputs: DatasetInput, tmp_path: Path) -> list[str]:
    """Select actual isolated candidate/tensor/report outputs for the real public CLI."""
    return [
        "--candidate-report",
        str(inputs.candidate_report),
        "--storage-root",
        str(inputs.storage_root),
        "--output-npz",
        str(inputs.output_npz),
        "--json-out",
        str(tmp_path / "launch.json"),
        "--report-out",
        str(tmp_path / "launch.md"),
        "--train-shots",
        "1",
        "--validation-shots",
        "2",
        "--test-shots",
        "3",
    ]


def test_actual_public_and_cold_cli(tmp_path: Path) -> None:
    """Exercise main and a foreign-cwd actual process on retained real file format, then inspect declared blocked output."""
    inputs = _format_inputs(tmp_path)
    assert main(_cli_args(inputs, tmp_path)) == 0
    selected = tmp_path / "cold"
    selected.mkdir()
    inputs = _format_inputs(selected)
    script = Path(__file__).resolve().parents[1] / "validation/build_mast_efm_neural_equilibrium_dataset.py"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, str(script), *_cli_args(inputs, selected)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0 and "Traceback" not in proc.stderr
    assert json.loads((selected / "launch.json").read_text())["admission_ready"] is False


@pytest.mark.parametrize("kind", ["missing", "alias", "usage", "help", "bad_shots", "zero_shot", "duplicate_shot"])
def test_actual_cli_refusals(tmp_path: Path, kind: str, capsys: pytest.CaptureFixture[str]) -> None:
    """Actual public CLI keeps domain1/help0/usage2 and prevents selected-input aliases before data persistence."""
    inputs = _format_inputs(tmp_path)
    args = _cli_args(inputs, tmp_path)
    if kind == "missing":
        inputs.candidate_report.unlink()
    elif kind == "alias":
        args[args.index("--json-out") + 1] = str(inputs.candidate_report)
    elif kind == "usage":
        with pytest.raises(SystemExit) as e:
            main(["--unknown"])
        assert e.value.code == 2
        return
    elif kind == "help":
        with pytest.raises(SystemExit) as e:
            main(["--help"])
        assert e.value.code == 0
        return
    elif kind == "bad_shots":
        args[args.index("--train-shots") + 1] = "bad"
    elif kind == "zero_shot":
        args[args.index("--train-shots") + 1] = "0"
    else:
        args[args.index("--train-shots") + 1] = "1,1"
    if kind in {"bad_shots", "zero_shot", "duplicate_shot"}:
        with pytest.raises(SystemExit) as e:
            parse_args(args)
        assert e.value.code == 2
    else:
        assert main(args) == 1 and "FAIL:" in capsys.readouterr().err
        assert not inputs.output_npz.exists()


def _feature_reference(tmp_path: Path) -> dict[str, np.ndarray[Any, Any]]:
    """Read actual retained converter-shaped input arrays for the public feature API's documented broadcast/fallback contracts."""
    inputs = _format_inputs(tmp_path)
    shot = json.loads(inputs.candidate_report.read_text())["shots"][0]
    with np.load(shot["output_path"], allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


@pytest.mark.parametrize(
    "kind",
    [
        "missing_defaults",
        "broadcast_profiles",
        "scalar_axis",
        "masked_profiles",
        "degenerate_radius",
        "flat_vertical",
        "missing_masks",
    ],
)
def test_actual_public_feature_defaults_and_geometry(tmp_path: Path, kind: str) -> None:
    """Public feature construction derives finite documented defaults/broadcasts from retained input schemas without claiming physics."""
    data = _feature_reference(tmp_path)
    if kind == "missing_defaults":
        data = {"psirz_Wb_per_rad": data["psirz_Wb_per_rad"]}
    elif kind == "broadcast_profiles":
        for key in (
            "q_profile",
            "q_profile_valid_mask",
            "pprime_Pa_per_Wb_rad",
            "pprime_valid_mask",
            "lcfs_r_m",
            "lcfs_z_m",
            "lcfs_valid_mask",
        ):
            data[key] = data[key][0]
    elif kind == "scalar_axis":
        data["magnetic_axis_r_m"] = np.asarray(0.7)
    elif kind == "masked_profiles":
        data["q_profile_valid_mask"][:] = False
        data["q_profile"][:] = np.nan
        data["pprime_Pa_per_Wb_rad"][:] = 0
    elif kind == "degenerate_radius":
        data["lcfs_r_m"][:] = 0.7
    elif kind == "flat_vertical":
        data["lcfs_z_m"][:] = 0
    else:
        for key in ("pprime_valid_mask", "q_profile_valid_mask", "lcfs_valid_mask"):
            del data[key]
    matrix = build_feature_matrix(data)
    assert matrix.shape == (2, 12) and np.all(np.isfinite(matrix))
    if kind == "missing_defaults":
        np.testing.assert_allclose(matrix[:, 0], 8.0)
        np.testing.assert_allclose(matrix[:, 1], 5.0)
        np.testing.assert_allclose(matrix[:, -1], 4.0)
    elif kind == "masked_profiles":
        np.testing.assert_allclose(matrix[:, -1], 4.0)
        np.testing.assert_allclose(matrix[:, 4], 0.25)
    elif kind == "scalar_axis":
        np.testing.assert_allclose(matrix[:, 2], 0.7)
    elif kind in {"degenerate_radius", "flat_vertical"}:
        np.testing.assert_allclose(matrix[:, 8], 1.7)


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("missing", "must contain"),
        ("psi_rank", "shape"),
        ("psi_dtype", "real numeric"),
        ("scalar_count", "per-equilibrium"),
        ("q_rows", "row count"),
        ("q_mask", "shape"),
        ("pprime_rows", "row count"),
        ("pprime_mask", "shape"),
        ("rms_count", "row count"),
        ("rms_invalid", "positive"),
        ("scalar_source_count", "row count"),
        ("scalar_source_invalid", "non-finite"),
        ("lcfs_rows", "align"),
        ("lcfs_mask", "shape"),
        ("mask_type", "boolean"),
    ],
)
def test_actual_public_feature_contract_refusals(tmp_path: Path, kind: str, message: str) -> None:
    """Actual public feature API refuses malformed source/broadcast/mask declarations without testing private arithmetic."""
    data = _feature_reference(tmp_path)
    if kind == "missing":
        del data["psirz_Wb_per_rad"]
    elif kind == "psi_rank":
        data["psirz_Wb_per_rad"] = data["psirz_Wb_per_rad"][:, 0, :]
    elif kind == "psi_dtype":
        data["psirz_Wb_per_rad"] = data["psirz_Wb_per_rad"].astype(str)
    elif kind == "scalar_count":
        data["magnetic_axis_r_m"] = np.asarray([0.7])
    elif kind == "q_rows":
        data["q_profile"] = data["q_profile"][:1]
    elif kind == "q_mask":
        data["q_profile_valid_mask"] = data["q_profile_valid_mask"][:, :1]
    elif kind == "pprime_rows":
        data["pprime_Pa_per_Wb_rad"] = data["pprime_Pa_per_Wb_rad"][:1]
    elif kind == "pprime_mask":
        data["pprime_valid_mask"] = data["pprime_valid_mask"][:, :1]
    elif kind == "rms_count":
        data["ffprime_rms_T_rad"] = np.asarray([2.0])
    elif kind == "rms_invalid":
        data["ffprime_rms_T_rad"][0] = 0
    elif kind == "scalar_source_count":
        data["Ip_MA"] = np.asarray([0.8])
    elif kind == "scalar_source_invalid":
        data["Ip_MA"][0] = np.nan
    elif kind == "lcfs_rows":
        data["lcfs_r_m"] = data["lcfs_r_m"][:1]
    elif kind == "lcfs_mask":
        data["lcfs_valid_mask"] = data["lcfs_valid_mask"][:, :1]
    else:
        data["q_profile_valid_mask"] = data["q_profile_valid_mask"].astype(str)
    with pytest.raises(ValueError, match=message):
        build_feature_matrix(data)


@pytest.mark.parametrize("reference", [True, "2", float("nan"), -1.0, 0.0, 10**400])
def test_public_feature_conditioning_refusals(tmp_path: Path, reference: Any) -> None:
    """Public feature controls refuse type/finite/range errors, including genuine integer overflow."""
    with pytest.raises(ValueError, match="finite and positive"):
        build_feature_matrix(_feature_reference(tmp_path), ffprime_reference=reference)


@pytest.mark.parametrize("kind", ["partial", "all"])
def test_actual_builder_fallback_provenance(tmp_path: Path, kind: str) -> None:
    """Public builder retains explicit fallback declarations when actual selected references omit sourced features."""
    inputs = _format_inputs(tmp_path)
    candidate = json.loads(inputs.candidate_report.read_text())
    shots = candidate["shots"][:1] if kind == "partial" else candidate["shots"]
    for shot in shots:
        path = Path(shot["output_path"])
        with np.load(path, allow_pickle=False) as f:
            data = {key: f[key] for key in f.files}
        for key in ("Ip_MA", "Bt_T", "ffprime_rms_T_rad"):
            del data[key]
        np.savez_compressed(path, **data)
    _rebind_candidate(inputs)
    report = build_dataset(inputs)
    assert report["fallback_features"] == ["Ip_MA", "Bt_T", "ffprime_scale"]
    assert report["feature_source_policy"] == {} and report["admission_ready"] is False


@pytest.mark.parametrize(
    "kind",
    [
        "empty_split",
        "duplicate_split",
        "shot_assignment",
        "reference_grid",
        "counts",
        "next_step",
        "policy_type",
        "stale_hash",
    ],
)
def test_public_dataset_report_binding_refusals(produced_report: dict[str, Any], kind: str) -> None:
    """Actual generated producer metadata refuses resealed shot/reference/policy/count drift."""
    report = json.loads(json.dumps(produced_report))
    if kind == "empty_split":
        report["split_policy"]["train_shots"] = []
    elif kind == "duplicate_split":
        report["split_policy"]["train_shots"] = [2]
    elif kind == "shot_assignment":
        report["shots"][0]["shot_id"] = 2
    elif kind == "reference_grid":
        report["shots"][0]["grid_shape"] = [1, 1]
    elif kind == "counts":
        report["shots"][0]["equilibria_count"] = 3
    elif kind == "next_step":
        report["next_processing_steps"] = [None]
    elif kind == "policy_type":
        report["feature_source_policy"]["Ip_MA"] = []
    else:
        report["payload_sha256"] = "a" * 64
    if kind != "stale_hash":
        _reseal(report)
    with pytest.raises(ValueError):
        validate_dataset_report(report)


@pytest.mark.parametrize("n", [1, 2])
def test_actual_extreme_positive_source_statistics_stay_finite(tmp_path: Path, n: int) -> None:
    """Actual producer preserves complete source provenance and unit normalisation without overflowing finite mean/median values."""
    inputs = _format_inputs(tmp_path)
    payload = json.loads(inputs.candidate_report.read_text())
    references = []
    for shot in payload["shots"]:
        path = Path(shot["output_path"])
        _write_reference(path, shot["shot_id"], n=n, lcfs_points=shot["shot_id"] + 2)
        with np.load(path, allow_pickle=False) as f:
            data = {key: f[key] for key in f.files}
        data["ffprime_rms_T_rad"][:] = 1.0e308
        data["pprime_Pa_per_Wb_rad"][:] = 1.0e308
        np.savez_compressed(path, **data)
        references.append((shot["shot_id"], path))
    inputs.candidate_report.write_text(json.dumps(_candidate_payload(references)))
    report = build_dataset(inputs)
    assert report["fallback_features"] == []
    assert report["feature_source_policy"]["ffprime_scale"]["campaign_reference"] == 1.0e308
    with np.load(inputs.output_npz, allow_pickle=False) as data:
        np.testing.assert_allclose(data["features"][:, 4:6], 1.0)
        assert np.all(np.isfinite(data["features"]))


@pytest.mark.parametrize("kind", ["absent_reference", "missing_target", "nonnumeric_target", "zero_rms"])
def test_actual_reference_admission_edges(tmp_path: Path, kind: str) -> None:
    """Public producer refuses genuinely absent bytes or unsupported real target/source fields."""
    inputs = _format_inputs(tmp_path)
    payload = json.loads(inputs.candidate_report.read_text())
    path = Path(payload["shots"][0]["output_path"])
    if kind == "absent_reference":
        path.unlink()
    else:
        with np.load(path, allow_pickle=False) as f:
            data = {key: f[key] for key in f.files}
        if kind == "missing_target":
            del data["q_profile"]
        elif kind == "nonnumeric_target":
            data["q_profile"] = data["q_profile"].astype(bool)
        else:
            data["ffprime_rms_T_rad"][0] = 0
        np.savez_compressed(path, **data)
        _rebind_candidate(inputs)
    with pytest.raises(ValueError):
        build_dataset(inputs)
    assert not inputs.output_npz.exists()


def test_public_producer_json_hash_refuses_nonjson() -> None:
    """The original public hash API rejects unsupported JSON declarations rather than emitting a noncanonical digest."""
    with pytest.raises(ValueError, match="finite JSON"):
        sha256_json({"unsupported": {1, 2}})


def test_public_report_shot_object_contract(produced_report: dict[str, Any]) -> None:
    """A producer-sized resealed report still requires every shot declaration to be an object."""
    report = json.loads(json.dumps(produced_report))
    report["shots"][0] = []
    _reseal(report)
    with pytest.raises(ValueError, match="shot must be an object"):
        validate_dataset_report(report)


@pytest.mark.parametrize(
    "kind",
    [
        "reference_absolute",
        "reference_drive",
        "reference_traversal",
        "reference_backslash",
        "reference_empty_component",
        "candidate_path",
        "grid_object",
        "grid_count",
        "grid_boolean",
        "grid_equal",
        "grid_reversed",
        "grid_nonfinite",
        "grid_overflow",
        "time_boolean",
        "time_negative",
        "time_reversed",
        "time_equal",
        "single_time_range",
        "policy_units",
        "policy_source",
        "policy_transform",
        "policy_reference_zero",
        "policy_reference_boolean",
        "policy_clip",
    ],
)
def test_public_report_storage_time_grid_and_source_bindings(produced_report: dict[str, Any], kind: str) -> None:
    """Resealed public report metadata retains supported storage/time/grid/source declarations without physical promotion."""
    report = json.loads(json.dumps(produced_report))
    if kind.startswith("reference_"):
        paths = {
            "reference_absolute": "/external.npz",
            "reference_drive": "C:/external.npz",
            "reference_traversal": "../external.npz",
            "reference_backslash": "unsafe" + chr(92) + "external.npz",
            "reference_empty_component": "unsafe//external.npz",
        }
        report["reference_paths"][0] = report["shots"][0]["reference_path"] = paths[kind]
    elif kind == "candidate_path":
        report["candidate_report"] = "./candidate.json"
    elif kind == "grid_object":
        report["r_grid_m"] = []
    elif kind == "grid_count":
        report["r_grid_m"]["count"] = 5
    elif kind == "grid_boolean":
        report["r_grid_m"]["min"] = False
    elif kind == "grid_equal":
        report["r_grid_m"]["max"] = report["r_grid_m"]["min"]
    elif kind == "grid_reversed":
        report["r_grid_m"]["min"] = 99.0
    elif kind == "grid_nonfinite":
        report["r_grid_m"]["min"] = float("inf")
    elif kind == "grid_overflow":
        report["r_grid_m"]["min"] = 10**500
    elif kind == "time_boolean":
        report["shots"][0]["time_start_s"] = False
    elif kind == "time_negative":
        report["shots"][0]["time_start_s"] = -1.0
    elif kind == "time_reversed":
        report["shots"][0]["time_end_s"] = -1.0
    elif kind == "time_equal":
        report["shots"][0]["time_end_s"] = report["shots"][0]["time_start_s"]
    elif kind == "single_time_range":
        report["shots"][0]["equilibria_count"] = 1
        report["split_counts"]["train"] = 1
        report["equilibria_count"] -= 1
    elif kind == "policy_units":
        report["feature_source_policy"]["Ip_MA"]["units"] = "A"
    elif kind == "policy_source":
        report["feature_source_policy"]["Ip_MA"]["source_key"] = "unknown"
    elif kind == "policy_transform":
        report["feature_source_policy"]["Bt_T"]["transform"] = "rescale"
    elif kind == "policy_reference_zero":
        report["feature_source_policy"]["ffprime_scale"]["campaign_reference"] = 0.0
    elif kind == "policy_reference_boolean":
        report["feature_source_policy"]["ffprime_scale"]["campaign_reference"] = True
    else:
        report["feature_source_policy"]["ffprime_scale"]["clip"] = [0.0, 8.0]
    if kind != "grid_nonfinite":
        _reseal(report)
    with pytest.raises(ValueError):
        validate_dataset_report(report)


@pytest.mark.parametrize("key", ["pprime_Pa_per_Wb_rad", "q_profile", "lcfs_r_m", "lcfs_z_m"])
@pytest.mark.parametrize("shape", [(), (2, 3, 4)])
def test_public_features_refuse_unsupported_profile_geometry_rank(
    tmp_path: Path, key: str, shape: tuple[int, ...]
) -> None:
    """Actual feature API refuses scalar/higher-rank profiles and geometry with authored errors instead of raw indexing errors."""
    inputs = _format_inputs(tmp_path)
    # Select the actual candidate's first source rather than assuming a fixture filename.
    selected = Path(json.loads(inputs.candidate_report.read_text())["shots"][0]["output_path"])
    with np.load(selected, allow_pickle=False) as payload:
        data = {name: payload[name] for name in payload.files}
    data[key] = np.ones(shape, dtype=np.float64)
    with pytest.raises(ValueError, match="one or two dimensions"):
        build_feature_matrix(data)


def test_public_features_preserve_partially_unobserved_pressure_rows(tmp_path: Path) -> None:
    """A wholly masked row uses the documented scale default while another real observed row informs the shot median."""
    inputs = _format_inputs(tmp_path)
    selected = Path(json.loads(inputs.candidate_report.read_text())["shots"][0]["output_path"])
    with np.load(selected, allow_pickle=False) as payload:
        data = {name: payload[name] for name in payload.files}
    data["pprime_Pa_per_Wb_rad"][:] = 3.0
    data["pprime_valid_mask"][0] = False
    features = build_feature_matrix(data)
    assert features[:, 4].tolist() == [0.5, 1.5]


def test_public_candidate_refuses_overflowed_json_exponent(tmp_path: Path) -> None:
    """Actual selected candidate JSON refuses exponent overflow before it becomes a nonfinite metadata value."""
    inputs = _format_inputs(tmp_path)
    inputs.candidate_report.write_text('{"unselected":1e999}')
    with pytest.raises(ValueError, match="must be finite"):
        build_dataset(inputs)


def test_public_candidate_accepts_finite_unselected_metadata(tmp_path: Path) -> None:
    """The actual finite JSON decoder accepts a finite optional value while preserving all candidate and byte custody."""
    inputs = _format_inputs(tmp_path)
    candidate = json.loads(inputs.candidate_report.read_text())
    candidate["optional_conversion_metadata"] = {"ratio": 0.125}
    _reseal(candidate)
    inputs.candidate_report.write_text(json.dumps(candidate))
    report = build_dataset(inputs)
    assert report["candidate_payload_sha256"] == candidate["payload_sha256"]
    assert report["status"] == "blocked" and report["admission_ready"] is False


@pytest.mark.parametrize("rebind", [False, True])
def test_public_producer_binds_exact_archive_bytes_even_when_arrays_match(tmp_path: Path, rebind: bool) -> None:
    """A valid NPZ container-only change must update candidate byte custody even when every decoded array is unchanged."""
    inputs = _format_inputs(tmp_path)
    candidate = json.loads(inputs.candidate_report.read_text())
    selected = Path(candidate["shots"][0]["output_path"])
    before = hashlib.sha256(selected.read_bytes()).hexdigest()
    with np.load(selected, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    with ZipFile(selected, "a") as archive:
        archive.comment = b"engineering format container custody regression"
    after = hashlib.sha256(selected.read_bytes()).hexdigest()
    assert after != before
    with np.load(selected, allow_pickle=False) as payload:
        assert set(payload.files) == set(arrays)
        for name in payload.files:
            assert np.array_equal(payload[name], arrays[name], equal_nan=True)
    if not rebind:
        with pytest.raises(ValueError, match="SHA-256 does not match candidate"):
            build_dataset(inputs)
        assert not inputs.output_npz.exists()
    else:
        _rebind_candidate(inputs)
        report = build_dataset(inputs)
        assert report["shots"][0]["reference_sha256"] == after
        assert report["status"] == "blocked" and report["admission_ready"] is False
        with np.load(inputs.output_npz, allow_pickle=False) as payload:
            assert np.allclose(payload["features"][:2, 0], arrays["Ip_MA"])


class NonfiniteNumericAdapter(float):
    """A finite JSON scalar adapter whose explicit numeric conversion returns infinity."""

    def __float__(self) -> float:
        """Expose a nonfinite converted value to test the public mapping's coercion boundary."""
        return float("inf")


def test_public_report_refuses_nonfinite_numeric_adapter(produced_report: dict[str, Any]) -> None:
    """A JSON-encodable scalar must still convert to a finite declared time value in the actual public report API."""
    report = json.loads(json.dumps(produced_report))
    report["shots"][0]["time_start_s"] = NonfiniteNumericAdapter(0.0)
    # Standard JSON encodes the finite underlying scalar, while float(value)
    # uses the adapter. No production or numerical function is replaced.
    assert json.dumps(report["shots"][0]["time_start_s"], allow_nan=False) == "0.0"
    _reseal(report)
    with pytest.raises(ValueError, match="time_start_s must be a finite number"):
        validate_dataset_report(report)


@pytest.mark.parametrize("kind", ["missing", "directory", "nul", "symlink_loop"])
def test_public_streaming_hash_refuses_unreadable_paths(tmp_path: Path, kind: str) -> None:
    """The retained public streaming hash helper reports real selected-file failures as authored ValueError."""
    selected = tmp_path / "selected.npz"
    if kind == "directory":
        selected.mkdir()
    elif kind == "nul":
        selected = Path("invalid" + chr(0))
    elif kind == "symlink_loop":
        selected.symlink_to(selected)
    with pytest.raises(ValueError, match="cannot hash selected dataset bytes"):
        sha256_file(selected)


@pytest.mark.parametrize("row_count", [True, 0, -1, "6"])
def test_public_feature_validator_requires_genuine_positive_rows(tmp_path: Path, row_count: Any) -> None:
    """The advertised public shared validator rejects ambiguous row controls while inspecting actual produced feature bytes."""
    inputs = _format_inputs(tmp_path)
    build_dataset(inputs)
    with np.load(inputs.output_npz, allow_pickle=False) as payload:
        features = payload["features"]
    with pytest.raises(ValueError, match="feature row_count must be a positive integer"):
        validate_feature_matrix(features, row_count=row_count)
