# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural equilibrium reference converter tests

"""Exercise the public converter on genuine labelled datasets and consolidated files."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import xarray as xr
from mast_efm_zarr_fixtures import sample_dataset, write_zarr

from validation.convert_mast_efm_neural_equilibrium_reference import (
    CANDIDATE_SCHEMA,
    convert_campaign,
    convert_shot_zarr,
    extract_reference_arrays,
    main,
    read_reference_zarr,
)


def test_extract_reference_arrays_keeps_only_converged_time_slices() -> None:
    """Keep the original observed arrays and convergence selection."""
    arrays = extract_reference_arrays(sample_dataset(), shot_id=30419)

    assert arrays["time_s"].tolist() == [0.1, 0.3]
    assert arrays["shot_id"].tolist() == [30419, 30419]
    assert arrays["psirz_Wb_per_rad"].shape == (2, 2, 2)
    assert arrays["psirz_valid_mask"].shape == (2, 2, 2)
    assert arrays["psirz_valid_mask"][0].tolist() == [[True, False], [True, True]]
    assert arrays["psi_axis_Wb_per_rad"].tolist() == [0.1, 0.3]
    assert arrays["psi_boundary_Wb_per_rad"].tolist() == [1.1, 1.3]
    assert arrays["Ip_MA"].tolist() == [8.0, 8.2]
    assert arrays["Bt_T"].tolist() == [5.0, 5.2]
    assert np.allclose(arrays["ffprime_rms_T_rad"], [np.sqrt(10.0), np.sqrt(50.0)])
    assert arrays["lcfs_r_m"].shape == arrays["lcfs_z_m"].shape
    assert arrays["r_grid_m"].tolist() == [0.4, 0.5]
    assert arrays["z_grid_m"].tolist() == [-0.1, 0.1]


def test_extract_reference_arrays_rejects_missing_exact_coordinate_grid() -> None:
    """Refuse a real dataset without its exact radial grid."""
    ds = sample_dataset()
    ds = ds.drop_vars("profile_r")

    try:
        extract_reference_arrays(ds, shot_id=30419)
    except ValueError as exc:
        assert "profile_r" in str(exc)
    else:
        raise AssertionError("missing profile_r was not rejected")


def test_extract_reference_arrays_rejects_missing_required_variable() -> None:
    """Refuse a real dataset without observed equilibrium flux."""
    ds = sample_dataset()
    ds = ds.drop_vars("psirz")

    try:
        extract_reference_arrays(ds, shot_id=30419)
    except ValueError as exc:
        assert "psirz" in str(exc)
    else:
        raise AssertionError("missing psirz was not rejected")


def test_convert_campaign_reports_blocked_reference_candidate(tmp_path: Path) -> None:
    """Convert actual consolidated files while predictive admission remains blocked."""
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "converted"
    manifest = tmp_path / "campaign.json"
    manifest.write_text(
        json.dumps(
            {"shots": [{"status": "acquired", "shot_id": 30419, "local_path": "mast/level1/shot_30419/efm.zarr"}]}
        ),
        encoding="utf-8",
    )

    write_zarr(dataset_root / "mast/level1/shot_30419/efm.zarr")

    report = convert_campaign(dataset_root=dataset_root, campaign_manifest=manifest, output_root=output_root)

    assert report["schema_version"] == CANDIDATE_SCHEMA
    assert report["status"] == "pass"
    assert report["admission_ready"] is False
    assert report["reference_equilibria_count"] == 2
    assert report["shots"][0]["sha256"]
    assert Path(report["shots"][0]["output_path"]).exists()
    assert "exact-model predictions" in report["blocked_reason"]
    assert len(report["payload_sha256"]) == 64


@pytest.mark.parametrize("name", ["time", "profile_z"])
def test_converter_refuses_missing_source_coordinates(name: str) -> None:
    """No absent coordinate is replaced with an index or synthetic grid."""
    with pytest.raises(ValueError, match=name):
        extract_reference_arrays(sample_dataset().drop_vars(name), shot_id=30419)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_converter_requires_genuine_positive_controls(value: object) -> None:
    """Boolean, fractional and nonpositive shot/time controls cannot select observations."""
    with pytest.raises(ValueError, match="positive integer"):
        extract_reference_arrays(sample_dataset(), shot_id=cast(Any, value))
    with pytest.raises(ValueError, match="positive integer"):
        extract_reference_arrays(sample_dataset(), shot_id=30419, max_times=cast(Any, value))


def test_converter_retains_explicit_transposed_time_alignment() -> None:
    """Labelled time axes are moved without reordering spatial observations."""
    ds = sample_dataset()
    for name in ["psirz", "ffprime", "pprime", "qpsi_c", "lcfs_r", "lcfs_z"]:
        ds[name] = ds[name].transpose(..., "time")
    observed = extract_reference_arrays(ds, shot_id=30419, max_times=1)
    expected = extract_reference_arrays(sample_dataset(), shot_id=30419, max_times=1)
    assert observed.keys() == expected.keys()
    for key in observed:
        np.testing.assert_equal(observed[key], expected[key])


@pytest.mark.parametrize("channel", ["status", "cnvrgd_times"])
@pytest.mark.parametrize("value", [np.nan, -1.0, 0.0])
def test_converter_does_not_replace_failed_convergence(channel: str, value: float) -> None:
    """Missing convergence is never supplied by status-only fallback or resizing."""
    if channel == "status" and value == 0:
        # Zero is the retained successful EFIT status; the convergence flag remains required.
        ds = sample_dataset()
        ds["status"].values[:] = value
        arrays = extract_reference_arrays(ds, shot_id=30419)
        assert arrays["time_s"].tolist() == [0.1, 0.2, 0.3]
    else:
        ds = sample_dataset()
        ds[channel] = xr.DataArray(np.full(3, value), dims=("time",))
        with pytest.raises(ValueError, match="no converged"):
            extract_reference_arrays(ds, shot_id=30419)


@pytest.mark.parametrize("name", ["status", "ffprime", "psirz"])
@pytest.mark.parametrize("dtype", [np.bool_, np.complex128, str])
def test_converter_refuses_nonreal_observation_coercion(name: str, dtype: Any) -> None:
    """Real source data cannot be fabricated by boolean, text or complex conversion."""
    ds = sample_dataset()
    ds[name] = ds[name].astype(dtype)
    with pytest.raises(ValueError, match="real numeric"):
        extract_reference_arrays(ds, shot_id=30419)


@pytest.mark.parametrize("name", ["plasma_current_x", "bphi_rmag", "ffprime"])
def test_converter_requires_declared_feature_units(name: str) -> None:
    """The declared source transform is admitted only for its explicit physical units."""
    ds = sample_dataset()
    ds[name].attrs["units"] = "unrelated"
    with pytest.raises(ValueError, match=f"{name} must declare units"):
        extract_reference_arrays(ds, shot_id=30419)


@pytest.mark.parametrize("name", ["status", "ffprime", "psirz"])
def test_converter_refuses_implicit_time_and_wrong_rank(name: str) -> None:
    """An unrelated dimension cannot be treated as time by its leading length."""
    ds = sample_dataset()
    ds[name] = ds[name].rename({"time": "other_time"})
    with pytest.raises(ValueError, match="explicit time dimension"):
        extract_reference_arrays(ds, shot_id=30419)


@pytest.mark.parametrize("name", ["plasma_current_x", "bphi_rmag"])
def test_converter_refuses_nonfinite_feature_values(name: str) -> None:
    """Selected measured feature channels must remain finite in the converted bundle."""
    ds = sample_dataset()
    ds[name].values[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        extract_reference_arrays(ds, shot_id=30419)


@pytest.mark.parametrize("name", ["psirz", "pprime", "qpsi_c", "lcfs_r"])
def test_converter_requires_valid_observations_on_every_selected_row(name: str) -> None:
    """One observed row cannot stand in for an empty row elsewhere in the campaign."""
    ds = sample_dataset()
    ds[name].values[0] = np.nan
    with pytest.raises(ValueError, match="each converted"):
        extract_reference_arrays(ds, shot_id=30419)


@pytest.mark.parametrize("profile", [[np.nan, np.inf], [0.0, 0.0]])
def test_converter_refuses_unobservable_or_zero_ffprime(profile: list[float]) -> None:
    """Unavailable and identically zero FF-prime rows cannot provide a positive RMS feature."""
    ds = sample_dataset()
    ds["ffprime"].values[0] = profile
    with pytest.raises(ValueError, match="profile RMS|must be positive"):
        extract_reference_arrays(ds, shot_id=30419)


def test_converter_handles_large_finite_profile_rms_on_real_zarr(tmp_path: Path) -> None:
    """The actual file reader preserves valid high-magnitude observations without overflow."""
    ds = sample_dataset()
    ds["ffprime"].values[:] = 1e308
    store = write_zarr(tmp_path / "large.zarr", ds)
    with np.errstate(over="raise", invalid="raise"):
        arrays = read_reference_zarr(shot_id=30419, zarr_path=store)
    assert arrays["ffprime_rms_T_rad"].tolist() == [1e308, 1e308]


def test_converter_filters_only_explicit_bad_equilibrium_scalars() -> None:
    """Finite convergence flags do not admit equal axis/boundary flux values."""
    ds = sample_dataset()
    ds["psi_boundary"].values[0] = ds["psi_axis"].values[0]
    arrays = extract_reference_arrays(ds, shot_id=30419)
    assert arrays["time_s"].tolist() == [0.3]
    ds["psi_axis"].values[2] = np.nan
    with pytest.raises(ValueError, match="no finite converged"):
        extract_reference_arrays(ds, shot_id=30419)


@pytest.mark.parametrize("values", [[0.2, 0.1, 0.3], [0.1, 0.1, 0.3], [-0.1, 0.2, 0.3], [0.1, np.nan, 0.3]])
def test_converter_refuses_invalid_exact_time(values: list[float]) -> None:
    """Time observations must be finite, nonnegative and strictly increasing."""
    ds = sample_dataset().assign_coords(time=values)
    with pytest.raises(ValueError, match="time"):
        extract_reference_arrays(ds, shot_id=30419)


def test_converter_preserves_descending_spatial_grids_and_rejects_duplicate_cells() -> None:
    """Source grids are retained exactly, with duplicates refused before serialization."""
    ds = sample_dataset().isel(profile_r=slice(None, None, -1))
    arrays = extract_reference_arrays(ds, shot_id=30419)
    assert arrays["r_grid_m"].tolist() == [0.5, 0.4]
    ds = ds.assign_coords(profile_r=[0.5, 0.5])
    with pytest.raises(ValueError, match="strictly monotonic"):
        extract_reference_arrays(ds, shot_id=30419)


def test_converter_refuses_noncoordinate_grid_and_empty_profile() -> None:
    """Real xarray dimensional metadata must describe usable coordinate and profile arrays."""
    ds = sample_dataset().drop_vars("profile_r")
    ds["profile_r"] = xr.DataArray([0.4, 0.5], dims=("different",))
    with pytest.raises(ValueError, match="one-dimensional coordinate"):
        extract_reference_arrays(ds, shot_id=30419)
    ds = sample_dataset().isel(psi_norm=slice(0, 0))
    with pytest.raises(ValueError, match="nonempty exact time"):
        extract_reference_arrays(ds, shot_id=30419)


def test_converter_refuses_mismatched_real_lcfs_shapes() -> None:
    """Independently dimensioned LCFS coordinates cannot be zipped or broadcast together."""
    ds = sample_dataset()
    ds["lcfs_z"] = xr.DataArray(np.ones((3, 2)), dims=("time", "other_lcfs"))
    with pytest.raises(ValueError, match="matching shapes"):
        extract_reference_arrays(ds, shot_id=30419)


def _campaign(tmp_path: Path, shots: object) -> tuple[Path, Path, Path]:
    """Write finite campaign input for real public converter and cold CLI tests."""
    manifest = tmp_path / "campaign.json"
    manifest.write_text(json.dumps({"shots": shots}), encoding="utf-8")
    return tmp_path / "dataset", manifest, tmp_path / "converted"


@pytest.mark.parametrize("shots", [[], [5], [{"shot_id": True}], [{"shot_id": 1}, {"shot_id": 1}]])
def test_campaign_refuses_invalid_shot_inventory_before_outputs(tmp_path: Path, shots: object) -> None:
    """Whole manifest identity validation happens before any reference output is created."""
    root, manifest, output = _campaign(tmp_path, shots)
    with pytest.raises(ValueError):
        convert_campaign(dataset_root=root, campaign_manifest=manifest, output_root=output)
    assert not output.exists()


@pytest.mark.parametrize("source", ["[]", '{"shots":[],"shots":[]}', '{"shots":[{"shot_id":1,"x":NaN}]}'])
def test_campaign_refuses_noncanonical_json(tmp_path: Path, source: str) -> None:
    """Nonobject roots, duplicate keys and nonfinite constants are normal input refusals."""
    root, manifest, output = _campaign(tmp_path, [])
    manifest.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError):
        convert_campaign(dataset_root=root, campaign_manifest=manifest, output_root=output)
    assert not output.exists()


@pytest.mark.parametrize(
    "local_path", [None, "", "../escape", "/absolute", "C:/source", "\\\\server\\source", "bad\x00name"]
)
def test_campaign_refuses_storage_escape_before_outputs(tmp_path: Path, local_path: object) -> None:
    """Relative local storage custody is checked before writes or original source access."""
    root, manifest, output = _campaign(tmp_path, [{"status": "acquired", "shot_id": 30419, "local_path": local_path}])
    with pytest.raises(ValueError, match="local_path"):
        convert_campaign(dataset_root=root, campaign_manifest=manifest, output_root=output)
    assert not output.exists()


def test_campaign_refuses_symlink_escape(tmp_path: Path) -> None:
    """A syntactically relative source cannot resolve outside selected dataset storage."""
    root, manifest, output = _campaign(tmp_path, [{"status": "acquired", "shot_id": 30419, "local_path": "linked"}])
    root.mkdir()
    (root / "linked").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="inside dataset_root"):
        convert_campaign(dataset_root=root, campaign_manifest=manifest, output_root=output)


def test_campaign_reports_unacquired_and_missing_actual_sources(tmp_path: Path) -> None:
    """Ordinary shot conversion refusals produce finite blocked candidate reports."""
    root, manifest, output = _campaign(
        tmp_path,
        [{"status": "pending", "shot_id": 1}, {"status": "acquired", "shot_id": 2, "local_path": "absent.zarr"}],
    )
    report = convert_campaign(dataset_root=root, campaign_manifest=manifest, output_root=output)
    assert report["status"] == "fail"
    assert report["reference_dataset_id"] == "mast-efm-empty"
    assert report["reference_equilibria_count"] == 0
    assert len(report["errors"]) == 2
    assert "not acquired" in report["errors"][0]["error"]
    assert "does not exist" in report["errors"][1]["error"]


def test_converter_protects_original_store_and_explicit_npz_suffix(tmp_path: Path) -> None:
    """Neither public single-shot nor campaign outputs may overwrite original sources."""
    root, manifest, output = _campaign(tmp_path, [{"status": "acquired", "shot_id": 30419, "local_path": "data.zarr"}])
    store = write_zarr(root / "data.zarr")
    before = (store / ".zmetadata").read_bytes()
    with pytest.raises(ValueError, match=".npz suffix"):
        convert_shot_zarr(shot_id=30419, zarr_path=store, output_path=tmp_path / "missing-suffix")
    with pytest.raises(ValueError, match="original Zarr"):
        convert_shot_zarr(shot_id=30419, zarr_path=store, output_path=store / "overwrite.npz")
    with pytest.raises(ValueError, match="original Zarr"):
        convert_campaign(dataset_root=root, campaign_manifest=manifest, output_root=store)
    with pytest.raises(ValueError, match="campaign manifest"):
        convert_campaign(dataset_root=root, campaign_manifest=manifest, output_root=manifest)
    with pytest.raises(ValueError, match="positive integer"):
        convert_campaign(dataset_root=root, campaign_manifest=manifest, output_root=output, max_times_per_shot=0)
    assert (store / ".zmetadata").read_bytes() == before


def test_actual_zarr_to_reference_to_producer_custody(tmp_path: Path) -> None:
    """The emitted reference bytes are consumable by the real producer's SHA-bound loader."""
    root, manifest, output = _campaign(tmp_path, [{"status": "acquired", "shot_id": 30419, "local_path": "data.zarr"}])
    write_zarr(root / "data.zarr")
    report = convert_campaign(dataset_root=root, campaign_manifest=manifest, output_root=output)
    selected = report["shots"][0]
    from validation.neural_equilibrium_dataset_tensors import load_reference

    arrays = load_reference(selected, Path(selected["output_path"]))
    assert arrays["Ip_MA"].tolist() == [8.0, 8.2]
    assert arrays["Bt_T"].tolist() == [5.0, 5.2]
    assert arrays["time_s"].tolist() == [0.1, 0.3]
    assert arrays["shot_id"].tolist() == [30419, 30419]


@pytest.mark.parametrize("target", ["campaign", "original", "reference"])
def test_cli_protects_selected_inputs_before_report_write(
    tmp_path: Path, target: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """The real CLI refuses report destinations that overwrite its input or output tensors."""
    root, manifest, output = _campaign(tmp_path, [{"status": "acquired", "shot_id": 30419, "local_path": "data.zarr"}])
    store = write_zarr(root / "data.zarr")
    targets = {
        "campaign": manifest,
        "original": store / ".zmetadata",
        "reference": output / "mast_efm_shot_30419_reference.npz",
    }
    before = manifest.read_bytes(), (store / ".zmetadata").read_bytes()
    assert (
        main(
            [
                "--dataset-root",
                str(root),
                "--campaign-manifest",
                str(manifest),
                "--output-root",
                str(output),
                "--report-out",
                str(targets[target]),
            ]
        )
        == 1
    )
    assert "must not overwrite" in capsys.readouterr().err
    assert (manifest.read_bytes(), (store / ".zmetadata").read_bytes()) == before
    assert not output.exists()


def test_cli_writes_actual_conversion_json_and_normal_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Public CLI succeeds with actual source files and refuses missing input without a traceback."""
    root, manifest, output = _campaign(tmp_path, [{"status": "acquired", "shot_id": 30419, "local_path": "data.zarr"}])
    write_zarr(root / "data.zarr")
    argv = ["--dataset-root", str(root), "--campaign-manifest", str(manifest), "--output-root", str(output)]
    assert main(argv + ["--json-out", "--max-times-per-shot", "1"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["reference_equilibria_count"] == 1
    assert json.loads((output / "mast_efm_neural_equilibrium_reference_candidate.json").read_text()) == report
    assert main(argv) == 0
    assert "pass shots=1 equilibria=2 admission_ready=False" in capsys.readouterr().out
    manifest.unlink()
    assert main(argv) == 1
    assert "refused" in capsys.readouterr().err


def test_cold_direct_converter_cli_with_actual_zarr(tmp_path: Path) -> None:
    """A fresh direct script process resolves imports and decodes real consolidated source chunks."""
    root, manifest, output = _campaign(tmp_path, [{"status": "acquired", "shot_id": 30419, "local_path": "data.zarr"}])
    write_zarr(root / "data.zarr")
    script = Path(__file__).resolve().parents[1] / "validation/convert_mast_efm_neural_equilibrium_reference.py"
    process = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset-root",
            str(root),
            "--campaign-manifest",
            str(manifest),
            "--output-root",
            str(output),
            "--json-out",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["reference_equilibria_count"] == 2
    help_process = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path, text=True, capture_output=True)
    assert help_process.returncode == 0
    assert "max-times-per-shot" in help_process.stdout
    bad = subprocess.run([sys.executable, str(script), "--unknown"], cwd=tmp_path, text=True, capture_output=True)
    assert bad.returncode == 2


def test_converter_requires_real_labelled_source_dataset() -> None:
    """The public extractor refuses unrelated input rather than accepting an unvalidated adapter."""
    with pytest.raises(ValueError, match="real xarray Dataset"):
        extract_reference_arrays(cast(Any, None), shot_id=30419)


def test_actual_zarr_refuses_coordinate_length_inconsistency(tmp_path: Path) -> None:
    """The real xarray reader enforces matching coordinate lengths before extracting arrays."""
    store = write_zarr(tmp_path / "inconsistent.zarr")
    path = store / ".zmetadata"
    metadata = json.loads(path.read_text())
    metadata["metadata"]["profile_r/.zarray"]["shape"] = [3]
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting sizes|dimension|length"):
        read_reference_zarr(shot_id=30419, zarr_path=store)


def test_converter_refuses_duplicate_source_dimension_names() -> None:
    """A repeated time dimension cannot be confused with either spatial coordinate."""
    ds = sample_dataset()
    with pytest.warns(UserWarning, match="Duplicate dimension"):
        ds["psirz"] = xr.DataArray(np.ones((3, 3, 2)), dims=("time", "time", "profile_r"))
    with pytest.raises(ValueError, match="explicit time dimension"):
        extract_reference_arrays(ds, shot_id=30419)


def test_converter_refuses_shot_identity_outside_serialized_int64() -> None:
    """A positive Python integer that cannot be persisted is a normal admission refusal."""
    with pytest.raises(ValueError, match="representable in int64"):
        extract_reference_arrays(sample_dataset(), shot_id=1 << 63)


def test_converter_refuses_unrepresentable_real_feature_observations() -> None:
    """Float64 conversion cannot silently discard a finite wider source observation."""
    ds = sample_dataset()
    with np.errstate(over="ignore"):
        value = np.longdouble(np.finfo(np.float64).max) * np.longdouble(2)
    ds["bphi_rmag"] = xr.DataArray(np.full(3, value, dtype=np.longdouble), dims=("time",), attrs={"units": "T"})
    with pytest.raises(ValueError, match="representable in float64|non-finite"):
        extract_reference_arrays(ds, shot_id=30419)


def test_campaign_refuses_json_float_overflow(tmp_path: Path) -> None:
    """A numeric JSON exponent that decodes to infinity is refused before any output."""
    root, manifest, output = _campaign(tmp_path, [])
    manifest.write_text('{"shots":[{"shot_id":30419,"status":"pending","unused":1e999}]}')
    with pytest.raises(ValueError, match="Out of range float"):
        convert_campaign(dataset_root=root, campaign_manifest=manifest, output_root=output)
    assert not output.exists()
