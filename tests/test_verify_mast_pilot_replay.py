# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Exact-object FAIR-MAST pilot replay tests
"""Offline file-boundary tests for the L2F-13 preserved-pilot gate."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from validation import verify_mast_pilot_replay as replay
from validation.mast_source_object_manifest import canonical_json_sha256
from validation.verify_mast_pilot_replay import (
    MastPilotReplayError,
    _open_local_group,
    main,
    verify_mast_pilot_replay,
)

_SHOT_ID = 27707


class _FakeVar:
    def __init__(self, values: NDArray[np.float64], *, dims: tuple[str, ...], units: str | None) -> None:
        self.values = values
        self.dims = dims
        self.attrs = {"units": units} if units is not None else {}
        self.chunks: None = None


class _FakeDataset:
    def __init__(self, variables: Mapping[str, _FakeVar]) -> None:
        self.variables = dict(variables)
        self.closed = False

    def __getitem__(self, key: str) -> _FakeVar:
        return self.variables[key]

    def close(self) -> None:
        self.closed = True


def _fake_groups() -> dict[str, _FakeDataset]:
    time = np.linspace(0.0, 0.02, 21, dtype=np.float64)
    time_mirnov = np.linspace(0.0, 0.02, 201, dtype=np.float64)
    time_saddle = np.linspace(0.0, 0.02, 101, dtype=np.float64)
    angles = np.tile(np.linspace(0.0, 330.0, 12, dtype=np.float64).reshape(-1, 1), (1, 28))
    summary = _FakeDataset(
        {
            "time": _FakeVar(time, dims=("time",), units="s"),
            "ip": _FakeVar(np.linspace(6.0e5, 4.0e5, 21), dims=("time",), units="A"),
            "line_average_n_e": _FakeVar(
                np.asarray([2.0e19] * 10 + [np.nan] + [1.5e19] * 10),
                dims=("time",),
                units="1 / m ** 3",
            ),
            "greenwald_density": _FakeVar(np.full(21, 2.5e19), dims=("time",), units="1 / m ** 3"),
        }
    )
    magnetics = _FakeDataset(
        {
            "time_saddle": _FakeVar(time_saddle, dims=("time_saddle",), units="s"),
            "time_mirnov": _FakeVar(time_mirnov, dims=("time_mirnov",), units="s"),
            "b_field_tor_probe_saddle_field": _FakeVar(
                np.ones((12, 101), dtype=np.float64),
                dims=("b_field_tor_probe_saddle_field_channel", "time_saddle"),
                units="T",
            ),
            "b_field_tor_probe_saddle_m_phi": _FakeVar(
                angles,
                dims=("b_field_tor_probe_saddle_m_geometry_channel", "coordinate"),
                units="SI, degrees, m",
            ),
        }
    )
    return {"summary": summary, "magnetics": magnetics}


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, str, str]:
    object_root = tmp_path / "disruption_shots"
    objects = {
        "raw/27707.zarr/summary/zarr.json": b'{"zarr_format":3,"node_type":"group"}',
        "raw/27707.zarr/summary/time/c/0": b"summary-time-chunk",
        "raw/27707.zarr/magnetics/zarr.json": b'{"zarr_format":3,"node_type":"group"}',
        "raw/27707.zarr/magnetics/time/c/0": b"magnetics-time-chunk",
    }
    files: list[dict[str, object]] = []
    records: list[str] = []
    for relative_path, content in sorted(objects.items()):
        path = object_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        size = len(content)
        suffix = "/".join(Path(relative_path).parts[2:])
        files.append(
            {
                "path": relative_path,
                "sha256": digest,
                "size_bytes": size,
                "source_url": f"https://s3.echo.stfc.ac.uk/mast/level2/shots/27707.zarr/{suffix}",
            }
        )
        records.append(f"{digest}:{size}:{relative_path}\n")
    aggregate = hashlib.sha256("".join(records).encode()).hexdigest()
    provenance_payload = {
        "schema": "scpn-fusion-open-disruption-data-provenance.v1",
        "retrieved_at_utc": "2026-07-22T00:23:35Z",
        "dataset": {
            "device": "MAST",
            "shot_id": _SHOT_ID,
            "downloaded_groups": ["summary", "magnetics"],
            "object_count": len(files),
            "total_size_bytes": sum(len(content) for content in objects.values()),
            "download_manifest_sha256": aggregate,
            "files": files,
        },
    }
    provenance = tmp_path / "PROVENANCE.json"
    provenance.write_text(json.dumps(provenance_payload, sort_keys=True) + "\n", encoding="utf-8")
    provenance_sha = hashlib.sha256(provenance.read_bytes()).hexdigest()
    return provenance, object_root, provenance_sha, aggregate


def _opener(groups: Mapping[str, _FakeDataset]) -> Any:
    def open_group(path: Path) -> _FakeDataset:
        return groups[path.name]

    return open_group


def test_exact_pilot_replay_verifies_objects_masks_and_fail_closed_groups(tmp_path: Path) -> None:
    """The real boundary binds native objects and exposes no unmasked aligned channel."""
    provenance, object_root, provenance_sha, aggregate = _write_fixture(tmp_path)
    groups = _fake_groups()

    report = verify_mast_pilot_replay(
        provenance,
        object_root=object_root,
        shot_id=_SHOT_ID,
        expected_provenance_sha256=provenance_sha,
        expected_aggregate_sha256=aggregate,
        open_group=_opener(groups),
    )

    assert report["status"] == "expected_missing_groups_fail_closed"
    assert report["source_object_integrity_verified"] is True
    assert report["pilot_download_manifest_sha256"] == aggregate
    assert report["pilot_object_count"] == 4
    assert report["missing_groups"] == ["equilibrium", "interferometer"]
    assert report["source_generation_pinned"] is False
    assert report["source_generation_blocker"] == "historical_pilot_root_zarr_json_not_preserved"
    assert report["n_bound_channels_aligned"] == 3
    assert report["n_channels_not_aligned"] == 8
    assert report["selected_array_count"] == 8
    assert all(dataset.closed for dataset in groups.values())
    aligned = [channel for channel in report["channels"] if channel["status"] == "aligned_with_validity_mask"]
    assert {channel["channel"] for channel in aligned} == {"time_s", "Ip_MA", "ne_1e19"}
    assert all(len(channel["valid_mask_sha256"]) == 64 for channel in aligned)
    assert all("values" not in channel and "valid_mask" not in channel for channel in report["channels"])
    equilibrium = report["missing_group_outcomes"][0]
    assert equilibrium["status"] == "fail_closed"
    assert {channel["channel"] for channel in equilibrium["channel_outcomes"]} == {
        "BT_T",
        "beta_N",
        "q95",
        "vertical_position_m",
    }
    assert report["missing_group_outcomes"][1] == {
        "group": "interferometer",
        "status": "absent_no_current_binding_consumer",
        "channel_outcomes": [],
    }
    assert report["payload_sha256"] == canonical_json_sha256({**report, "payload_sha256": None})
    for claim in (
        "scientific_validity_claim_admissible",
        "cohort_claim_admissible",
        "model_training_admissible",
        "facility_claim_admissible",
        "control_admission_admissible",
    ):
        assert report[claim] is False


def test_pilot_replay_rejects_wrong_pins_and_unexpected_missing_groups(tmp_path: Path) -> None:
    """Provenance, aggregate, shot identity, and expected group absence are pinned."""
    provenance, object_root, provenance_sha, aggregate = _write_fixture(tmp_path)
    kwargs = {
        "object_root": object_root,
        "shot_id": _SHOT_ID,
        "expected_provenance_sha256": provenance_sha,
        "expected_aggregate_sha256": aggregate,
        "open_group": _opener(_fake_groups()),
    }
    with pytest.raises(MastPilotReplayError, match="provenance SHA-256"):
        verify_mast_pilot_replay(provenance, **{**kwargs, "expected_provenance_sha256": "0" * 64})
    with pytest.raises(MastPilotReplayError, match="aggregate SHA-256"):
        verify_mast_pilot_replay(provenance, **{**kwargs, "expected_aggregate_sha256": "0" * 64})
    with pytest.raises(MastPilotReplayError, match="requested MAST shot"):
        verify_mast_pilot_replay(provenance, **{**kwargs, "shot_id": 1})
    with pytest.raises(MastPilotReplayError, match="do not match expected"):
        verify_mast_pilot_replay(provenance, **kwargs, expected_missing_groups=("equilibrium",))
    with pytest.raises(MastPilotReplayError, match="positive integer"):
        verify_mast_pilot_replay(provenance, **{**kwargs, "shot_id": 0})
    with pytest.raises(MastPilotReplayError, match="must be unique"):
        verify_mast_pilot_replay(
            provenance,
            **kwargs,
            expected_missing_groups=("equilibrium", "equilibrium"),
        )


def test_pilot_replay_detects_native_object_change_during_read(tmp_path: Path) -> None:
    """A pilot object changed after the first integrity pass cannot satisfy replay."""
    provenance, object_root, provenance_sha, aggregate = _write_fixture(tmp_path)
    groups = _fake_groups()
    changed = False

    def mutating_opener(path: Path) -> _FakeDataset:
        nonlocal changed
        if not changed:
            changed = True
            (object_root / "raw/27707.zarr/summary/time/c/0").write_bytes(b"substituted-after-pin")
        return groups[path.name]

    with pytest.raises(MastPilotReplayError, match="object integrity mismatch"):
        verify_mast_pilot_replay(
            provenance,
            object_root=object_root,
            shot_id=_SHOT_ID,
            expected_provenance_sha256=provenance_sha,
            expected_aggregate_sha256=aggregate,
            open_group=mutating_opener,
        )


def test_pilot_replay_rejects_unsafe_or_inconsistent_object_metadata(tmp_path: Path) -> None:
    """Paths, URLs, object counts, bytes, and downloaded groups are hard boundaries."""
    provenance, object_root, provenance_sha, aggregate = _write_fixture(tmp_path)
    payload = json.loads(provenance.read_text())
    cases = (
        (lambda p: p["dataset"].update(object_count=99), "object_count"),
        (lambda p: p["dataset"].update(total_size_bytes=99), "total_size_bytes"),
        (lambda p: p["dataset"].update(downloaded_groups=["summary", "summary"]), "downloaded_groups"),
        (lambda p: p["dataset"]["files"][0].update(source_url="https://example.invalid"), "source_url"),
        (lambda p: p["dataset"]["files"][0].update(path="../escape"), "outside its declared"),
    )
    for mutate, message in cases:
        candidate_payload = json.loads(json.dumps(payload))
        mutate(candidate_payload)
        candidate = tmp_path / f"bad-{message}.json"
        candidate.write_text(json.dumps(candidate_payload), encoding="utf-8")
        candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
        with pytest.raises(MastPilotReplayError, match=message):
            verify_mast_pilot_replay(
                candidate,
                object_root=object_root,
                shot_id=_SHOT_ID,
                expected_provenance_sha256=candidate_sha,
                expected_aggregate_sha256=aggregate,
                open_group=_opener(_fake_groups()),
            )


def test_native_manifest_shape_guards_cover_every_rejected_field(tmp_path: Path) -> None:
    """Malformed provenance fields cannot reach Zarr or the mapping layer."""
    provenance, object_root, _provenance_sha, aggregate = _write_fixture(tmp_path)
    valid = json.loads(provenance.read_text())
    cases = (
        (lambda p: p.update(schema="wrong"), "provenance schema"),
        (lambda p: p.update(dataset=[]), "dataset must be an object"),
        (lambda p: p["dataset"].update(files=[], object_count=0), "files must be a non-empty"),
        (lambda p: p["dataset"].update(downloaded_groups="summary"), "downloaded_groups"),
        (lambda p: p["dataset"].update(files=[None], object_count=1), r"files\[0\] must be an object"),
        (
            lambda p: p["dataset"].update(
                files=[p["dataset"]["files"][0], p["dataset"]["files"][0]],
                object_count=2,
            ),
            "path must be a unique string",
        ),
        (lambda p: p["dataset"]["files"][0].update(path=None), "path must be a unique string"),
        (lambda p: p["dataset"]["files"][0].update(sha256="bad"), "sha256 must be a SHA-256"),
        (lambda p: p["dataset"]["files"][0].update(size_bytes=True), "size_bytes"),
        (
            lambda p: p["dataset"].update(download_manifest_sha256="f" * 64),
            "recomputed pilot aggregate",
        ),
        (lambda p: p.update(retrieved_at_utc=""), "retrieved_at_utc"),
    )
    for mutate, message in cases:
        candidate = json.loads(json.dumps(valid))
        mutate(candidate)
        candidate_dataset = candidate.get("dataset")
        expected_aggregate = (
            candidate_dataset.get("download_manifest_sha256", aggregate)
            if isinstance(candidate_dataset, Mapping)
            else aggregate
        )
        with pytest.raises(MastPilotReplayError, match=message):
            replay._verify_native_objects(
                candidate,
                provenance_sha256="a" * 64,
                object_root=object_root,
                shot_id=_SHOT_ID,
                expected_aggregate_sha256=expected_aggregate,
            )


def test_path_and_file_io_guards_are_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing provenance, unsafe paths, directories, and hash I/O fail cleanly."""
    with pytest.raises(MastPilotReplayError, match="cannot read pilot provenance"):
        replay._load_pinned_provenance(tmp_path / "missing.json", expected_sha256="0" * 64)

    root_list = tmp_path / "list.json"
    root_list.write_text("[]", encoding="utf-8")
    with pytest.raises(MastPilotReplayError, match="root must be an object"):
        replay._load_pinned_provenance(
            root_list,
            expected_sha256=hashlib.sha256(root_list.read_bytes()).hexdigest(),
        )

    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(MastPilotReplayError, match="unsafe pilot object path"):
        replay._object_path(root, "../escape")
    with pytest.raises(MastPilotReplayError, match="does not resolve"):
        replay._object_path(root, "missing")
    directory = root / "directory"
    directory.mkdir()
    with pytest.raises(MastPilotReplayError, match="not a file"):
        replay._object_path(root, "directory")
    with pytest.raises(MastPilotReplayError, match="cannot hash pilot object"):
        replay._hash_file_once(directory)

    symlink = root / "outside-link"
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    symlink.symlink_to(outside)
    with pytest.raises(MastPilotReplayError, match="escapes its root"):
        replay._object_path(root, "outside-link")

    monkeypatch.setattr(Path, "resolve", lambda _self, strict=False: (_ for _ in ()).throw(OSError("bad-root")))
    with pytest.raises(OSError, match="bad-root"):
        replay._object_path(root, "file")


def test_group_directory_and_nonclosable_dataset_paths(tmp_path: Path) -> None:
    """Declared groups must exist, while a reader without close remains supported."""
    zarr_root = tmp_path / "27707.zarr"
    (zarr_root / "summary").mkdir(parents=True)
    with pytest.raises(MastPilotReplayError, match="does not resolve to a directory"):
        replay._read_selected_arrays(
            zarr_root,
            downloaded_groups=("summary", "magnetics"),
            open_group=lambda _path: _fake_groups()["summary"],
        )

    (zarr_root / "magnetics").mkdir()
    groups = _fake_groups()

    class _NoCloseDataset:
        def __init__(self, dataset: _FakeDataset) -> None:
            self.variables = dataset.variables

        def __getitem__(self, key: str) -> _FakeVar:
            return self.variables[key]

    arrays, _metadata, missing = replay._read_selected_arrays(
        zarr_root,
        downloaded_groups=("summary", "magnetics"),
        open_group=lambda path: _NoCloseDataset(groups[path.name]),
    )
    assert "magnetics.b_field_tor_probe_saddle_field" in arrays
    assert missing == ("equilibrium", "interferometer")


def test_impossible_alignment_and_post_pin_states_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Defence-in-depth rejects aligned absent groups, changed pins, and missing mask digests."""
    with pytest.raises(MastPilotReplayError, match="unexpectedly produced an aligned channel"):
        replay._missing_group_outcomes(
            ("equilibrium",),
            [{"channel": "q95", "status": "aligned_with_validity_mask", "reason_code": None}],
        )

    provenance, object_root, provenance_sha, aggregate = _write_fixture(tmp_path)
    real_verify = replay._verify_native_objects
    call_count = 0

    def changed_pin(*args: Any, **kwargs: Any) -> replay.PilotObjectPin:
        nonlocal call_count
        call_count += 1
        pin = real_verify(*args, **kwargs)
        if call_count == 2:
            return replay.PilotObjectPin(
                shot_id=pin.shot_id,
                provenance_sha256=pin.provenance_sha256,
                aggregate_sha256=pin.aggregate_sha256,
                object_count=pin.object_count,
                total_size_bytes=pin.total_size_bytes,
                downloaded_groups=pin.downloaded_groups,
                retrieved_at="changed",
            )
        return pin

    monkeypatch.setattr(replay, "_verify_native_objects", changed_pin)
    with pytest.raises(MastPilotReplayError, match="identity changed during replay"):
        verify_mast_pilot_replay(
            provenance,
            object_root=object_root,
            shot_id=_SHOT_ID,
            expected_provenance_sha256=provenance_sha,
            expected_aggregate_sha256=aggregate,
            open_group=_opener(_fake_groups()),
        )

    monkeypatch.setattr(replay, "_verify_native_objects", real_verify)
    real_alignment = replay.verify_real_object_alignment

    def missing_mask(*args: Any, **kwargs: Any) -> dict[str, Any]:
        report = real_alignment(*args, **kwargs)
        channel = next(item for item in report["channels"] if item["status"] == "aligned_with_validity_mask")
        channel["valid_mask_sha256"] = None
        return report

    monkeypatch.setattr(replay, "verify_real_object_alignment", missing_mask)
    with pytest.raises(MastPilotReplayError, match="lacks a validity-mask digest"):
        verify_mast_pilot_replay(
            provenance,
            object_root=object_root,
            shot_id=_SHOT_ID,
            expected_provenance_sha256=provenance_sha,
            expected_aggregate_sha256=aggregate,
            open_group=_opener(_fake_groups()),
        )


def test_pilot_replay_rejects_duplicate_keys_and_invalid_json(tmp_path: Path) -> None:
    """Ambiguous or undecodable provenance bytes fail before object access."""
    object_root = tmp_path / "objects"
    object_root.mkdir()
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    duplicate_sha = hashlib.sha256(duplicate.read_bytes()).hexdigest()
    with pytest.raises(MastPilotReplayError, match="duplicate JSON key"):
        verify_mast_pilot_replay(
            duplicate,
            object_root=object_root,
            shot_id=_SHOT_ID,
            expected_provenance_sha256=duplicate_sha,
            expected_aggregate_sha256="0" * 64,
        )
    invalid = tmp_path / "invalid.json"
    invalid.write_bytes(b"\xff")
    invalid_sha = hashlib.sha256(invalid.read_bytes()).hexdigest()
    with pytest.raises(MastPilotReplayError, match="cannot decode"):
        verify_mast_pilot_replay(
            invalid,
            object_root=object_root,
            shot_id=_SHOT_ID,
            expected_provenance_sha256=invalid_sha,
            expected_aggregate_sha256="0" * 64,
        )


def test_pilot_replay_normalises_group_open_and_missing_saddle_failures(tmp_path: Path) -> None:
    """Optional Zarr errors and incomplete magnetic selection remain bounded."""
    provenance, object_root, provenance_sha, aggregate = _write_fixture(tmp_path)
    kwargs = {
        "object_root": object_root,
        "shot_id": _SHOT_ID,
        "expected_provenance_sha256": provenance_sha,
        "expected_aggregate_sha256": aggregate,
    }

    def broken_opener(_path: Path) -> None:
        raise RuntimeError("broken-zarr")

    with pytest.raises(MastPilotReplayError, match="cannot open preserved pilot group"):
        verify_mast_pilot_replay(provenance, **kwargs, open_group=broken_opener)
    groups = _fake_groups()
    groups["magnetics"].variables.pop("b_field_tor_probe_saddle_field")
    with pytest.raises(MastPilotReplayError, match="no toroidal saddle"):
        verify_mast_pilot_replay(provenance, **kwargs, open_group=_opener(groups))


def test_local_group_opener_and_cli_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The lazy xarray seam and CLI write only the bounded JSON report."""
    seen: dict[str, Any] = {}
    fake_xarray = types.ModuleType("xarray")
    fake_xarray.open_zarr = lambda path, consolidated: (
        seen.update(  # type: ignore[attr-defined]
            path=path, consolidated=consolidated
        )
        or "dataset"
    )
    monkeypatch.setitem(sys.modules, "xarray", fake_xarray)
    assert _open_local_group(tmp_path) == "dataset"
    assert seen == {"path": tmp_path, "consolidated": False}

    output = tmp_path / "evidence" / "pilot.json"
    report = {
        "shot_id": _SHOT_ID,
        "n_bound_channels_aligned": 3,
        "payload_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        "validation.verify_mast_pilot_replay.verify_mast_pilot_replay",
        lambda *_args, **_kwargs: report,
    )
    assert (
        main(
            [
                "--provenance",
                str(tmp_path / "PROVENANCE.json"),
                "--object-root",
                str(tmp_path),
                "--shot-id",
                str(_SHOT_ID),
                "--expected-provenance-sha256",
                "b" * 64,
                "--expected-aggregate-sha256",
                "c" * 64,
                "--json-out",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text()) == report
