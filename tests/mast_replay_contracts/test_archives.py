# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST replay archive public boundary tests.
"""Exercise genuine NPZ snapshots through public inspection and dataset CLI."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from scpn_control._npz import save_npz_arrays
from validation.build_disruption_replay_channels import inspect_replay_archive, inspect_replay_archive_bytes
from validation.build_mast_disruption_dataset import main as dataset_main

from ._fixtures import STAMP, channels, write_channels


def arguments(source: Path, output: Path) -> list[str]:
    """Return real dataset command arguments for a manufactured archive."""
    return [
        "--channels-npz",
        str(source),
        "--dataset-id",
        "candidate",
        "--out-dir",
        str(output),
        "--json-out",
        str(output / "report.json"),
        "--retrieved-at",
        STAMP,
    ]


@pytest.mark.parametrize("identifiers", [[101.5], [np.nan], [np.inf], [101.0, 101.0], [-1.0], [float(2**63)]])
def test_dataset_refuses_lossy_or_invalid_identities(tmp_path: Path, identifiers: list[float]) -> None:
    """A real input archive cannot silently truncate or duplicate shot identity."""
    source = tmp_path / "channels.npz"
    write_channels(source, identifiers=np.array(identifiers, dtype=np.float64))
    original = source.read_bytes()
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        dataset_main(arguments(source, output))
    assert source.read_bytes() == original and not output.exists()


def test_legacy_integral_float_identity_is_preserved(tmp_path: Path) -> None:
    """Keep established exact-integral float-ID compatibility without admission."""
    source = tmp_path / "channels.npz"
    write_channels(source, identifiers=np.array([101.0], dtype=np.float64))
    assert dataset_main(arguments(source, tmp_path / "output")) == 0
    report = json.loads((tmp_path / "output/report.json").read_text())
    assert report["shots"][0]["shot_id"] == 101 and report["admission_ready"] is False
    with pytest.raises(ValueError, match="integer vector"):
        inspect_replay_archive(source)


@pytest.mark.parametrize("clock", [[0.0] * 8, list(reversed(range(8))), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan]])
def test_archive_clock_is_shared_by_reader_and_inspector(tmp_path: Path, clock: list[float]) -> None:
    """Both public surfaces refuse a corrupt clock before dataset publication."""
    values = channels()
    values["time_s"] = np.array(clock, dtype=np.float64)
    source = tmp_path / "channels.npz"
    save_npz_arrays(
        source,
        {"shot_ids": np.array([101], dtype=np.int64), **{f"101:{name}": array for name, array in values.items()}},
        allow_pickle=False,
    )
    before = source.read_bytes()
    with pytest.raises(ValueError):
        inspect_replay_archive(source)
    with pytest.raises(ValueError):
        dataset_main(arguments(source, tmp_path / "output"))
    assert source.read_bytes() == before and not (tmp_path / "output").exists()


def test_duplicate_member_names_are_refused(tmp_path: Path) -> None:
    """Reject ambiguous ZIP names even when their set matches the schema."""
    source = tmp_path / "channels.npz"
    write_channels(source)
    stream = BytesIO()
    with ZipFile(source) as archive, ZipFile(stream, "w") as duplicate:
        for name in archive.namelist():
            duplicate.writestr(name, archive.read(name))
        with pytest.warns(UserWarning, match="Duplicate name"):
            duplicate.writestr("shot_ids.npy", archive.read("shot_ids.npy"))
    with pytest.raises(ValueError, match="member names must be unique"):
        inspect_replay_archive_bytes(stream.getvalue(), path_name="duplicate.npz")


def test_unreadable_real_file_has_authored_refusal(tmp_path: Path) -> None:
    """Real filesystem permission refusal exposes no underlying OS message."""
    source = tmp_path / "channels.npz"
    write_channels(source)
    source.chmod(0)
    try:
        with pytest.raises(ValueError, match="^cannot read replay archive$"):
            inspect_replay_archive(source)
    finally:
        source.chmod(0o600)


@pytest.mark.parametrize("raw", [b"", b"not an archive"])
def test_malformed_snapshot_has_authored_refusal(raw: bytes) -> None:
    """The byte entry point normalises actual NumPy decoder failures."""
    with pytest.raises(ValueError, match="^cannot validate replay archive bytes$"):
        inspect_replay_archive_bytes(raw, path_name="bad.npz")


def test_snapshot_binding_preserves_source_and_value_digests(tmp_path: Path) -> None:
    """File and byte APIs inspect the same immutable snapshot and channel schema."""
    source = tmp_path / "channels.npz"
    write_channels(source)
    original = source.read_bytes()
    bound = inspect_replay_archive_bytes(original, path_name=source.name, expected_shot_ids=[101])
    assert bound == inspect_replay_archive(source, expected_shot_ids=[101])
    assert bound["shot_count"] == 1 and bound["shot_members"][0]["n_samples"] == 8
    assert source.read_bytes() == original
