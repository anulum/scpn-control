# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Public MAST mirror-to-dataset workflow contracts.
"""Exercise native file/CLI paths with manufactured data and no admission."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scpn_control._npz import save_npz_arrays
from scpn_control.core.real_data_manifest import load_real_data_manifest
from validation.build_disruption_replay_channels import build_channels, derive_replay_channels
from validation.build_mast_disruption_dataset import build_dataset, build_shot_npz
from validation.build_mast_disruption_dataset import main as dataset_main

from ._fixtures import STAMP, channels, mirror, write_channels
from .test_archives import arguments

ROOT = Path(__file__).resolve().parents[2]


def test_native_mirror_to_labelled_dataset(tmp_path: Path) -> None:
    """Run both maintained module CLIs and verify genuine persisted artefacts."""
    material = tmp_path / "material"
    material.mkdir()
    save_npz_arrays(material / "shot_101.npz", mirror(), allow_pickle=False)
    source_bytes = (material / "shot_101.npz").read_bytes()
    replay = tmp_path / "replay"
    output = tmp_path / "dataset"
    environment = os.environ.copy()
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / "src"),
        TMPDIR=str(tmp_path),
        TMP=str(tmp_path),
        TEMP=str(tmp_path),
    )
    commands = [
        [
            sys.executable,
            "-m",
            "validation.build_disruption_replay_channels",
            "--material-dir",
            str(material),
            "--out-dir",
            str(replay),
            "--json-out",
            str(replay / "report.json"),
            "--locked-window",
            "3",
            "--generated-at",
            STAMP,
        ],
        [sys.executable, "-m", "validation.build_mast_disruption_dataset", *arguments(replay / "channels.npz", output)],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True, timeout=45)
        assert result.returncode == 0, result.stderr
    candidate = json.loads((replay / "report.json").read_text())
    assert candidate["n_derived"] == 1 and all(v is False for v in candidate["claim_boundary"].values())
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "blocked" and report["admission_ready"] is False
    assert report["independent_label_count"] == 0
    load_real_data_manifest(output / "candidate.manifest.json", verify_artifact=True)
    assert (material / "shot_101.npz").read_bytes() == source_bytes


@pytest.mark.parametrize("mode", ["direct", "symlink", "hardlink"])
def test_report_cannot_replace_input(tmp_path: Path, mode: str) -> None:
    """Refuse all existing filesystem aliases before writing dataset outputs."""
    source = tmp_path / "channels.npz"
    write_channels(source)
    original = source.read_bytes()
    target = source
    if mode != "direct":
        target = tmp_path / "report.json"
        if mode == "symlink":
            target.symlink_to(source)
        else:
            os.link(source, target)
    argv = arguments(source, tmp_path / "output")
    argv[argv.index("--json-out") + 1] = str(target)
    with pytest.raises(ValueError, match="aliases a selected input"):
        dataset_main(argv)
    assert source.read_bytes() == original and not (tmp_path / "output").exists()


@pytest.mark.parametrize("dataset_id", ["../escape", "/absolute", "", ".", "with/slash", "with\\slash"])
def test_dataset_id_is_a_single_filename(tmp_path: Path, dataset_id: str) -> None:
    """A dataset identifier cannot redirect its manifest outside the output."""
    destination = tmp_path / "output"
    with pytest.raises(ValueError, match="filename identifier"):
        build_dataset(
            [{"shot_id": 101, "channels": channels()}],
            dataset_id=dataset_id,
            out_dir=destination,
            retrieved_at=STAMP,
            generated_at=STAMP,
        )
    assert not destination.exists()


def test_invalid_later_shot_is_refused_before_first_write(tmp_path: Path) -> None:
    """Validate the entire batch before producing an earlier valid shot."""
    valid = channels()
    invalid = channels()
    invalid["time_s"][1] = invalid["time_s"][0]
    with pytest.raises(ValueError, match="strictly increasing"):
        build_dataset(
            [{"shot_id": 101, "channels": valid}, {"shot_id": 102, "channels": invalid}],
            dataset_id="candidate",
            out_dir=tmp_path / "output",
            retrieved_at=STAMP,
            generated_at=STAMP,
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("failure", ["empty", "missing_retrieval", "duplicate_ids"])
def test_dataset_batch_metadata_is_refused_before_publication(tmp_path: Path, failure: str) -> None:
    """Refuse empty batches, absent acquisition labels and duplicate shot identities."""
    shots = [{"shot_id": 101, "channels": channels()}]
    retrieval = STAMP
    if failure == "empty":
        shots = []
    elif failure == "missing_retrieval":
        retrieval = " "
    else:
        shots.append({"shot_id": 101, "channels": channels()})
    with pytest.raises(ValueError):
        build_dataset(
            shots,
            dataset_id="candidate",
            out_dir=tmp_path / "output",
            retrieved_at=retrieval,
            generated_at=STAMP,
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("identity", [True, 0, -1, 1.5, 2**63])
def test_shot_writer_identity_domain(tmp_path: Path, identity: object) -> None:
    """Refuse invalid shot identities through the actual public writer."""
    from collections.abc import Callable

    writer: Callable[..., object] = build_shot_npz
    with pytest.raises(ValueError, match="positive signed-int64 integer"):
        writer(identity, channels(), out_dir=tmp_path / "output", drop_fraction=0.8, quench_window_ms=5.0)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("key", ["summary.time", "equilibrium.time", "magnetics.time_saddle"])
def test_mirror_clocks_are_chronological(key: str) -> None:
    """Public mirror derivation refuses duplicate group time samples."""
    values = mirror()
    values[key][1] = values[key][0]
    with pytest.raises(ValueError, match="strictly increasing"):
        derive_replay_channels(values, locked_window=3)


@pytest.mark.parametrize("names", [["shot_x.npz"], ["shot_0.npz"], ["shot_1.npz", "shot_01.npz"]])
def test_source_filename_inventory_precedes_publication(tmp_path: Path, names: list[str]) -> None:
    """Refuse ambiguous/invalid identities before producing an archive."""
    material = tmp_path / "material"
    material.mkdir()
    for name in names:
        save_npz_arrays(material / name, mirror(), allow_pickle=False)
    with pytest.raises(ValueError, match="identit"):
        build_channels(material, out_dir=tmp_path / "output", generated_at=STAMP, locked_window=3)
    assert not (tmp_path / "output").exists()
