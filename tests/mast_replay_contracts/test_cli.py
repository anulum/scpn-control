# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Native MAST candidate CLI refusal and parser contracts.
"""Run genuine module commands with local archives and filesystem failures."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scpn_control._npz import save_npz_arrays

from ._fixtures import STAMP, mirror, write_channels
from .test_archives import arguments

ROOT = Path(__file__).resolve().parents[2]


def run_command(module: str, argv: list[str], temporary: Path) -> subprocess.CompletedProcess[str]:
    """Execute the installed Python's real module command with bounded local I/O."""
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "src"))),
        TMPDIR=str(temporary),
        TMP=str(temporary),
        TEMP=str(temporary),
        PYTHONDONTWRITEBYTECODE="1",
    )
    return subprocess.run(
        [sys.executable, "-m", module, *argv],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=45,
    )


@pytest.mark.parametrize(
    "module",
    [
        "validation.build_disruption_replay_channels",
        "validation.build_mast_disruption_dataset",
    ],
)
def test_help_and_parser_exits(module: str, tmp_path: Path) -> None:
    """Help succeeds and missing required arguments fail without a traceback."""
    help_result = run_command(module, ["--help"], tmp_path)
    assert help_result.returncode == 0 and "usage:" in help_result.stdout
    missing = run_command(module, [], tmp_path)
    assert missing.returncode == 2 and "required" in missing.stderr
    assert "Traceback" not in missing.stderr


@pytest.mark.parametrize("stage", ["replay", "dataset"])
@pytest.mark.parametrize("failure", ["invalid_input", "output_is_file", "input_alias"])
def test_native_refusal_preserves_input_and_authored_text(
    tmp_path: Path,
    stage: str,
    failure: str,
) -> None:
    """Native input/path/I/O refusals preserve source bytes and expose fixed text."""
    output = tmp_path / "output"
    if stage == "replay":
        source = tmp_path / "material"
        source.mkdir()
        source_file = source / "shot_101.npz"
        save_npz_arrays(source_file, mirror(), allow_pickle=False)
        argv = [
            "--material-dir",
            str(source),
            "--out-dir",
            str(output),
            "--json-out",
            str(output / "report.json"),
            "--locked-window",
            "3",
            "--generated-at",
            STAMP,
        ]
        module = "validation.build_disruption_replay_channels"
        expected = "Could not build MAST replay channels.\n"
        if failure == "invalid_input":
            argv[argv.index("--locked-window") + 1] = "2"
    else:
        source_file = tmp_path / "channels.npz"
        write_channels(source_file)
        argv = arguments(source_file, output)
        module = "validation.build_mast_disruption_dataset"
        expected = "Could not build the MAST candidate dataset.\n"
        if failure == "invalid_input":
            source_file.write_bytes(b"not an NPZ archive")
    if failure == "output_is_file":
        output.write_bytes(b"existing output marker")
    if failure == "input_alias":
        argv[argv.index("--json-out") + 1] = str(source_file)
    before = source_file.read_bytes()
    result = run_command(module, argv, tmp_path)
    assert result.returncode == 2 and result.stderr == expected
    assert source_file.read_bytes() == before
    if failure == "output_is_file":
        assert output.read_bytes() == b"existing output marker"
