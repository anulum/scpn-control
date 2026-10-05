# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real phase-video encoders, output custody and command contracts.

"""Exercise actual GIF/FFmpeg files and real CLI refusals without backend mocks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
import pytest
from PIL import Image
from PIL.GifImagePlugin import GifImageFile

from tools.generate_phase_video import ROOT, generate, main
from tools.phase_video_model import capture_trajectory
from tools.phase_video_rendering import VideoResult, render_trajectory, validate_render_options

SCRIPT = ROOT / "tools/generate_phase_video.py"


def _cli(cwd: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Invoke the actual checkout source entry without inherited import or tracer flags."""
    env = os.environ.copy()
    for key in ("PYTHONPATH", "COVERAGE_PROCESS_START", "COVERAGE_PROCESS_CONFIG"):
        env.pop(key, None)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments], cwd=cwd, env=env, capture_output=True, text=True, check=False
    )


@pytest.fixture(scope="module")
def actual_bundle(tmp_path_factory: pytest.TempPathFactory) -> VideoResult:
    """Render the real eight-tick seeded model through both installed encoders."""
    output = tmp_path_factory.mktemp("actual-phase-encoders") / "bundle"
    return generate(8, 2, 3, 0.5, 5, output_dir=output)


def test_actual_gif_and_mp4_bind_to_displayed_model_values(actual_bundle: VideoResult) -> None:
    """Decode actual files and independently replay every value in the written display projection."""
    assert actual_bundle.mp4_path is not None
    payload = json.loads(actual_bundle.metadata_path.read_text())
    assert payload["physical_reference_admitted"] is False and payload["model"]["dt"] == 0.001
    assert payload["sampled_ticks"] == list(range(1, 9))
    assert payload["model_elapsed_time"] == 0.008 and payload["ideal_playback_seconds"] == 1.6
    expected = capture_trajectory(8, 2, 3, 0.5)
    for actual, sample in zip(payload["displayed_metrics"], expected.snapshots, strict=True):
        for key in ("tick", "R_global", "R_layer", "V_global", "lambda_exp", "guard_approved"):
            assert actual[key] == sample[key]
    with Image.open(actual_bundle.gif_path) as gif:
        assert isinstance(gif, GifImageFile)
        assert gif.format == "GIF" and gif.n_frames == 8 and gif.size == (1200, 700)
        assert gif.info["duration"] == 200
        for index in (0, gif.n_frames - 1):
            gif.seek(index)
            footer = gif.convert("RGB").crop((0, 675, 1200, 700))
            assert footer.getextrema() != ((15, 15), (23, 23), (42, 42))
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_frames",
            "-of",
            "json",
            str(actual_bundle.mp4_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["codec_name"] == "h264" and stream["width"] == 1200 and stream["height"] == 700
    assert stream["r_frame_rate"] == "5/1" and stream["nb_frames"] == "8"


@pytest.mark.parametrize("ticks,layers,population,zeta", [(1, 17, 1, 0.0), (3, 1, 3, 0.0)])
def test_real_gif_only_nonnegative_and_negative_exponent_layouts(
    tmp_path: Path, ticks: int, layers: int, population: int, zeta: float
) -> None:
    """Render actual one-/many-layer models with explicit GIF-only output and restored rc state."""
    before = dict(matplotlib.rcParams)
    output = generate(ticks, layers, population, zeta, 5, output_dir=tmp_path / "gif-only", gif_only=True)
    assert output.mp4_path is None and json.loads(output.metadata_path.read_text())["mp4"] is None
    with Image.open(output.gif_path) as gif:
        assert isinstance(gif, GifImageFile)
        assert gif.n_frames == ticks
        footer = gif.convert("RGB").crop((0, 675, 1200, 700))
        assert footer.getextrema() != ((15, 15), (23, 23), (42, 42))
    assert dict(matplotlib.rcParams) == before


def test_public_main_and_actual_relative_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Use the genuine imported main for completed GIF-only output and caller-safe repeat refusal."""
    output = tmp_path / "main-bundle"
    args = ["--ticks", "1", "--layers", "1", "--n-per", "1", "--output-dir", str(output), "--gif-only"]
    assert main(args) == 0
    before = (output / "phase_video.json").read_bytes()
    assert main(args) == 2 and (output / "phase_video.json").read_bytes() == before
    messages = capsys.readouterr()
    assert "Created" in messages.out and "refused" in messages.err
    result = _cli(tmp_path, "--ticks", "1", "--layers", "1", "--n-per", "1", "--gif-only")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "phase-video/phase_sync_live.gif").is_file()


@pytest.mark.parametrize(
    "arguments,expected",
    [
        (("--help",), 0),
        (("--unknown-option",), 2),
        (("--ticks", "1.5"), 2),
        (("--ticks", "0"), 2),
        (("--fps", "0"), 2),
        (("--zeta", "nan"), 2),
    ],
)
def test_actual_parser_and_domain_refusals(tmp_path: Path, arguments: tuple[str, ...], expected: int) -> None:
    """Run real help/usage/domain errors outside the checkout before any bundle is created."""
    result = _cli(tmp_path, *arguments)
    assert result.returncode == expected
    assert "Traceback" not in result.stderr and not (tmp_path / "phase-video").exists()


@pytest.mark.parametrize("kind", ["directory", "file", "symlink", "dangling", "missing_parent", "encoder"])
def test_real_cli_preserves_existing_targets_and_refuses_missing_inputs(tmp_path: Path, kind: str) -> None:
    """Exercise real filesystem/encoder lookup inputs without a backend or outcome substitute."""
    output = tmp_path / "bundle"
    original = tmp_path / "original.txt"
    original.write_text("preserve source custody")
    args = ["--ticks", "1", "--layers", "1", "--n-per", "1"]
    if kind == "directory":
        output.mkdir()
    elif kind == "file":
        output.write_text("preserve existing file")
    elif kind == "symlink":
        output.symlink_to(original)
    elif kind == "dangling":
        output.symlink_to(tmp_path / "absent")
    elif kind == "missing_parent":
        output = tmp_path / "absent-parent/bundle"
    else:
        args += ["--ffmpeg-path", str(tmp_path / "absent-encoder")]
    before = original.read_bytes()
    with pytest.raises(OSError):
        generate(
            1, 1, 1, 0.5, 5, output_dir=output, ffmpeg_path=tmp_path / "absent-encoder" if kind == "encoder" else None
        )
    result = _cli(tmp_path, *args, "--output-dir", str(output))
    assert result.returncode == 2
    assert result.stderr == "Phase video refused: model input, output directory or rendering is invalid.\n"
    assert original.read_bytes() == before
    if kind in ("missing_parent", "encoder"):
        assert not output.exists()


def test_public_option_validation_boolean_and_actual_encoder_path(tmp_path: Path) -> None:
    """Resolve the real installed encoder and reject a nonboolean format selector through its public API."""
    encoder = shutil.which("ffmpeg")
    assert encoder is not None
    assert validate_render_options(tmp_path / "one", 20, False, Path(encoder)) == encoder
    assert validate_render_options(tmp_path / "two", 20, True, tmp_path / "absent-ignored-encoder") is None
    with pytest.raises(ValueError, match="boolean"):
        validate_render_options(tmp_path / "three", 20, json.loads("0"), None)


def test_actual_native_misconfiguration_restores_style_and_retains_partial_bundle(tmp_path: Path) -> None:
    """A real ffprobe mistakenly configured as encoder fails natively, without mocking FFmpeg outcomes."""
    executable = shutil.which("ffprobe")
    assert executable is not None
    before = dict(matplotlib.rcParams)
    output = tmp_path / "native-refusal"
    with pytest.raises(subprocess.CalledProcessError):
        render_trajectory(capture_trajectory(1, 1, 1, 0), output, fps=5, ffmpeg_path=Path(executable))
    assert (output / "phase_sync_live.gif").is_file()
    assert not (output / "phase_video.json").exists()
    assert dict(matplotlib.rcParams) == before


def test_actual_render_includes_appended_terminal_sample(tmp_path: Path) -> None:
    """Render the real floor-stride trajectory and bind its appended final frame to metadata."""
    result = generate(42, 2, 3, 0.5, 2, output_dir=tmp_path / "sampled", gif_only=True)
    assert result.frame_indices == (*range(0, 42, 2), 41)
    payload = json.loads(result.metadata_path.read_text())
    assert payload["sampled_ticks"] == [index + 1 for index in result.frame_indices]
    assert payload["displayed_metrics"][-1]["tick"] == 42
    with Image.open(result.gif_path) as gif:
        assert isinstance(gif, GifImageFile)
        assert gif.n_frames == len(result.frame_indices) and gif.info["duration"] == 500
