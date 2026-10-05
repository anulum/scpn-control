#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Capture and render the Paper 27 phase-model trajectory.

"""Create a fresh GIF/MP4 model visualization and its displayed-value JSON.

The CLI uses a fresh caller-relative phase-video directory, preserving existing
documentation media. Both formats are required by default; --gif-only explicitly
selects GIF when MP4 is unnecessary. Model dt=0.001 is independent of playback
FPS. Model guard PASS/HALT is not a reactor protection or actuation decision.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.phase_video_model import capture_trajectory
from tools.phase_video_rendering import VideoResult, render_trajectory, validate_render_options


def generate(
    n_ticks: int,
    L: int,
    N_per: int,
    zeta: float,
    fps: int,
    *,
    output_dir: Path = Path("phase-video"),
    gif_only: bool = False,
    ffmpeg_path: Path | None = None,
) -> VideoResult:
    """Capture the actual seeded monitor and render a new diagnostic bundle.

    Parameters
    ----------
    n_ticks, L, N_per : int
        Positive, nonboolean tick/layer/population counts. The original model
        factory remains unchanged, with seed 42, external driver 0 and dt=0.001.
    zeta : float
        Finite nonnegative dimensionless gain in the declared phase model.
    fps : int
        Nonboolean playback rate in [1, 100], separate from model time.
    output_dir : pathlib.Path
        New caller-relative or absolute directory with an existing parent.
        Existing directories, files and symlinks refuse before model execution.
    gif_only : bool
        Explicit GIF-only output. Otherwise require actual FFmpeg and both files.
    ffmpeg_path : pathlib.Path or None
        Encoder executable path or PATH name; None resolves ffmpeg. Ignored for
        GIF-only output. No encoder is installed or emulated.

    Returns
    -------
    VideoResult
        Actual file paths and zero-based selected sample indices. The JSON
        records displayed model metrics and false physical-reference admission.

    Raises
    ------
    ValueError
        Model/display parameters or captured snapshots are invalid.
    OSError
        Output/encoder lookup, creation or writing failed. Existing artifacts
        are preserved; a later failure may leave a new partial bundle.
    subprocess.CalledProcessError
        Native encoding failed. There is no automatic GIF-only fallback.
    """
    validate_render_options(output_dir, fps, gif_only, ffmpeg_path)
    trajectory = capture_trajectory(n_ticks, L, N_per, zeta)
    return render_trajectory(trajectory, output_dir, fps=fps, gif_only=gif_only, ffmpeg_path=ffmpeg_path)


def main(argv: Sequence[str] | None = None) -> int:
    """Return 0 for a completed bundle or fixed caller-safe refusal 2; argparse owns help/usage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--n-per", type=int, default=50)
    parser.add_argument("--zeta", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("phase-video"))
    parser.add_argument("--gif-only", action="store_true", help="Explicitly produce GIF without requiring FFmpeg")
    parser.add_argument("--ffmpeg-path", type=Path, help="Actual MP4 encoder executable; default resolves ffmpeg")
    args = parser.parse_args(argv)
    try:
        result = generate(
            args.ticks,
            args.layers,
            args.n_per,
            args.zeta,
            args.fps,
            output_dir=args.output_dir,
            gif_only=args.gif_only,
            ffmpeg_path=args.ffmpeg_path,
        )
    except Exception:
        print("Phase video refused: model input, output directory or rendering is invalid.", file=sys.stderr)
        return 2
    print(f"Created {result.gif_path}")
    if result.mp4_path is not None:
        print(f"Created {result.mp4_path}")
    print(f"Created {result.metadata_path} ({len(result.frame_indices)} sampled frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
