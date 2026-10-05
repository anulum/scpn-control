# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Actual Pillow/FFmpeg phase-model video rendering.

"""Render validated model metrics with real encoders into a new output directory.

An unmanaged FigureCanvasAgg avoids switching pyplot's backend. Temporary style
and FFmpeg rc settings restore on exit; Matplotlib global rc contexts still make
concurrent rendering unsupported. Artifacts are written sequentially, not as an
atomic bundle; failure can leave partial files. No physical or performance
admission follows from rendering or the displayed monitor-tick latency.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from matplotlib import style
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.artist import Artist
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from tools.phase_video_model import PhaseTrajectory, sample_frame_indices


@dataclass(frozen=True)
class VideoResult:
    """Written GIF, optional MP4, displayed-value JSON and zero-based frame indices."""

    gif_path: Path
    mp4_path: Path | None
    metadata_path: Path
    frame_indices: tuple[int, ...]


def validate_render_options(output_dir: Path, fps: int, gif_only: bool, ffmpeg_path: Path | None) -> str | None:
    """Validate a fresh destination and actual requested encoder without writing.

    Require fps in [1,100], a boolean gif_only flag, a nonexistent directory path
    (including no dangling symlink), and an existing directory parent. GIF-only
    ignores ffmpeg_path. Otherwise shutil.which resolves that path/name or
    ffmpeg on current PATH and a missing/nonexecutable encoder refuses. Resolution
    is not codec or executable-provenance certification; real encoding can fail.
    """
    sample_frame_indices(1, fps)
    if not isinstance(gif_only, bool):
        raise ValueError("GIF-only selection must be boolean")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Phase video output directory already exists")
    if not output_dir.absolute().parent.is_dir():
        raise FileNotFoundError("Phase video output parent does not exist")
    if gif_only:
        return None
    encoder = shutil.which(str(ffmpeg_path) if ffmpeg_path is not None else "ffmpeg")
    if encoder is None:
        raise FileNotFoundError("Requested FFmpeg encoder is unavailable")
    return encoder


def render_trajectory(
    trajectory: PhaseTrajectory,
    output_dir: Path,
    *,
    fps: int = 20,
    gif_only: bool = False,
    ffmpeg_path: Path | None = None,
) -> VideoResult:
    """Render all selected model samples with real Pillow and optional FFmpeg.

    Validate display inputs and options before exclusive directory creation.
    Model-error snapshots refuse; guard HALT remains visibly labelled as a model
    diagnostic. Trace x coordinates are actual ticks1..n, correcting the old
    zero-based trace offset. The original floor-stride/terminal-frame policy is
    unchanged. GIF centisecond timing can differ from ideal frame_count/fps;
    MP4 uses requested fps, codec h264, bitrate2000 and Matplotlib's encoder args.

    Returns absolute paths and selected zero-based indices. phase_video.json
    contains the displayed-field projection, sampled ticks, model dt, ideal
    playback duration and false physical-reference admission. Provenance/numerical
    truth of supplied snapshots is not established. Expected validation failures
    raise ValueError, filesystem/lookup failures OSError, and native encoder
    failures can raise subprocess.CalledProcessError. Existing data is preserved;
    partial newly created bundles remain after later failures. Figure resources
    and rc contexts are released/restored in success and error paths.
    """
    trajectory.validate()
    encoder = validate_render_options(output_dir, fps, gif_only, ffmpeg_path)
    output_dir = output_dir.absolute()
    indices = sample_frame_indices(len(trajectory.snapshots), fps)
    scalar_keys = ("tick", "R_global", "V_global", "lambda_exp", "latency_us", "guard_approved")
    snapshots = [
        {**{key: sample[key] for key in scalar_keys}, "R_layer": list(sample["R_layer"])}
        for sample in trajectory.snapshots
    ]
    r_global = [sample["R_global"] for sample in snapshots]
    v_global = [sample["V_global"] for sample in snapshots]
    lam_exp = [sample["lambda_exp"] for sample in snapshots]
    r_layer_all = np.asarray([sample["R_layer"] for sample in snapshots], dtype=np.float64)
    n_ticks = len(snapshots)
    output_dir.mkdir()
    gif_path = output_dir / "phase_sync_live.gif"
    mp4_path = None if gif_only else output_dir / "phase_sync_live.mp4"
    metadata_path = output_dir / "phase_video.json"
    with style.context("dark_background"), matplotlib.rc_context({"animation.ffmpeg_path": encoder or "ffmpeg"}):
        fig = Figure(figsize=(12, 7), facecolor="#0f172a")
        FigureCanvasAgg(fig)
        try:
            axes = fig.subplots(2, 2)
            for ax in axes.flat:
                ax.set_facecolor("#1e293b")
                ax.tick_params(colors="#94a3b8", labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color("#334155")
            fig.suptitle(
                f"SCPN Phase Model — {trajectory.L} layers × {trajectory.N_per} osc, ζ={trajectory.zeta}",
                color="#e2e8f0",
                fontsize=13,
                fontweight="bold",
            )
            ax_r, ax_v, ax_l, ax_b = axes.flat
            (line_r,) = ax_r.plot([], [], color="#3b82f6", linewidth=1.5, marker=".", markersize=3)
            ax_r.set(xlim=(0, n_ticks), ylim=(0, 1.05), ylabel="R_global", title="Global Coherence")
            ax_r.axhline(0.9, color="#22c55e", linewidth=0.5, linestyle="--", alpha=0.5)
            (line_v,) = ax_v.plot([], [], color="#8b5cf6", linewidth=1.5, marker=".", markersize=3)
            ax_v.set(
                xlim=(0, n_ticks), ylim=(0, max(v_global) * 1.1 + 0.01), ylabel="V_global", title="Model Lyapunov V(t)"
            )
            (line_l,) = ax_l.plot([], [], color="#f59e0b", linewidth=1.5, marker=".", markersize=3)
            lam_min = min(lam_exp) * 1.2 if min(lam_exp) < 0 else -1
            ax_l.set(
                xlim=(0, n_ticks),
                ylim=(lam_min, max(max(lam_exp) * 1.2, 0.5)),
                ylabel="λ / model time",
                xlabel="tick",
                title="Finite-history Model Exponent",
            )
            ax_l.axhline(0, color="#ef4444", linewidth=0.5, linestyle="--", alpha=0.5)
            bars = ax_b.bar(np.arange(trajectory.L), np.zeros(trajectory.L), color="#3b82f6", width=0.7, alpha=0.85)
            ax_b.set(
                xlim=(-0.5, trajectory.L - 0.5),
                ylim=(0, 1.05),
                ylabel="R_layer",
                xlabel="Layer",
                title="Per-Layer Coherence",
            )
            if trajectory.L <= 16:
                ax_b.set_xticks(range(0, trajectory.L, max(1, trajectory.L // 8)))
            metrics_text = fig.text(
                0.5, 0.01, "", ha="center", va="bottom", color="#94a3b8", fontsize=10, fontfamily="monospace"
            )
            fig.tight_layout(rect=(0, 0.04, 1, 0.95))
            cmap = matplotlib.colormaps["Blues"]

            def update(frame_idx: int) -> list[Artist]:
                """Update one actual selected model sample without claiming reactor protection."""
                idx = indices[frame_idx]
                t = idx + 1
                line_r.set_data(range(1, t + 1), r_global[:t])
                line_v.set_data(range(1, t + 1), v_global[:t])
                line_l.set_data(range(1, t + 1), lam_exp[:t])
                for bar, value in zip(bars, r_layer_all[idx], strict=True):
                    bar.set_height(value)
                    bar.set_color(cmap(0.3 + 0.7 * value))
                sample = snapshots[idx]
                guard = "PASS" if sample["guard_approved"] else "HALT"
                metrics_text.set_text(
                    f"tick {t}/{n_ticks}    R={sample['R_global']:.3f}    V={sample['V_global']:.3f}    λ={sample['lambda_exp']:.3f}    model guard={guard}    tick={sample['latency_us']:.0f}µs"
                )
                return [line_r, line_v, line_l, metrics_text, *bars]

            # Figure-level footer text must participate in the actual saved frames.
            animation = FuncAnimation(fig, update, frames=len(indices), interval=1000 / fps, blit=False)
            animation.save(str(gif_path), writer=PillowWriter(fps=fps))
            if mp4_path is not None:
                animation.save(str(mp4_path), writer=FFMpegWriter(fps=fps, codec="h264", bitrate=2000))
        finally:
            fig.clear()
    payload: dict[str, Any] = {
        "schema_version": "scpn-control.phase-video.v1",
        "physical_reference_admitted": False,
        "model": {"L": trajectory.L, "N_per": trajectory.N_per, "zeta": trajectory.zeta, "dt": trajectory.dt},
        "fps": fps,
        "sampled_ticks": [index + 1 for index in indices],
        "ideal_playback_seconds": len(indices) / fps,
        "model_elapsed_time": n_ticks * trajectory.dt,
        "gif": gif_path.name,
        "mp4": mp4_path.name if mp4_path is not None else None,
        "displayed_metrics": snapshots,
    }
    metadata_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return VideoResult(gif_path, mp4_path, metadata_path, indices)
